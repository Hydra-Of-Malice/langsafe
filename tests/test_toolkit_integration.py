"""CPU regressions at the pinned-toolkit and llama.cpp process boundaries."""
import hashlib
import io
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from langsafe.quant import gguf
from langsafe.repair import arms
from langsafe.circuits import discover as circuits
from langsafe.circuits import task


def fake_server(monkeypatch, failures):
    """urlopen that raises each failure in turn, then answers."""
    calls = []

    def urlopen(request, timeout):
        calls.append(request.full_url)
        if len(calls) <= len(failures):
            raise failures[len(calls) - 1]
        return io.BytesIO(b'{"ok": 1}')

    monkeypatch.setattr(gguf.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(gguf.time, "sleep", lambda seconds: None)
    return calls


def http_error(code):
    return gguf.urllib.error.HTTPError("http://x", code, "no", {}, None)


def test_gguf_post_survives_a_dropped_local_connection(monkeypatch):
    calls = fake_server(monkeypatch, [gguf.urllib.error.URLError(TimeoutError(10060, "timed out")), http_error(503)])
    assert gguf._post("http://x/completion", {}) == {"ok": 1}
    assert len(calls) == 3


def test_gguf_post_gives_up_when_the_connection_stays_down(monkeypatch):
    calls = fake_server(monkeypatch, [ConnectionResetError(10054, "reset")] * 5)
    with pytest.raises(ConnectionResetError):
        gguf._post("http://x/completion", {})
    assert len(calls) == 5


def test_gguf_post_does_not_repeat_a_rejected_request(monkeypatch):
    calls = fake_server(monkeypatch, [http_error(400)])
    with pytest.raises(gguf.urllib.error.HTTPError):
        gguf._post("http://x/completion", {})
    assert len(calls) == 1


class Tokenizer:
    bos_token = "<BOS_TOKEN>"
    chat_template = "tiny-aya"
    padding_side = "right"

    def __len__(self):
        return 128

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def apply_chat_template(self, msgs, tokenize, add_generation_prompt):
        return ("<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>safe<|END_OF_TURN_TOKEN|>"
                f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{msgs[0]['content']}<|END_OF_TURN_TOKEN|>"
                "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>")


ITEM = {"id": "q1", "question": "2+2?", "option_a": "3", "option_b": "4", "option_c": "5",
        "option_d": "6", "answer": "B"}


def module(monkeypatch, name, **exports):
    obj = ModuleType(name)
    obj.__dict__.update(exports)
    monkeypatch.setitem(sys.modules, name, obj)
    return obj


def test_gguf_fingerprint_streams_and_includes_bytes_after_the_header(tmp_path, monkeypatch):
    a, b = tmp_path / "a.gguf", tmp_path / "b.gguf"
    prefix = b"x" * (1 << 24)
    a.write_bytes(prefix + b"a")
    b.write_bytes(prefix + b"b")

    def forbidden_read_bytes(self):
        raise AssertionError("GGUF weights must be hashed in bounded chunks")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read_bytes)
    one = gguf.GGUFGenerator(Tokenizer(), {}, "m", str(a), cache_dir=tmp_path / "cache")
    two = gguf.GGUFGenerator(Tokenizer(), {}, "m", str(b), cache_dir=tmp_path / "cache")
    assert one.fp == hashlib.sha256(prefix + b"a").hexdigest()
    assert one.fp != two.fp


def test_gguf_cache_changes_when_hf_prompt_rendering_changes(tmp_path):
    from langsafe.data.records import Prompt

    class MutableTokenizer(Tokenizer):
        def apply_chat_template(self, *args, **kwargs):
            return super().apply_chat_template(*args, **kwargs) + self.chat_template

    tokenizer = MutableTokenizer()
    checkpoint = tmp_path / 'model.gguf'
    checkpoint.write_bytes(b'unchanged-gguf')
    generator = gguf.GGUFGenerator(tokenizer, {}, 'q4', str(checkpoint), cache_dir=tmp_path / 'cache')
    prompt = Prompt(id='g::en', group='g', lang='en', kind='benign', text='Hello', source='test', source_id='g')
    before = generator._key(prompt, 'default')
    tokenizer.chat_template = 'updated-chat-template'
    assert generator._key(prompt, 'default') != before


def test_gguf_scores_all_exact_candidates_and_honors_preamble(tmp_path, monkeypatch):
    payloads = []

    def post(url, body):
        if url.endswith("/tokenize"):
            return {"tokens": [ord(body["content"])]}
        payloads.append(body)
        return {"content": "A", "completion_probabilities": [{"id": 65, "token": "A", "prob": .1,
                "top_probs": [{"id": 65, "token": "A", "prob": .1},
                              {"id": 66, "token": "B", "prob": .7},
                              {"id": 67, "token": "C", "prob": .1},
                              {"id": 68, "token": "D", "prob": .1}]}]}

    monkeypatch.setattr(gguf, "_post", post)
    result = gguf.gguf_mmlu_accuracy(Tokenizer(), [ITEM], preamble="none")
    assert result["accuracy"] == 1.0 and result["per_item"] == [1]
    assert result["outcomes"][0]["prediction"] == "B"
    assert result["complete"] and result["n_unscorable"] == 0
    assert "SYSTEM_TOKEN" not in ''.join(chr(token) for token in payloads[0]["prompt"])
    assert payloads[0]["n_predict"] == 1
    assert payloads[0]["post_sampling_probs"] is True
    assert payloads[0]["samplers"] == []
    banned = {token_id for token_id, bias in payloads[0]["logit_bias"] if bias is False}
    assert banned == set(range(128)) - {65, 66, 67, 68}


def test_gguf_missing_letter_is_unscorable_instead_of_incorrect(monkeypatch):
    def post(url, body):
        if url.endswith("/tokenize"):
            return {"tokens": [ord(body["content"])]}
        return {"content": "A", "completion_probabilities": [{"top_probs": [
            {"id": 65, "token": "A", "prob": .2}, {"id": 66, "token": "B", "prob": .1}]}]}

    monkeypatch.setattr(gguf, "_post", post)
    result = gguf.gguf_mmlu_accuracy(Tokenizer(), [ITEM])
    assert result["accuracy"] is None and result["per_item"] == [None]
    assert result["n_scored"] == 0 and result["n_unscorable"] == 1
    assert not result["complete"] and result["outcomes"][0]["status"] == "unscorable"


def test_gguf_uses_exact_hf_input_ids_without_server_inserting_another_bos(monkeypatch, tmp_path):
    from langsafe.data.records import Prompt

    class BoundaryTokenizer(Tokenizer):
        def encode(self, text, add_special_tokens=False):
            if text in 'ABCD':
                return [ord(text)]
            assert add_special_tokens is False
            return [2, 71, 99]

    payloads = []

    def post(url, body):
        if url.endswith('/tokenize'):
            return {'tokens': [ord(body['content'])]}
        payloads.append(body)
        return {'content': 'B', 'tokens_predicted': 1, 'stop_type': 'eos',
                'completion_probabilities': [{'top_probs': [
                    {'id': 65, 'prob': .1}, {'id': 66, 'prob': .7},
                    {'id': 67, 'prob': .1}, {'id': 68, 'prob': .1}]}]}

    monkeypatch.setattr(gguf, '_post', post)
    tokenizer = BoundaryTokenizer()
    assert gguf.gguf_mmlu_accuracy(tokenizer, [ITEM])['complete']
    checkpoint = tmp_path / 'm.gguf'
    checkpoint.write_bytes(b'weights')
    generator = gguf.GGUFGenerator(tokenizer, {'max_new_tokens': 8}, 'q4', str(checkpoint), cache_dir=tmp_path / 'cache')
    prompt = Prompt(id='g::en', group='g', lang='en', kind='benign', text='Say B', source='test', source_id='g', split='heldout')
    row = generator.run([prompt], 'default')[0]
    assert [body['prompt'] for body in payloads] == [[2, 71, 99], [2, 71, 99]]
    assert row['split'] == 'heldout' and row['group'] == 'g'


def test_gguf_server_tokenizer_mismatch_fails_closed(monkeypatch):
    monkeypatch.setattr(gguf, "_post", lambda url, body: {"tokens": [999]})
    result = gguf.gguf_mmlu_accuracy(Tokenizer(), [ITEM])
    assert result["n_unscorable"] == 1 and "token" in result["outcomes"][0]["reason"]


def test_gguf_conversion_and_quantization_use_checked_argument_lists(tmp_path, monkeypatch):
    kit = tmp_path / "llama cpp"
    (kit / "build/bin").mkdir(parents=True)
    (kit / "convert_hf_to_gguf.py").touch()
    (kit / "build/bin/llama-quantize").touch()
    checkpoint = tmp_path / "model with spaces"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    fp16, q4 = tmp_path / "model f16.gguf", tmp_path / "model q4.gguf"
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        assert isinstance(command, list) and kwargs["check"] is True
        assert not kwargs.get("shell", False)
        Path(command[command.index("--outfile") + 1] if "--outfile" in command else command[2]).write_bytes(b"gguf")

    monkeypatch.setattr("subprocess.run", run)
    assert gguf.convert_hf_to_gguf(checkpoint, fp16, kit) == fp16
    assert gguf.quantize_gguf(fp16, q4, kit) == q4
    assert commands[0] == [sys.executable, str(kit / "convert_hf_to_gguf.py"), str(checkpoint),
                           "--outtype", "f16", "--outfile", str(fp16.with_suffix('.gguf.partial'))]
    assert commands[1] == [str(kit / "build/bin/llama-quantize"), str(fp16), str(q4.with_suffix('.gguf.partial')), "Q4_K_M"]


@pytest.mark.parametrize('generation_eos,expected', [(3, 3), (None, 2)])
def test_gguf_export_preserves_hf_generation_eos_over_tokenizer_eot(tmp_path, monkeypatch,
                                                                 generation_eos, expected):
    library = pytest.importorskip('gguf')
    kit = tmp_path / 'llama'
    kit.mkdir()
    (kit / 'convert_hf_to_gguf.py').touch()
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'config.json').write_text(json.dumps({'eos_token_id': 2}))
    tokenizer_config = checkpoint / 'tokenizer_config.json'
    tokenizer_config.write_text(json.dumps({'eos_token': '<|END_OF_TURN_TOKEN|>'}))
    if generation_eos is not None:
        (checkpoint / 'generation_config.json').write_text(json.dumps({'eos_token_id': generation_eos}))

    def convert(command, **kwargs):
        path = Path(command[command.index('--outfile') + 1])
        writer = library.GGUFWriter(path, 'cohere2')
        writer.add_token_list([f'token-{i}' for i in range(8)])
        writer.add_eos_token_id(6)  # Converter takes EOT from tokenizer_config, unlike HF generation.
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    monkeypatch.setattr(subprocess, 'run', convert)
    output = gguf.convert_hf_to_gguf(checkpoint, tmp_path / 'export.gguf', kit)
    reader = library.GGUFReader(output)
    assert reader.get_field('tokenizer.ggml.eos_token_id').contents() == expected
    assert json.loads(tokenizer_config.read_text())['eos_token'] == '<|END_OF_TURN_TOKEN|>'


def test_converter_resolves_cached_model_without_optional_hub_assets(tmp_path, monkeypatch):
    from fnmatch import fnmatch
    import huggingface_hub
    from langsafe.cli import _local_checkpoint

    def snapshot(model, *, local_files_only, allow_patterns=None):
        assert local_files_only is True
        names = ('config.json', 'model-00001-of-00002.safetensors',
                 'model.safetensors.index.json', 'tokenizer.json', 'chat_template.jinja')
        if allow_patterns is None:
            raise FileNotFoundError('Optional README and images are not cached')
        assert all(any(fnmatch(name, pattern) for pattern in allow_patterns) for name in names)
        assert not any(fnmatch(name, pattern) for name in ('README.md', 'assets/model.png')
                       for pattern in allow_patterns)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, 'snapshot_download', snapshot)
    assert _local_checkpoint('CohereLabs/tiny-aya-global') == tmp_path


def test_circuit_discovery_reads_report_and_the_current_artifact(monkeypatch, tmp_path):
    class Pipeline:
        @classmethod
        def from_custom_data(cls, model_name, data_path, **kwargs):
            assert model_name == "local/model"
            result = cls()
            result.out = Path(kwargs["output_dir"])
            result._custom_data_cfg = {'type': 'template', 'path': data_path}
            return result

        def discover(self, **kwargs):
            assert self._custom_data_cfg.get('max_records') == kwargs['n_examples']
            self.out.mkdir(parents=True, exist_ok=True)
            self.artifact_path = str(self.out / "current.pt")
            Path(self.artifact_path).touch()
            (self.out / "current_scores.json").write_text(json.dumps({"node_scores": {"MLP 2": .9}}))
            (self.out / "zzz_stale_scores.json").write_text(json.dumps({"node_scores": {"MLP 9": .1}}))
            return self

        def evaluate(self, **kwargs):
            self.report = {"patching_score": .8, "ablation_score": .7}
            return self

    module(monkeypatch, "circuitkit")
    module(monkeypatch, "circuitkit.pipeline", Pipeline=Pipeline)
    monkeypatch.setattr(circuits, "ROOT", tmp_path)
    monkeypatch.setattr(circuits, "OUT", tmp_path / "circuits")
    monkeypatch.setattr(circuits, "parity_check", lambda *a, **k: {"passed": True}, raising=False)
    monkeypatch.setattr(circuits, "_release_memory", lambda: None, raising=False)
    monkeypatch.setattr("langsafe.gen.model.resolve_model", lambda _: "local/model")
    monkeypatch.setattr("langsafe.gen.model.best_device", lambda: "cpu")
    monkeypatch.setattr(circuits.Ledger, "default", lambda: SimpleNamespace(append=lambda *a, **k: None))
    pairs = tmp_path / "pairs.csv"
    pairs.write_text("harmful,harmless,refuse,comply\na,b,c,d\n")
    result = circuits.discover("model", "fire", "en", pairs)
    assert result["faithfulness"] == {"patching_score": .8, "ablation_score": .7}
    assert Path(result["scores_json"]).name == "current_scores.json"
    assert result["parity"]["passed"]


def test_circuit_task_preamble_conditions_get_distinct_files(monkeypatch, tmp_path):
    monkeypatch.setattr(task, "TASK_DIR", tmp_path)
    pairs = [{"harmful": "a", "harmless": "b"}]
    default = task.write_task_csv(Tokenizer(), pairs, "en", "fire", ("I", "Sure"), "default")
    stripped = task.write_task_csv(Tokenizer(), pairs, "en", "fire", ("I", "Sure"), "none")
    assert default != stripped
    assert "SYSTEM_TOKEN" in default.read_text()
    assert "SYSTEM_TOKEN" not in stripped.read_text()
    assert "<BOS_TOKEN>" not in default.read_text()


def test_circuit_report_serializes_arrays_and_nonfinite_values():
    import numpy as np

    report = SimpleNamespace(
        patching_score=np.float32(.75),
        stability={'overlap_matrix': np.array([[1., .5], [.5, 1.]]),
                   'undefined': np.array([np.nan, np.inf])})
    saved = circuits._plain(report)
    assert saved == {'patching_score': .75,
                     'stability': {'overlap_matrix': [[1., .5], [.5, 1.]],
                                   'undefined': [None, None]}}
    json.dumps(saved, allow_nan=False)


def test_resta_rejects_shape_mismatch_before_any_mutation(monkeypatch):
    models = {"fire": SimpleNamespace(state_dict=lambda: {"w": SimpleNamespace(shape=(2,))}),
              "base": SimpleNamespace(state_dict=lambda: {"w": SimpleNamespace(shape=(3,))}),
              "global": SimpleNamespace(state_dict=lambda: {"w": SimpleNamespace(shape=(2,))})}
    module(monkeypatch, "safetune.recover", apply_resta=lambda *a, **k: None)
    monkeypatch.setattr(arms, "load_spec", lambda: {"models": {"base": "base", "reference": "global"}})
    monkeypatch.setattr(arms, "_load_cpu", models.__getitem__)
    monkeypatch.setattr(arms, "_save", lambda *a, **k: Path("saved"))
    with pytest.raises(ValueError, match="shape"):
        arms.resta("fire", .25, "fire")


def test_resta_explicitly_disables_dare_and_selects_cpu(monkeypatch):
    model = SimpleNamespace(state_dict=lambda: {"w": SimpleNamespace(shape=(2,))})
    def apply(target, *, base, aligned, alpha, dare, device):
        assert target is model and base is model and aligned is model
        assert alpha == .25 and dare is False and device == "cpu"
    module(monkeypatch, "safetune.recover", apply_resta=apply)
    monkeypatch.setattr(arms, "load_spec", lambda: {"models": {"base": "base", "reference": "global"}})
    monkeypatch.setattr(arms, "_load_cpu", lambda _: model)
    monkeypatch.setattr(arms, "_save", lambda *a, **k: Path("saved"))
    assert arms.resta("fire", .25, "fire") == Path("saved")


def test_circuit_rejects_missing_layer_restriction_before_loading_weights(monkeypatch, tmp_path):
    info = SimpleNamespace(layer_suggestions=SimpleNamespace(layer_subset=None, target_modules=["q_proj"]))
    module(monkeypatch, "safetune.core.circuit_kit.adapter", load_circuit_info_from_file=lambda *a, **k: info)
    module(monkeypatch, "safetune.recover", apply_ctheta=lambda *a, **k: None)
    monkeypatch.setattr(arms, "load_spec", lambda: {"models": {"base": "base", "reference": "global"}})
    monkeypatch.setattr(arms, "_load_cpu", lambda _: pytest.fail("an unlocalized circuit must not load weights"))
    scores = tmp_path / "scores.json"
    scores.write_text(json.dumps({"layer_suggestions": {"target_modules": ["q_proj"]}}))
    with pytest.raises(ValueError, match="layer"):
        arms.circuit("fire", str(scores), 1.0, "fire")


def test_circuit_resolves_root_relative_path_and_passes_explicit_mask(monkeypatch, tmp_path):
    info = SimpleNamespace(layer_suggestions=SimpleNamespace(layer_subset=[2], target_modules=["q_proj"]))
    def load(path, *, top_fraction):
        assert Path(path) == tmp_path / "scores.json"
        return info
    def apply(target, positive, negative, circuit_info, *, strength, layer_subset, target_modules):
        assert strength == .5 and layer_subset == [2] and target_modules == ["q_proj"]
    module(monkeypatch, "safetune.core.circuit_kit.adapter", load_circuit_info_from_file=load)
    module(monkeypatch, "safetune.recover", apply_ctheta=apply)
    monkeypatch.setattr(arms, "ROOT", tmp_path)
    monkeypatch.setattr(arms, "load_spec", lambda: {"models": {"base": "base", "reference": "global"}})
    model = SimpleNamespace(state_dict=lambda: {"model.layers.2.self_attn.q_proj.weight": SimpleNamespace(shape=(2, 2))})
    monkeypatch.setattr(arms, "_load_cpu", lambda _: model)
    monkeypatch.setattr(arms, "_save", lambda *a, **k: Path("saved"))
    (tmp_path / "scores.json").write_text(json.dumps({"layer_suggestions": {"layer_subset": [2], "target_modules": ["q_proj"]}}))
    assert arms.circuit("fire", "scores.json", .5, "fire") == Path("saved")


def test_steering_extraction_honors_preamble_and_extracts_only_requested_layer(monkeypatch, tmp_path):
    import torch

    model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=3, hidden_size=2, _name_or_path="fire"))
    def extract(model, tokenizer, harmful, harmless, config):
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": harmful[0]}],
                                                  tokenize=False, add_generation_prompt=True)
        assert "SYSTEM_TOKEN" not in rendered
        assert config.target_layers == [1] and config.pick_layer == 1
        assert config.pool_method == "last_token" and config.select_directions is False
        tokenizer.padding_side = "left"
        assert tokenizer.tokenizer.padding_side == "left"
        return torch.tensor([1., 0.]), 1, {1: torch.tensor([1., 0.])}
    module(monkeypatch, "safetune.steer", RefusalDirectionConfig=SimpleNamespace, extract_refusal_direction=extract)
    monkeypatch.setattr(arms, "CKPT", tmp_path)
    monkeypatch.setattr(arms.Ledger, "default", lambda: SimpleNamespace(append=lambda *a, **k: None))
    out = arms.extract_direction(model, Tokenizer(), ["harmful"], ["harmless"], "fire", pick_layer=1, preamble="none")
    assert json.loads((out / "meta.json").read_text())["preamble"] == "none"
    assert torch.load(out / "refusal_direction.pt", weights_only=True)["layer"] == 1


def test_steering_rejects_wrong_hidden_width(monkeypatch, tmp_path):
    import torch

    torch.save({"direction": torch.tensor([1., 0.]), "layer": 1}, tmp_path / "refusal_direction.pt")
    model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=3, hidden_size=3), device="cpu")
    module(monkeypatch, "safetune.steer", RefusalDirectionModel=lambda *a, **k: object())
    with pytest.raises(ValueError, match="hidden"):
        arms.steered(model, tmp_path, .5)


def test_circuit_top_fraction_recomputes_upstream_preselected_mask(monkeypatch, tmp_path):
    """CircuitKIT's interop fields hard-code 20%; a requested 50% must replace them."""
    narrow = SimpleNamespace(layer_suggestions=SimpleNamespace(layer_subset=[1], target_modules=["q_proj"]))
    broad = SimpleNamespace(layer_suggestions=SimpleNamespace(layer_subset=[1, 2], target_modules=["q_proj"]))
    def from_scores(scores, *, top_fraction, raw_output_path):
        assert scores == {"A1.0": .8, "A2.0": .7, "A3.0": .2, "A4.0": .1}
        assert top_fraction == .5
        return broad
    def apply(target, positive, negative, circuit_info, *, strength, layer_subset, target_modules):
        assert layer_subset == [1, 2]
    module(monkeypatch, "safetune.core.circuit_kit.adapter", load_circuit_info_from_file=lambda *a, **k: narrow,
           circuit_info_from_node_scores=from_scores)
    module(monkeypatch, "safetune.recover", apply_ctheta=apply)
    monkeypatch.setattr(arms, "load_spec", lambda: {"models": {"base": "base", "reference": "global"}})
    model = SimpleNamespace(state_dict=lambda: {f"model.layers.{i}.self_attn.q_proj.weight": SimpleNamespace(shape=(2, 2))
                                               for i in (1, 2)})
    monkeypatch.setattr(arms, "_load_cpu", lambda _: model)
    monkeypatch.setattr(arms, "_save", lambda *a, **k: Path("saved"))
    scores = tmp_path / "scores.json"
    scores.write_text(json.dumps({"node_scores": {"A1.0": .8, "A2.0": .7, "A3.0": .2, "A4.0": .1},
                                  "layer_suggestions": {"layer_subset": [1], "target_modules": ["q_proj"]}}))
    assert arms.circuit("fire", str(scores), .5, "fire", top_fraction=.5) == Path("saved")


@pytest.mark.parametrize("changed,passed", [(False, True), (True, False)])
def test_parity_gate_compares_real_next_token_distributions(monkeypatch, changed, passed):
    import torch

    class Model:
        def __init__(self, logits):
            self.logits = torch.tensor([[logits]], dtype=torch.float32)
        def eval(self):
            return self
        def parameters(self):
            yield torch.tensor(0.)
        def __call__(self, *args, **kwargs):
            return SimpleNamespace(logits=self.logits)

    hf, tl = Model([0., 1., 2., 3.]), Model([3., 2., 1., 0.] if changed else [0., 1., 2., 3.])
    class TL:
        def eval(self):
            return self
        def parameters(self):
            return tl.parameters()
        def __call__(self, ids):
            return tl.logits
    def load(model_id, *, dtype, device, hf_model, **kwargs):
        assert hf_model is hf and dtype == "float32" and device == "cpu"
        assert not kwargs["fold_ln"] and not kwargs["center_unembed"]
        return TL()
    tokenizer = lambda text, **kwargs: {"input_ids": torch.tensor([[0, 1]])}
    module(monkeypatch, "circuitkit")
    module(monkeypatch, "circuitkit.quick", load_model=load)
    module(monkeypatch, "transformers", AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **k: hf),
           AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer))
    monkeypatch.setattr("langsafe.gen.model.resolve_model", lambda value: value)
    monkeypatch.setattr(circuits, "_release_memory", lambda: None)
    result = circuits.parity_check("model", prompts=["probe"])
    assert result["passed"] is passed
    assert result["per_prompt"][0]["argmax_agrees"] is passed


@pytest.mark.parametrize("mask", [([1, 9], ["q_proj"]), ([1], ["q_proj", "missing_proj"])])
def test_circuit_rejects_partly_missing_mask(monkeypatch, tmp_path, mask):
    layers, targets = mask
    info = SimpleNamespace(layer_suggestions=SimpleNamespace(layer_subset=layers, target_modules=targets))
    module(monkeypatch, "safetune.core.circuit_kit.adapter", load_circuit_info_from_file=lambda *a, **k: info)
    module(monkeypatch, "safetune.recover", apply_ctheta=lambda *a, **k: None)
    monkeypatch.setattr(arms, "load_spec", lambda: {"models": {"base": "base", "reference": "global"}})
    model = SimpleNamespace(state_dict=lambda: {"model.layers.1.self_attn.q_proj.weight": SimpleNamespace(shape=(2, 2))})
    monkeypatch.setattr(arms, "_load_cpu", lambda _: model)
    monkeypatch.setattr(arms, "_save", lambda *a, **k: Path("saved"))
    scores = tmp_path / "scores.json"
    scores.write_text(json.dumps({"layer_suggestions": {"layer_subset": layers, "target_modules": targets}}))
    with pytest.raises(ValueError, match="match"):
        arms.circuit("fire", str(scores), .5, "fire")


@pytest.mark.parametrize("arm,want", [("resta", 3.), ("merge_back", 2.75), ("circuit", 4.), ("circuit_merge", 3.5)])
def test_pinned_safetune_weight_arithmetic_round_trips_on_cpu(monkeypatch, tmp_path, arm, want):
    """Exercise the real pinned arithmetic on tiny tensors without any model load."""
    import torch
    pytest.importorskip("safetune")

    class Model(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
            self.model.layers[0].self_attn = torch.nn.Module()
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(self.model.layers[0].self_attn, name, torch.nn.Linear(2, 2, bias=False))
                with torch.no_grad():
                    getattr(self.model.layers[0].self_attn, name).weight.fill_(value)

    models = {"fire": Model(2), "base": Model(1), "global": Model(5)}
    monkeypatch.setattr(arms, "_load_cpu", models.__getitem__)
    monkeypatch.setattr(arms, "load_spec", lambda: {"models": {"base": "base", "reference": "global"}})
    def save(model, *args, **kwargs):
        out = tmp_path / "state.pt"
        torch.save(model.state_dict(), out)
        return out
    monkeypatch.setattr(arms, "_save", save)
    if arm in ("circuit", "circuit_merge"):
        scores = tmp_path / "scores.json"
        scores.write_text(json.dumps({"node_scores": {"A0.0": 1.0}}))
        out = arms.circuit("fire", str(scores), .5, "fire", vector="safety" if arm == "circuit" else "drift")
    else:
        out = getattr(arms, arm)("fire", .25, "fire")
    state = torch.load(out, map_location="cpu", weights_only=True)
    assert torch.equal(state["model.layers.0.self_attn.q_proj.weight"], torch.full((2, 2), want))


@pytest.mark.skipif(os.name == "nt", reason="POSIX installer script")
def test_mac_installer_preserves_runtime_and_does_not_resolve_cuda_dependencies(tmp_path):
    harness = tmp_path / "harness.py"
    log = tmp_path / "calls.jsonl"
    harness.write_text('''import importlib.metadata as metadata
import json, os, platform, sys, types
from pathlib import Path
args = sys.argv[1:]
if args[:2] == ["-m", "pip"]:
    if args[2:] == ["--version"]:
        raise SystemExit(0)
    constraints = Path(args[args.index("--constraint") + 1]).read_text()
    with open(os.environ["INSTALL_TEST_LOG"], "a") as stream:
        stream.write(json.dumps({"args": args, "constraints": constraints}) + "\\n")
    raise SystemExit(0)
versions = {"torch": "2.15.0", "transformers": "5.22.1", "huggingface-hub": "2.4.0"}
metadata.version = lambda name: versions.get(name, "0.1")
platform.system = lambda: "Darwin"
def module(name, **exports):
    result = types.ModuleType(name); result.__dict__.update(exports); sys.modules[name] = result
module("torch", cuda=types.SimpleNamespace(is_available=lambda: False),
       backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)))
module("transformers")
module("auditkit"); module("curatorkit")
module("safetune.recover", apply_resta=None, apply_ctheta=None, apply_wise_ft=None)
module("safetune.steer", RefusalDirectionConfig=None, RefusalDirectionModel=None, extract_refusal_direction=None)
sys.argv = args
exec(compile(sys.stdin.read(), "<installer>", "exec"))
''')
    launcher = tmp_path / "fake python"
    launcher.write_text(f"#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(harness))} \"$@\"\n")
    launcher.chmod(0o755)
    env = {**os.environ, "LANGSAFE_PYTHON": str(launcher), "INSTALL_TEST_LOG": str(log)}
    script = Path(__file__).parents[1] / "scripts/install_kits.sh"
    result = subprocess.run(["bash", str(script), "--repair"], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 4
    assert all("torch==2.15.0" in call["constraints"] and "transformers==5.22.1" in call["constraints"] for call in calls)
    assert calls[0]["args"][-1] == ".[repair,dev]"
    assert "--no-deps" in calls[1]["args"]
    assert not any("bitsandbytes" in str(call["args"]) or "torchao" in str(call["args"]) for call in calls)
    log.unlink()
    result = subprocess.run(["bash", str(script), "--all"], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and "huggingface-hub>=1.5,<2" in result.stderr
    assert not log.exists()
    result = subprocess.run(["bash", str(script), "--repair", "--cuda"], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and "Darwin" in result.stderr
    assert not log.exists()


def test_saved_repair_provenance_contains_runtime_and_local_input_lineage(tmp_path, monkeypatch):
    pytest.importorskip("safetune")
    import transformers

    source = tmp_path / "source"
    source.mkdir()
    (source / "lexsi_provenance.json").write_text(json.dumps({"schema": "lexsi.provenance/1", "method": "parent"}))
    tokenizer = SimpleNamespace(save_pretrained=lambda out: (out / "tokenizer.json").write_text("{}"))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda _: tokenizer)
    monkeypatch.setattr("langsafe.gen.model.resolve_model", lambda _: str(source))
    monkeypatch.setattr(arms.Ledger, "default", lambda: SimpleNamespace(append=lambda *a, **k: None))
    monkeypatch.setattr(arms, "_load_cpu", lambda _: None)
    monkeypatch.setattr(arms, "edit_stats", lambda *a: {"edit_l2": 1.0})
    model = SimpleNamespace(save_pretrained=lambda out, **kwargs: (out / "model.safetensors").write_bytes(b"weights"))
    out = arms._save(model, "fire", tmp_path / "repaired", "resta", ["fire"], {"alpha": .25})
    record = json.loads((out / "lexsi_provenance.json").read_text())
    assert record["inputs"][0]["provenance"]["method"] == "parent"
    assert record["params"]["alpha"] == .25
    assert record["params"]["safetune_commit"] == "3901a9f"
    assert record["params"]["runtime"]["torch"]
    assert record["params"]["runtime"]["transformers"]


def test_failed_gguf_conversion_removes_partial_output_for_retry(tmp_path, monkeypatch):
    kit = tmp_path / "llama"
    kit.mkdir()
    (kit / "convert_hf_to_gguf.py").touch()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    out = tmp_path / "out.gguf"
    def fail(command, **kwargs):
        Path(command[command.index('--outfile') + 1]).write_bytes(b"partial")
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        gguf.convert_hf_to_gguf(checkpoint, out, kit)
    assert not out.exists()


def test_interrupted_gguf_export_can_replace_only_after_success(tmp_path, monkeypatch):
    kit = tmp_path / 'llama'
    kit.mkdir()
    (kit / 'convert_hf_to_gguf.py').touch()
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'config.json').write_text('{}')
    out = tmp_path / 'out.gguf'
    out.write_bytes(b'previous-valid-file')
    out.with_suffix('.gguf.partial').write_bytes(b'interrupted-partial')

    def fail(command, **kwargs):
        Path(command[command.index('--outfile') + 1]).write_bytes(b'incomplete')
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, 'run', fail)
    with pytest.raises(subprocess.CalledProcessError):
        gguf.convert_hf_to_gguf(checkpoint, out, kit, overwrite=True)
    assert out.read_bytes() == b'previous-valid-file'

    def succeed(command, **kwargs):
        Path(command[command.index('--outfile') + 1]).write_bytes(b'completed-retry')

    monkeypatch.setattr(subprocess, 'run', succeed)
    gguf.convert_hf_to_gguf(checkpoint, out, kit, overwrite=True)
    assert out.read_bytes() == b'completed-retry'
    assert not out.with_suffix('.gguf.partial').exists()


def test_cli_records_real_auditkit_outputs_with_precision_identity(monkeypatch, tmp_path):
    pytest.importorskip('auditkit')
    from argparse import Namespace
    from langsafe import cli
    from langsafe.data.records import Prompt
    prompt = Prompt(id='g::en', group='g', lang='en', kind='benign', text='Say hello.',
                    source='test', source_id='g', split='heldout')
    monkeypatch.setattr('langsafe.data.build.load_split', lambda split: [prompt])
    monkeypatch.setattr(cli, 'RUNS', tmp_path)
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'private-cache'))
    (tmp_path / 'gens').mkdir()
    (tmp_path / 'judged').mkdir()
    (tmp_path / 'gens' / 'fire-q4.jsonl').write_text(json.dumps({
        'id': 'g::en', 'preamble': 'default', 'response': 'Hello!', 'fingerprint': 'q4-hash'}) + '\n')
    (tmp_path / 'judged' / 'fire-q4.rules.jsonl').write_text(json.dumps({
        'id': 'g::en', 'preamble': 'default', 'kind': 'benign', 'lang': 'en',
        'label': 'full_compliance', 'judges': {'rules': 'full_compliance'}}) + '\n')
    cli.cmd_audit(Namespace(tag='fire-q4', judges='rules', split='heldout'))
    directory = tmp_path / 'auditkit' / 'fire-q4.rules'
    saved = json.loads((directory / 'fire-q4.rules.json').read_text())
    assert (directory / 'lexsi_provenance.json').is_file()
    assert saved['config']['extra']['variant'] == 'fire-q4.rules'
    assert saved['config']['extra']['model_fingerprints'] == ['q4-hash']
    assert saved['headline']['judged_full_refusal_v1'] == 0.0

    # AuditKIT scores precomputed labels from metadata. A corrected judgment
    # must invalidate its cache even when the prompt and response are unchanged.
    judged_path = tmp_path / 'judged' / 'fire-q4.rules.jsonl'
    corrected = json.loads(judged_path.read_text())
    corrected.update(label='full_refusal', judges={'rules': 'full_refusal'})
    judged_path.write_text(json.dumps(corrected) + '\n')
    cli.cmd_audit(Namespace(tag='fire-q4', judges='rules', split='heldout'))
    revised = json.loads((directory / 'fire-q4.rules.json').read_text())
    assert revised['headline']['judged_full_refusal_v1'] == 1.0
    assert revised['fingerprint'] != saved['fingerprint']

    # A regenerated completion must also replace the saved prediction.
    gen_path = tmp_path / 'gens' / 'fire-q4.jsonl'
    regenerated = json.loads(gen_path.read_text())
    regenerated['response'] = 'Good morning!'
    gen_path.write_text(json.dumps(regenerated) + '\n')
    cli.cmd_audit(Namespace(tag='fire-q4', judges='rules', split='heldout'))
    final = json.loads((directory / 'fire-q4.rules.json').read_text())
    assert final['fingerprint'] != revised['fingerprint']
    assert final['predictions'][0]['raw_output'] == 'Good morning!'


def test_top_layers_ranks_layers_by_mass_not_node_membership():
    scores = {"A0.0": .1, "A1.0": .9, "A1.1": .8, "MLP 2": .7, "A3.0": .05, "MLP 4": .05, "A5.0": .05, "MLP 6": .05,
              "A7.0": .05, "MLP 8": .05, "A9.0": .05}
    assert circuits.layer_mass(scores)[1] == pytest.approx(1.7)
    assert circuits.top_layers(scores, .2) == [1, 2]


def test_edit_stats_records_what_changed_and_where():
    import torch

    def model(value):
        state = {"model.layers.3.mlp.up_proj.weight": torch.full((2,), value), "model.norm.weight": torch.ones(2)}
        return SimpleNamespace(state_dict=lambda: state)

    stats = arms.edit_stats(model(2.), model(1.))
    assert stats["edited_tensors"] == 1 and stats["edited_layers"] == [3]
    assert stats["edit_l2"] == pytest.approx(2 ** .5)


@pytest.mark.parametrize("vector,seed,negative,layers", [("safety", None, "base", [1]), ("drift", None, "fire", [1]),
                                                        ("safety", 7, "base", None)])
def test_circuit_vectors_and_random_control_stay_outside_the_circuit(monkeypatch, tmp_path, vector, seed, negative, layers):
    seen = {}

    def apply(target, positive, neg, circuit_info, *, strength, layer_subset, target_modules):
        seen.update(negative=neg, layers=layer_subset)

    info = SimpleNamespace(layer_suggestions=SimpleNamespace(layer_subset=[0, 1, 2, 3], target_modules=["q_proj"]))
    module(monkeypatch, "safetune.core.circuit_kit.adapter", circuit_info_from_node_scores=lambda *a, **k: info,
           load_circuit_info_from_file=lambda *a, **k: info)
    module(monkeypatch, "safetune.recover", apply_ctheta=apply)
    monkeypatch.setattr(arms, "load_spec", lambda: {"models": {"base": "base", "reference": "global"}})
    state = {f"model.layers.{i}.self_attn.q_proj.weight": SimpleNamespace(shape=(2, 2)) for i in range(4)}
    monkeypatch.setattr(arms, "_load_cpu", lambda name: SimpleNamespace(state_dict=lambda: state, name=name))
    monkeypatch.setattr(arms, "_save", lambda *a, **k: Path("saved"))
    scores = tmp_path / "scores.json"
    scores.write_text(json.dumps({"node_scores": {"A0.0": .1, "A1.0": .9, "A2.0": .1, "A3.0": .1}}))
    arms.circuit("fire", str(scores), .5, "fire", top_fraction=.25, vector=vector, random_seed=seed)
    assert seen["negative"].name == negative
    if layers:
        assert seen["layers"] == layers
    else:
        assert len(seen["layers"]) == 1 and 1 not in seen["layers"]


def _fetch_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("fetch_models", Path(__file__).parents[1] / "scripts/fetch_models.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_fetch_continues_a_partial_file_and_survives_a_dropped_connection(tmp_path, monkeypatch):
    fetcher = _fetch_module()
    payload = b"0123456789"
    (tmp_path / "w.bin.part").write_bytes(payload[:4])
    calls = []

    class Reply:
        def __init__(self, start, drop):
            self.status_code, self.start, self.drop = 206, start, drop

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def iter_content(self, size):
            yield payload[self.start:self.start + 3]
            if self.drop:
                raise fetcher.requests.ConnectionError("reset")
            yield payload[self.start + 3:]

    def get(url, headers, **kwargs):
        start = int(headers["Range"].split("=")[1].rstrip("-"))
        calls.append(start)
        return Reply(start, drop=len(calls) == 1)

    monkeypatch.setattr(fetcher.requests, "get", get)
    monkeypatch.setattr(fetcher.time, "sleep", lambda _: None)
    fetcher.fetch("url", tmp_path / "w.bin", len(payload), {})
    assert calls == [4, 7]                       # resumed from the partial file, then from where it dropped
    assert (tmp_path / "w.bin").read_bytes() == payload and not (tmp_path / "w.bin.part").exists()


def test_fetch_restarts_when_the_server_ignores_the_range(tmp_path, monkeypatch):
    fetcher = _fetch_module()
    payload = b"abcdef"
    (tmp_path / "w.bin.part").write_bytes(b"xx")

    class Reply:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def iter_content(self, size):
            yield payload

    monkeypatch.setattr(fetcher.requests, "get", lambda *a, **k: Reply())
    fetcher.fetch("url", tmp_path / "w.bin", len(payload), {})
    assert (tmp_path / "w.bin").read_bytes() == payload


class WordTokenizer:
    """First "token" is the first word, as `I'm` and `Aku` are in Tiny Aya's own tokenizer."""
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids):
        return " ".join(ids)


def _opener_rows(answers):
    judged = [{"id": str(i), "preamble": "default", "lang": lang, "label": label} for i, (lang, label, _) in enumerate(answers)]
    return judged, {(str(i), "default"): {"response": text} for i, (_, _, text) in enumerate(answers)}


def test_openers_are_what_the_reference_model_actually_says():
    judged, gens = _opener_rows([("jv", "full_refusal", "Aku ora bisa"), ("jv", "full_refusal", "Aku njaluk ngapura"),
                                 ("jv", "full_refusal", "Nuwun sewu"), ("jv", "full_compliance", "Iki jawabane"),
                                 ("jv", "full_compliance", "Iki carane"), ("en", "full_refusal", "Sorry no")])
    assert task.openers_from_judged(judged, gens, "jv", WordTokenizer()) == ("Aku", "Iki")


def test_openers_skip_markdown_and_never_use_one_token_for_both_sides():
    judged, gens = _opener_rows([("jv", "full_refusal", "** Aku"), ("jv", "full_refusal", "** Aku"), ("jv", "full_refusal", "Aku"),
                                 ("jv", "full_compliance", "Aku iki"), ("jv", "full_compliance", "Aku iku"),
                                 ("jv", "full_compliance", "Iki"), ("jv", "full_compliance", "** Iki")])
    assert task.openers_from_judged(judged, gens, "jv", WordTokenizer()) == ("Aku", "Iki")


def test_openers_fall_back_to_the_stock_phrases_without_labelled_answers():
    assert task.openers_from_judged([], {}, "jv", WordTokenizer()) == task.DEFAULT_OPENERS["jv"]
