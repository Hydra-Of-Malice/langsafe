"""Checks on the reproduction notebook. Nothing here runs a cell, loads a model or needs a GPU.

The notebook is the part of the project a judge reads, and it cannot be run in CI. So these tests read it as text: it
must be valid, it must compile, every command must exist in the code as it is today, and no cell may touch model text.
"""
from __future__ import annotations

import ast
import functools
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "01_reproduce_results.ipynb"
GENERATOR = ROOT / "scripts" / "make_reproduce_notebook.py"
SCRIPTS_WITH_HELP = {"scripts/localisation_analysis.py", "scripts/render_results.py"}
NF4_WRAPPER = "scripts/run_nf4_evaluation.sh"
NF4_PROGRAM = "scripts/check_nf4_environment.py"      # the wrapper passes its arguments straight to this

CELLS = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def _source(cell: dict) -> str:
    return "".join(cell["source"])


CODE = [_source(cell) for cell in CELLS if cell["cell_type"] == "code"]
MARKDOWN = [_source(cell) for cell in CELLS if cell["cell_type"] == "markdown"]
PROSE = "\n".join(MARKDOWN)
TREES = [ast.parse(source) for source in CODE]


# ---- reading commands out of the notebook --------------------------------------------------------------------------

def _assignments() -> dict[str, list[ast.expr]]:
    table: dict[str, list[ast.expr]] = {}
    for tree in TREES:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                table.setdefault(node.targets[0].id, []).append(node.value)
    return table


ASSIGNED = _assignments()


def _variants(node: ast.expr) -> list[str]:
    """Every text an expression can take. A value the notebook only knows at run time, like a path, becomes `x`."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        parts = [_variants(part.value if isinstance(part, ast.FormattedValue) else part) for part in node.values]
        return ["".join(choice) for choice in itertools.product(*parts)]
    if isinstance(node, ast.IfExp):
        return _variants(node.body) + _variants(node.orelse)
    if isinstance(node, ast.Name) and node.id in ASSIGNED:
        found = [text for value in ASSIGNED[node.id] for text in _variants(value)]
        return found or ["x"]
    return ["x"]


def _calls(name: str) -> list[str]:
    found = []
    for tree in TREES:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name and node.args:
                found += [" ".join(text.split()) for text in _variants(node.args[0])]
    return sorted(set(found))


def _documented_commands() -> list[str]:
    """CLI lines shown in markdown are held to the same standard as the ones that run."""
    lines = re.findall(r"^python -m langsafe\.cli (.+)$", PROSE, flags=re.M)
    return [re.sub(r"<\w+>", "x", line) for line in lines]


CLI_COMMANDS = sorted(set(_calls("lx") + _documented_commands()))
SCRIPT_COMMANDS = _calls("script")
SHELL_COMMANDS = [command for command in _calls("run") if command.startswith("bash scripts/")]


@functools.cache
def _help(*command: str) -> str:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONUTF8": "1"}
    done = subprocess.run([sys.executable, *command, "--help"], cwd=ROOT, env=env, capture_output=True, text=True,
                          encoding="utf-8", timeout=120)
    assert done.returncode == 0, done.stderr
    return done.stdout


def _flags(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token.startswith("--")]


def _check_against_help(tokens: list[str], text: str) -> None:
    for flag in _flags(tokens):
        assert re.search(rf"(?<![\w-]){re.escape(flag)}(?![\w-])", text), f"{flag} is not in --help"
    for flag, choices in re.findall(r"(--[a-z-]+) \{([^}]+)\}", text):
        if flag in tokens:
            value = tokens[tokens.index(flag) + 1]
            assert value in choices.split(",") or value == "x", f"{flag} {value} is not one of {choices}"


# ---- validity -------------------------------------------------------------------------------------------------------

def test_notebook_matches_its_generator(tmp_path):
    output = tmp_path / "notebook.ipynb"
    subprocess.run([sys.executable, str(GENERATOR), str(output)], check=True, capture_output=True)
    assert output.read_bytes() == NOTEBOOK.read_bytes(), "Run scripts/make_reproduce_notebook.py and commit the result"


def test_notebook_has_the_nbformat_structure():
    """A check that needs no extra package. nbformat itself is not a project dependency yet."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert (notebook["nbformat"], notebook["nbformat_minor"]) == (4, 5)
    assert isinstance(notebook["metadata"], dict) and notebook["cells"]
    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(ids) == len(set(ids)) and all(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value) for value in ids)
    for cell in notebook["cells"]:
        assert cell["cell_type"] in {"markdown", "code"} and isinstance(cell["metadata"], dict)
        assert isinstance(cell["source"], list) and all(isinstance(line, str) for line in cell["source"])
        if cell["cell_type"] == "code":
            assert set(cell) == {"cell_type", "execution_count", "id", "metadata", "outputs", "source"}
        else:
            assert set(cell) == {"cell_type", "id", "metadata", "source"}


def test_notebook_passes_nbformat_validation():
    nbformat = pytest.importorskip("nbformat")
    nbformat.validate(nbformat.read(NOTEBOOK, as_version=4))


def test_notebook_is_saved_without_outputs():
    for cell in CELLS:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None


@pytest.mark.parametrize("index", range(len(CODE)))
def test_code_cell_compiles(index):
    compile(CODE[index], f"cell {index}", "exec")


def test_code_cells_are_plain_python():
    """The tests can only see commands in plain Python, and a failed `!` line would not stop Run All."""
    for source in CODE:
        assert not re.search(r"^\s*[!%]", source, flags=re.M)


# ---- commands -------------------------------------------------------------------------------------------------------

def test_the_command_scan_finds_the_pipeline():
    """Guards the tests below against passing because the scan found nothing."""
    used = {shlex.split(command)[0] for command in CLI_COMMANDS}
    assert used >= {"experiment", "judge-warm", "judge", "ledger-verify", "calibrate"}
    stops = {tokens[tokens.index("--stop-after") + 1] for tokens in map(shlex.split, CLI_COMMANDS)
             if "--stop-after" in tokens}
    assert stops == {"data", "baseline", "circuit", "repairs"}
    assert any("--skip-circuit" in command for command in CLI_COMMANDS)
    assert any("--skip-quantization" in command for command in CLI_COMMANDS)
    assert any("scripts/localisation_analysis.py" in command for command in SCRIPT_COMMANDS)
    assert any(NF4_WRAPPER in command for command in SHELL_COMMANDS)


@pytest.mark.parametrize("command", CLI_COMMANDS)
def test_cli_command_and_flags_exist_in_help(command):
    tokens = shlex.split(command)
    assert re.search(rf"\b{re.escape(tokens[0])}\b", _help("-m", "langsafe.cli")), f"no subcommand {tokens[0]}"
    _check_against_help(tokens, _help("-m", "langsafe.cli", tokens[0]))


@pytest.mark.parametrize("command", CLI_COMMANDS)
def test_cli_command_parses_with_the_real_parser(command, monkeypatch):
    """Help text cannot show a missing required flag. This runs the real parser with every handler stubbed out."""
    from langsafe import cli

    called = []
    for name in dir(cli):
        if name.startswith("cmd_"):
            monkeypatch.setattr(cli, name, called.append)
    for variable in ("LANGSAFE_SPEC", "LANGSAFE_RUNS_DIR", "LANGSAFE_DATA_DIR"):
        monkeypatch.setenv(variable, "unused")
    monkeypatch.setattr(cli, "RUNS", cli.RUNS)
    cli.main(["--spec", "spec.yaml", "--runs-dir", "run", *shlex.split(command)])
    assert len(called) == 1


@pytest.mark.parametrize("command", SCRIPT_COMMANDS)
def test_script_exists_and_its_flags_are_real(command):
    tokens = shlex.split(command)
    assert (ROOT / tokens[0]).is_file()
    if tokens[0] in SCRIPTS_WITH_HELP:
        _check_against_help(tokens, _help(tokens[0]))
    else:
        assert tokens[0] == "scripts/fetch_models.py" and not _flags(tokens)      # it treats every argument as a model


@pytest.mark.parametrize("command", SHELL_COMMANDS)
def test_shell_script_exists_and_its_flags_are_real(command):
    tokens = shlex.split(command)[1:]
    assert (ROOT / tokens[0]).is_file()
    if tokens[0] == NF4_WRAPPER:
        _check_against_help(tokens, _help(NF4_PROGRAM))
    else:
        installer = (ROOT / tokens[0]).read_text(encoding="utf-8")
        assert all(flag in installer for flag in _flags(tokens))


def test_the_plots_grid_has_a_column_for_every_metric():
    source = (ROOT / "scripts" / "render_results.py").read_text(encoding="utf-8")
    assert "plt.subplots(2, len(METRICS)" in source


def test_notebook_arms_exist_in_the_cli():
    choices = re.search(r"--arm \{([^}]+)\}", _help("-m", "langsafe.cli", "repair")).group(1).split(",")
    arms = re.findall(r'"id": "(\w+)"', "\n".join(CODE))
    assert arms and set(arms) <= set(choices)


def test_development_profile_still_has_what_the_notebook_patches():
    spec = yaml.safe_load((ROOT / "configs" / "development.yaml").read_text(encoding="utf-8"))
    assert spec["profile"] == "development" and len(spec["judging"]["local_judges"]) == 2
    assert isinstance(spec["experiment"], dict) and isinstance(spec["repair_arms"], list)


# ---- model text -----------------------------------------------------------------------------------------------------

LEAKS = (
    (r"\bresponses?\b", "the answer text field"),
    (r"""["'](?:verdicts|raw|reason)["']""", "a judge text field"),
    (r"""\b(?:gens|generations|hand_check|review|circuit_tasks)[/\\]""", "a private folder"),
    (r"""/\s*["'](?:gens|generations|hand_check|review|circuit_tasks)\b""", "a private folder"),
    (r"llm_cache", "the verdict cache"),
    (r"\.jsonl\b", "a file of row-level records"),
    (r"\.csv\b", "a hand-check sheet"),
    (r"\bdemo\b", "the demo command, which prints answers"),
)


def leaks(source: str) -> list[str]:
    return [why for pattern, why in LEAKS if re.search(pattern, source, flags=re.I)]


@pytest.mark.parametrize("index", range(len(CODE)))
def test_cell_never_touches_model_text(index):
    assert leaks(CODE[index]) == []


@pytest.mark.parametrize("bad", [
    'print(row["response"])',
    "print(row['response'][:80])",
    'frame["response"].head()',
    'row.get("response")',
    'lx("demo --prompt hello")',
    'run("head runs/gens/fire.jsonl")',
    'open(RUN / "gens" / "fire.txt")',
    'print(row["verdicts"])',
    'pd.read_json(path, lines=True)  # judged/fire.local.jsonl',
    'pd.read_csv("hand_check/fire.local.csv")',
])
def test_leak_check_catches_a_bad_cell(bad):
    assert leaks(bad)


def test_aggregate_reader_refuses_private_folders():
    helpers = next(source for source in CODE if "def aggregate(" in source)
    assert "PRIVATE" in helpers and 'path.suffix != ".json"' in helpers


def test_prose_says_why_model_text_is_hidden():
    assert "never shows a model's answer to a harmful prompt" in PROSE
    assert "aggregates" in PROSE and "acceptable-use policy" in PROSE


def test_export_uses_a_fixed_list():
    export = next(source for source in CODE if "EXPORT.mkdir" in source)
    assert "copytree" not in export and "rglob" not in export


# ---- honesty and style ----------------------------------------------------------------------------------------------

def test_no_token_is_written_down():
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert not re.search(r"\bhf_[A-Za-z0-9]{20,}|\bghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}", text)


def test_secrets_come_from_the_platform():
    source = "\n".join(CODE)
    assert "kaggle_secrets" in source and "google.colab" in source
    assert not re.search(r"print\([^)]*(?:HF_TOKEN|GH_TOKEN)", source)


def test_placeholders_are_all_declared():
    sys.path.insert(0, str(GENERATOR.parent))
    try:
        from make_reproduce_notebook import PLACEHOLDERS
    finally:
        sys.path.pop(0)
    used = set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", NOTEBOOK.read_text(encoding="utf-8")))
    assert used == set(PLACEHOLDERS)


def test_results_are_marked_as_awaiting_real_runs():
    assert "AWAITING REAL RUNS" in PROSE
    assert PROSE.count("AWAITING T4 RUN") >= 6


def test_only_supported_percentages_appear():
    """Report Table 7, the split, the circuit mask and the interval level. Anything else would be a result."""
    allowed = {"87.0%", "78.1%", "77.5%", "60%", "40%", "20%", "95%"}
    assert set(re.findall(r"\d+(?:\.\d+)?%", PROSE)) <= allowed


def test_every_stage_has_a_section():
    headings = "\n".join(re.findall(r"^## .+$", PROSE, flags=re.M))
    for stage in ("Setup", "small profile", "Data and split", "Generation", "Two-judge labelling", "Circuits",
                  "Repair arms and the random-layer control", "Localisation analysis", "4-bit checks", "Reports",
                  "Published results", "Limits"):
        assert stage in headings


def test_both_rubrics_are_explained():
    assert "`SYSTEM`" in PROSE and "`SAFETY_SYSTEM`" in PROSE and "`disputed`" in PROSE


def test_runtime_and_disk_are_stated_and_marked_as_estimates():
    assert "Expected runtime" in PROSE and "Expected disk use" in PROSE and "planning estimates" in PROSE


def test_no_dashes_that_the_style_rules_forbid():
    for path in (NOTEBOOK, GENERATOR):
        text = path.read_text(encoding="utf-8")
        assert "—" not in text and "–" not in text


def test_figures_section_can_run_alone():
    """A reader with no GPU starts at section 12, so its first cell cannot lean on the setup cells."""
    first = next(source for source in CODE if "def load_aggregate(" in source)
    names = {node.id for node in ast.walk(ast.parse(first)) if isinstance(node, ast.Name)}
    assert not names & {"RUN", "PROJECT", "PY", "WORK", "lx", "run", "script", "aggregate"}
    assert "import json" in first and "import matplotlib.pyplot as plt" in first
