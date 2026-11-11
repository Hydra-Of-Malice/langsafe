"""Start (or resume) the NF4 run in a Kaggle notebook. `remote_nf4.py bundle` puts this file at the root of the private
bundle repo, so the notebook cell only has to fetch and run it (see docs/REMOTE_NF4.md):

    import os,sys,subprocess;from kaggle_secrets import UserSecretsClient as U;os.environ['HF_TOKEN']=U().get_secret('HF_TOKEN');subprocess.run([sys.executable,'-m','pip','install','-q','huggingface_hub==1.33.0'],check=True);from huggingface_hub import hf_hub_download as d;exec(open(d('HydraOfMalice/lexsi-nf4-private','kaggle_start.py',repo_type='dataset')).read())

It needs a T4 accelerator, internet on, and a Kaggle secret HF_TOKEN that can read and write the private repo.
"""
import os
import subprocess
import glob
import shutil
import sys
import tarfile

from huggingface_hub import hf_hub_download

REPO = 'HydraOfMalice/lexsi-nf4-private'
WORK = '/tmp/lexsi'
if not os.environ.get('HF_TOKEN'):
    raise SystemExit('Add a Kaggle secret named HF_TOKEN first (Add-ons > Secrets)')


def sh(command: str) -> str:
    return subprocess.run(['bash', '-c', command], capture_output=True, text=True).stdout.strip()


# The llama.cpp build needs nvcc. Kaggle images keep it in a versioned CUDA folder, not always on PATH.
nvcc = shutil.which('nvcc') or next(iter(sorted(glob.glob('/usr/local/cuda*/bin/nvcc'), reverse=True)), None) \
    or (sh('find / -xdev -name nvcc -type f -perm -u+x 2>/dev/null | head -n 1') or None)
print('nvcc:', nvcc, '|', sh(f'"{nvcc}" --version | tail -n 1') if nvcc else 'not found')
print('cmake:', sh('cmake --version | head -n 1') or 'not found', '| g++:', sh('g++ --version | head -n 1') or 'not found')
path = os.environ.get('PATH', '')
if nvcc:
    path = os.path.dirname(nvcc) + ':' + path
    os.environ['CUDACXX'] = nvcc  # CMake's CUDA compiler
# Kaggle's CUDA toolkit has no libcuda stub, so CMake cannot make its CUDA::cuda_driver target and ggml-cuda fails to
# configure. The GPU machine has the driver's own libcuda.so.1; a libcuda.so link to it on CMAKE_LIBRARY_PATH fills in.
libcuda = (sh("ldconfig -p | grep -m 1 'libcuda.so.1 ' | awk '{print $NF}'")
           or next(iter(sorted(glob.glob('/usr/lib/x86_64-linux-gnu/libcuda.so*') + glob.glob('/usr/local/nvidia/lib64/libcuda.so*')
                               + glob.glob('/usr/local/cuda*/**/stubs/libcuda.so', recursive=True))), None))
print('libcuda:', libcuda or 'not found')
if libcuda:
    stub = '/tmp/lexsi-cuda-driver'
    os.makedirs(stub, exist_ok=True)
    if not os.path.lexists(os.path.join(stub, 'libcuda.so')):
        os.symlink(libcuda, os.path.join(stub, 'libcuda.so'))
    os.environ['CMAKE_LIBRARY_PATH'] = ':'.join(filter(None, [stub, os.environ.get('CMAKE_LIBRARY_PATH')]))
# Kaggle stops a session at 12 hours; stop cleanly at 11.5, sync, and let the next session resume.
os.environ.update(BUNDLE_REPO=REPO, PLATFORM='kaggle-t4', WORK=WORK, DEADLINE_MINUTES='690', CUDA_VISIBLE_DEVICES='0',
                  PATH=path)
os.makedirs(WORK, exist_ok=True)
with tarfile.open(hf_hub_download(REPO, 'code.tar', repo_type='dataset', local_dir=WORK)) as archive:
    archive.extractall(os.path.join(WORK, 'code'), filter='data')
sys.path.insert(0, os.path.join(WORK, 'code', 'scripts'))
import remote_nf4  # noqa: E402

result = subprocess.run(['bash', '-c', remote_nf4.BOOTSTRAP])
print('NF4 run exit code', result.returncode)
if result.returncode:
    for name in ('llama-cmake.log', 'llama-build.log'):
        log = os.path.join(WORK, name)
        if os.path.isfile(log):
            print(f'--- last lines of {name} ---')
            print(sh(f'tail -n 25 "{log}"'))
