"""Environment setup helper (Windows)."""
from __future__ import annotations
import subprocess, sys

STEPS = [
    [sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'],
    [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
]

if __name__ == '__main__':
    print('Install PyTorch for your CUDA/CPU from https://pytorch.org/get-started/locally/ first if needed.')
    for cmd in STEPS:
        print('>', ' '.join(cmd))
        subprocess.check_call(cmd)
