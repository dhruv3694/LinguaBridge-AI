"""Run default notebook-aligned tests."""
import subprocess, sys
raise SystemExit(subprocess.call([sys.executable, '-m', 'pytest', '-q']))
