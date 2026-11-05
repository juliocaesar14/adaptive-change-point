# Run the whole project, one step after another.
# See the README for how to run it, with or without fresh data.

import argparse
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--source", default="bundled", choices=["bundled", "yahoo"])
source = parser.parse_args().source

# Each step is its own file, so any step can also be run alone.
commands = [["tests/test_methods.py"],
            ["step1_prepare_data.py", "--source", source],
            ["step2_run_simulation.py"],
            ["step3_run_sp500.py", "--source", source],
            ["step4_make_figures.py", "--source", source],
            ["evaluate.py"]]
for command in commands:
    print("\nRunning", command[0], flush=True)
    subprocess.run([sys.executable] + command, check=True)
