#!/usr/bin/env python3
import sys, subprocess
if __name__ == "__main__":
    scenario = sys.argv[1] if len(sys.argv) > 1 else "tiny_truth"
    subprocess.run(["python", "-m", "src.main", "--scenario", scenario], check=True)
