#!/usr/bin/env python3
"""Generate the canonical MNIST training report."""

import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXAMPLES_DIR))

from analyze_classification_experiment import main  # noqa: E402


if __name__ == "__main__":
    if "--dataset" not in sys.argv:
        sys.argv[1:1] = ["--dataset", "mnist"]
    main()
