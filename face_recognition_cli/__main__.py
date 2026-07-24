"""Entry point for ``python -m face_recognition_cli``."""

from __future__ import annotations

import sys

from face_recognition_cli.cli import main

if __name__ == "__main__":
    sys.exit(main())
