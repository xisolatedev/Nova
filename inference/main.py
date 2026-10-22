"""Compatibility entry point for target-verified DFlash chat."""
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference.cli import main

if __name__ == "__main__":
    main(default_dflash=True)
