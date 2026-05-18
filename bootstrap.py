# Bootstrap script for PyInstaller
# This properly loads the dinov3 package and calls the CLI entry point

import sys
import os

# Add the current directory to path to find the package
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    # Running as compiled executable
    os.chdir(sys._MEIPASS)
    sys.path.insert(0, sys._MEIPASS)

# Now import and call the entry point
from dinov3_cli.cli import app

if __name__ == "__main__":
    app()
