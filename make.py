"""
Run this file to redo all processing and figure creation.
"""

import importlib
import re
from pathlib import Path

pattern = r"^[A-Za-z]\d{3}_"

if __name__ == "__main__":
    print("Running all Elk Creek Processing ....")  # noqa
    for path in sorted(Path(__file__).parent.glob("*.py")):
        name = path.stem
        if re.match(pattern, name):
            mod = importlib.import_module(name)
            print(f"Running {path.name}")  # noqa
            assert hasattr(mod, "main"), f"No main method in {name}"
            mod.main()
    print("Finished all Elk Creek Processing. Check the outputs folder.")  # noqa
