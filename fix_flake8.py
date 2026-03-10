"""
Run this from the root of your adversarial-robustness-engine repo:
    python fix_flake8.py
"""
import os, re

def fix(relpath, replacements):
    if not os.path.exists(relpath):
        print(f"  SKIP (not found): {relpath}")
        return
    with open(relpath) as f:
        src = f.read()
    for old, new in replacements:
        if old in src:
            src = src.replace(old, new)
        else:
            print(f"  MISS in {relpath}: {repr(old[:60])}")
    with open(relpath, 'w') as f:
        f.write(src)
    print(f"  OK: {relpath}")

# F401: unused BaseModel import in fgsm.py
fix("src/adversarial_robustness/attacks/fgsm.py", [
    ("from adversarial_robustness.models.base_model import BaseModel\n", ""),
])

# F401 + F841: unused sys import and unused 'report' variable in cli.py
fix("src/adversarial_robustness/cli.py", [
    ("import sys\nimport os\n", "import os\n"),
    ("import sys\n", ""),
    ("    report = benchmarker.run(\n", "    benchmarker.run(\n"),
])

# F401: unused Tuple in adversarial_training.py
fix("src/adversarial_robustness/defenses/adversarial_training.py", [
    ("from typing import List, Tuple\n", "from typing import List\n"),
    ("from typing import Tuple, List\n", "from typing import List\n"),
])

# F841: unused half_k in denoiser.py
fix("src/adversarial_robustness/defenses/denoiser.py", [
    ("        half_k = self.kernel_size // 2\n        for i in range(x.shape[0]):",
     "        for i in range(x.shape[0]):"),
])

# F401: unused Any, Dict, Optional in base_model.py
fix("src/adversarial_robustness/models/base_model.py", [
    ("from typing import Any, Dict, Optional, Tuple\n", "from typing import Tuple\n"),
    ("from typing import Any, Dict, Tuple\n", "from typing import Tuple\n"),
    ("from typing import Any, Optional, Tuple\n", "from typing import Tuple\n"),
    ("from typing import Dict, Optional, Tuple\n", "from typing import Tuple\n"),
])

# F401: unused Optional in dummy_model.py
fix("src/adversarial_robustness/models/dummy_model.py", [
    ("from typing import Optional, Tuple\n", "from typing import Tuple\n"),
    ("from typing import Tuple, Optional\n", "from typing import Tuple\n"),
])

# F401: unused os in logger.py
fix("src/adversarial_robustness/utils/logger.py", [
    ("import os\nimport", "import"),
    ("import os\n", ""),
])

# F811: redefinition of torch in tensor_ops.py — remove the inner import inside get_device
fix("src/adversarial_robustness/utils/tensor_ops.py", [
    ("    if not _TORCH_AVAILABLE:\n        return \"numpy\"\n    import torch\n    if preference == \"auto\":",
     "    if not _TORCH_AVAILABLE:\n        return \"numpy\"\n    if preference == \"auto\":"),
    ("    if not _TORCH_AVAILABLE:\n        return \"numpy\"\n    import torch\n\n    if preference == \"auto\":",
     "    if not _TORCH_AVAILABLE:\n        return \"numpy\"\n    if preference == \"auto\":"),
])

# E402: module-level imports not at top in conftest.py
# The sys.path.insert must come before other project imports
with open("tests/conftest.py") as f:
    src = f.read()
# Check if sys.path.insert is already at top (after stdlib imports)
if "import sys\nimport os\n\nsys.path.insert" not in src and "import os\nimport sys\n\nsys.path.insert" not in src:
    # Rebuild: move sys.path stuff before numpy/pytest
    src = re.sub(
        r'(""".*?""")\n(.*?)(sys\.path\.insert[^\n]+\n)\n(import numpy)',
        lambda m: m.group(1) + "\nimport os\nimport sys\n\n" + m.group(3) + "\n" + m.group(4),
        src, flags=re.DOTALL
    )
    with open("tests/conftest.py", 'w') as f:
        f.write(src)
    print("  OK: tests/conftest.py")
else:
    print("  OK (already fixed): tests/conftest.py")

# E402: fix all test files by moving sys.path.insert above other imports
test_files = [
    "tests/integration/test_full_pipeline.py",
    "tests/unit/test_attacks.py",
    "tests/unit/test_config.py",
    "tests/unit/test_defenses.py",
    "tests/unit/test_metrics.py",
    "tests/unit/test_models.py",
]

for tf in test_files:
    if not os.path.exists(tf):
        print(f"  SKIP: {tf}")
        continue
    with open(tf) as f:
        src = f.read()
    # Pattern: """docstring"""\nimport something\nimport numpy as np\n\nsys.path.insert(...)
    # Fix: """docstring"""\nimport os\nimport sys\n\nsys.path.insert(...)\n\nimport something\n...
    new = re.sub(
        r'("""[^"]+""")\n(import sys, os[^\n]*\n)(import numpy[^\n]*\n)\n(sys\.path\.insert[^\n]+\n)',
        lambda m: m.group(1) + "\nimport os\nimport sys\n\n" + m.group(4) + "\n" + m.group(3),
        src
    )
    # Also handle "import sys, os, ..." style
    new = re.sub(
        r'("""[^"]+""")\n(import sys, os,[^\n]+\n)(import numpy[^\n]*\n)\n(sys\.path\.insert[^\n]+\n)',
        lambda m: m.group(1) + "\nimport os\nimport sys\n\n" + m.group(4) + "\n" + m.group(3),
        new
    )
    if new != src:
        with open(tf, 'w') as f:
            f.write(new)
        print(f"  OK (reordered): {tf}")
    else:
        # Fallback: just add # noqa: E402 to the import lines after sys.path.insert
        # Find the sys.path.insert line number and add noqa to everything after it
        lines = src.splitlines(keepends=True)
        path_insert_idx = next((i for i, l in enumerate(lines) if "sys.path.insert" in l), None)
        if path_insert_idx is not None:
            for i in range(path_insert_idx + 1, len(lines)):
                line = lines[i]
                if re.match(r'^from |^import ', line) and "# noqa" not in line:
                    lines[i] = line.rstrip('\n') + "  # noqa: E402\n"
            with open(tf, 'w') as f:
                f.writelines(lines)
            print(f"  OK (noqa): {tf}")
        else:
            print(f"  SKIP (no sys.path.insert found): {tf}")

# F401: unused Config in test_config.py
fix("tests/unit/test_config.py", [
    ("from adversarial_robustness.utils.config import load_config, Config\n",
     "from adversarial_robustness.utils.config import load_config\n"),
    ("from adversarial_robustness.utils.config import load_config, Config  # noqa: E402\n",
     "from adversarial_robustness.utils.config import load_config  # noqa: E402\n"),
])

# F401: unused adversarial_accuracy in test_metrics.py
fix("tests/unit/test_metrics.py", [
    ("    clean_accuracy,\n    adversarial_accuracy,\n    attack_success_rate,\n",
     "    clean_accuracy,\n    attack_success_rate,\n"),
])

# F841: unused y1 in test_models.py
fix("tests/unit/test_models.py", [
    ("        x1 = X[:1]\n        y1 = Y[:1]\n        self.assertEqual(MODEL.forward(x1).shape, (1, 5))\n        self.assertEqual(MODEL.predict(x1).shape, (1,))",
     "        x1 = X[:1]\n        self.assertEqual(MODEL.forward(x1).shape, (1, 5))\n        self.assertEqual(MODEL.predict(x1).shape, (1,))"),
])

print("\nDone! Now run: git add -A && git commit -m 'fix: resolve all flake8 warnings' && git push")
