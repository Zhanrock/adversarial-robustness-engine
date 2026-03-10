"""tests/unit/test_config.py — Config loader tests."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from adversarial_robustness.utils.config import Config, load_config

YAML = "model:\n  architecture: resnet18\n  num_classes: 10\nattacks:\n  fgsm:\n    epsilon: 0.03\n"


class TestConfig(unittest.TestCase):
    def _yaml(self, content):
        fd, path = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(fd, "w") as f:
            f.write(content)
        return path

    def test_basic_load(self):
        p = self._yaml(YAML)
        cfg = load_config(p)
        os.unlink(p)
        self.assertEqual(cfg.model.architecture, "resnet18")
        self.assertEqual(cfg.model.num_classes, 10)

    def test_nested_access(self):
        p = self._yaml(YAML)
        cfg = load_config(p)
        os.unlink(p)
        self.assertAlmostEqual(cfg.attacks.fgsm.epsilon, 0.03)

    def test_to_dict(self):
        p = self._yaml(YAML)
        cfg = load_config(p)
        os.unlink(p)
        d = cfg.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["model"]["num_classes"], 10)

    def test_get_default(self):
        p = self._yaml(YAML)
        cfg = load_config(p)
        os.unlink(p)
        self.assertEqual(cfg.model.get("missing", "fb"), "fb")

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_config("/no/such.yaml")

    def test_inheritance(self):
        with tempfile.TemporaryDirectory() as d:
            base = os.path.join(d, "base.yaml")
            child = os.path.join(d, "child.yaml")
            with open(base, "w") as f:
                f.write("base_key: 42\noverride_key: original\n")
            with open(child, "w") as f:
                f.write("_base_: base.yaml\noverride_key: overridden\nnew_key: hi\n")
            cfg = load_config(child)
            self.assertEqual(cfg.base_key, 42)
            self.assertEqual(cfg.override_key, "overridden")
            self.assertEqual(cfg.new_key, "hi")


if __name__ == "__main__":
    unittest.main(verbosity=2)
