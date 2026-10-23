import unittest
import tempfile
from pathlib import Path
from tf_scripts.xai_utils import test_images


class TestBatchSelection(unittest.TestCase):
    def test_excludes_normal_unless_requested(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / "screw" / "test"
            for name in ("good", "scratch_head", "thread_side"):
                folder = base / name
                folder.mkdir(parents=True)
                (folder / "005.png").touch()
            self.assertEqual(len(test_images(temp, "screw")), 2)
            self.assertEqual(len(test_images(temp, "screw", include_good=True)), 3)
            good = test_images(temp, "screw", anomaly_type="good")
            self.assertEqual(good[0].parent.name, "good")


if __name__ == "__main__":
    unittest.main()
