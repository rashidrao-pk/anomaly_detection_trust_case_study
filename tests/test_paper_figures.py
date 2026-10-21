import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from tf_scripts.paper_figures import image_paths, gt_for_image


class PaperFigureTests(unittest.TestCase):
    def test_image_mask_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/"screw"
            for folder in (root/"test"/"good",root/"test"/"scratch_head",
                           root/"ground_truth"/"scratch_head"):
                folder.mkdir(parents=True)
            Image.new("RGB",(32,16),"gray").save(root/"test"/"good"/"000.png")
            Image.new("RGB",(32,16),"gray").save(root/"test"/"scratch_head"/"000.png")
            mask=np.zeros((16,32),dtype=np.uint8)
            mask[5:8,3:6]=255
            Image.fromarray(mask).save(root/"ground_truth"/"scratch_head"/"000_mask.png")
            paths=image_paths(root,one_per_class=True)
            self.assertEqual([p.parent.name for p in paths],["good","scratch_head"])
            self.assertEqual(gt_for_image(root,paths[0]).sum(),0)
            self.assertGreater(gt_for_image(root,paths[1]).sum(),0)


if __name__ == "__main__":
    unittest.main()
