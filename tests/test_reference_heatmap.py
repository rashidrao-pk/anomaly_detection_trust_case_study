import unittest
import numpy as np
from tf_scripts.xai_utils import reference_heatmap

class ReferenceStyleTests(unittest.TestCase):
    def test_paper_bwr_symmetric_limits(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        try:
            artist = reference_heatmap(ax, np.array([[-0.5, 0.2],[0.1, 0.4]]))
            self.assertEqual(artist.get_cmap().name, 'bwr')
            self.assertEqual(artist.get_clim(), (-0.5, 0.5))
        finally:
            plt.close(fig)

if __name__ == '__main__':
    unittest.main()
