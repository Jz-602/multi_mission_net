"""CPU GT [N,7] current-car xyz/lwh/yaw checks for half-open XY encoding."""
from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from Assign import AssignLabel


class AssignXYBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.assign = AssignLabel([['vehicle'], ['pedestrian', 'cyclist']], .1, 500, 2,
                                  [-24, -24, -7, 24, 24, 5.8], [.05, .05, .05], [4, 4])

    def Encode(self, x, y, name='vehicle'):
        box = np.array([[x, y, -1., 4., 2., 1.5, .75]], dtype=np.float32)
        return self.assign({'annotations': {'gt_boxes': box, 'gt_names': np.array([name])}})

    def TestOutsideAndRealLowerBoundaryFrames(self):
        for x, y in [(-24.1, 0), (0, -24.1), (-24.111726760864258, -3.1060948371887207),
                     (-24.183916091918945, .47848770022392273), (24., 0), (0, 24.)]:
            with self.subTest(x=x, y=y):
                result = self.Encode(x, y)
                self.assertEqual(sum(int(mask.sum()) for mask in result['mask']), 0)
                self.assertEqual(sum(float(hm.sum()) for hm in result['hm']), 0.)

    def TestInsideAndExactLowerBoundary(self):
        for name, task, local_class in [('vehicle', 0, 0), ('pedestrian', 1, 0), ('cyclist', 1, 1)]:
            for x, y in [(-24., -24.), (-23.999, 0.), (0., -23.999), (23.999, 23.999), (12.37, -7.83)]:
                with self.subTest(name=name, x=x, y=y):
                    result = self.Encode(x, y, name)
                    self.assertEqual(int(result['mask'][task].sum()), 1)
                    self.assertEqual(int(result['cat'][task][0]), local_class)
                    ct = np.asarray([(np.float32(x)+24)/.2, (np.float32(y)+24)/.2], dtype=np.float32)
                    integer = ct.astype(np.int32)
                    self.assertEqual(int(result['ind'][task][0]), int(integer[1]*240+integer[0]))
                    expected = np.r_[ct-integer, -1., np.log([4., 2., 1.5]), np.sin(.75), np.cos(.75)]
                    np.testing.assert_allclose(result['anno_box'][task][0], expected, atol=1e-7)
                    self.assertEqual(float(result['hm'][task][local_class, integer[1], integer[0]]), 1.)

    def TestZDefinitionUnchanged(self):
        # This patch preserves existing XY-only training eligibility; z is not newly filtered.
        box = np.array([[0., 0., 6., 4., 2., 1.5, 0.]], dtype=np.float32)
        result = self.assign({'annotations': {'gt_boxes': box, 'gt_names': np.array(['vehicle'])}})
        self.assertEqual(int(result['mask'][0].sum()), 1)
        self.assertEqual(float(result['anno_box'][0][0, 2]), 6.)


if __name__ == '__main__':
    loader = unittest.TestLoader()
    loader.testMethodPrefix = 'Test'
    unittest.main(testLoader=loader)
