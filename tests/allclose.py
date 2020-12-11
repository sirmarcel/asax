import numpy as np


class AllClose:
    """Mixin class for np.test.assert_allclose"""

    def assertAllClose(self, actual, desired, rtol=1e-7, atol=1e-15, equal_nan=True):
        np.testing.assert_allclose(
            actual, desired, rtol=rtol, atol=atol, equal_nan=True
        )
