"""Test for module opw """


from itertools import product
import numpy as np

import pytest

import ot
from ot.backend import torch, tf

@pytest.mark.parametrize("verbose, warn", product([True, False], [True, False]))
def test_sinkhorn(verbose, warn):
    n = 100
    rng = np.random.RandomState(0)

    x = rng.randn(n, 2)
    u = ot.utils.unif(n)

    M = ot.dist(x, x)
    G = ot.opw_sinkhorn(u, u, M, 50, 0.1, 1, stopThr=1e-10, verbose=verbose, warn=warn)

    #check constraints
    np.testing.assert_allclose(
        u, G.sum(1), atol=1e-05
    )
    np.testing.assert_allclose(
        u, G.sum(0), atol=1e-05
    )

    with pytest.warns(UserWarning):
        ot.opw_sinkhorn(u, u, M, 50, 0.1, 1, stopThr=0, numItermax=1)
