import random
import numpy as np

from ocp_app.core.seeds import fix_all


def test_fix_all_makes_random_and_numpy_reproducible():
    fix_all(123)

    py_rand_1 = random.random()
    np_rand_1 = np.random.rand(5)

    fix_all(123)

    py_rand_2 = random.random()
    np_rand_2 = np.random.rand(5)

    assert py_rand_1 == py_rand_2
    assert np.allclose(np_rand_1, np_rand_2)
