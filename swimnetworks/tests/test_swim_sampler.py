from unittest import TestCase
from swimnetworks import Dense
import numpy as np

class TestSWIMSampler(TestCase):
    """
    For testing SWIM (data-driven) specific sampling.
    """
    layer_width: int = 10
    input_dim: int = 2
    data_distributions = [
        np.random.uniform,
        np.random.normal,
        np.random.exponential,
        np.random.standard_cauchy,
    ]
    param_samplers = [ "tanh", "relu" ]

    def setUp(self):
        """
        Initializes a random-feature layer and a random 2D input.
        """
        self.dense = Dense(layer_width=self.layer_width, random_seed=19587346)

    def assert_all_arrays_different(self, arr_list, err_msg):
        for i in range(len(arr_list)):
            for j in range(i + 1, len(arr_list)):
                np.testing.assert_raises(
                    AssertionError,
                    np.testing.assert_array_almost_equal, arr_list[i], arr_list[j],
                    err_msg=err_msg + f"Parameters sampled from data distribution \
                            {self.data_distributions[i]} and {self.data_distributions[j]} resulted \
                            in the same parameter set."
                )

    def assert_probabilities_swim(self, n_points, n_dim, indices=[], expect_uniform=True):
        """
        Sets the gradients of candidates at the specified indices very large and asserts the
        SWIM probabilities according to the expect_uniform parameter.
        """
        dy = np.zeros((n_points, n_dim), dtype=np.float64)
        dy[indices] = 100.0 # large gradients, it should be the most weighted one in the SWIM probability distribution
        dists = np.ones((n_points, 1), dtype=np.float64)
        probs = self.dense.weight_probabilities(dy, dists)

        if expect_uniform:
            expected_probs = np.full_like(probs, 1.0 / len(probs))
        else:
            expected_probs = np.zeros_like(probs)
            expected_probs[indices] = 1.0 / len(indices) # expect very high probability at this candidate

        np.testing.assert_array_almost_equal(probs, expected_probs,
                                             err_msg=f"Got probabilities:\n f{probs}\nExpected:\n{expected_probs}")

    def test_idxfrom_idxto_set_after_sampling(self):
        self.dense.sample_uniformly = True
        x = np.random.uniform(-1.0, 1.0, size=(10, 2))
        for param_sampler in self.param_samplers:
            self.dense.parameter_sampler = param_sampler
            self.dense.__post_init__()
            self.assertIsNone(self.dense.idx_from)
            self.assertIsNone(self.dense.idx_to)
            self.dense.fit(x)
            # assert idx_to and idx_from
            self.assertTrue(isinstance(self.dense.idx_from, np.ndarray),
                            f"Expected np.ndarray but got idx_from type = {type(self.dense.idx_from)}")
            self.assertTrue(isinstance(self.dense.idx_to, np.ndarray),
                            f"Expected np.ndarray but got idx_to type = {type(self.dense.idx_to)}")
            # assert that idx values are integers
            self.assertTrue(all(idx.is_integer() for idx in self.dense.idx_from),
                            f"Expected all values in idx_from to be of type integer, "
                            f"but got {[idx.is_integer() for idx in self.dense.idx_from]}")
            self.assertTrue(all(idx.is_integer() for idx in self.dense.idx_to),
                            f"Expected all values in idx_to to be of type integer, "
                            f"but got {[idx.is_integer() for idx in self.dense.idx_to]}")
            # reset
            self.dense.idx_from = None
            self.dense.idx_to = None

    def test_probabilities_swim(self):
        """
        Tests the SWIM probability distribution (high probabilities where the gradients are large)
        """
        # Test with one candidate having very large gradient others zero
        self.dense.sample_uniformly = False # for gradient based sampling
        self.assert_probabilities_swim(10, 1, indices=[-1], expect_uniform=False)
        # Test with three candidates having very large gradients others zero
        self.assert_probabilities_swim(10, 1, indices=[1, -1, -2], expect_uniform=False)

        self.dense.sample_uniformly = True # for uniform sampling
        self.assert_probabilities_swim(10, 1, indices=[-1], expect_uniform=True)
        self.assert_probabilities_swim(10, 1, indices=[1, -1, -2], expect_uniform=True)

    def test_probabilities_uniform_if_gradients_zero(self):
        """
        Tests whether the SWIM probability distribution fallback to uniform when having gradients
        close to zero.
        """
        self.dense.sample_uniformly = False # for gradient based sampling, but we will fallback to
                                            # uniform sampling due to zero gradients hopefully

        # Here we specify indices=[] which leaves all the gradients around zero and this should fallback
        # to uniform sampling
        self.assert_probabilities_swim(10, 1, indices=[], expect_uniform=True)

    def test_data_driven_sampling(self):
        """
        Asserts that for changing datasets the parameters are changing even with a fixed seed (data-driven sampling)
        """
        n_points = 10_000
        self.dense.sample_uniformly = True
        for param_sampler in self.param_samplers:
            weights = []
            biases = []
            for data_distribution in self.data_distributions:
                x = data_distribution(size=(n_points, self.input_dim))
                self.dense.parameter_sampler = param_sampler
                self.dense.__post_init__()
                self.dense.fit(x)
                weights.append(self.dense.weights.copy())
                biases.append(self.dense.biases.copy())

            # assert all the parameters are different
            self.assert_all_arrays_different(weights,
                                             err_msg=f"Data-driven test failed when using the \
                                             parameter sampler '{param_sampler}' to sample the weights.")
            self.assert_all_arrays_different(biases,
                                             err_msg=f"Data-driven test failed when using the \
                                             parameter sampler '{param_sampler}' to sample the biases.")
