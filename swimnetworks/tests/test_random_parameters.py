from unittest import TestCase

import numpy as np

from swimnetworks import Dense


class TestDataAgnosticSampler(TestCase):
    """
    For testing a data-agnostic sampler behavior in random feature networks. (e.g. ELM)

    Note: this test class does not look into specifics of data-driven sampling.
    """
    width: int = 10000
    input_dim: int = 2
    parameter_sampler = "random"  # (data-agnostic)
    data_distribution = [np.random.uniform,
                         np.random.normal,
                         np.random.exponential,
                         np.random.standard_cauchy,
                         ]

    def setUp(self):
        """
        Initializes a random-feature layer network.
        """
        self.dense = Dense(layer_width=self.width, parameter_sampler=self.parameter_sampler)
        # this test class does not look into specifics of data-driven sampling
        self.dense.sample_uniformly = True

    def assert_param_statistics(self, x):
        """
        Asserts statistics of weights and biases after random-feature sampling with the random sampler.
        """
        self.dense.__post_init__()

        self.dense.fit(x)

        # assert that weights have mean 0 and std 1 (standard Gaussian)
        self.assertAlmostEqual(np.mean(self.dense.weights), 0, places=1,
                               msg=f"The mean of weights is {np.mean(self.dense.weights)} but should be close to 0.")

        self.assertAlmostEqual(np.std(self.dense.weights), 1, places=1,
                               msg=f"The standard deviation of weights is {np.mean(self.dense.weights)} but should be close to 1.")

        # assert that biases have mean 0, min=-π, max=π (uniform from -π to π)
        self.assertAlmostEqual(np.mean(self.dense.biases), 0, places=1,
                               msg=f"The mean of biases is {np.mean(self.dense.biases)} but should be close to 0.")

        self.assertAlmostEqual(np.min(self.dense.biases), -np.pi, places=1,
                               msg=f"The minimum of biases is {np.mean(self.dense.biases)} but should be close to -3.14 (=-π).")

        self.assertAlmostEqual(np.max(self.dense.biases), np.pi, places=1,
                               msg=f"The maximum of biases is {np.mean(self.dense.biases)} but should be close to 3.14 (=π).")

    def assert_indices_not_set(self):
        """
        Asserts that indices in Dense are not set.
        """
        # assert that idx are None
        self.assertIsNone(self.dense.idx_to,
                          msg="Indices to are set for data-agnostic sampling, this should not happen.")
        self.assertIsNone(self.dense.idx_from,
                          msg="Indices to are set for data-agnostic sampling, this should not happen.")

    def test_param_statistics(self):
        """
        Asserts that for changing datasets parameters have same statistics and indices are not set.
        """
        for dist in self.data_distribution:
            x = dist(size=(10000, self.input_dim))
            self.assert_param_statistics(x)
            self.assert_indices_not_set()

    def test_data_agnostic_sampling(self):
        """
        Asserts that for changing datasets the parameters will be exactly the same if a seed is set.
        """
        layer_width = 10
        n_points = 10000
        seed = 123
        self.dense = Dense(layer_width=layer_width, parameter_sampler=self.parameter_sampler, random_seed=seed)

        # initialize with data from Gamma(10) distribution
        x = np.random.gamma(10, size=(n_points, self.input_dim))
        self.dense.fit(x)
        # store parameters
        weights = np.array(self.dense.weights)
        biases = np.array(self.dense.biases)

        for dist in self.data_distribution:
            x = dist(size=(n_points, self.input_dim))
            # assert that parameters are same as for dataset Gamma distribution
            self.dense = Dense(layer_width=layer_width, parameter_sampler=self.parameter_sampler, random_seed=seed)
            self.dense.fit(x)
            np.testing.assert_array_almost_equal(weights, self.dense.weights,
                                                 err_msg="Weights have changed for a different dataset, "
                                                         "even though data-agnostic sampling is used.")
            np.testing.assert_array_almost_equal(biases, self.dense.biases,
                                                 err_msg="Biases have changed for a different dataset, "
                                                         "even though data-agnostic sampling is used.")
