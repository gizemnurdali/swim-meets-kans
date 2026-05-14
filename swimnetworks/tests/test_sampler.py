from unittest import TestCase

import numpy as np

from swimnetworks import Dense


class TestSampler(TestCase):
    """
    For testing general sampler behavior in random feature networks.

    Note: this test class does not look into specifics of data-driven sampling.
    """
    width: int = 16
    input_dim: int = 2
    parameter_samplers = [
        "random",  #  data-agnostic
        "relu",  # relu-SWIM (data-driven)
        "tanh",  # tanh-SWIM (data-driven)
    ]
    dtypes = [ np.float64 ]

    def setUp(self):
        """
        Initializes a random-feature layer and a random 2D input.
        """
        self.dense = Dense(layer_width=self.width)
        # this test class does not look into specifics of data-driven sampling
        self.dense.sample_uniformly = True
        # 2D input
        self.x = np.random.uniform(low=-1.0, high=1.0, size=(100, self.input_dim))

    def assert_param_dimensions(self, parameter_sampler):
        """
        Asserts dimensions after random-feature sampling with the specified sampler.
        """
        self.dense.parameter_sampler = parameter_sampler
        self.dense.__post_init__()

        self.dense.fit(self.x)

        self.assertEqual(self.dense.weights.shape, (2, self.width),
                         f"Dense weights shape wrong after sampling with {parameter_sampler}.")
        self.assertEqual(self.dense.biases.shape, (1, self.width),
                         f"Dense biases shape wrong after sampling with {parameter_sampler}.")

    def assert_parameter_initialization(self, parameter_sampler):
        """
        Checks that weights and biases are not None
        """
        self.dense.parameter_sampler = parameter_sampler
        self.dense.__post_init__()

        self.dense.fit(self.x)

        # Assert that the following parameters are being updated after sampling
        self.assertIsNotNone(self.dense.weights,
                             f"Weighs are None after sampling with {parameter_sampler}")
        self.assertIsNotNone(self.dense.biases,
                             f"Biases are None after sampling with {parameter_sampler}")

        self.assertEqual(self.dense.input_shape, self.input_dim,
                         f"Input shape is not matching when sampling with {parameter_sampler}.")
        self.assertEqual(self.dense.output_shape, self.width,
                         f"Output shape is not matching when sampling with {parameter_sampler}.")

    def assert_changed_params(self, parameter_sampler):
        """
        Asserts whether the parameters (weights and biases) are actually being updated when sampling.
        """
        self.dense.parameter_sampler = parameter_sampler
        self.dense.__post_init__()
        # set all weights to a huge value to see whether the values are changing or not
        self.dense.weights = np.full_like(self.dense.weights, np.inf)
        self.dense.biases = np.full_like(self.dense.biases, np.inf)
        weights_before_sampling = self.dense.weights
        biases_before_sampling = self.dense.biases

        self.dense.fit(self.x)

        self.assertFalse(np.isclose(weights_before_sampling, self.dense.weights).any(),
                         f"Dense weights did not change after sampling with {parameter_sampler}.")
        self.assertFalse(np.isclose(biases_before_sampling, self.dense.biases).any(),
                         f"Dense biases did not change after sampling with {parameter_sampler}.")

    def assert_n_parameters(self, parameter_sampler):
        """
        Checks if parameter n_parameters is set properly.
        """
        self.dense.parameter_sampler = parameter_sampler
        self.dense.__post_init__()
        self.dense.fit(self.x)

        # Assert that the following parameters are being updated after sampling
        assert (np.issubdtype(self.dense.n_parameters, np.integer)
                or np.issubdtype(self.dense.n_parameters, int)), \
            "Attribute n_parameters of Dense class is not of type int."
        self.assertEqual(self.dense.n_parameters, (self.input_dim * self.width + self.width),
                         f"Number of parameters is set to {self.dense.n_parameters} when sampling with "
                         f"{parameter_sampler} but it should be {self.input_dim * self.width + self.width}.")

    def assert_flag_is_classifer(self, parameter_sampler):
        """
        Checks correctness when is_classifier is changed
        """
        self.dense.parameter_sampler = parameter_sampler
        self.dense.__post_init__()
        # Save constant parameters before sampling
        assert isinstance(self.dense.is_classifier, bool), \
            "Attribute is_classifier of Dense class is not of type boolean."
        is_classifier = self.dense.is_classifier

        # Verify that for both can be set
        for flag in [True, False]:
            self.dense = Dense(layer_width=self.width, is_classifier=flag)
            self.dense.sample_uniformly = True
            is_classifier = self.dense.is_classifier

            self.dense.fit(self.x)

            # Assert that the following parameters are staying constant after sampling
            self.assertEqual(is_classifier, self.dense.is_classifier,
                             f"After sampling the is_classifier parameter has changed to "
                             f"{self.dense.is_classifier} but it should be {is_classifier}.")

    def assert_flag_prune_duplicates(self, parameter_sampler):
        """
        Checks correctness when prune_duplicates is changed
        """
        self.dense.parameter_sampler = parameter_sampler
        self.dense.__post_init__()
        # Save constant parameters before sampling
        assert isinstance(self.dense.prune_duplicates, bool), \
            "Attribute prune_duplicates of Dense class is not of type boolean."

        # Verify that for both can be set
        for flag in [True, False]:
            self.dense = Dense(layer_width=self.width, prune_duplicates=flag)
            self.dense.sample_uniformly = True
            prune_duplicates = self.dense.prune_duplicates

            self.dense.fit(self.x)

            # Assert that the following parameters are staying constant after sampling
            self.assertEqual(prune_duplicates, self.dense.prune_duplicates,
                             f"After sampling the prune_duplicates parameter has changed to "
                             f"{self.dense.prune_duplicates} but it should be {prune_duplicates}.")

    def assert_layer_variables(self, parameter_sampler):
        """
        Checks correct behavior for is_classifier attribute
        """
        self.dense.parameter_sampler = parameter_sampler
        self.dense.__post_init__()
        # Save constant parameters before sampling
        is_classifier = self.dense.is_classifier
        layer_width = self.dense.layer_width
        activation = self.dense.activation
        parameter_sampler = self.dense.parameter_sampler
        sample_uniformly = self.dense.sample_uniformly
        random_seed = self.dense.random_seed
        dist_min = self.dense.dist_min
        repetition_scaler = self.dense.repetition_scaler

        self.dense.fit(self.x)

        # Assert that the following parameters are staying constant after sampling
        self.assertEqual(layer_width, self.dense.layer_width,
                         f"After sampling the layer_width parameter has changed to "
                         f"{self.dense.layer_width} but it should be {layer_width}.")
        self.assertEqual(activation, self.dense.activation,
                         f"After sampling the activation parameter has changed to "
                         f"{self.dense.activation} but it should be {activation}.")
        self.assertEqual(parameter_sampler, self.dense.parameter_sampler,
                         f"After sampling the parameter_sampler has changed to "
                         f"{self.dense.parameter_sampler} but it should be {parameter_sampler}.")
        self.assertEqual(sample_uniformly, self.dense.sample_uniformly,
                         f"After sampling the sample_uniformly has changed to "
                         f"{self.dense.sample_uniformly} but it should be {sample_uniformly}.")
        self.assertEqual(random_seed, self.dense.random_seed,
                         f"After sampling the random_seed has changed to "
                         f"{self.dense.random_seed} but it should be {random_seed}.")
        self.assertEqual(dist_min, self.dense.dist_min,
                         f"After sampling the dist_min has changed to "
                         f"{self.dense.dist_min} but it should be {dist_min}.")
        self.assertEqual(repetition_scaler, self.dense.repetition_scaler,
                         f"After sampling the repetition_scaler has changed to "
                         f"{self.dense.repetition_scaler} but it should be {repetition_scaler}.")

    def assert_param_dtypes(self, dtype, parameter_sampler):
        """
        Sets the dtype of the input to the network as given using the specified parameter sampler
        and asserts that the fitted network has the parameters of the same dtype as the input.
        """
        self.dense.parameter_sampler = parameter_sampler
        self.dense.__post_init__()

        # set the dtype of input
        self.x = self.x.astype(dtype)

        self.dense.fit(self.x)

        # assert
        self.assertEqual(np.isdtype(self.dense.weights.dtype, dtype), True,
                         f"weights dtype are not matching, got: {self.dense.weights.dtype} but expected {dtype}")
        self.assertEqual(np.isdtype(self.dense.biases.dtype, dtype), True,
                         f"biases dtype are not matching, got: {self.dense.biases.dtype} but expected {dtype}")

    #####################################################################################

    def test_unsupported_sampler_names(self):
        """
        Verify that for unknown sampler names an error is thrown.
        """
        self.dense.parameter_sampler = "dummy_name"
        with self.assertRaises(ValueError):
            self.dense.__post_init__()

    def test_sampler_param_dimensions(self):
        """
        Asserts dimensions for all available samplers.
        """
        for sampler in self.parameter_samplers:
            self.assert_param_dimensions(sampler)

    def test_sampler_changed_params(self):
        """
        Asserts dimensions for all available samplers.
        """
        for sampler in self.parameter_samplers:
            self.assert_parameter_initialization(sampler)
            self.assert_changed_params(sampler)

    def test_layer_params_after_sampling(self):
        """
        Asserts that the variables in the layer (Dense) are changing or not changing properly.
        """
        for sampler in self.parameter_samplers:
            self.assert_layer_variables(sampler)
            self.assert_n_parameters(sampler)
            self.assert_flag_is_classifer(sampler)
            self.assert_flag_prune_duplicates(sampler)
            for dtype in self.dtypes:
                self.assert_param_dtypes(dtype, sampler)

