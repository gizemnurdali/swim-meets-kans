from keras.models import Sequential
from keras.layers import Input, Dense
from sklearn.pipeline import Pipeline
from swimnetworks import Base as BaseSWIM, Dense as DenseSWIM, Linear as LinearSWIM


def get_activation(activation):
    if activation == BaseSWIM.identity_activation:
        return lambda x: x
    elif activation == BaseSWIM.relu_activation:
        return "relu"
    elif activation == BaseSWIM.tanh_activation:
        return "tanh"
    else:
        raise ValueError(f"Unknown activation function: {activation}.")


def swim_to_keras_layer(layer):
    """Transforms a SWIM Dense or Linear layer into the corresponding keras one."""
    if not(isinstance(layer, DenseSWIM)) and not(isinstance(layer, LinearSWIM)):
        raise ValueError("Could transform only Dense and Linear layers into keras.")
    activation = get_activation(layer.activation)
    fcn_layer = Dense(layer.layer_width, activation=activation)
    return fcn_layer

def set_keras_weights(swim_layer, keras_layer):
    """Sets weights of a keras layer to the weight of a swim layer."""
    if not(isinstance(swim_layer, DenseSWIM)) and not(isinstance(swim_layer, LinearSWIM)):
        raise ValueError("Could set weights only SWIM from Dense and Linear layers.")
    bias = swim_layer.biases.reshape(-1)
    keras_layer.set_weights([swim_layer.weights, bias])


def swim_to_keras_model(
    random_basis_pipeline: Pipeline, input_shape: tuple, set_weights=False, **compile_params
):
    if input_shape is not None:
        layers = [Input(input_shape)]
    else:
        layers = []
    for transformer_name in random_basis_pipeline.named_steps.keys():
        transformer_ = random_basis_pipeline.named_steps[transformer_name]
        if isinstance(transformer_, BaseSWIM):
            layers.append(swim_to_keras_layer(transformer_))
    model = Sequential(layers)
    model.compile(**compile_params)

    if set_weights:
        k_layer = 0
        for transformer_name in random_basis_pipeline.named_steps.keys():
            transformer_ = random_basis_pipeline.named_steps[transformer_name]
            if isinstance(transformer_, BaseSWIM):
                set_keras_weights(transformer_, model.layers[k_layer])
                k_layer += 1
    return model
