import tensorflow as tf
import tensorflow.keras as keras


class MLP(keras.Model):
    
    def __init__(
            self,
            input_size: int,
            hidden_layers_sizes: list[int],
            output_size: int,
            hidden_activation: str = 'leaky_relu',
            last_activation: str = 'leaky_relu',
            add_dropout: bool = False,
            dropout_prob: float = 0.2,
            add_batchnorm: bool = True,
            use_bias: bool = True,
            name: str = 'MLP',
        ):
        super().__init__(name=name)
        
        # Define the layers of the MLP
        layers = []
        previous_size = input_size
        self.dropout_prob = dropout_prob

        # Add hidden layers
        for i, hidden_size in enumerate(hidden_layers_sizes):
            layers.append(
                keras.layers.Dense(
                    units = hidden_size,
                    activation = hidden_activation,
                    input_shape = (previous_size,),
                    use_bias = use_bias,
                    # kernel_regularizer = keras.regularizers.L2(1e-5),
                    name = f"dense{i:d}",
                    ),
            )
            if add_batchnorm:
                layers.append(keras.layers.BatchNormalization())
            if add_dropout:
                layers.append(keras.layers.Dropout(rate = dropout_prob, name = f"dropout{i:d}",))
            previous_size = hidden_size

        # Add output layer
        layers.append(
            keras.layers.Dense(
                units = output_size,
                activation = last_activation,
                input_shape = (previous_size,),
                use_bias = use_bias,
                name = f"dense{i+1:d}",
                ),
        )
        
        # Create sequential model
        self.model = keras.models.Sequential(layers, name=name)
        
    def call(self, inputs):
        return self.model(inputs)