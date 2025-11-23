import tensorflow as tf
from tensorflow import keras
from keras import backend as K

class DefuzzyLayer(keras.layers.Layer):

    def __init__(self,
                 output_dim,
                 initial_rules_outcomes=None,
                 **kwargs):

        super(DefuzzyLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.initial_rules_outcomes = initial_rules_outcomes

    def build(self, input_shape):
        self.input_dimensions = list(input_shape)[:-1:-1]

        if self.initial_rules_outcomes is None:
            init_values = tf.random_uniform_initializer()(
                shape=(input_shape[-1], self.output_dim),
                dtype="float32"
            )
        else:
            init_values = tf.convert_to_tensor(self.initial_rules_outcomes, dtype="float32")

        self.rules_outcome = self.add_weight(
            name="rules",
            shape=(input_shape[-1], self.output_dim),
            initializer=tf.initializers.constant(init_values),
            trainable=True
        )

        super(DefuzzyLayer, self).build(input_shape)

    def call(self, x):
        aligned_x = K.repeat_elements(K.expand_dims(x, axis=-1), self.output_dim, -1)
        aligned_rules = self.rules_outcome

        for dim in self.input_dimensions:
            aligned_rules = K.repeat_elements(K.expand_dims(aligned_rules, 0), dim, 0)

        return K.sum(aligned_x * aligned_rules, axis=-2, keepdims=False)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.output_dim,)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "output_dim": self.output_dim,
            "initial_rules_outcomes": self.initial_rules_outcomes
        })
        return cfg
