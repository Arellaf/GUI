from keras import backend as K
from tensorflow import keras
import tensorflow as tf

class FuzzyLayer(keras.layers.Layer):

    def __init__(self,
                 output_dim,
                 initial_centers=None,
                 initial_sigmas=None,
                 **kwargs):
        super(FuzzyLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.initial_centers = initial_centers
        self.initial_sigmas = initial_sigmas

    def build(self, input_shape):
        self.input_dimensions = list(input_shape)[:-1:-1]

        if self.initial_centers is None:
            c_init_values = tf.random_uniform_initializer(-1, 1)(
                shape=(input_shape[-1], self.output_dim), dtype="float32"
            )
        else:
            c_init_values = tf.convert_to_tensor(self.initial_centers, dtype="float32")

        self.c = self.add_weight(
            name="centers",
            shape=(input_shape[-1], self.output_dim),
            initializer=tf.initializers.constant(c_init_values),
            trainable=True
        )

        if self.initial_sigmas is None:
            a_init_values = tf.ones_initializer()(
                shape=(input_shape[-1], self.output_dim), dtype="float32"
            )
        else:
            a_init_values = tf.convert_to_tensor(self.initial_sigmas, dtype="float32")

        self.a = self.add_weight(
            name="sigmas",
            shape=(input_shape[-1], self.output_dim),
            initializer=tf.initializers.constant(a_init_values),
            trainable=True
        )

        super(FuzzyLayer, self).build(input_shape)

    def call(self, x):
        aligned_x = K.repeat_elements(K.expand_dims(x, axis=-1), self.output_dim, -1)
        aligned_c = self.c
        aligned_a = self.a

        for dim in self.input_dimensions:
            aligned_c = K.repeat_elements(K.expand_dims(aligned_c, 0), dim, 0)
            aligned_a = K.repeat_elements(K.expand_dims(aligned_a, 0), dim, 0)

        xc = K.exp(-K.sum(K.square((aligned_x - aligned_c) / (2 * aligned_a)),
                          axis=-2, keepdims=False))
        return xc

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.output_dim,)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "output_dim": self.output_dim,
            "initial_centers": self.initial_centers,
            "initial_sigmas": self.initial_sigmas
        })
        return cfg
