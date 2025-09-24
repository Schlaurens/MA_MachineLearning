import tensorflow as tf


class BaseEncoder(tf.keras.Model):
    def __init__(self, height, width, category_names, n_context, **kwargs):
        super().__init__(**kwargs)
        self.height = height
        self.width = width
        self.category_names = category_names
        self.n_context = n_context

    def build_common_output(self, x):
        output = []
        for name in self.category_names:
            # TODO: some activated stuff here?
            offset = tf.keras.layers.Conv2D(2, 1)(x)
            x = tf.keras.layers.Conv2D(1, 1)(x)  # < ???
            interest = tf.keras.layers.Activation("sigmoid")(x)
            output += [tf.keras.layers.Concatenate(name=name)([offset, interest])]
        if self.n_context > 0:
            context = tf.keras.layers.Conv2D(self.n_context, 1, name="context")(x)
            output += [context]
        return output  # output: [offset, interest] for each category + context

    def call(self, inputs, training=False):
        raise NotImplementedError("Subclasses must implement this method.")


class DefaultEncoder(BaseEncoder):
    def __init__(self, height, width, category_names, n_context, **kwargs):
        super().__init__(height, width, category_names, n_context, **kwargs)

    def call(self, image, training=False):
        x = image
        # 480x320x4
        x = tf.keras.layers.Conv2D(32, 3, strides=(2, 1), padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(scale=False)(x)
        x = tf.keras.layers.ReLU(6.0)(x)
        # 240x320x32
        x = tf.keras.layers.Conv2D(32, 3, strides=(2, 2), padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(scale=False)(x)
        x = tf.keras.layers.ReLU(6.0)(x)
        # 120x160x32
        x = tf.keras.layers.Conv2D(32, 3, strides=(2, 2), padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(scale=False)(x)
        x = tf.keras.layers.ReLU(6.0)(x)
        # 30x40x32
        x = tf.keras.layers.Conv2D(32, 3, strides=(2, 2), padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(scale=False)(x)
        x = tf.keras.layers.ReLU(6.0)(x)
        # 30x40x32
        x = tf.keras.layers.Conv2D(32, 3, strides=(2, 2), padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(scale=False)(x)
        x = tf.keras.layers.ReLU(6.0)(x)

        return self.build_common_output(x)


def get_encoder(encoder_name, height, width, category_names, n_context, **kwargs):
    if encoder_name == "default":
        return DefaultEncoder(height, width, category_names, n_context, **kwargs)
    else:
        raise ValueError(f"Unknown encoder name: {encoder_name}")
