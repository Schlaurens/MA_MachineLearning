import tensorflow as tf


def get_encoder(
    encoder_architecture: str,
    height: int,
    width: int,
    category_names: list[str],
    n_context: int,
    **kwargs,
):
    """Return the specified encoder model

    Args:
        encoder_architecture: The name of the encoder architecture
        height: The height of the input image
        width: The width of the input image
        category_names: The names of the different object categories the encoder needs to be able to detect
        n_context: The size of the context vector

    Raises:
        ValueError: When the provided encoder architecture is unknown

    Returns:
        A tf.keras.Model with the provided architecture
    """
    if encoder_architecture == "default":
        return _get_encoder_default(height, width, category_names, n_context, **kwargs)
    if encoder_architecture == "default_heavy":
        return _get_encoder_heavy(height, width, category_names, n_context, **kwargs)
    else:
        raise ValueError(f"Unknown encoder name: {encoder_architecture}")


def _get_common_output(x, category_names, n_context, image):
    """Return the common output logic for every encoder architecture. The different outputs for each categories are concatenated here.

    Args:
        x: The tensor output of the hidden encoder layers
        category_names: The names of the object categories. Used to build the output
        n_context: The size of the context vector
        image: The input image. Used to build the tf.keras.Model

    Returns:
        A tf.keras.Model
    """
    output = []
    for name in category_names:
        # TODO: some activated stuff here?
        offset = tf.keras.layers.Conv2D(2, 1)(x)

        x = tf.keras.layers.Conv2D(1, 1)(x)
        interest = tf.keras.layers.Activation("sigmoid")(x)

        output += [tf.keras.layers.Concatenate(name=name)([offset, interest])]

    if n_context > 0:
        context = tf.keras.layers.Conv2D(n_context, 1, name="context")(x)
        output += [context]

    return tf.keras.Model(
        image, output, name="encoder"
    )  # input: image, output: [offset, interest] for each category + context


def _get_encoder_heavy(height, width, category_names, n_context):
    image = tf.keras.layers.Input((height, width, 4))
    x = image
    
    # 480x320x4
    x = tf.keras.layers.Conv2D(16, 3, strides=(2, 1), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # 240x320x16
    x = tf.keras.layers.Conv2D(32, 3, strides=(2, 2), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # 120x160x32
    x = tf.keras.layers.Conv2D(32, 3, strides=(2, 2), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # 60x80x32
    x = tf.keras.layers.Conv2D(64, 3, strides=(2, 2), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # 30x40x64
    x = tf.keras.layers.Conv2D(64, 3, strides=(2, 2), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)
    
    # 15x20x64
    return _get_common_output(x, category_names, n_context, image)

def _get_encoder_default(height, width, category_names, n_context):
    image = tf.keras.layers.Input((height, width, 4))
    # TODO: input [B, H, W/2, 4] (treat each YUYV tuple as a pixel)
    x = image
    
    # 480x320x4
    x = tf.keras.layers.Conv2D(32, 3, strides=(2, 1), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # 240x320x32
    # ires-block(16, expansion=1)

    # 240x320x16
    # ires-block(24, stride=2, expansion=6)
    # ires-block(24, stride=1, expansion=6)

    x = tf.keras.layers.Conv2D(32, 3, strides=(2, 2), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # 120x160x24
    # ires-block(32, stride=2, expansion=6)
    # ires-block(32, stride=1, expansion=6)
    # ires-block(32, stride=1, expansion=6)

    x = tf.keras.layers.Conv2D(32, 3, strides=(2, 2), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # 60x80x32

    # ires-block(64, stride=2, expansion=6)
    # ires-block(64, stride=1, expansion=6)
    # ires-block(64, stride=1, expansion=6)
    # ires-block(64, stride=1, expansion=6)

    x = tf.keras.layers.Conv2D(32, 3, strides=(2, 2), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # 30x40x64

    # ires-block(96, stride=1, expansion=6)
    # ires-block(96, stride=1, expansion=6)
    # ires-block(96, stride=1, expansion=6)

    # 30x40x96

    # ires-block(160, stride=2, expansion=6)
    # ires-block(160, stride=1, expansion=6)
    # ires-block(160, stride=1, expansion=6)

    x = tf.keras.layers.Conv2D(32, 3, strides=(2, 2), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    # 15x20x160

    # ires-block(320, stride=1, expansion=6)

    return _get_common_output(x, category_names, n_context, image)
