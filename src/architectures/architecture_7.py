import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore

# # Custom PixelShuffle Layer
# class PixelShuffle(layers.Layer):
#     def __init__(self, scale, **kwargs):
#         super(PixelShuffle, self).__init__(**kwargs)
#         self.scale = scale

#     def call(self, inputs):
#         return tf.nn.depth_to_space(inputs, block_size=self.scale)

# class AdaptiveAveragePooling2D(layers.Layer):
#     def __init__(self, pool_size=(2, 2), **kwargs):
#         super(AdaptiveAveragePooling2D, self).__init__(**kwargs)
#         self.pool_size = pool_size

#     def call(self, inputs):
#         input_shape = tf.shape(inputs)
#         target_height = input_shape[1] // self.pool_size[0]
#         target_width = input_shape[2] // self.pool_size[1]
#         return tf.image.resize(inputs, [target_height, target_width])

#     def get_config(self):
#         config = super().get_config()
#         config.update({"pool_size": self.pool_size})
#         return config



# def generator(input_shape=(None, None, 3)):
#     inputs = layers.Input(shape=input_shape)
    
#     # Original scale
#     x = layers.Conv2D(128, (3, 3), padding='same')(inputs)
#     x = layers.Activation('relu')(x)
    
#     # Original scale processing (scale1)
#     scale1 = x
#     for _ in range(3):
#         scale1 = rrdb(scale1, 128)

#     # Downscale by 2 (scale2) dynamically
#     scale2 = tf.image.resize(scale1, [tf.shape(scale1)[1] // 2, tf.shape(scale1)[2] // 2])
#     for _ in range(3):
#         scale2 = rrdb(scale2, 128)
#     # Upscale by 2 dynamically
#     scale2 = PixelShuffle(scale=2)(scale2)
    
#     # Upscale by 2 (scale4) dynamically
#     scale4 = tf.image.resize(scale1, [tf.shape(scale1)[1] * 2, tf.shape(scale1)[2] * 2])
#     for _ in range(3):
#         scale4 = rrdb(scale4, 128)
#     # Downscale by 2 dynamically
#     scale4 = tf.image.resize(scale4, [tf.shape(scale4)[1] // 2, tf.shape(scale4)[2] // 2])
    
#     # Concatenate multi-scale features
#     multi_scale = layers.Concatenate()([scale1, scale2, scale4])
    
#     # Additional convolutional layers
#     multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
#     multi_scale = layers.Activation('relu')(multi_scale)
    
#     # Upscale by 2 dynamically
#     multi_scale = PixelShuffle(scale=2)(multi_scale)
#     multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
#     multi_scale = layers.Activation('relu')(multi_scale)
    
#     # Upscale by 2 dynamically
#     multi_scale = PixelShuffle(scale=2)(multi_scale)
#     multi_scale = layers.Conv2D(3, (3, 3), padding='same')(multi_scale)
    
#     # Final output
#     outputs = layers.Conv2D(3, (3, 3), padding='same', activation='tanh')(multi_scale)
    
#     return Model(inputs, outputs)


# def discriminator(input_shape=(None, None, 3)):
#     inputs = layers.Input(shape=input_shape)

#     # Initial convolution block
#     x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
#     x = layers.LeakyReLU(alpha=0.2)(x)

#     # Multi-scale processing branches
#     scale1 = res_block(x, 32)
#     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
#     scale2 = res_block(scale2, 32)
#     scale2 = PixelShuffle(scale=2)(scale2)
#     scale3 = layers.AveragePooling2D(pool_size=(4, 4))(x)
#     scale3 = res_block(scale3, 32)
#     scale3 = PixelShuffle(scale=4)(scale3)

#     multi_scale = layers.Concatenate()([scale1, scale2, scale3])

#     # Additional convolutional layers after concatenation
#     x = layers.Conv2D(32, (3, 3), padding='same')(multi_scale)
#     x = layers.LeakyReLU(alpha=0.2)(x)
    
#     for filters in [64, 128, 256, 512]:
#         x = layers.Conv2D(filters, (4, 4), strides=(2, 2), padding='same')(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.LeakyReLU(negative_slope=0.2)(x)
    
#     x = layers.Conv2D(1, (4, 4), padding='same')(x)
#     return Model(inputs, x)

# 1st one
# def generator(input_shape=(None, None, 3)):
#     inputs = layers.Input(shape=input_shape)
    
#     # Original scale
#     x = layers.Conv2D(128, (3, 3), padding='same')(inputs)
#     x = layers.Activation('relu')(x)
    
#     # Original scale processing (scale1)
#     scale1 = x
#     for _ in range(3):
#         scale1 = rrdb(scale1, 128)

#     # Downscale by 2 (scale2)
#     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(scale1)
#     for _ in range(3):
#         scale2 = rrdb(scale2, 128)
#     # Upscale by 2
#     scale2 = PixelShuffle(scale=2)(scale2)
    
#     # Upscale by 2 (scale4)
#     scale4 = PixelShuffle(scale=2)(scale1)
#     for _ in range(3):
#         scale4 = rrdb(scale4, 128)
#     # Downscale by 2
#     scale4 = layers.AveragePooling2D(pool_size=(2, 2))(scale4)
    
#     # Concatenate multi-scale features
#     multi_scale = layers.Concatenate()([scale1, scale2, scale4])
    
#     # Additional convolutional layers
#     multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
#     multi_scale = layers.Activation('relu')(multi_scale)
    
#     # Upscale by 2
#     multi_scale = PixelShuffle(scale=2)(multi_scale)
#     multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
#     multi_scale = layers.Activation('relu')(multi_scale)
    
#     # Upscale by 2
#     multi_scale = PixelShuffle(scale=2)(multi_scale)
#     multi_scale = layers.Conv2D(3, (3, 3), padding='same')(multi_scale)
    
#     # Final output
#     outputs = layers.Conv2D(3, (3, 3), padding='same', activation='tanh')(multi_scale)
    
#     return Model(inputs, outputs)
# def discriminator(input_shape=(None, None, 3)):
#     inputs = layers.Input(shape=input_shape)

#     # Initial convolution block
#     x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
#     x = layers.LeakyReLU(alpha=0.2)(x)

#     # Multi-scale processing branches
#     scale1 = res_block(x, 32)
#     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
#     scale2 = res_block(scale2, 32)
#     scale2 = PixelShuffle(scale=2)(scale2)
#     scale3 = layers.AveragePooling2D(pool_size=(4, 4))(x)
#     scale3 = res_block(scale3, 32)
#     scale3 = PixelShuffle(scale=4)(scale3)

#     multi_scale = layers.Concatenate()([scale1, scale2, scale3])

#     # Additional convolutional layers after concatenation
#     x = layers.Conv2D(32, (3, 3), padding='same')(multi_scale)
#     x = layers.LeakyReLU(alpha=0.2)(x)
    
#     for filters in [64, 128, 256, 512]:
#         x = layers.Conv2D(filters, (4, 4), strides=(2, 2), padding='same')(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.LeakyReLU(negative_slope=0.2)(x)
    
#     x = layers.Conv2D(1, (4, 4), padding='same')(x)
#     return Model(inputs, x)


class PixelShuffle(layers.Layer):
    def __init__(self, scale, **kwargs):
        super(PixelShuffle, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        output = tf.nn.depth_to_space(inputs, block_size=self.scale)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"scale": self.scale})
        return config

# Define Residual Dense Block (RDB) with Mixed Convolution Types
def residual_dense_block(x, filters, growth_rate=4, layers_in_block=4):
    concat_features = [x]
    for i in range(layers_in_block):
        x = layers.Conv2D(growth_rate, (3, 3), padding='same')(x)  # Standard convolution
        x = layers.Activation('relu')(x)
        
        # Uncomment the following lines to use Dilated convolution
        x = layers.Conv2D(growth_rate, (3, 3), padding='same', dilation_rate=2)(x)
        x = layers.Activation('relu')(x)

        # Uncomment the following lines to use Depthwise separable convolution
        x = layers.SeparableConv2D(growth_rate, (3, 3), padding='same')(x)
        x = layers.Activation('relu')(x)
        
        concat_features.append(x)
        x = layers.Concatenate()(concat_features)
    
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    return x

# Define Residual-in-Residual Dense Block (RRDB)
def rrdb(x, filters, growth_rate=4, res_block=4):
    res = layers.Conv2D(filters, (3, 3), padding='same')(x)
    for i in range(res_block):
        x = residual_dense_block(x, filters, growth_rate)
    output = layers.Add()([x, layers.Lambda(lambda x: x * 0.2)(res)])
    return output

# Residual Block for Discriminator
def res_block(x, filters):
    res = x
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    output = layers.Add()([x, res])
    return output

# Generator
def generator(input_shape=(None, None, 3)):
    inputs = layers.Input(shape=input_shape)
    
    # Original scale
    x = layers.Conv2D(128, (3, 3), padding='same')(inputs)
    x = layers.Activation('relu')(x)
    print("Generator scale1 initial output:", x)
    
    # Original scale processing (scale1)
    scale1 = x
    for i in range(3):
        print(f"Generator scale1 RRDB block {i} input:", scale1)
        scale1 = rrdb(scale1, 128)
        print(f"Generator scale1 RRDB block {i} output:", scale1)

    # Downscale by 2 (scale2)
    scale2 = layers.AveragePooling2D(pool_size=(2, 2))(scale1)
    print("Generator scale2 input after pooling:", scale2)
    for i in range(3):
        print(f"Generator scale2 RRDB block {i} input:", scale2)
        scale2 = rrdb(scale2, 128)
        print(f"Generator scale2 RRDB block {i} output:", scale2)
    # Upscale by 2
    scale2 = PixelShuffle(scale=2)(scale2)
    print("Generator scale2 output after PixelShuffle:", scale2)
    
    # Upscale by 2 (scale4)
    scale4 = PixelShuffle(scale=2)(scale1)
    print("Generator scale4 input after PixelShuffle:", scale4)
    for i in range(3):
        print(f"Generator scale4 RRDB block {i} input:", scale4)
        scale4 = rrdb(scale4, 128)
        print(f"Generator scale4 RRDB block {i} output:", scale4)
    # Downscale by 2
    scale4 = layers.AveragePooling2D(pool_size=(2, 2))(scale4)
    print("Generator scale4 output after pooling:", scale4)

    # Concatenate multi-scale features
    multi_scale = layers.Concatenate()([scale1, scale2, scale4])
    print("Generator multi_scale output:", multi_scale)
    
    # Additional convolutional layers
    multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
    multi_scale = layers.Activation('relu')(multi_scale)
    print("Generator additional conv layers output:", multi_scale)
    
    # Upscale by 2
    multi_scale = PixelShuffle(scale=2)(multi_scale)
    multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
    multi_scale = layers.Activation('relu')(multi_scale)
    print("Generator output before final upscaling:", multi_scale)
    
    # Upscale by 2
    multi_scale = PixelShuffle(scale=2)(multi_scale)
    multi_scale = layers.Conv2D(3, (3, 3), padding='same')(multi_scale)
    print("Generator output before final activation:", multi_scale)
    
    # Final output
    outputs = layers.Conv2D(3, (3, 3), padding='same', activation='tanh')(multi_scale)
    print("Generator final output:", outputs)
    
    return Model(inputs, outputs)

# Discriminator
def discriminator(input_shape=(None, None, 3)):
    inputs = layers.Input(shape=input_shape)

    # Initial convolution block
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.LeakyReLU(alpha=0.2)(x)
    print("Discriminator initial output:", x)

    # Multi-scale processing branches
    scale1 = res_block(x, 32)
    print("Discriminator scale1 output:", scale1)
    scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
    print("Discriminator scale2 input after pooling:", scale2)
    scale2 = res_block(scale2, 32)
    print("Discriminator scale2 output before PixelShuffle:", scale2)
    scale2 = PixelShuffle(scale=2)(scale2)
    print("Discriminator scale2 output after PixelShuffle:", scale2)
    scale3 = layers.AveragePooling2D(pool_size=(4, 4))(x)
    print("Discriminator scale3 input after pooling:", scale3)
    scale3 = res_block(scale3, 32)
    print("Discriminator scale3 output before PixelShuffle:", scale3)
    scale3 = PixelShuffle(scale=4)(scale3)
    print("Discriminator scale3 output after PixelShuffle:", scale3)

    multi_scale = layers.Concatenate()([scale1, scale2, scale3])
    print("Discriminator multi_scale output:", multi_scale)

    # Additional convolutional layers after concatenation
    x = layers.Conv2D(32, (3, 3), padding='same')(multi_scale)
    x = layers.LeakyReLU(alpha=0.2)(x)
    print("Discriminator additional conv layers output:", x)
    
    for filters in [64, 128, 256, 512]:
        x = layers.Conv2D(filters, (4, 4), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)
        print(f"Discriminator conv layer with {filters} filters output:", x)
    
    x = layers.Conv2D(1, (4, 4), padding='same')(x)
    print("Discriminator final output:", x)
    return Model(inputs, x)
