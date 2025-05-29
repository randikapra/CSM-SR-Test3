
################## -- Test -- ############################
## VERSION 1.0 ##
import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore
from tensorflow.keras.applications import VGG19 # type: ignore

# # Custom PixelShuffle Layer
# class PixelShuffle(layers.Layer):
#     def __init__(self, scale, **kwargs):
#         super(PixelShuffle, self).__init__(**kwargs)
#         self.scale = scale

#     def call(self, inputs):
#         return tf.nn.depth_to_space(inputs, block_size=self.scale)

# # Define Residual Dense Block (RDB) with Mixed Convolution Types
# def residual_dense_block(x, filters, growth_rate=4, layers_in_block=4):
#     concat_features = [x]
#     for _ in range(layers_in_block):
#         x = layers.Conv2D(growth_rate, (3, 3), padding='same')(x)  # Standard convolution
#         x = layers.Activation('relu')(x)
        
#         # Dilated convolution
#         x = layers.Conv2D(growth_rate, (3, 3), padding='same', dilation_rate=2)(x)
#         x = layers.Activation('relu')(x)

#         # Depthwise separable convolution
#         x = layers.SeparableConv2D(growth_rate, (3, 3), padding='same')(x)
#         x = layers.Activation('relu')(x)
        
#         concat_features.append(x)
#         x = layers.Concatenate()(concat_features)
    
#     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
#     return x

# # Define Residual-in-Residual Dense Block (RRDB)
# def rrdb(x, filters, growth_rate=4, res_block=4):
#     res = layers.Conv2D(filters, (3, 3), padding='same')(x)
#     for _ in range(res_block):
#         x = residual_dense_block(x, filters, growth_rate)
#     return layers.Add()([x, layers.Lambda(lambda x: x * 0.2)(res)])

# # def vgg_feature_extractor(input_shape):
# #     vgg = VGG19(include_top=False, weights='imagenet', input_shape=input_shape)
# #     vgg.trainable = False
# #     layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']
# #     feature_extractor = tf.keras.Model(inputs=vgg.input, outputs=[vgg.get_layer(name).output for name in layers])
# #     return feature_extractor
# def vgg_feature_extractor(input_shape):
#     vgg = VGG19(include_top=False, weights='imagenet', input_shape=input_shape)
#     vgg.trainable = False
#     return vgg

# def generator(input_shape=(192, 256, 3)):
#     inputs = layers.Input(shape=input_shape)
#     extracted_features = layers.Input(shape=(None, None, 512))  # VGG19 features have shape (height, width, 512)

#     # Use the last VGG19 block's features and reshape them
#     reshaped_features = tf.image.resize(extracted_features, size=(inputs.shape[1], inputs.shape[2]))  # Resize to match input dimensions
#     reshaped_features = layers.Conv2D(128, (1, 1), padding='same', activation='relu')(reshaped_features)  # Reduce dimensionality
#     concatenated_input = layers.Concatenate()([inputs, reshaped_features])

#     # Original scale
#     x = layers.Conv2D(128, (3, 3), padding='same')(concatenated_input)
#     x = layers.Activation('relu')(x)
    
#     # Original scale processing (scale1)
#     scale1 = x
#     for _ in range(3):
#         scale1 = rrdb(scale1, 128)

#     # Downscale by 2 (scale2)
#     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
#     for _ in range(3):
#         scale2 = rrdb(scale2, 128)
#     # Upscale by 2
#     scale2 = PixelShuffle(scale=2)(scale2)

#     # Upscale by 2 (scale4)
#     scale4 = PixelShuffle(scale=2)(x)
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
    
#     return Model([inputs, extracted_features], outputs, name='Generator')


# Custom PixelShuffle Layer
class PixelShuffle(layers.Layer):
    def __init__(self, scale, **kwargs):
        super(PixelShuffle, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, block_size=self.scale)

# Define Residual Dense Block (RDB) with Mixed Convolution Types
def residual_dense_block(x, filters, growth_rate=4, layers_in_block=4):
    concat_features = [x]
    for _ in range(layers_in_block):
        x = layers.Conv2D(growth_rate, (3, 3), padding='same')(x)  # Standard convolution
        x = layers.Activation('relu')(x)
        
        # Dilated convolution
        x = layers.Conv2D(growth_rate, (3, 3), padding='same', dilation_rate=2)(x)
        x = layers.Activation('relu')(x)

        # Depthwise separable convolution
        x = layers.SeparableConv2D(growth_rate, (3, 3), padding='same')(x)
        x = layers.Activation('relu')(x)
        
        concat_features.append(x)
        x = layers.Concatenate()(concat_features)
    
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    return x

# Define Residual-in-Residual Dense Block (RRDB)
def rrdb(x, filters, growth_rate=4, res_block=4):
    res = layers.Conv2D(filters, (3, 3), padding='same')(x)
    for _ in range(res_block):
        x = residual_dense_block(x, filters, growth_rate)
    return layers.Add()([x, layers.Lambda(lambda x: x * 0.2)(res)])

def vgg_feature_extractor(input_shape):
    if isinstance(input_shape, list):
        input_shape = tuple(input_shape)
    if len(input_shape) == 2:  # Add channel dimension if not present
        input_shape = input_shape + (3,)
    vgg = VGG19(include_top=False, weights='imagenet', input_shape=input_shape)
    vgg.trainable = False
    layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']
    feature_extractor = tf.keras.Model(inputs=vgg.input, outputs=[vgg.get_layer(name).output for name in layers])
    return feature_extractor

# Custom Layer to resize extracted features
class ResizeLayer(layers.Layer):
    def __init__(self, target_height, target_width, **kwargs):
        super(ResizeLayer, self).__init__(**kwargs)
        self.target_height = target_height
        self.target_width = target_width

    def call(self, inputs):
        return tf.image.resize(inputs, size=(self.target_height, self.target_width))

def generator(input_shape=(192, 256, 3)):
    inputs = layers.Input(shape=input_shape)
    extracted_features = layers.Input(shape=(None, None, 512))  # VGG19 features have shape (height, width, 512)

    # Use the last VGG19 block's features and reshape them
    reshaped_features = ResizeLayer(inputs.shape[1], inputs.shape[2])(extracted_features)  # Resize to match input dimensions
    reshaped_features = layers.Conv2D(128, (1, 1), padding='same', activation='relu')(reshaped_features)  # Reduce dimensionality
    reshaped_features = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(reshaped_features)  # Extra conv layer for better compatibility
    concatenated_input = layers.Concatenate()([inputs, reshaped_features])

    # Original scale
    x = layers.Conv2D(128, (3, 3), padding='same')(concatenated_input)
    x = layers.Activation('relu')(x)
    
    # Original scale processing (scale1)
    scale1 = x
    for _ in range(3):
        scale1 = rrdb(scale1, 128)

    # Downscale by 2 (scale2)
    scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
    for _ in range(3):
        scale2 = rrdb(scale2, 128)
    # Upscale by 2
    scale2 = PixelShuffle(scale=2)(scale2)

    # Upscale by 2 (scale4)
    scale4 = PixelShuffle(scale=2)(x)
    for _ in range(3):
        scale4 = rrdb(scale4, 128)
    # Downscale by 2
    scale4 = layers.AveragePooling2D(pool_size=(2, 2))(scale4)
    
    # Concatenate multi-scale features
    multi_scale = layers.Concatenate()([scale1, scale2, scale4])
    
    # Additional convolutional layers
    multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
    multi_scale = layers.Activation('relu')(multi_scale)
    
    # Upscale by 2
    multi_scale = PixelShuffle(scale=2)(multi_scale)
    multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
    multi_scale = layers.Activation('relu')(multi_scale)
    
    # Upscale by 2
    multi_scale = PixelShuffle(scale=2)(multi_scale)
    multi_scale = layers.Conv2D(3, (3, 3), padding='same')(multi_scale)
    
    # Final output
    outputs = layers.Conv2D(3, (3, 3), padding='same', activation='tanh')(multi_scale)
    
    return Model([inputs, extracted_features], outputs, name='Generator')

# Residual Block for Discriminator
def res_block(x, filters):
    res = x
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.Add()([x, res])
    return x

def discriminator(input_shape=(768, 1024, 3)):
    inputs = layers.Input(shape=input_shape)

    # Initial convolution block
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Multi-scale processing branches
    scale1 = res_block(x, 32)
    scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
    scale2 = res_block(scale2, 32)
    scale2 = PixelShuffle(scale=2)(scale2)
    scale3 = layers.AveragePooling2D(pool_size=(4, 4))(x)
    scale3 = res_block(scale3, 32)
    scale3 = PixelShuffle(scale=4)(scale3)

    multi_scale = layers.Concatenate()([scale1, scale2, scale3])

    # Additional convolutional layers after concatenation
    x = layers.Conv2D(32, (3, 3), padding='same')(multi_scale)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    for filters in [64, 128, 256, 512]:
        x = layers.Conv2D(filters, (4, 4), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)
    
    x = layers.Conv2D(1, (4, 4), padding='same')(x)
    return Model(inputs, x)



'''# Custom PixelShuffle Layer
class PixelShuffle(layers.Layer):
    def __init__(self, scale, **kwargs):
        super(PixelShuffle, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, block_size=self.scale)

# Define Residual Dense Block (RDB) with Mixed Convolution Types
def residual_dense_block(x, filters, growth_rate=4, layers_in_block=4):
    concat_features = [x]
    for _ in range(layers_in_block):
        x = layers.Conv2D(growth_rate, (3, 3), padding='same')(x)
        x = layers.Activation('relu')(x)
        
        # Dilated convolution
        x = layers.Conv2D(growth_rate, (3, 3), padding='same', dilation_rate=2)(x)
        x = layers.Activation('relu')(x)

        # Depthwise separable convolution
        x = layers.SeparableConv2D(growth_rate, (3, 3), padding='same')(x)
        x = layers.Activation('relu')(x)
        
        concat_features.append(x)
        x = layers.Concatenate()(concat_features)
    
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    return x

# Define Residual-in-Residual Dense Block (RRDB)
def rrdb(x, filters, growth_rate=4, res_block=4):
    res = layers.Conv2D(filters, (3, 3), padding='same')(x)
    for _ in range(res_block):
        x = residual_dense_block(x, filters, growth_rate)
    return layers.Add()([x, layers.Lambda(lambda x: x * 0.2)(res)])

def generator(input_shape=(192, 256, 3), use_self_conditioning=False):
    inputs = layers.Input(shape=input_shape)
    conditioning = None

    if use_self_conditioning:
        conditioning_input = layers.Input(shape=(256,))
        conditioning = layers.Dense(128, activation='relu')(conditioning_input)

    # Original scale
    x = layers.Conv2D(128, (3, 3), padding='same')(inputs)
    x = layers.Activation('relu')(x)
    
    if conditioning is not None:
        conditioning = layers.Reshape((1, 1, 128))(conditioning)
        conditioning = tf.repeat(conditioning, tf.shape(x)[1], axis=1)
        conditioning = tf.repeat(conditioning, tf.shape(x)[2], axis=2)
        x = layers.Concatenate()([x, conditioning])

    # Original scale processing (scale1)
    scale1 = x
    for _ in range(2):
        scale1 = rrdb(scale1, 128)

    # Downscale by 2 (scale2)
    scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
    for _ in range(2):
        scale2 = rrdb(scale2, 128)
    # Upscale by 2
    scale2 = PixelShuffle(scale=2)(scale2)

    # Upscale by 2 (scale4)
    scale4 = PixelShuffle(scale=2)(x)
    for _ in range(2):
        scale4 = rrdb(scale4, 128)
    # Downscale by 2
    scale4 = layers.AveragePooling2D(pool_size=(2, 2))(scale4)
    
    # Concatenate multi-scale features
    multi_scale = layers.Concatenate()([scale1, scale2, scale4])
    
    # Additional convolutional layers
    multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
    multi_scale = layers.Activation('relu')(multi_scale)
    
    # Upscale by 2
    multi_scale = PixelShuffle(scale=2)(multi_scale)
    multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
    multi_scale = layers.Activation('relu')(multi_scale)
    
    # Upscale by 2
    multi_scale = PixelShuffle(scale=2)(multi_scale)
    multi_scale = layers.Conv2D(3, (3, 3), padding='same')(multi_scale)
    
    # Final output
    outputs = layers.Conv2D(3, (3, 3), padding='same', activation='tanh')(multi_scale)

    if conditioning is not None:
        return Model([inputs, conditioning_input], outputs)

    return Model(inputs, outputs)

# Residual Block for Discriminator
def res_block(x, filters):
    res = x
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.Add()([x, res])
    return x

def discriminator(input_shape=(768, 1024, 3)):
    inputs = layers.Input(shape=input_shape)

    # Initial convolution block
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Multi-scale processing branches
    scale1 = res_block(x, 32)
    scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
    scale2 = res_block(scale2, 32)
    scale2 = PixelShuffle(scale=2)(scale2)
    scale3 = layers.AveragePooling2D(pool_size=(4, 4))(x)
    scale3 = res_block(scale3, 32)
    scale3 = PixelShuffle(scale=4)(scale3)

    multi_scale = layers.Concatenate()([scale1, scale2, scale3])

    # Additional convolutional layers after concatenation
    x = layers.Conv2D(32, (3, 3), padding='same')(multi_scale)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    for filters in [64, 128, 256, 512]:
        x = layers.Conv2D(filters, (4, 4), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)
    
    x = layers.Conv2D(1, (4, 4), padding='same')(x)
    return Model(inputs, x)'''

# """
# ## WORKING BETTER CODE ##
# # Custom PixelShuffle Layer
# class PixelShuffle(layers.Layer):
#     def __init__(self, scale, **kwargs):
#         super(PixelShuffle, self).__init__(**kwargs)
#         self.scale = scale

#     def call(self, inputs):
#         return tf.nn.depth_to_space(inputs, block_size=self.scale)

# # Define Residual Dense Block (RDB) with Mixed Convolution Types
# def residual_dense_block(x, filters, growth_rate=8, layers_in_block=5):
#     concat_features = [x]
#     for _ in range(layers_in_block):
#         x = layers.Conv2D(growth_rate, (3, 3), padding='same')(x)  # Standard convolution
#         x = layers.Activation('relu')(x)
        
#         # Dilated convolution
#         x = layers.Conv2D(growth_rate, (3, 3), padding='same', dilation_rate=2)(x)
#         x = layers.Activation('relu')(x)
        
#         # Depthwise separable convolution
#         x = layers.SeparableConv2D(growth_rate, (3, 3), padding='same')(x)
#         x = layers.Activation('relu')(x)
        
#         concat_features.append(x)
#         x = layers.Concatenate()(concat_features)
    
#     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
#     return x

# # Define Residual-in-Residual Dense Block (RRDB)
# def rrdb(x, filters, growth_rate=8, res_block=4):
#     res = layers.Conv2D(filters, (3, 3), padding='same')(x)
#     for _ in range(res_block):
#         x = residual_dense_block(x, filters, growth_rate)
#     return layers.Add()([x, layers.Lambda(lambda x: x * 0.2)(res)])


# # Feature Pyramid Network (FPN) Block with PixelShuffle
# def fpn_block(inputs, filters):
#     P3 = layers.Conv2D(filters, (3, 3), padding='same')(inputs)
#     P3 = layers.Activation('relu')(P3)
    
#     P4 = layers.MaxPooling2D(pool_size=(2, 2))(P3)
#     P4 = layers.Conv2D(filters, (3, 3), padding='same')(P4)
#     P4 = layers.Activation('relu')(P4)

#     P5 = layers.MaxPooling2D(pool_size=(2, 2))(P4)
#     P5 = layers.Conv2D(filters * 4, (3, 3), padding='same')(P5)  # Increase filters for PixelShuffle
#     P5 = layers.Activation('relu')(P5)
    
#     P4_up = PixelShuffle(scale=2)(P5)
#     P4_up = layers.Conv2D(filters, (3, 3), padding='same')(P4_up)  # Match the number of filters
#     P4 = layers.Add()([P4, P4_up])
    
#     P3_up = PixelShuffle(scale=2)(P4)
#     P3_up = layers.Conv2D(filters, (3, 3), padding='same')(P3_up)  # Match the number of filters
#     P3 = layers.Add()([P3, P3_up])
    
#     return P3

# def generator(input_shape=(192, 256, 3)):
#     inputs = layers.Input(shape=input_shape)
    
#     # Initial convolution
#     x = layers.Conv2D(128, (3, 3), padding='same')(inputs)
#     x = layers.Activation('relu')(x)
    
#     # Multi-Scale Processing Branches
#     scale1 = x
#     for _ in range(2):
#         scale1 = rrdb(scale1, 128)

#     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
#     for _ in range(2):
#         scale2 = rrdb(scale2, 128)
#     scale2 = PixelShuffle(scale=2)(scale2)
    
#     scale3 = PixelShuffle(scale=2)(x)
#     for _ in range(2):
#         scale3 = rrdb(scale3, 128)
#     scale3 = layers.AveragePooling2D(pool_size=(2, 2))(scale3)
    
#     # # Additional multi-scale level
#     # scale4 = PixelShuffle(scale=4)(x)
#     # for _ in range(3):
#     #     scale4 = rrdb(scale4, 128)
#     # scale4 = layers.AveragePooling2D(pool_size=(2, 2))(scale4)
    
#     # Concatenate multi-scale features
#     multi_scale = layers.Concatenate()([scale1, scale2, scale3])
    
#     # Additional convolutional layers
#     multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
#     multi_scale = layers.Activation('relu')(multi_scale)
    
#     # Incorporate FPN
#     multi_scale = fpn_block(multi_scale, 128)
    
#     # Upsampling to the final resolution
#     multi_scale = PixelShuffle(scale=2)(multi_scale)
#     multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
#     multi_scale = layers.Activation('relu')(multi_scale)
    
#     multi_scale = PixelShuffle(scale=2)(multi_scale)
#     multi_scale = layers.Conv2D(3, (3, 3), padding='same')(multi_scale)
    
#     # Final output
#     outputs = layers.Conv2D(3, (3, 3), padding='same', activation='tanh')(multi_scale)
    
#     return Model(inputs, outputs)

# # Residual Block for Discriminator
# def res_block(x, filters):
#     res = x
#     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
#     x = layers.LeakyReLU(alpha=0.2)(x)
#     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
#     x = layers.Add()([x, res])
#     return x

# def discriminator(input_shape=(768, 1024, 3)):
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
    
#     # # Additional multi-scale level
#     # scale4 = layers.AveragePooling2D(pool_size=(8, 8))(x)
#     # scale4 = res_block(scale4, 32)
#     # scale4 = PixelShuffle(scale=8)(scale4)

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
# """






# import tensorflow as tf
# from tensorflow.keras import layers, Model # type: ignore

# class AttentionGate(layers.Layer):
#     def __init__(self, filters, **kwargs):
#         super(AttentionGate, self).__init__(**kwargs)
#         self.filters = filters
#         self.conv_f = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)
#         self.conv_g = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)
#         self.conv_h = layers.Conv2D(1, (1, 1), padding='same', use_bias=False)
#         self.activation = layers.Activation('sigmoid')
        
#     def call(self, x, g):
#         f = self.conv_f(x)
#         g = self.conv_g(g)
#         h = layers.Add()([f, g])
#         h = self.activation(h)
#         h = self.conv_h(h)
#         return layers.Multiply()([x, h])

# class NoiseInjection(layers.Layer):
#     def __init__(self, **kwargs):
#         super(NoiseInjection, self).__init__(**kwargs)
        
#     def call(self, x, training=None):
#         if training:
#             noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=1.0)
#             return x + noise
#         return x

# class SelfAttentionLayer(tf.keras.layers.Layer):
#     def __init__(self, filters, **kwargs):
#         super(SelfAttentionLayer, self).__init__(**kwargs)
#         self.filters = filters
#         self.query_conv = layers.Conv2D(filters // 8, kernel_size=1)
#         self.key_conv = layers.Conv2D(filters // 8, kernel_size=1)
#         self.value_conv = layers.Conv2D(filters, kernel_size=1)
#         self.gamma = self.add_weight(name="gamma", shape=[1], initializer='zeros', trainable=True)
#         self.ln = layers.LayerNormalization()

#     def call(self, x):
#         batch_size, height, width, channels = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
#         query = tf.reshape(self.query_conv(x), [batch_size, height * width, self.filters // 8])
#         key = tf.reshape(self.key_conv(x), [batch_size, height * width, self.filters // 8])
#         value = tf.reshape(self.value_conv(x), [batch_size, height * width, self.filters])

#         attention = tf.nn.softmax(tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(tf.cast(self.filters // 8, tf.float32)))
#         out = tf.matmul(attention, value)
#         out = tf.reshape(out, [batch_size, height, width, self.filters])
#         out = self.gamma * out + x
#         return self.ln(out)
    
#     def compute_output_shape(self, input_shape):
#         return input_shape

# def se_block(x, filters, ratio=32):
#     se = layers.GlobalAveragePooling2D()(x)
#     se = layers.Dense(filters // ratio, activation='relu')(se)
#     se = layers.Dense(filters, activation='sigmoid')(se)
#     se = layers.Reshape((1, 1, filters))(se)
#     x = layers.Multiply()([x, se])
#     return x

# def residual_dense_block(x, filters, growth_rate=32, layers_in_block=4):
#     concat_features = [x]
#     for _ in range(layers_in_block):
#         x = layers.Conv2D(growth_rate, (3, 3), padding='same')(x)
#         x = layers.Activation('relu')(x)
#         concat_features.append(x)
#         x = layers.Concatenate()(concat_features)
#     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
#     return x

# def rrdb(x, filters, growth_rate=32, res_block=4):
#     res = x
#     for _ in range(res_block):
#         x = residual_dense_block(x, filters, growth_rate)
#     return layers.Add()([x, layers.Lambda(lambda x: x * 0.2)(res)])

# def fpn_block(x, filters):
#     p1 = layers.Conv2D(filters, (3, 3), padding='same')(x)
#     p1 = layers.LeakyReLU(alpha=0.2)(p1)
#     p2 = layers.Conv2D(filters, (3, 3), padding='same', strides=(2, 2))(p1)
#     p2 = layers.LeakyReLU(alpha=0.2)(p2)
#     p3 = layers.Conv2D(filters, (3, 3), padding='same', strides=(2, 2))(p2)
#     p3 = layers.LeakyReLU(alpha=0.2)(p3)
#     p4 = layers.Conv2D(filters, (3, 3), padding='same', strides=(2, 2))(p3)
#     p4 = layers.LeakyReLU(alpha=0.2)(p4)

#     p3_upsampled = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(p4)
#     p3_upsampled = layers.LeakyReLU(alpha=0.2)(p3_upsampled)
#     p3 = layers.Add()([p3, p3_upsampled])

#     p2_upsampled = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(p3)
#     p2_upsampled = layers.LeakyReLU(alpha=0.2)(p2_upsampled)
#     p2 = layers.Add()([p2, p2_upsampled])

#     p1_upsampled = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(p2)
#     p1_upsampled = layers.LeakyReLU(alpha=0.2)(p1_upsampled)
#     p1 = layers.Add()([p1, p1_upsampled])

#     return p1

# def generator(input_shape=(192, 256, 3)):
#     inputs = layers.Input(shape=input_shape)
    
#     # Noise Injection Layer
#     noisy_inputs = NoiseInjection()(inputs, training=True)

#     x = layers.Conv2D(128, (3, 3), padding='same')(noisy_inputs)
#     x = layers.LeakyReLU(alpha=0.2)(x)
#     x = se_block(x, 128)  # Adding SE Block

#     # Adding Residual-in-Residual Dense Blocks with Self-Attention
#     rrdb_outputs = []
#     for _ in range(4):
#         x = rrdb(x, 128)
#         # x = SelfAttentionLayer(128)(x)
#         rrdb_outputs.append(x)

#     # # Attention Gates
#     # for i in range(len(rrdb_outputs)):
#     #     if i > 0:
#     #         x = AttentionGate(64)(x, rrdb_outputs[i])

#     # Multi-Scale Feature Processing with FPN
#     scale1 = x
#     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
#     scale2 = rrdb(scale2, 128)
#     scale2 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(scale2)
#     scale2 = layers.LeakyReLU(alpha=0.2)(scale2)
#     scale3 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
#     scale3 = layers.AveragePooling2D(pool_size=(2, 2))(scale3)
#     scale3 = rrdb(scale3, 128)
#     scale4 = layers.AveragePooling2D(pool_size=(2, 2))(scale3)
#     scale4 = rrdb(scale4, 128)
#     scale4 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(scale4)
#     scale4 = layers.LeakyReLU(alpha=0.2)(scale4)
    
#     multi_scale = layers.Concatenate()([scale1, scale2, scale3, scale4])
#     multi_scale = fpn_block(multi_scale, 128)
    
#     # Upsampling to the final resolution
#     multi_scale = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(multi_scale)
#     multi_scale = layers.LeakyReLU(alpha=0.2)(multi_scale)
#     multi_scale = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(multi_scale)
#     multi_scale = layers.LeakyReLU(alpha=0.2)(multi_scale)

#     outputs = layers.Conv2D(3, (3, 3), padding='same', activation='tanh')(multi_scale)
    
#     return Model(inputs, outputs)

# def res_block(x, filters):
#     res = x
#     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
#     x = layers.LeakyReLU(alpha=0.2)(x)
#     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
#     x = layers.Add()([x, res])
#     return x


# def discriminator(input_shape=(768, 1024, 3)):
#     inputs = layers.Input(shape=input_shape)

#     # Initial Conv Layer
#     x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
#     x = layers.LeakyReLU(alpha=0.2)(x)
#     ## all 32 must be 64. I changed it 1/19
#     # Multi-Scale Feature Extraction
#     scale1 = res_block(x, 32)
#     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
#     scale2 = res_block(scale2,32)
#     scale2 = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(scale2)
#     scale2 = layers.LeakyReLU(alpha=0.2)(scale2)
#     scale3 = layers.AveragePooling2D(pool_size=(4, 4))(x)
#     scale3 = res_block(scale3, 32)
#     scale3 = layers.Conv2DTranspose(32, (3, 3), strides=(4, 4), padding='same')(scale3)
#     scale3 = layers.LeakyReLU(alpha=0.2)(scale3)

#     multi_scale = layers.Concatenate()([scale1, scale2, scale3])
#     x = layers.Conv2D(64, (3, 3), padding='same')(multi_scale)
#     x = layers.LeakyReLU(alpha=0.2)(x)

#     # Downsampling Layers
#     for filters in [64, 128, 256, 512]:
#         x = layers.Conv2D(filters, (4, 4), strides=(2, 2), padding='same')(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.LeakyReLU(alpha=0.2)(x)

#     # Final Conv Layer for PatchGAN
#     x = layers.Conv2D(1, (4, 4), padding='same')(x)
    
#     return Model(inputs, x)


# # import tensorflow as tf
# # from tensorflow.keras import layers, Model # type: ignore

# # class PixelShuffleLayer(layers.Layer):
# #     def __init__(self, upscale_factor, **kwargs):
# #         super(PixelShuffleLayer, self).__init__(**kwargs)
# #         self.upscale_factor = upscale_factor

# #     def call(self, x):
# #         return tf.nn.depth_to_space(x, self.upscale_factor)

# #     def get_config(self):
# #         config = super().get_config()
# #         config.update({"upscale_factor": self.upscale_factor})
# #         return config

# # def se_block(x, filters, ratio=16):
# #     se = layers.GlobalAveragePooling2D()(x)
# #     se = layers.Dense(filters // ratio, activation='relu')(se)
# #     se = layers.Dense(filters, activation='sigmoid')(se)
# #     se = layers.Reshape((1, 1, filters))(se)
# #     x = layers.Multiply()([x, se])
# #     return x

# # def fpn_block(x, filters):
# #     p1 = layers.Conv2D(filters, (3, 3), padding='same')(x)
# #     p1 = layers.LeakyReLU(alpha=0.2)(p1)
# #     p2 = layers.Conv2D(filters, (3, 3), padding='same', strides=(2, 2))(p1)
# #     p2 = layers.LeakyReLU(alpha=0.2)(p2)
# #     p3 = layers.Conv2D(filters, (3, 3), padding='same', strides=(2, 2))(p2)
# #     p3 = layers.LeakyReLU(alpha=0.2)(p3)

# #     p2_upsampled = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(p3)
# #     p2_upsampled = layers.LeakyReLU(alpha=0.2)(p2_upsampled)
# #     p2 = layers.Add()([p2, p2_upsampled])

# #     p1_upsampled = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(p2)
# #     p1_upsampled = layers.LeakyReLU(alpha=0.2)(p1_upsampled)
# #     p1 = layers.Add()([p1, p1_upsampled])

# #     return p1

# # def residual_dense_block(x, filters, growth_rate=32, layers_in_block=6):
# #     concat_features = [x]
# #     for _ in range(layers_in_block):
# #         x = layers.Conv2D(growth_rate, (3, 3), padding='same')(x)
# #         x = layers.Activation('relu')(x)
# #         concat_features.append(x)
# #         x = layers.Concatenate()(concat_features)
# #     x = layers.Conv2D(filters, (3, 3), padding='same')(x)  # Match filters with the initial input to rrdb
# #     return x

# # def rrdb(x, filters, growth_rate=32, res_block=6):
# #     res = x
# #     for _ in range(res_block):
# #         x = residual_dense_block(x, filters, growth_rate)  # Ensure growth rate matches filters
# #     return layers.Add()([x, layers.Lambda(lambda x: x * 0.2)(res)])

# # def generator(input_shape=(192, 256, 3)):
# #     inputs = layers.Input(shape=input_shape)

# #     # Initial convolution
# #     x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
# #     x = layers.Activation('relu')(x)
    
# #     # Increase the depth of the initial convolution
# #     for _ in range(4):
# #         x = layers.Conv2D(64, (3, 3), padding='same')(x)
# #         x = layers.Activation('relu')(x)
    
# #     # Multi-Scale Processing Branches
# #     # Scale 1: Original scale
# #     scale1 = x
# #     for _ in range(4):  # Increase the number of RRDB blocks for better feature extraction
# #         scale1 = rrdb(scale1, 64)

# #     # Scale 2: Half scale
# #     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
# #     for _ in range(4):
# #         scale2 = rrdb(scale2, 64)
# #     scale2 = PixelShuffleLayer(upscale_factor=2)(scale2)
    
# #     # Scale 3: Double scale
# #     scale3 = PixelShuffleLayer(upscale_factor=2)(x)
# #     for _ in range(4):
# #         scale3 = rrdb(scale3, 64)
# #     scale3 = layers.AveragePooling2D(pool_size=(2, 2))(scale3)
    
# #     # Concatenate multi-scale features
# #     multi_scale = layers.Concatenate()([scale1, scale2, scale3])
# #     multi_scale = fpn_block(multi_scale, 64)  # Adding FPN Block
    
# #     # Additional convolutional layers
# #     multi_scale = layers.Conv2D(64, (3, 3), padding='same')(multi_scale)
# #     multi_scale = layers.Activation('relu')(multi_scale)
    
# #     # Upsampling to the final resolution using Pixel Shuffle
# #     multi_scale = PixelShuffleLayer(upscale_factor=2)(multi_scale)
# #     multi_scale = PixelShuffleLayer(upscale_factor=2)(multi_scale)

# #     outputs = layers.Conv2D(3, (3, 3), padding='same', activation='tanh')(multi_scale)
    
# #     return Model(inputs, outputs)

# # def res_block(x, filters):
# #     res = x
# #     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
# #     x = layers.LeakyReLU(alpha=0.2)(x)
# #     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
# #     x = layers.Add()([x, res])
# #     return x

# # def discriminator(input_shape=(768, 1024, 3)):
# #     inputs = layers.Input(shape=input_shape)

# #     # Initial convolution block
# #     x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
# #     x = layers.LeakyReLU(alpha=0.2)(x)

# #     # Increase the depth of the initial convolution
# #     for _ in range(4):
# #         x = layers.Conv2D(64, (3, 3), padding='same')(x)
# #         x = layers.LeakyReLU(alpha=0.2)(x)
    
# #     # Multi-scale processing branches
# #     scale1 = res_block(x, 64)
# #     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
# #     scale2 = res_block(scale2, 64)
# #     scale2 = layers.UpSampling2D(size=(2, 2))(scale2)
# #     scale3 = layers.AveragePooling2D(pool_size=(4, 4))(x)
# #     scale3 = res_block(scale3, 64)
# #     scale3 = layers.UpSampling2D(size=(4, 4))(scale3)

# #     multi_scale = layers.Concatenate()([scale1, scale2, scale3])
# #     multi_scale = fpn_block(multi_scale, 64)  # Adding FPN Block

# #     # Additional convolutional layers after concatenation
# #     x = layers.Conv2D(64, (3, 3), padding='same')(multi_scale)
# #     x = layers.LeakyReLU(alpha=0.2)(x)
    
# #     for filters in [64, 128, 256, 512]:
# #         x = layers.Conv2D(filters, (4, 4), strides=(2, 2), padding='same')(x)
# #         x = layers.BatchNormalization()(x)
# #         x = layers.LeakyReLU(negative_slope=0.2)(x)
    
# #     x = layers.Conv2D(1, (4, 4), padding='same')(x)
# #     return Model(inputs, x)



# # # def se_block(x, filters, ratio=16):
# # #     se = layers.GlobalAveragePooling2D()(x)
# # #     se = layers.Dense(filters // ratio, activation='relu')(se)
# # #     se = layers.Dense(filters, activation='sigmoid')(se)
# # #     se = layers.Reshape((1, 1, filters))(se)
# # #     x = layers.Multiply()([x, se])
# # #     return x

# # # def fpn_block(x, filters):
# # #     p1 = layers.Conv2D(filters, (3, 3), padding='same')(x)
# # #     p1 = layers.LeakyReLU(alpha=0.2)(p1)
# # #     p2 = layers.Conv2D(filters, (3, 3), padding='same', strides=(2, 2))(p1)
# # #     p2 = layers.LeakyReLU(alpha=0.2)(p2)
# # #     p3 = layers.Conv2D(filters, (3, 3), padding='same', strides=(2, 2))(p2)
# # #     p3 = layers.LeakyReLU(alpha=0.2)(p3)

# # #     p2_upsampled = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(p3)
# # #     p2_upsampled = layers.LeakyReLU(alpha=0.2)(p2_upsampled)
# # #     p2 = layers.Add()([p2, p2_upsampled])

# # #     p1_upsampled = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(p2)
# # #     p1_upsampled = layers.LeakyReLU(alpha=0.2)(p1_upsampled)
# # #     p1 = layers.Add()([p1, p1_upsampled])

# # #     return p1

# # # def residual_dense_block(x, filters, growth_rate=32, layers_in_block=4):
# # #     concat_features = [x]
# # #     for _ in range(layers_in_block):
# # #         x = layers.Conv2D(growth_rate, (3, 3), padding='same')(x)
# # #         x = layers.Activation('relu')(x)
# # #         concat_features.append(x)
# # #         x = layers.Concatenate()(concat_features)
# # #     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
# # #     return x

# # # def rrdb(x, filters, growth_rate=32, res_block=4):
# # #     res = x
# # #     for _ in range(res_block):
# # #         x = residual_dense_block(x, filters, growth_rate)
# # #     return layers.Add()([x, res])

# # # def pixel_shuffle(scale):
# # #     return layers.Lambda(lambda x: tf.nn.depth_to_space(x, scale))

# # # def generator(input_shape=(192, 256, 3)):
# # #     inputs = layers.Input(shape=input_shape)

# # #     # Initial convolution
# # #     x = layers.Conv2D(128, (3, 3), padding='same')(inputs)
# # #     x = layers.Activation('relu')(x)
# # #     x = se_block(x, 128)  # Adding SE Block

# # #     # Initial convolution
# # #     scale1 = x
# # #     for _ in range(4):  # Increase the number of RRDB blocks for better feature extraction
# # #         scale1 = rrdb(scale1, 128)

# # #     # Multi-Scale Processing Branches
# # #     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
# # #     for _ in range(4):
# # #         scale2 = rrdb(scale2, 128)
# # #     scale2 = pixel_shuffle(2)(scale2)

# # #     scale3 = pixel_shuffle(2)(x)
# # #     for _ in range(4):
# # #         scale3 = rrdb(scale3, 128)
# # #     scale3 = layers.AveragePooling2D(pool_size=(2, 2))(scale3)

# # #     # Concatenate multi-scale features
# # #     multi_scale = layers.Concatenate()([scale1, scale2, scale3])
# # #     multi_scale = fpn_block(multi_scale, 128)  # Adding FPN Block

# # #     # Additional convolutional layers
# # #     multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
# # #     multi_scale = layers.Activation('relu')(multi_scale)

# # #     # Upsampling to the final resolution using Pixel Shuffle
# # #     multi_scale = pixel_shuffle(2)(multi_scale)
# # #     multi_scale = pixel_shuffle(2)(multi_scale)

# # #     outputs = layers.Conv2D(3, (3, 3), padding='same', activation='tanh')(multi_scale)

# # #     return Model(inputs, outputs)

# # # def res_block(x, filters):
# # #     res = x
# # #     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
# # #     x = layers.LeakyReLU(alpha=0.2)(x)
# # #     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
# # #     x = layers.Add()([x, res])
# # #     return x

# # # def discriminator(input_shape=(768, 1024, 3)):
# # #     inputs = layers.Input(shape=input_shape)

# # #     # Initial convolution block
# # #     x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
# # #     x = layers.LeakyReLU(alpha=0.2)(x)

# # #     # Multi-scale processing branches
# # #     scale1 = res_block(x, 64)
# # #     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
# # #     scale2 = res_block(scale2, 64)
# # #     scale2 = layers.UpSampling2D(size=(2, 2))(scale2)
# # #     scale3 = layers.AveragePooling2D(pool_size=(4, 4))(x)
# # #     scale3 = res_block(scale3, 64)
# # #     scale3 = layers.UpSampling2D(size=(4, 4))(scale3)

# # #     multi_scale = layers.Concatenate()([scale1, scale2, scale3])

# # #     # Additional convolutional layers after concatenation
# # #     multi_scale = fpn_block(multi_scale, 64)  # Adding FPN Block
# # #     x = layers.Conv2D(64, (3, 3), padding='same')(multi_scale)
# # #     x = layers.LeakyReLU(alpha=0.2)(x)

# # #     for filters in [64, 128, 256, 512]:
# # #         x = layers.Conv2D(filters, (4, 4), strides=(2, 2), padding='same')(x)
# # #         x = layers.BatchNormalization()(x)
# # #         x = layers.LeakyReLU(negative_slope=0.2)(x)

# # #     x = layers.Conv2D(1, (4, 4), padding='same')(x)
# # #     return Model(inputs, x)



# # '''
# # import tensorflow as tf
# # from tensorflow.keras import layers, Model # type: ignore

# # def se_block(x, filters, ratio=16):
# #     se = layers.GlobalAveragePooling2D()(x)
# #     se = layers.Dense(filters // ratio, activation='relu')(se)
# #     se = layers.Dense(filters, activation='sigmoid')(se)
# #     se = layers.Reshape((1, 1, filters))(se)
# #     x = layers.Multiply()([x, se])
# #     return x

# # def fpn_block(x, filters):
# #     p1 = layers.Conv2D(filters, (3, 3), padding='same')(x)
# #     p1 = layers.LeakyReLU(alpha=0.2)(p1)
# #     p2 = layers.Conv2D(filters, (3, 3), padding='same', strides=(2, 2))(p1)
# #     p2 = layers.LeakyReLU(alpha=0.2)(p2)
# #     p3 = layers.Conv2D(filters, (3, 3), padding='same', strides=(2, 2))(p2)
# #     p3 = layers.LeakyReLU(alpha=0.2)(p3)

# #     p2_upsampled = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(p3)
# #     p2_upsampled = layers.LeakyReLU(alpha=0.2)(p2_upsampled)
# #     p2 = layers.Add()([p2, p2_upsampled])

# #     p1_upsampled = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(p2)
# #     p1_upsampled = layers.LeakyReLU(alpha=0.2)(p1_upsampled)
# #     p1 = layers.Add()([p1, p1_upsampled])

# #     return p1

# # def residual_dense_block(x, filters, growth_rate=4, layers_in_block=4):
# #     concat_features = [x]
# #     for _ in range(layers_in_block):
# #         x = layers.Conv2D(growth_rate, (3, 3), padding='same')(x)
# #         x = layers.Activation('relu')(x)
# #         concat_features.append(x)
# #         x = layers.Concatenate()(concat_features)
# #     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
# #     return x

# # def rrdb(x, filters, growth_rate=4, res_block=4):
# #     res = x
# #     for _ in range(res_block):
# #         x = residual_dense_block(x, filters, growth_rate)
# #     return layers.Add()([x, layers.Lambda(lambda x: x * 0.2)(res)])

# # # def pixel_shuffle_block(x, filters, upscale_factor):
# # #     x = layers.Conv2D(filters * (upscale_factor ** 2), (3, 3), padding='same')(x)
# # #     x = tf.nn.depth_to_space(x, upscale_factor)
# # #     x = layers.Activation('relu')(x)
# # #     return x
# # def pixel_shuffle(scale):
# #     return layers.Lambda(lambda x: tf.nn.depth_to_space(x, scale))

# # def generator(input_shape=(192, 256, 3)):
# #     inputs = layers.Input(shape=input_shape)

# #     # Initial convolution
# #     x = layers.Conv2D(128, (3, 3), padding='same')(inputs)
# #     x = layers.Activation('relu')(x)
# #     x = se_block(x, 128)  # Adding SE Block

# #     # Initial convolution
# #     scale1 = x
# #     for _ in range(4):  # Increase the number of RRDB blocks for better feature extraction
# #         scale1 = rrdb(scale1, 128)

# #     # Multi-Scale Processing Branches
# #     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
# #     for _ in range(4):
# #         scale2 = rrdb(scale2, 128)
# #     scale2 = pixel_shuffle(2)(scale2)

# #     scale3 = pixel_shuffle(2)(x)
# #     for _ in range(4):
# #         scale3 = rrdb(scale3, 128)
# #     scale3 = layers.AveragePooling2D(pool_size=(2, 2))(scale3)

# #     # Concatenate multi-scale features
# #     multi_scale = layers.Concatenate()([scale1, scale2, scale3])
# #     multi_scale = fpn_block(multi_scale, 128)  # Adding FPN Block

# #     # Additional convolutional layers
# #     multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
# #     multi_scale = layers.Activation('relu')(multi_scale)

# #     # Upsampling to the final resolution using Pixel Shuffle 
# #     multi_scale = pixel_shuffle(2)(multi_scale) 
# #     multi_scale = pixel_shuffle(2)(multi_scale)

# #     outputs = layers.Conv2D(3, (3, 3), padding='same', activation='tanh')(multi_scale)

# #     return Model(inputs, outputs)

# # def res_block(x, filters):
# #     res = x
# #     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
# #     x = layers.LeakyReLU(alpha=0.2)(x)
# #     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
# #     x = layers.Add()([x, res])
# #     return x

# # def discriminator(input_shape=(768, 1024, 3)):
# #     inputs = layers.Input(shape=input_shape)

# #     # Initial convolution block
# #     x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
# #     x = layers.LeakyReLU(alpha=0.2)(x)

# #     # Multi-scale processing branches
# #     scale1 = res_block(x, 64)
# #     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
# #     scale2 = res_block(scale2, 64)
# #     scale2 = layers.UpSampling2D(size=(2, 2))(scale2)
# #     scale3 = layers.AveragePooling2D(pool_size=(4, 4))(x)
# #     scale3 = res_block(scale3, 64)
# #     scale3 = layers.UpSampling2D(size=(4, 4))(scale3)

# #     multi_scale = layers.Concatenate()([scale1, scale2, scale3])

# #     # Additional convolutional layers after concatenation
# #     multi_scale = fpn_block(multi_scale, 64)  # Adding FPN Block
# #     x = layers.Conv2D(64, (3, 3), padding='same')(multi_scale)
# #     x = layers.LeakyReLU(alpha=0.2)(x)

# #     for filters in [64, 128, 256, 512]:
# #         x = layers.Conv2D(filters, (4, 4), strides=(2, 2), padding='same')(x)
# #         x = layers.BatchNormalization()(x)
# #         x = layers.LeakyReLU(negative_slope=0.2)(x)

# #     x = layers.Conv2D(1, (4, 4), padding='same')(x)
# #     return Model(inputs, x)
# # '''
# # ## -- Version 1 -- ##
# # # import tensorflow as tf
# # # from tensorflow.keras import layers, Model # type: ignore

# # # def se_block(x, filters, ratio=16):
# # #     se = layers.GlobalAveragePooling2D()(x)
# # #     se = layers.Dense(filters // ratio, activation='relu')(se)
# # #     se = layers.Dense(filters, activation='sigmoid')(se)
# # #     se = layers.Reshape((1, 1, filters))(se)
# # #     x = layers.Multiply()([x, se])
# # #     return x

# # # def fpn_block(x, filters):
# # #     p1 = layers.Conv2D(filters, (3, 3), padding='same')(x)
# # #     p1 = layers.LeakyReLU(alpha=0.2)(p1)
# # #     p2 = layers.Conv2D(filters, (3, 3), padding='same', strides=(2, 2))(p1)
# # #     p2 = layers.LeakyReLU(alpha=0.2)(p2)
# # #     p3 = layers.Conv2D(filters, (3, 3), padding='same', strides=(2, 2))(p2)
# # #     p3 = layers.LeakyReLU(alpha=0.2)(p3)

# # #     p2_upsampled = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(p3)
# # #     p2_upsampled = layers.LeakyReLU(alpha=0.2)(p2_upsampled)
# # #     p2 = layers.Add()([p2, p2_upsampled])

# # #     p1_upsampled = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(p2)
# # #     p1_upsampled = layers.LeakyReLU(alpha=0.2)(p1_upsampled)
# # #     p1 = layers.Add()([p1, p1_upsampled])

# # #     return p1

# # # # Define Residual Dense Block (RDB)
# # # def residual_dense_block(x, filters, growth_rate=4, layers_in_block=4):
# # #     concat_features = [x]
# # #     for _ in range(layers_in_block):
# # #         x = layers.Conv2D(growth_rate, (3, 3), padding='same')(x)
# # #         x = layers.Activation('relu')(x)
# # #         concat_features.append(x)
# # #         x = layers.Concatenate()(concat_features)
# # #     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
# # #     return x

# # # # Define Residual-in-Residual Dense Block (RRDB)
# # # def rrdb(x, filters, growth_rate=4, res_block=4):
# # #     res = x
# # #     for _ in range(res_block):
# # #         x = residual_dense_block(x, filters, growth_rate)
# # #     return layers.Add()([x, layers.Lambda(lambda x: x * 0.2)(res)])

# # # def generator(input_shape=(192, 256, 3)):
# # #     inputs = layers.Input(shape=input_shape)
    
# # #     # Initial convolution
# # #     x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
# # #     x = layers.Activation('relu')(x)
# # #     x = se_block(x, 64)  # Adding SE Block

# # #     ## Initial convolution 
# # #     # x1 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs) 
# # #     # x2 = layers.Conv2D(64, (5, 5), padding='same', activation='relu')(inputs) 
# # #     # x = layers.Concatenate()([x1, x2])
    
# # #     # Multi-Scale Processing Branches
# # #     # Scale 1: Original scale
# # #     scale1 = x
# # #     for _ in range(2):  # Increase the number of RRDB blocks for better feature extraction
# # #         scale1 = rrdb(scale1, 64)

# # #     # Scale 2: Half scale
# # #     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
# # #     for _ in range(2):
# # #         scale2 = rrdb(scale2, 64)
# # #     scale2 = layers.UpSampling2D(size=(2, 2))(scale2)
    
# # #     # Scale 3: Double scale 
# # #     scale3 = layers.UpSampling2D(size=(2, 2))(x)
# # #     for _ in range(2):
# # #         scale3 = rrdb(scale3, 64)
# # #     scale3 = layers.AveragePooling2D(pool_size=(2, 2))(scale3)
    
# # #     # Concatenate multi-scale features
# # #     multi_scale = layers.Concatenate()([scale1, scale2, scale3])
# # #     multi_scale = fpn_block(multi_scale, 64)  # Adding FPN Block
    
# # #     # Additional convolutional layers
# # #     multi_scale = layers.Conv2D(64, (3, 3), padding='same')(multi_scale)
# # #     multi_scale = layers.Activation('relu')(multi_scale)
    
# # #     # Upsampling to the final resolution
# # #     multi_scale = layers.UpSampling2D(size=(2, 2))(multi_scale)
# # #     multi_scale = layers.Conv2D(64, (3, 3), padding='same')(multi_scale)
# # #     multi_scale = layers.Activation('relu')(multi_scale)
    
# # #     multi_scale = layers.UpSampling2D(size=(2, 2))(multi_scale)
# # #     multi_scale = layers.Conv2D(3, (3, 3), padding='same')(multi_scale)
    
# # #     # Final output
# # #     outputs = layers.Conv2D(3, (3, 3), padding='same', activation='tanh')(multi_scale)
    
# # #     return Model(inputs, outputs)

# # # # Residual Block for Discriminator
# # # def res_block(x, filters):
# # #     res = x
# # #     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
# # #     x = layers.LeakyReLU(alpha=0.2)(x)
# # #     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
# # #     x = layers.Add()([x, res])
# # #     return x

# # # def discriminator(input_shape=(768, 1024, 3)):
# # #     inputs = layers.Input(shape=input_shape)

# # #     # Initial convolution block
# # #     x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
# # #     x = layers.LeakyReLU(alpha=0.2)(x)

# # #     # Multi-scale processing branches
# # #     scale1 = res_block(x, 64)
# # #     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
# # #     scale2 = res_block(scale2, 64)
# # #     scale2 = layers.UpSampling2D(size=(2, 2))(scale2)  
# # #     scale3 = layers.AveragePooling2D(pool_size=(4, 4))(x)
# # #     scale3 = res_block(scale3, 64)
# # #     scale3 = layers.UpSampling2D(size=(4, 4))(scale3)  

# # #     multi_scale = layers.Concatenate()([scale1, scale2, scale3])

# # #     # Additional convolutional layers after concatenation
# # #     multi_scale = fpn_block(multi_scale, 64)  # Adding FPN Block
# # #     x = layers.Conv2D(64, (3, 3), padding='same')(multi_scale)
# # #     x = layers.LeakyReLU(alpha=0.2)(x)
    
# # #     for filters in [64, 128, 256, 512]:
# # #         x = layers.Conv2D(filters, (4, 4), strides=(2, 2), padding='same')(x)
# # #         x = layers.BatchNormalization()(x)
# # #         x = layers.LeakyReLU(negative_slope=0.2)(x)
    
# # #     x = layers.Conv2D(1, (4, 4), padding='same')(x)
# # #     return Model(inputs, x)


# # """
# # import tensorflow as tf
# # from tensorflow.keras.layers import Layer

# # class PixelShuffleLayer(Layer):
# #     def __init__(self, upscale_factor, **kwargs):
# #         super(PixelShuffleLayer, self).__init__(**kwargs)
# #         self.upscale_factor = upscale_factor

# #     def call(self, x):
# #         return tf.nn.depth_to_space(x, self.upscale_factor)

# #     def get_config(self):
# #         config = super().get_config()
# #         config.update({"upscale_factor": self.upscale_factor})
# #         return config
# # import tensorflow as tf
# # from tensorflow.keras import layers, Model

# # def se_block(x, filters, ratio=16):
# #     se = layers.GlobalAveragePooling2D()(x)
# #     se = layers.Dense(filters // ratio, activation='relu')(se)
# #     se = layers.Dense(filters, activation='sigmoid')(se)
# #     se = layers.Reshape((1, 1, filters))(se)
# #     x = layers.Multiply()([x, se])
# #     return x

# # def fpn_block(x, filters):
# #     p1 = layers.Conv2D(filters, (3, 3), padding='same')(x)
# #     p1 = layers.LeakyReLU(alpha=0.2)(p1)
# #     p2 = layers.Conv2D(filters, (3, 3), padding='same', strides=(2, 2))(p1)
# #     p2 = layers.LeakyReLU(alpha=0.2)(p2)
# #     p3 = layers.Conv2D(filters, (3, 3), padding='same', strides=(2, 2))(p2)
# #     p3 = layers.LeakyReLU(alpha=0.2)(p3)

# #     p2_upsampled = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(p3)
# #     p2_upsampled = layers.LeakyReLU(alpha=0.2)(p2_upsampled)
# #     p2 = layers.Add()([p2, p2_upsampled])

# #     p1_upsampled = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(p2)
# #     p1_upsampled = layers.LeakyReLU(alpha=0.2)(p1_upsampled)
# #     p1 = layers.Add()([p1, p1_upsampled])

# #     return p1

# # def residual_dense_block(x, filters, growth_rate=4, layers_in_block=4):
# #     concat_features = [x]
# #     for _ in range(layers_in_block):
# #         x = layers.Conv2D(growth_rate, (3, 3), padding='same')(x)
# #         x = layers.Activation('relu')(x)
# #         concat_features.append(x)
# #         x = layers.Concatenate()(concat_features)
# #     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
# #     return x

# # def rrdb(x, filters, growth_rate=4, res_block=4):
# #     res = x
# #     for _ in range(res_block):
# #         x = residual_dense_block(x, filters, growth_rate)
# #     return layers.Add()([x, layers.Lambda(lambda x: x * 0.2)(res)])

# # def pixel_shuffle_block(x, filters, upscale_factor):
# #     x = layers.Conv2D(filters * (upscale_factor ** 2), (3, 3), padding='same')(x)
# #     x = PixelShuffleLayer(upscale_factor)(x)
# #     x = layers.Activation('relu')(x)
# #     return x

# # def generator(input_shape=(192, 256, 3)):
# #     inputs = layers.Input(shape=input_shape)

# #     # Initial convolution
# #     x = layers.Conv2D(128, (3, 3), padding='same')(inputs)
# #     x = layers.Activation('relu')(x)
# #     x = se_block(x, 128)  # Adding SE Block

# #     # Initial convolution
# #     scale1 = x
# #     for _ in range(4):  # Increase the number of RRDB blocks for better feature extraction
# #         scale1 = rrdb(scale1, 128)

# #     # Multi-Scale Processing Branches
# #     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
# #     for _ in range(4):
# #         scale2 = rrdb(scale2, 128)
# #     scale2 = pixel_shuffle_block(scale2, 128, 2)

# #     scale3 = pixel_shuffle_block(x, 128, 2)
# #     for _ in range(4):
# #         scale3 = rrdb(scale3, 128)
# #     scale3 = layers.AveragePooling2D(pool_size=(2, 2))(scale3)

# #     # Concatenate multi-scale features
# #     multi_scale = layers.Concatenate()([scale1, scale2, scale3])
# #     multi_scale = fpn_block(multi_scale, 128)  # Adding FPN Block

# #     # Additional convolutional layers
# #     multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
# #     multi_scale = layers.Activation('relu')(multi_scale)

# #     # Upsampling to the final resolution using Pixel Shuffle
# #     multi_scale = pixel_shuffle_block(multi_scale, 128, 2)
# #     multi_scale = pixel_shuffle_block(multi_scale, 128, 2)

# #     outputs = layers.Conv2D(3, (3, 3), padding='same', activation='tanh')(multi_scale)

# #     return Model(inputs, outputs)

# # def res_block(x, filters):
# #     res = x
# #     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
# #     x = layers.LeakyReLU(alpha=0.2)(x)
# #     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
# #     x = layers.Add()([x, res])
# #     return x

# # def discriminator(input_shape=(768, 1024, 3)):
# #     inputs = layers.Input(shape=input_shape)

# #     # Initial convolution block
# #     x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
# #     x = layers.LeakyReLU(alpha=0.2)(x)

# #     # Multi-scale processing branches
# #     scale1 = res_block(x, 64)
# #     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
# #     scale2 = res_block(scale2, 64)
# #     scale2 = layers.UpSampling2D(size=(2, 2))(scale2)
# #     scale3 = layers.AveragePooling2D(pool_size=(4, 4))(x)
# #     scale3 = res_block(scale3, 64)
# #     scale3 = layers.UpSampling2D(size=(4, 4))(scale3)

# #     multi_scale = layers.Concatenate()([scale1, scale2, scale3])

# #     # Additional convolutional layers after concatenation
# #     multi_scale = fpn_block(multi_scale, 64)  # Adding FPN Block
# #     x = layers.Conv2D(64, (3, 3), padding='same')(multi_scale)
# #     x = layers.LeakyReLU(alpha=0.2)(x)

# #     for filters in [64, 128, 256, 512]:
# #         x = layers.Conv2D(filters, (4, 4), strides=(2, 2), padding='same')(x)
# #         x = layers.BatchNormalization()(x)
# #         x = layers.LeakyReLU(negative_slope=0.2)(x)

# #     x = layers.Conv2D(1, (4, 4), padding='same')(x)
# #     return Model(inputs, x)


# # """


# ## -- Best Model -- ##
# '''

# import tensorflow as tf
# from tensorflow.keras import layers, Model # type: ignore

# class AttentionGate(layers.Layer):
#     def __init__(self, filters, **kwargs):
#         super(AttentionGate, self).__init__(**kwargs)
#         self.filters = filters
#         self.conv_f = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)
#         self.conv_g = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)
#         self.conv_h = layers.Conv2D(1, (1, 1), padding='same', use_bias=False)
#         self.activation = layers.Activation('sigmoid')
        
#     def call(self, x, g):
#         f = self.conv_f(x)
#         g = self.conv_g(g)
#         h = layers.Add()([f, g])
#         h = self.activation(h)
#         h = self.conv_h(h)
#         return layers.Multiply()([x, h])

# class NoiseInjection(layers.Layer):
#     def __init__(self, **kwargs):
#         super(NoiseInjection, self).__init__(**kwargs)
        
#     def call(self, x, training=None):
#         if training:
#             noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=1.0)
#             return x + noise
#         return x

# class SelfAttentionLayer(tf.keras.layers.Layer):
#     def __init__(self, filters, **kwargs):
#         super(SelfAttentionLayer, self).__init__(**kwargs)
#         self.filters = filters
#         self.query_conv = layers.Conv2D(filters // 8, kernel_size=1)
#         self.key_conv = layers.Conv2D(filters // 8, kernel_size=1)
#         self.value_conv = layers.Conv2D(filters, kernel_size=1)
#         self.gamma = self.add_weight(name="gamma", shape=[1], initializer='zeros', trainable=True)
#         self.ln = layers.LayerNormalization()

#     def call(self, x):
#         batch_size, height, width, channels = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
#         query = tf.reshape(self.query_conv(x), [batch_size, height * width, self.filters // 8])
#         key = tf.reshape(self.key_conv(x), [batch_size, height * width, self.filters // 8])
#         value = tf.reshape(self.value_conv(x), [batch_size, height * width, self.filters])

#         attention = tf.nn.softmax(tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(tf.cast(self.filters // 8, tf.float32)))
#         out = tf.matmul(attention, value)
#         out = tf.reshape(out, [batch_size, height, width, self.filters])
#         out = self.gamma * out + x
#         return self.ln(out)
    
#     def compute_output_shape(self, input_shape):
#         return input_shape

# def se_block(x, filters, ratio=16):
#     se = layers.GlobalAveragePooling2D()(x)
#     se = layers.Dense(filters // ratio, activation='relu')(se)
#     se = layers.Dense(filters, activation='sigmoid')(se)
#     se = layers.Reshape((1, 1, filters))(se)
#     x = layers.Multiply()([x, se])
#     return x

# def residual_dense_block(x, filters, growth_rate=32, layers_in_block=4):
#     concat_features = [x]
#     for _ in range(layers_in_block):
#         x = layers.Conv2D(growth_rate, (3, 3), padding='same')(x)
#         x = layers.Activation('relu')(x)
#         concat_features.append(x)
#         x = layers.Concatenate()(concat_features)
#     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
#     return x

# def rrdb(x, filters, growth_rate=32, res_block=4):
#     res = x
#     for _ in range(res_block):
#         x = residual_dense_block(x, filters, growth_rate)
#     return layers.Add()([x, layers.Lambda(lambda x: x * 0.2)(res)])

# def fpn_block(x, filters):
#     p1 = layers.Conv2D(filters, (3, 3), padding='same')(x)
#     p1 = layers.LeakyReLU(alpha=0.2)(p1)
#     p2 = layers.Conv2D(filters, (3, 3), padding='same', strides=(2, 2))(p1)
#     p2 = layers.LeakyReLU(alpha=0.2)(p2)
#     p3 = layers.Conv2D(filters, (3, 3), padding='same', strides=(2, 2))(p2)
#     p3 = layers.LeakyReLU(alpha=0.2)(p3)
#     p4 = layers.Conv2D(filters, (3, 3), padding='same', strides=(2, 2))(p3)
#     p4 = layers.LeakyReLU(alpha=0.2)(p4)

#     p3_upsampled = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(p4)
#     p3_upsampled = layers.LeakyReLU(alpha=0.2)(p3_upsampled)
#     p3 = layers.Add()([p3, p3_upsampled])

#     p2_upsampled = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(p3)
#     p2_upsampled = layers.LeakyReLU(alpha=0.2)(p2_upsampled)
#     p2 = layers.Add()([p2, p2_upsampled])

#     p1_upsampled = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(p2)
#     p1_upsampled = layers.LeakyReLU(alpha=0.2)(p1_upsampled)
#     p1 = layers.Add()([p1, p1_upsampled])

#     return p1

# def generator(input_shape=(192, 256, 3)):
#     inputs = layers.Input(shape=input_shape)
    
#     # Noise Injection Layer
#     noisy_inputs = NoiseInjection()(inputs, training=True)

#     x = layers.Conv2D(128, (3, 3), padding='same')(noisy_inputs)
#     x = layers.LeakyReLU(alpha=0.2)(x)
#     x = se_block(x, 128)  # Adding SE Block

#     # Adding Residual-in-Residual Dense Blocks with Self-Attention
#     rrdb_outputs = []
#     for _ in range(4):
#         x = rrdb(x, 128)
#         # x = SelfAttentionLayer(128)(x)
#         rrdb_outputs.append(x)

#     # # Attention Gates
#     # for i in range(len(rrdb_outputs)):
#     #     if i > 0:
#     #         x = AttentionGate(64)(x, rrdb_outputs[i])

#     # Multi-Scale Feature Processing with FPN
#     scale1 = x
#     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
#     scale2 = rrdb(scale2, 128)
#     scale2 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(scale2)
#     scale2 = layers.LeakyReLU(alpha=0.2)(scale2)
#     scale3 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
#     scale3 = layers.AveragePooling2D(pool_size=(2, 2))(scale3)
#     scale3 = rrdb(scale3, 128)
#     scale4 = layers.AveragePooling2D(pool_size=(2, 2))(scale3)
#     scale4 = rrdb(scale4, 128)
#     scale4 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(scale4)
#     scale4 = layers.LeakyReLU(alpha=0.2)(scale4)
    
#     multi_scale = layers.Concatenate()([scale1, scale2, scale3, scale4])
#     multi_scale = fpn_block(multi_scale, 128)
    
#     # Upsampling to the final resolution
#     multi_scale = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(multi_scale)
#     multi_scale = layers.LeakyReLU(alpha=0.2)(multi_scale)
#     multi_scale = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(multi_scale)
#     multi_scale = layers.LeakyReLU(alpha=0.2)(multi_scale)

#     outputs = layers.Conv2D(3, (3, 3), padding='same', activation='tanh')(multi_scale)
    
#     return Model(inputs, outputs)

# def res_block(x, filters):
#     res = x
#     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
#     x = layers.LeakyReLU(alpha=0.2)(x)
#     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
#     x = layers.Add()([x, res])
#     return x


# def discriminator(input_shape=(768, 1024, 3)):
#     inputs = layers.Input(shape=input_shape)

#     # Initial Conv Layer
#     x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
#     x = layers.LeakyReLU(alpha=0.2)(x)
#     ## all 32 must be 64. I changed it 1/19
#     # Multi-Scale Feature Extraction
#     scale1 = res_block(x, 64)
#     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
#     scale2 = res_block(scale2,64)
#     scale2 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(scale2)
#     scale2 = layers.LeakyReLU(alpha=0.2)(scale2)
#     scale3 = layers.AveragePooling2D(pool_size=(4, 4))(x)
#     scale3 = res_block(scale3, 64)
#     scale3 = layers.Conv2DTranspose(64, (3, 3), strides=(4, 4), padding='same')(scale3)
#     scale3 = layers.LeakyReLU(alpha=0.2)(scale3)

#     multi_scale = layers.Concatenate()([scale1, scale2, scale3])
#     x = layers.Conv2D(64, (3, 3), padding='same')(multi_scale)
#     x = layers.LeakyReLU(alpha=0.2)(x)

#     # Downsampling Layers
#     for filters in [64, 128, 256, 512]:
#         x = layers.Conv2D(filters, (4, 4), strides=(2, 2), padding='same')(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.LeakyReLU(alpha=0.2)(x)

#     # Final Conv Layer for PatchGAN
#     x = layers.Conv2D(1, (4, 4), padding='same')(x)
    
#     return Model(inputs, x)


# '''