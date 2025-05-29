import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Reshape, Multiply, Add, Concatenate, Activation, Lambda, LeakyReLU, Input, AveragePooling2D, UpSampling2D, BatchNormalization # type: ignore
from tensorflow.keras.models import Model # type: ignore

## -- Model Version 1.0 -- ##
# Channel Attention
def channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]
    
    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    
    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)
    
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)
    
    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    
    return Multiply()([input_feature, cbam_feature])

# Spatial Attention Module
class SpatialAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SpatialAttentionLayer, self).__init__(**kwargs)
        
    def call(self, input_feature):
        avg_pool = tf.reduce_mean(input_feature, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(input_feature, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        cbam_feature = Conv2D(filters=1,
                              kernel_size=7,
                              strides=1,
                              padding='same',
                              activation='sigmoid',
                              kernel_initializer='he_normal',
                              use_bias=False)(concat)
        return input_feature * cbam_feature

#CBAM (Combine Channel and Spatial Attention)
def cbam_block(cbam_feature, ratio=8):
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = SpatialAttentionLayer()(cbam_feature)
    return cbam_feature

def residual_dense_block(x, filters, growth_rate=4, layers_in_block=8):
    concat_features = [x]
    for _ in range(layers_in_block):
        x = Conv2D(growth_rate, (3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        concat_features.append(x)
        x = Concatenate()(concat_features)
    x = Conv2D(filters, (3, 3), padding='same')(x)
    return x

def rrdb(x, filters, growth_rate=4, res_block=12):
    res = x
    for _ in range(res_block):
        x = residual_dense_block(x, filters, growth_rate)
        x = cbam_block(x)
    return Add()([x, Lambda(lambda x: x * 0.2)(res)])

def multi_scale_fusion(x):
    scale1 = x
    scale2 = AveragePooling2D(pool_size=(2, 2))(x)
    scale2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(scale2)
    scale2 = LeakyReLU(alpha=0.2)(scale2)
    scale3 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
    scale3 = LeakyReLU(alpha=0.2)(scale3)
    scale3 = AveragePooling2D(pool_size=(2, 2))(scale3)
    return Concatenate()([scale1, scale2, scale3])

def iterative_refinement(x, num_iterations=3):
    for _ in range(num_iterations):
        initial_upscaled = Conv2D(64, (3, 3), padding='same')(x)
        initial_upscaled = LeakyReLU(alpha=0.2)(initial_upscaled)
        
        residual = Conv2D(64, (3, 3), padding='same')(x)
        residual = LeakyReLU(alpha=0.2)(residual)
        
        x = Add()([initial_upscaled, residual])
    return x

def generator(input_shape=(192, 256, 3)):
    inputs = Input(shape=input_shape)
    
    x = Conv2D(64, (3, 3), padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)

    scale1 = x
    for _ in range(4):
        scale1 = rrdb(scale1, 64)

    scale2 = AveragePooling2D(pool_size=(2, 2))(x)
    for _ in range(4):
        scale2 = rrdb(scale2, 64)
    scale2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(scale2) 
    scale2 = LeakyReLU(alpha=0.2)(scale2)

    scale3 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
    for _ in range(4):
        scale3 = rrdb(scale3, 64)
    scale3 = AveragePooling2D(pool_size=(2, 2))(scale3)

    multi_scale = multi_scale_fusion(x)
    multi_scale = iterative_refinement(multi_scale, num_iterations=3)
    
    multi_scale = Conv2D(64, (3, 3), padding='same')(multi_scale)
    multi_scale = LeakyReLU(alpha=0.2)(multi_scale)
    
    multi_scale = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(multi_scale)
    multi_scale = LeakyReLU(alpha=0.2)(multi_scale)

    multi_scale = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(multi_scale) 
    multi_scale = LeakyReLU(alpha=0.2)(multi_scale)

    outputs = Conv2D(3, (3, 3), padding='same', activation='tanh')(multi_scale)
    
    return Model(inputs, outputs)

def res_block(x, filters):
    res = x
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = Add()([x, res])
    return x

def discriminator(input_shape=(768, 1024, 3)):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, (3, 3), padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)

    scale1 = res_block(x, 64)
    scale2 = AveragePooling2D(pool_size=(2, 2))(x)
    scale2 = res_block(scale2, 64)
    scale2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(scale2) 
    scale2 = LeakyReLU(alpha=0.2)(scale2)
    scale3 = AveragePooling2D(pool_size=(4, 4))(x)
    scale3 = res_block(scale3, 64)
    scale3 = Conv2DTranspose(64, (3, 3), strides=(4, 4), padding='same')(scale3) 
    scale3 = LeakyReLU(alpha=0.2)(scale3)
    
    multi_scale = Concatenate()([scale1, scale2, scale3])

    x = Conv2D(64, (3, 3), padding='same')(multi_scale)
    x = LeakyReLU(alpha=0.2)(x)
    
    for filters in [64, 128, 256, 512]:
        x = Conv2D(filters, (4, 4), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(negative_slope=0.2)(x)
    
    x = Conv2D(1, (4, 4), padding='same')(x)
    return Model(inputs, x)

# # Channel Attention Module
# def channel_attention(input_feature, ratio=8):
#     channel = input_feature.shape[-1]
    
#     shared_layer_one = Dense(channel // ratio,
#                              activation='relu',
#                              kernel_initializer='he_normal',
#                              use_bias=True,
#                              bias_initializer='zeros')
#     shared_layer_two = Dense(channel,
#                              kernel_initializer='he_normal',
#                              use_bias=True,
#                              bias_initializer='zeros')
    
#     avg_pool = GlobalAveragePooling2D()(input_feature)
#     avg_pool = Reshape((1, 1, channel))(avg_pool)
#     avg_pool = shared_layer_one(avg_pool)
#     avg_pool = shared_layer_two(avg_pool)
    
#     max_pool = GlobalMaxPooling2D()(input_feature)
#     max_pool = Reshape((1, 1, channel))(max_pool)
#     max_pool = shared_layer_one(max_pool)
#     max_pool = shared_layer_two(max_pool)
    
#     cbam_feature = Add()([avg_pool, max_pool])
#     cbam_feature = Activation('sigmoid')(cbam_feature)
    
#     return Multiply()([input_feature, cbam_feature])

# # Spatial Attention Module using Custom Keras Layer
# class SpatialAttentionLayer(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super(SpatialAttentionLayer, self).__init__(**kwargs)
        
#     def call(self, input_feature):
#         avg_pool = tf.reduce_mean(input_feature, axis=-1, keepdims=True)
#         max_pool = tf.reduce_max(input_feature, axis=-1, keepdims=True)
#         concat = tf.concat([avg_pool, max_pool], axis=-1)
#         cbam_feature = Conv2D(filters=1,
#                               kernel_size=7,
#                               strides=1,
#                               padding='same',
#                               activation='sigmoid',
#                               kernel_initializer='he_normal',
#                               use_bias=False)(concat)
#         return input_feature * cbam_feature

# # CBAM (Combine Channel and Spatial Attention)
# def cbam_block(cbam_feature, ratio=8):
#     cbam_feature = channel_attention(cbam_feature, ratio)
#     cbam_feature = SpatialAttentionLayer()(cbam_feature)
#     return cbam_feature

# # Define Residual Dense Block (RDB)
# def residual_dense_block(x, filters, growth_rate=4, layers_in_block=8):
#     concat_features = [x]
#     for _ in range(layers_in_block):
#         x = Conv2D(growth_rate, (3, 3), padding='same')(x)
#         x = LeakyReLU(alpha=0.2)(x)
#         concat_features.append(x)
#         x = Concatenate()(concat_features)
#     x = Conv2D(filters, (3, 3), padding='same')(x)
#     return x

# # Define Residual-in-Residual Dense Block (RRDB)
# def rrdb(x, filters, growth_rate=4, res_block=12):
#     res = x
#     for _ in range(res_block):
#         x = residual_dense_block(x, filters, growth_rate)
#         x = cbam_block(x)  # Integrate CBAM
#     return Add()([x, Lambda(lambda x: x * 0.2)(res)])

# # Multi-Scale Fusion
# def multi_scale_fusion(x):
#     scale1 = x
#     scale2 = AveragePooling2D(pool_size=(2, 2))(x)
#     scale2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(scale2)
#     # scale2 = UpSampling2D(size=(2, 2))(scale2)
#     scale2 = LeakyReLU(alpha=0.2)(scale2)
#     # scale3 = UpSampling2D(size=(2, 2))(x)
#     scale3 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x) 
#     scale3 = LeakyReLU(alpha=0.2)(scale3)
#     scale3 = AveragePooling2D(pool_size=(2, 2))(scale3)
    
#     return Concatenate()([scale1, scale2, scale3])

# # Corrected Iterative Refinement
# def iterative_refinement(x, num_iterations=3):
#     for _ in range(num_iterations):
#         initial_upscaled = Conv2D(64, (3, 3), padding='same')(x)
#         initial_upscaled = LeakyReLU(alpha=0.2)(initial_upscaled)
        
#         # Ensure residual has the same shape as initial_upscaled
#         residual = Conv2D(64, (3, 3), padding='same')(x)
#         residual = LeakyReLU(alpha=0.2)(residual)
        
#         x = Add()([initial_upscaled, residual])
#     return x

# # Generator with CBAM
# def generator(input_shape=(192, 256, 3)):
#     inputs = Input(shape=input_shape)
    
#     # Initial convolution
#     x = Conv2D(64, (3, 3), padding='same')(inputs)
#     x = LeakyReLU(alpha=0.2)(x)

#     # Multi-Scale Processing Branches
#     scale1 = x
#     for _ in range(4):  # Add 2 RRDB blocks
#         scale1 = rrdb(scale1, 64)

#     scale2 = AveragePooling2D(pool_size=(2, 2))(x)
#     for _ in range(4):  # Add 2 RRDB blocks
#         scale2 = rrdb(scale2, 64)
#     # scale2 = UpSampling2D(size=(2, 2))(scale2)
#     scale2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(scale2) 
#     scale2 = LeakyReLU(alpha=0.2)(scale2)

#     # scale3 = UpSampling2D(size=(2, 2))(x)
#     scale3 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
#     for _ in range(4):  # Add 2 RRDB blocks
#         scale3 = rrdb(scale3, 64)
#     scale3 = AveragePooling2D(pool_size=(2, 2))(scale3)

#     # Multi-Scale Fusion
#     multi_scale = multi_scale_fusion(x)
    
#     # Iterative Refinement
#     multi_scale = iterative_refinement(multi_scale, num_iterations=3)
    
#     # Additional convolutional layers
#     multi_scale = Conv2D(64, (3, 3), padding='same')(multi_scale)
#     multi_scale = LeakyReLU(alpha=0.2)(multi_scale)
    
#     # # Upsampling to the final resolution
#     # multi_scale = UpSampling2D(size=(2, 2))(multi_scale)
#     # multi_scale = Conv2D(64, (3, 3), padding='same')(multi_scale)
#     # multi_scale = LeakyReLU(alpha=0.2)(multi_scale)
#     # Upsampling to the final resolution using Transposed Convolution 
#     multi_scale = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(multi_scale) 
#     multi_scale = LeakyReLU(alpha=0.2)(multi_scale)

#     # multi_scale = UpSampling2D(size=(2, 2))(multi_scale)
#     # multi_scale = Conv2D(3, (3, 3), padding='same')(multi_scale)
#     # Upsampling to the final resolution using Transposed Convolution 
#     multi_scale = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(multi_scale) 
#     multi_scale = LeakyReLU(alpha=0.2)(multi_scale)

#     # Final output, ensure shape (768, 1024, 3)
#     outputs = Conv2D(3, (3, 3), padding='same', activation='tanh')(multi_scale)
    
#     return Model(inputs, outputs)

# # Residual Block for Discriminator
# def res_block(x, filters):
#     res = x
#     x = Conv2D(filters, (3, 3), padding='same')(x)
#     x = LeakyReLU(alpha=0.2)(x)
#     x = Conv2D(filters, (3, 3), padding='same')(x)
#     x = Add()([x, res])
#     return x

# # Enhanced Discriminator with Multi-Scale Processing
# def discriminator(input_shape=(768, 1024, 3)):
#     inputs = Input(shape=input_shape)

#     # Initial convolution block
#     x = Conv2D(64, (3, 3), padding='same')(inputs)
#     x = LeakyReLU(alpha=0.2)(x)

#     # Multi-scale processing branches
#     scale1 = res_block(x, 64)
#     scale2 = AveragePooling2D(pool_size=(2, 2))(x)
#     scale2 = res_block(scale2, 64)
#     # scale2 = UpSampling2D(size=(2, 2))(scale2)  # Upsample back to original scale
#     scale2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(scale2) 
#     scale2 = LeakyReLU(alpha=0.2)(scale2)
#     scale3 = AveragePooling2D(pool_size=(4, 4))(x)
#     scale3 = res_block(scale3, 64)
#     # scale3 = UpSampling2D(size=(4, 4))(scale3)  # Upsample back to original scale
#     scale3 = Conv2DTranspose(64, (3, 3), strides=(4, 4), padding='same')(scale3) 
#     scale3 = LeakyReLU(alpha=0.2)(scale3)
    
#     multi_scale = Concatenate()([scale1, scale2, scale3])

#     # Additional convolutional layers after concatenation
#     x = Conv2D(64, (3, 3), padding='same')(multi_scale)
#     x = LeakyReLU(alpha=0.2)(x)
    
#     # PatchGAN-style convolutions
#     for filters in [64, 128, 256, 512]:
#         x = Conv2D(filters, (4, 4), strides=(2, 2), padding='same')(x)
#         x = BatchNormalization()(x)
#         x = LeakyReLU(negative_slope=0.2)(x)
    
#     x = Conv2D(1, (4, 4), padding='same')(x)
#     return Model(inputs, x)


# import tensorflow as tf
# from tensorflow.keras.layers import Conv2D, Conv2DTranspose, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Reshape, Multiply, Add, Concatenate, Activation, Lambda, LeakyReLU, Input, AveragePooling2D, UpSampling2D, BatchNormalization # type: ignore
# from tensorflow.keras.models import Model # type: ignore

# # Channel Attention Module
# def channel_attention(input_feature, ratio=8):
#     channel = input_feature.shape[-1]
    
#     shared_layer_one = Dense(channel // ratio,
#                              activation='relu',
#                              kernel_initializer='he_normal',
#                              use_bias=True,
#                              bias_initializer='zeros')
#     shared_layer_two = Dense(channel,
#                              kernel_initializer='he_normal',
#                              use_bias=True,
#                              bias_initializer='zeros')
    
#     avg_pool = GlobalAveragePooling2D()(input_feature)
#     avg_pool = Reshape((1, 1, channel))(avg_pool)
#     avg_pool = shared_layer_one(avg_pool)
#     avg_pool = shared_layer_two(avg_pool)
    
#     max_pool = GlobalMaxPooling2D()(input_feature)
#     max_pool = Reshape((1, 1, channel))(max_pool)
#     max_pool = shared_layer_one(max_pool)
#     max_pool = shared_layer_two(max_pool)
    
#     cbam_feature = Add()([avg_pool, max_pool])
#     cbam_feature = Activation('sigmoid')(cbam_feature)
    
#     return Multiply()([input_feature, cbam_feature])

# # Spatial Attention Module using Custom Keras Layer
# class SpatialAttentionLayer(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super(SpatialAttentionLayer, self).__init__(**kwargs)
        
#     def call(self, input_feature):
#         avg_pool = tf.reduce_mean(input_feature, axis=-1, keepdims=True)
#         max_pool = tf.reduce_max(input_feature, axis=-1, keepdims=True)
#         concat = tf.concat([avg_pool, max_pool], axis=-1)
#         cbam_feature = Conv2D(filters=1,
#                               kernel_size=7,
#                               strides=1,
#                               padding='same',
#                               activation='sigmoid',
#                               kernel_initializer='he_normal',
#                               use_bias=False)(concat)
#         return input_feature * cbam_feature

# # CBAM (Combine Channel and Spatial Attention)
# def cbam_block(cbam_feature, ratio=8):
#     cbam_feature = channel_attention(cbam_feature, ratio)
#     cbam_feature = SpatialAttentionLayer()(cbam_feature)
#     return cbam_feature

# # Define Residual Dense Block (RDB)
# def residual_dense_block(x, filters, growth_rate=4, layers_in_block=8):
#     concat_features = [x]
#     for _ in range(layers_in_block):
#         x = Conv2D(growth_rate, (3, 3), padding='same')(x)
#         x = LeakyReLU(alpha=0.2)(x)
#         concat_features.append(x)
#         x = Concatenate()(concat_features)
#     x = Conv2D(filters, (3, 3), padding='same')(x)
#     return x

# # Define Residual-in-Residual Dense Block (RRDB)
# def rrdb(x, filters, growth_rate=4, res_block=12):
#     res = x
#     for _ in range(res_block):
#         x = residual_dense_block(x, filters, growth_rate)
#         x = cbam_block(x)  # Integrate CBAM
#     return Add()([x, Lambda(lambda x: x * 0.2)(res)])

# # Multi-Scale Fusion
# def multi_scale_fusion(x):
#     scale1 = x
#     scale2 = AveragePooling2D(pool_size=(2, 2))(x)
#     scale2 = UpSampling2D(size=(2, 2))(scale2)
#     scale3 = UpSampling2D(size=(2, 2))(x)
#     scale3 = AveragePooling2D(pool_size=(2, 2))(scale3)
#     return Concatenate()([scale1, scale2, scale3])

# # Corrected Iterative Refinement
# def iterative_refinement(x, num_iterations=3):
#     for _ in range(num_iterations):
#         initial_upscaled = Conv2D(64, (3, 3), padding='same')(x)
#         initial_upscaled = LeakyReLU(alpha=0.2)(initial_upscaled)
        
#         # Ensure residual has the same shape as initial_upscaled
#         residual = Conv2D(64, (3, 3), padding='same')(x)
#         residual = LeakyReLU(alpha=0.2)(residual)
        
#         x = Add()([initial_upscaled, residual])
#     return x

# # Generator with CBAM
# def generator(input_shape=(192, 256, 3)):
#     inputs = Input(shape=input_shape)
    
#     # Initial convolution
#     x = Conv2D(64, (3, 3), padding='same')(inputs)
#     x = LeakyReLU(alpha=0.2)(x)

#     # Multi-Scale Processing Branches
#     scale1 = x
#     for _ in range(8):  # Add 2 RRDB blocks
#         scale1 = rrdb(scale1, 64)

#     scale2 = AveragePooling2D(pool_size=(2, 2))(x)
#     for _ in range(8):  # Add 2 RRDB blocks
#         scale2 = rrdb(scale2, 64)
#     scale2 = UpSampling2D(size=(2, 2))(scale2)

#     scale3 = UpSampling2D(size=(2, 2))(x)
#     for _ in range(8):  # Add 2 RRDB blocks
#         scale3 = rrdb(scale3, 64)
#     scale3 = AveragePooling2D(pool_size=(2, 2))(scale3)

#     # Multi-Scale Fusion
#     multi_scale = multi_scale_fusion(x)
    
#     # Iterative Refinement
#     multi_scale = iterative_refinement(multi_scale, num_iterations=3)
    
#     # Additional convolutional layers
#     multi_scale = Conv2D(64, (3, 3), padding='same')(multi_scale)
#     multi_scale = LeakyReLU(alpha=0.2)(multi_scale)
    
#     # Upsampling to the final resolution
#     multi_scale = UpSampling2D(size=(2, 2))(multi_scale)
#     multi_scale = Conv2D(64, (3, 3), padding='same')(multi_scale)
#     multi_scale = LeakyReLU(alpha=0.2)(multi_scale)

#     multi_scale = UpSampling2D(size=(2, 2))(multi_scale)
#     multi_scale = Conv2D(3, (3, 3), padding='same')(multi_scale)

#     # Final output, ensure shape (768, 1024, 3)
#     outputs = Conv2D(3, (3, 3), padding='same', activation='tanh')(multi_scale)
    
#     return Model(inputs, outputs)

# # Residual Block for Discriminator
# def res_block(x, filters):
#     res = x
#     x = Conv2D(filters, (3, 3), padding='same')(x)
#     x = LeakyReLU(alpha=0.2)(x)
#     x = Conv2D(filters, (3, 3), padding='same')(x)
#     x = Add()([x, res])
#     return x

# # Enhanced Discriminator with Multi-Scale Processing
# def discriminator(input_shape=(768, 1024, 3)):
#     inputs = Input(shape=input_shape)

#     # Initial convolution block
#     x = Conv2D(64, (3, 3), padding='same')(inputs)
#     x = LeakyReLU(alpha=0.2)(x)

#     # Multi-scale processing branches
#     scale1 = res_block(x, 64)
#     scale2 = AveragePooling2D(pool_size=(2, 2))(x)
#     scale2 = res_block(scale2, 64)
#     scale2 = UpSampling2D(size=(2, 2))(scale2)  # Upsample back to original scale
#     scale3 = AveragePooling2D(pool_size=(4, 4))(x)
#     scale3 = res_block(scale3, 64)
#     scale3 = UpSampling2D(size=(4, 4))(scale3)  # Upsample back to original scale
#     multi_scale = Concatenate()([scale1, scale2, scale3])

#     # Additional convolutional layers after concatenation
#     x = Conv2D(64, (3, 3), padding='same')(multi_scale)
#     x = LeakyReLU(alpha=0.2)(x)
    
#     # PatchGAN-style convolutions
#     for filters in [64, 128, 256, 512]:
#         x = Conv2D(filters, (4, 4), strides=(2, 2), padding='same')(x)
#         x = BatchNormalization()(x)
#         x = LeakyReLU(negative_slope=0.2)(x)
    
#     x = Conv2D(1, (4, 4), padding='same')(x)
#     return Model(inputs, x)

# ## -- Version 2.0 -- ##
# def channel_attention(input_feature, ratio=8):
#     channel = input_feature.shape[-1]
    
#     shared_layer_one = Dense(channel // ratio,
#                              activation='relu',
#                              kernel_initializer='he_normal',
#                              use_bias=True,
#                              bias_initializer='zeros')
#     shared_layer_two = Dense(channel,
#                              kernel_initializer='he_normal',
#                              use_bias=True,
#                              bias_initializer='zeros')
    
#     avg_pool = GlobalAveragePooling2D()(input_feature)
#     avg_pool = Reshape((1, 1, channel))(avg_pool)
#     avg_pool = shared_layer_one(avg_pool)
#     avg_pool = shared_layer_two(avg_pool)
    
#     max_pool = GlobalMaxPooling2D()(input_feature)
#     max_pool = Reshape((1, 1, channel))(max_pool)
#     max_pool = shared_layer_one(max_pool)
#     max_pool = shared_layer_two(max_pool)
    
#     cbam_feature = Add()([avg_pool, max_pool])
#     cbam_feature = Activation('sigmoid')(cbam_feature)
    
#     return Multiply()([input_feature, cbam_feature])

# class SpatialAttentionLayer(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super(SpatialAttentionLayer, self).__init__(**kwargs)
        
#     def call(self, input_feature):
#         avg_pool = tf.reduce_mean(input_feature, axis=-1, keepdims=True)
#         max_pool = tf.reduce_max(input_feature, axis=-1, keepdims=True)
#         concat = tf.concat([avg_pool, max_pool], axis=-1)
#         cbam_feature = Conv2D(filters=1,
#                               kernel_size=7,
#                               strides=1,
#                               padding='same',
#                               activation='sigmoid',
#                               kernel_initializer='he_normal',
#                               use_bias=False)(concat)
#         return input_feature * cbam_feature

# def cbam_block(cbam_feature, ratio=8):
#     cbam_feature = channel_attention(cbam_feature, ratio)
#     cbam_feature = SpatialAttentionLayer()(cbam_feature)
#     return cbam_feature

# def residual_dense_block(x, filters, growth_rate=4, layers_in_block=8):
#     concat_features = [x]
#     for _ in range(layers_in_block):
#         x = Conv2D(growth_rate, (3, 3), padding='same')(x)
#         x = LeakyReLU(alpha=0.2)(x)
#         concat_features.append(x)
#         x = Concatenate()(concat_features)
#     x = Conv2D(filters, (3, 3), padding='same')(x)
#     return x

# def rrdb(x, filters, growth_rate=4, res_block=12):
#     res = x
#     for _ in range(res_block):
#         x = residual_dense_block(x, filters, growth_rate)
#         x = cbam_block(x)
#     return Add()([x, Lambda(lambda x: x * 0.2)(res)])

# def multi_scale_fusion(x):
#     scale1 = x
#     scale2 = AveragePooling2D(pool_size=(2, 2))(x)
#     scale2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(scale2)
#     scale2 = LeakyReLU(alpha=0.2)(scale2)
#     scale3 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
#     scale3 = LeakyReLU(alpha=0.2)(scale3)
#     scale3 = AveragePooling2D(pool_size=(2, 2))(scale3)
#     return Concatenate()([scale1, scale2, scale3])

# def iterative_refinement(x, num_iterations=3):
#     for _ in range(num_iterations):
#         initial_upscaled = Conv2D(64, (3, 3), padding='same')(x)
#         initial_upscaled = LeakyReLU(alpha=0.2)(initial_upscaled)
        
#         residual = Conv2D(64, (3, 3), padding='same')(x)
#         residual = LeakyReLU(alpha=0.2)(residual)
        
#         x = Add()([initial_upscaled, residual])
#     return x

# def generator(input_shape=(192, 256, 3)):
#     inputs = Input(shape=input_shape)
    
#     x = Conv2D(64, (3, 3), padding='same')(inputs)
#     x = LeakyReLU(alpha=0.2)(x)

#     scale1 = x
#     for _ in range(4):
#         scale1 = rrdb(scale1, 64)

#     scale2 = AveragePooling2D(pool_size=(2, 2))(x)
#     for _ in range(4):
#         scale2 = rrdb(scale2, 64)
#     scale2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(scale2) 
#     scale2 = LeakyReLU(alpha=0.2)(scale2)

#     scale3 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
#     for _ in range(4):
#         scale3 = rrdb(scale3, 64)
#     scale3 = AveragePooling2D(pool_size=(2, 2))(scale3)

#     multi_scale = multi_scale_fusion(x)
#     multi_scale = iterative_refinement(multi_scale, num_iterations=3)
    
#     multi_scale = Conv2D(64, (3, 3), padding='same')(multi_scale)
#     multi_scale = LeakyReLU(alpha=0.2)(multi_scale)
    
#     multi_scale = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(multi_scale)
#     multi_scale = LeakyReLU(alpha=0.2)(multi_scale)

#     multi_scale = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(multi_scale) 
#     multi_scale = LeakyReLU(alpha=0.2)(multi_scale)

#     outputs = Conv2D(3, (3, 3), padding='same', activation='tanh')(multi_scale)
    
#     return Model(inputs, outputs)

# def res_block(x, filters):
#     res = x
#     x = Conv2D(filters, (3, 3), padding='same')(x)
#     x = LeakyReLU(alpha=0.2)(x)
#     x = Conv2D(filters, (3, 3), padding='same')(x)
#     x = Add()([x, res])
#     return x

# def discriminator(input_shape=(768, 1024, 3)):
#     inputs = Input(shape=input_shape)

#     x = Conv2D(64, (3, 3), padding='same')(inputs)
#     x = LeakyReLU(alpha=0.2)(x)

#     scale1 = res_block(x, 64)
#     scale2 = AveragePooling2D(pool_size=(2, 2))(x)
#     scale2 = res_block(scale2, 64)
#     scale2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(scale2) 
#     scale2 = LeakyReLU(alpha=0.2)(scale2)
#     scale3 = AveragePooling2D(pool_size=(4, 4))(x)
#     scale3 = res_block(scale3, 64)
#     scale3 = Conv2DTranspose(64, (3, 3), strides=(4, 4), padding='same')(scale3) 
#     scale3 = LeakyReLU(alpha=0.2)(scale3)
    
#     multi_scale = Concatenate()([scale1, scale2, scale3])

#     x = Conv2D(64, (3, 3), padding='same')(multi_scale)
#     x = LeakyReLU(alpha=0.2)(x)
    
#     for filters in [64, 128, 256, 512]:
#         x = Conv2D(filters, (4, 4), strides=(2, 2), padding='same')(x)
#         x = BatchNormalization()(x)
#         x = LeakyReLU(negative_slope=0.2)(x)
    
#     x = Conv2D(1, (4, 4), padding='same')(x)
#     return Model(inputs, x)


# ## -- Self Attention for Upsampling -- ##
# from tensorflow.keras.layers import LayerNormalization, Input, Conv2D, Conv2DTranspose, LeakyReLU, AveragePooling2D, Concatenate, Add # type: ignore
# from tensorflow.keras.models import Model # type: ignore

# class SelfAttentionLayer(tf.keras.layers.Layer):
#     def __init__(self, filters, **kwargs):
#         super(SelfAttentionLayer, self).__init__(**kwargs)
#         self.filters = filters
#         self.query_conv = Conv2D(filters // 8, kernel_size=1)
#         self.key_conv = Conv2D(filters // 8, kernel_size=1)
#         self.value_conv = Conv2D(filters, kernel_size=1)
#         self.gamma = self.add_weight(name="gamma", shape=[1], initializer='zeros', trainable=True)
#         self.ln = LayerNormalization()

#     def call(self, x):
#         batch_size, height, width, channels = x.get_shape().as_list()
#         query = tf.reshape(self.query_conv(x), [batch_size, -1, height * width])
#         key = tf.reshape(self.key_conv(x), [batch_size, -1, height * width])
#         value = tf.reshape(self.value_conv(x), [batch_size, -1, height * width])
        
#         attention = tf.nn.softmax(tf.matmul(query, key, transpose_b=True) / (self.filters ** 0.5), axis=-1)
#         out = tf.reshape(tf.matmul(attention, value), [batch_size, height, width, channels])
#         out = self.gamma * out + x
#         return self.ln(out)

# def residual_dense_block(x, filters, growth_rate=4, layers_in_block=8):
#     concat_features = [x]
#     for _ in range(layers_in_block):
#         x = Conv2D(growth_rate, (3, 3), padding='same')(x)
#         x = LeakyReLU(alpha=0.2)(x)
#         concat_features.append(x)
#         x = Concatenate()(concat_features)
#     x = Conv2D(filters, (3, 3), padding='same')(x)
#     return x

# def rrdb(x, filters, growth_rate=4, res_block=12):
#     res = x
#     for _ in range(res_block):
#         x = residual_dense_block(x, filters, growth_rate)
#         x = cbam_block(x)
#     return Add()([x, Lambda(lambda x: x * 0.2)(res)])

# def multi_scale_fusion(x):
#     scale1 = x
#     scale2 = AveragePooling2D(pool_size=(2, 2))(x)
#     scale2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(scale2)
#     scale2 = LeakyReLU(alpha=0.2)(scale2)
#     scale3 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
#     scale3 = LeakyReLU(alpha=0.2)(scale3)
#     scale3 = AveragePooling2D(pool_size=(2, 2))(scale3)
#     return Concatenate()([scale1, scale2, scale3])

# def iterative_refinement(x, num_iterations=3):
#     for _ in range(num_iterations):
#         initial_upscaled = Conv2D(64, (3, 3), padding='same')(x)
#         initial_upscaled = LeakyReLU(alpha=0.2)(initial_upscaled)
        
#         residual = Conv2D(64, (3, 3), padding='same')(x)
#         residual = LeakyReLU(alpha=0.2)(residual)
        
#         x = Add()([initial_upscaled, residual])
#     return x

# def generator(input_shape=(192, 256, 3)):
#     inputs = Input(shape=input_shape)
    
#     x = Conv2D(64, (3, 3), padding='same')(inputs)
#     x = LeakyReLU(alpha=0.2)(x)

#     scale1 = x
#     for _ in range(4):
#         scale1 = rrdb(scale1, 64)

#     scale2 = AveragePooling2D(pool_size=(2, 2))(x)
#     for _ in range(4):
#         scale2 = rrdb(scale2, 64)
#     scale2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(scale2) 
#     scale2 = LeakyReLU(alpha=0.2)(scale2)

#     scale3 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
#     for _ in range(4):
#         scale3 = rrdb(scale3, 64)
#     scale3 = AveragePooling2D(pool_size=(2, 2))(scale3)

#     multi_scale = multi_scale_fusion(x)
#     multi_scale = iterative_refinement(multi_scale, num_iterations=3)
    
#     multi_scale = Conv2D(64, (3, 3), padding='same')(multi_scale)
#     multi_scale = LeakyReLU(alpha=0.2)(multi_scale)

#     # Adding Self-Attention Layer before upsampling
#     multi_scale = SelfAttentionLayer(filters=64)(multi_scale)
    
#     multi_scale = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(multi_scale)
#     multi_scale = LeakyReLU(alpha=0.2)(multi_scale)

#     multi_scale = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(multi_scale) 
#     multi_scale = LeakyReLU(alpha=0.2)(multi_scale)

#     outputs = Conv2D(3, (3, 3), padding='same', activation='tanh')(multi_scale)
    
#     return Model(inputs, outputs)

# def res_block(x, filters):
#     res = x
#     x = Conv2D(filters, (3, 3), padding='same')(x)
#     x = LeakyReLU(alpha=0.2)(x)
#     x = Conv2D(filters, (3, 3), padding='same')(x)
#     x = Add()([x, res])
#     return x

# def discriminator(input_shape=(768, 1024, 3)):
#     inputs = Input(shape=input_shape)

#     x = Conv2D(64, (3, 3), padding='same')(inputs)
#     x = LeakyReLU(alpha=0.2)(x)

#     scale1 = res_block(x, 64)
#     scale2 = AveragePooling2D(pool_size=(2, 2))(x)
#     scale2 = res_block(scale2, 64)
#     scale2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(scale2) 
#     scale2 = LeakyReLU(alpha=0.2)(scale2)
#     scale3 = AveragePooling2D(pool_size=(4, 4))(x)
#     scale3 = res_block(scale3, 64)
#     scale3 = Conv2DTranspose(64, (3, 3), strides=(4, 4), padding='same')(scale3) 
#     scale3 = LeakyReLU(alpha=0.2)(scale3)
    
#     multi_scale = Concatenate()([scale1, scale2, scale3])

#     x = Conv2D(64, (3, 3), padding='same')(multi_scale)
#     x = LeakyReLU(alpha=0.2)(x)
    
#     for filters in [64, 128, 256, 512]:
#         x = Conv2D(filters, (4, 4), strides=(2, 2), padding='same')(x)
#         x = BatchNormalization()(x)
#         x = LeakyReLU(negative_slope=0.2)(x)
    
#     x = Conv2D(1, (4, 4), padding='same')(x)
#     return Model(inputs, x)
