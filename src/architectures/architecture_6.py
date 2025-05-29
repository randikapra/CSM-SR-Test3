
################## -- Test -- ############################
## VERSION 1.0 ##
import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore



# Custom DynamicUpsampling Layer
class DynamicUpsampling(layers.Layer):
    def __init__(self, scale, **kwargs):
        super(DynamicUpsampling, self).__init__(**kwargs)
        self.scale = scale
        self.filters = tf.Variable(initial_value=tf.random.normal([3, 3, self.scale, self.scale]), trainable=True)

    def call(self, inputs):
        batch_size, height, width, channels = tf.shape(inputs)
        upscaled = tf.nn.conv2d_transpose(inputs, self.filters, output_shape=[batch_size, height * self.scale, width * self.scale, channels], strides=[1, self.scale, self.scale, 1], padding='SAME')
        return upscaled

class PixelAdaptiveConvolution(layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(PixelAdaptiveConvolution, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dynamic_filter_gen = layers.Conv2D(filters * kernel_size[0] * kernel_size[1] * filters, (1, 1))

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        channels = tf.shape(inputs)[3]
        
        dynamic_filters = self.dynamic_filter_gen(inputs)
        dynamic_filters = tf.reshape(dynamic_filters, [batch_size, height, width, self.kernel_size[0], self.kernel_size[1], channels, self.filters])
        
        # Create patches of the input with the same size as the dynamic filters
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.kernel_size[0], self.kernel_size[1], 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding='SAME'
        )
        patches = tf.reshape(patches, [batch_size, height, width, self.kernel_size[0], self.kernel_size[1], channels])
        
        # Perform element-wise multiplication between patches and dynamic filters
        outputs = tf.einsum('bhwklc,bhwklcf->bhwf', patches, dynamic_filters)
        
        return outputs

# Custom PixelShuffle Layer
class PixelShuffle(layers.Layer):
    def __init__(self, scale, **kwargs):
        super(PixelShuffle, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, block_size=self.scale)

# # Define Residual Dense Block (RDB) with Mixed Convolution Types
def residual_dense_block(x, filters, growth_rate=4, layers_in_block=4):
    concat_features = [x]
    for _ in range(layers_in_block):
        x = layers.Conv2D(growth_rate, (3, 3), padding='same')(x)  # Standard convolution
        x = layers.Activation('relu')(x)
        
        # Dilated convolution
        x = layers.Conv2D(growth_rate, (3, 3), padding='same', dilation_rate=2)(x)
        x = layers.Activation('relu')(x)
        
        # # PixelAdaptiveConvolution
        # x = PixelAdaptiveConvolution(growth_rate, (3, 3))(x)
        # x = layers.Activation('relu')(x)

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

def generator(input_shape=(192, 256, 3)):
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
    scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
    print("Generator scale2 input after pooling:", scale2)
    for i in range(3):
        print(f"Generator scale2 RRDB block {i} input:", scale2)
        scale2 = rrdb(scale2, 128)
        print(f"Generator scale2 RRDB block {i} output:", scale2)
    # Upscale by 2
    scale2 = PixelShuffle(scale=2)(scale2)
    # scale2 = DynamicUpsampling(scale=2)(scale2)

    # # Downscale by 4 (scale3)
    # scale3 = layers.AveragePooling2D(pool_size=(4, 4))(x)
    # for _ in range(2):
    #     scale3 = rrdb(scale3, 128)
    # # Upscale by 4
    # scale3 = PixelShuffle(scale=4)(scale3)
    
    # Upscale by 2 (scale4)
    scale4 = PixelShuffle(scale=2)(x)
    print("Generator scale4 input after PixelShuffle:", scale4)
    for i in range(3):
        print(f"Generator scale4 RRDB block {i} input:", scale4)
        scale4 = rrdb(scale4, 128)
        print(f"Generator scale4 RRDB block {i} output:", scale4)
    # Downscale by 2
    scale4 = layers.AveragePooling2D(pool_size=(2, 2))(scale4)
    print("Generator scale4 output after pooling:", scale4)

    # # Upscale by 4 and Downscale by 4 (scale5)
    # scale5 = PixelShuffle(scale=4)(x)
    # for _ in range(2):
    #     scale5 = rrdb(scale5, 128)
    # scale5 = layers.AveragePooling2D(pool_size=(4, 4))(scale5)
    
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

## TEST ##
# import tensorflow as tf
# from tensorflow.keras import layers, Model # type: ignore

# ## --Attetion-- ##
# # Channel Attention Layer
# def channel_attention(x, filters, reduction=16):
#     avg_pool = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
#     avg_pool = layers.Conv2D(filters // reduction, (1, 1), activation='relu', padding='same')(avg_pool)
#     avg_pool = layers.Conv2D(filters, (1, 1), activation='sigmoid', padding='same')(avg_pool)
    
#     max_pool = tf.reduce_max(x, axis=[1, 2], keepdims=True)
#     max_pool = layers.Conv2D(filters // reduction, (1, 1), activation='relu', padding='same')(max_pool)
#     max_pool = layers.Conv2D(filters, (1, 1), activation='sigmoid', padding='same')(max_pool)
    
#     return layers.Add()([x, avg_pool, max_pool])

# # Spatial Attention Layer
# def spatial_attention(x):
#     avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
#     max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
#     concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
#     return layers.Conv2D(1, (7, 7), activation='sigmoid', padding='same')(concat)
# ## --end attention-- ##
# '''
# # Feature Pyramid Network (FPN) Block
# def fpn_block(inputs, filters):
#     P3 = layers.Conv2D(filters, (3, 3), padding='same')(inputs)
#     P3 = layers.Activation('relu')(P3)
    
#     P4 = layers.MaxPooling2D(pool_size=(2, 2))(P3)
#     P4 = layers.Conv2D(filters, (3, 3), padding='same')(P4)
#     P4 = layers.Activation('relu')(P4)

#     P5 = layers.MaxPooling2D(pool_size=(2, 2))(P4)
#     P5 = layers.Conv2D(filters, (3, 3), padding='same')(P5)
#     P5 = layers.Activation('relu')(P5)
    
#     # P4_up = layers.UpSampling2D(size=(2, 2))(P5)
#     P4_up = PixelShuffle(scale=2)(P5)
#     P4 = layers.Add()([P4, P4_up])
    
#     # P3_up = layers.UpSampling2D(size=(2, 2))(P4)
#     P3_up = PixelShuffle(scale=2)(P4)
#     P3 = layers.Add()([P3, P3_up])
    
#     return P3

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
# '''
# # Custom PixelShuffle Layer
# class PixelShuffle(layers.Layer):
#     def __init__(self, scale, **kwargs):
#         super(PixelShuffle, self).__init__(**kwargs)
#         self.scale = scale

#     def call(self, inputs):
#         return tf.nn.depth_to_space(inputs, block_size=self.scale)

# # Define Squeeze-and-Excitation Block
# def se_block(input_tensor, reduction=16):
#     channel = input_tensor.shape[-1]
#     se = layers.GlobalAveragePooling2D()(input_tensor)
#     se = layers.Dense(channel // reduction, activation='relu')(se)
#     se = layers.Dense(channel, activation='sigmoid')(se)
#     se = layers.Multiply()([input_tensor, se])
#     return se

# # Define Residual Dense Block (RDB) with Mixed Convolution Types
# def residual_dense_block(x, filters, growth_rate=16, layers_in_block=4):
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
        
#         # SE Block
#         # x = se_block(x)

#         concat_features.append(x)
#         x = layers.Concatenate()(concat_features)
    
#     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
#     return x

# # Define Residual-in-Residual Dense Block (RRDB)
# def rrdb(x, filters, growth_rate=16, res_block=4):
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
#     P4_up = layers.Activation('relu')(P4_up)
#     P4 = layers.Add()([P4, P4_up])
    
#     P3_up = PixelShuffle(scale=2)(P4)
#     P3_up = layers.Conv2D(filters, (3, 3), padding='same')(P3_up)  # Match the number of filters
#     P3_up = layers.Activation('relu')(P3_up)
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

# # Define the multi-stage generator function
# def multi_stage_generator(input_shape=(192, 256, 3)):
#     inputs = layers.Input(shape=input_shape)
    
#     # First stage
#     stage1_output = generator(input_shape)(inputs)
    
#     # Second stage
#     stage2_input = layers.Concatenate()([inputs, stage1_output])
#     stage2_output = generator(input_shape)(stage2_input)
    
#     return Model(inputs, stage2_output)

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

"""
## WORKING BETTER CODE ##
# Custom PixelShuffle Layer
class PixelShuffle(layers.Layer):
    def __init__(self, scale, **kwargs):
        super(PixelShuffle, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, block_size=self.scale)

# Define Residual Dense Block (RDB) with Mixed Convolution Types
def residual_dense_block(x, filters, growth_rate=8, layers_in_block=5):
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
def rrdb(x, filters, growth_rate=8, res_block=4):
    res = layers.Conv2D(filters, (3, 3), padding='same')(x)
    for _ in range(res_block):
        x = residual_dense_block(x, filters, growth_rate)
    return layers.Add()([x, layers.Lambda(lambda x: x * 0.2)(res)])


# Feature Pyramid Network (FPN) Block with PixelShuffle
def fpn_block(inputs, filters):
    P3 = layers.Conv2D(filters, (3, 3), padding='same')(inputs)
    P3 = layers.Activation('relu')(P3)
    
    P4 = layers.MaxPooling2D(pool_size=(2, 2))(P3)
    P4 = layers.Conv2D(filters, (3, 3), padding='same')(P4)
    P4 = layers.Activation('relu')(P4)

    P5 = layers.MaxPooling2D(pool_size=(2, 2))(P4)
    P5 = layers.Conv2D(filters * 4, (3, 3), padding='same')(P5)  # Increase filters for PixelShuffle
    P5 = layers.Activation('relu')(P5)
    
    P4_up = PixelShuffle(scale=2)(P5)
    P4_up = layers.Conv2D(filters, (3, 3), padding='same')(P4_up)  # Match the number of filters
    P4 = layers.Add()([P4, P4_up])
    
    P3_up = PixelShuffle(scale=2)(P4)
    P3_up = layers.Conv2D(filters, (3, 3), padding='same')(P3_up)  # Match the number of filters
    P3 = layers.Add()([P3, P3_up])
    
    return P3

def generator(input_shape=(192, 256, 3)):
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv2D(128, (3, 3), padding='same')(inputs)
    x = layers.Activation('relu')(x)
    
    # Multi-Scale Processing Branches
    scale1 = x
    for _ in range(2):
        scale1 = rrdb(scale1, 128)

    scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
    for _ in range(2):
        scale2 = rrdb(scale2, 128)
    scale2 = PixelShuffle(scale=2)(scale2)
    
    scale3 = PixelShuffle(scale=2)(x)
    for _ in range(2):
        scale3 = rrdb(scale3, 128)
    scale3 = layers.AveragePooling2D(pool_size=(2, 2))(scale3)
    
    # # Additional multi-scale level
    # scale4 = PixelShuffle(scale=4)(x)
    # for _ in range(3):
    #     scale4 = rrdb(scale4, 128)
    # scale4 = layers.AveragePooling2D(pool_size=(2, 2))(scale4)
    
    # Concatenate multi-scale features
    multi_scale = layers.Concatenate()([scale1, scale2, scale3])
    
    # Additional convolutional layers
    multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
    multi_scale = layers.Activation('relu')(multi_scale)
    
    # Incorporate FPN
    multi_scale = fpn_block(multi_scale, 128)
    
    # Upsampling to the final resolution
    multi_scale = PixelShuffle(scale=2)(multi_scale)
    multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
    multi_scale = layers.Activation('relu')(multi_scale)
    
    multi_scale = PixelShuffle(scale=2)(multi_scale)
    multi_scale = layers.Conv2D(3, (3, 3), padding='same')(multi_scale)
    
    # Final output
    outputs = layers.Conv2D(3, (3, 3), padding='same', activation='tanh')(multi_scale)
    
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
    
    # # Additional multi-scale level
    # scale4 = layers.AveragePooling2D(pool_size=(8, 8))(x)
    # scale4 = res_block(scale4, 32)
    # scale4 = PixelShuffle(scale=8)(scale4)

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
"""

## VERSION 2.0 ##
'''
import tensorflow as tf
from tensorflow.keras import layers, Model

# Custom PixelShuffle Layer
class PixelShuffle(layers.Layer):
    def __init__(self, scale, **kwargs):
        super(PixelShuffle, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, block_size=self.scale)

# Define Residual Dense Block (RDB)
def residual_dense_block(x, filters, growth_rate=8, layers_in_block=4):
    concat_features = [x]
    for _ in range(layers_in_block):
        x = layers.Conv2D(growth_rate, (3, 3), padding='same')(x)
        x = layers.Activation('relu')(x)
        concat_features.append(x)
        x = layers.Concatenate()(concat_features)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    return x

# Define Residual-in-Residual Dense Block (RRDB)
def rrdb(x, filters, growth_rate=8, res_block=4):
    res = layers.Conv2D(filters, (3, 3), padding='same')(x)
    for _ in range(res_block):
        x = residual_dense_block(x, filters, growth_rate)
    return layers.Add()([x, layers.Lambda(lambda x: x * 0.2)(res)])

# Feature Pyramid Network (FPN) Block
def fpn_block(inputs, filters):
    P3 = layers.Conv2D(filters, (3, 3), padding='same')(inputs)
    P3 = layers.Activation('relu')(P3)
    
    P4 = layers.MaxPooling2D(pool_size=(2, 2))(P3)
    P4 = layers.Conv2D(filters, (3, 3), padding='same')(P4)
    P4 = layers.Activation('relu')(P4)

    P5 = layers.MaxPooling2D(pool_size=(2, 2))(P4)
    P5 = layers.Conv2D(filters, (3, 3), padding='same')(P5)
    P5 = layers.Activation('relu')(P5)
    
    P4_up = layers.UpSampling2D(size=(2, 2))(P5)
    P4 = layers.Add()([P4, P4_up])
    
    P3_up = layers.UpSampling2D(size=(2, 2))(P4)
    P3 = layers.Add()([P3, P3_up])
    
    return P3

def generator(input_shape=(192, 256, 3)):
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv2D(128, (3, 3), padding='same')(inputs)
    x = layers.Activation('relu')(x)
    
    # Multi-Scale Processing Branches
    scale1 = x
    for _ in range(3):
        scale1 = rrdb(scale1, 128)

    scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
    for _ in range(3):
        scale2 = rrdb(scale2, 128)
    scale2 = PixelShuffle(scale=2)(scale2)
    
    scale3 = PixelShuffle(scale=2)(x)
    for _ in range(3):
        scale3 = rrdb(scale3, 128)
    scale3 = layers.AveragePooling2D(pool_size=(2, 2))(scale3)
    
    # Additional multi-scale level
    scale4 = PixelShuffle(scale=4)(x)
    for _ in range(3):
        scale4 = rrdb(scale4, 128)
    scale4 = layers.AveragePooling2D(pool_size=(8, 8))(scale4)
    scale4 = layers.UpSampling2D(size=(8, 8))(scale4)  # Correct upsampling to match spatial dimensions with other scales
    
    # Concatenate multi-scale features
    scale4 = layers.UpSampling2D(size=(2, 2))(scale4)  # Ensure further matching dimensions
    multi_scale = layers.Concatenate()([scale1, scale2, scale3, scale4])
    
    # Additional convolutional layers
    multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
    multi_scale = layers.Activation('relu')(multi_scale)
    
    # Incorporate FPN
    multi_scale = fpn_block(multi_scale, 128)
    
    # Upsampling to the final resolution
    multi_scale = PixelShuffle(scale=2)(multi_scale)
    multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
    multi_scale = layers.Activation('relu')(multi_scale)
    
    multi_scale = PixelShuffle(scale=2)(multi_scale)
    multi_scale = layers.Conv2D(3, (3, 3), padding='same')(multi_scale)
    
    # Final output
    outputs = layers.Conv2D(3, (3, 3), padding='same', activation='tanh')(multi_scale)
    
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
    
    # Additional multi-scale level
    scale4 = layers.AveragePooling2D(pool_size=(8, 8))(x)
    scale4 = res_block(scale4, 32)
    scale4 = PixelShuffle(scale=8)(scale4)
    scale4 = layers.UpSampling2D(size=(8, 8))(scale4)  # Correct upsampling to match spatial dimensions with other scales
    scale4 = layers.UpSampling2D(size=(2, 2))(scale4)  # Further match dimensions

    multi_scale = layers.Concatenate()([scale1, scale2, scale3, scale4])

    # Additional convolutional layers after concatenation
    x = layers.Conv2D(32, (3, 3), padding='same')(multi_scale)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    for filters in [64, 128, 256, 512]:
        x = layers.Conv2D(filters, (4, 4), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)
    
    x = layers.Conv2D(1, (4, 4), padding='same')(x)
    return Model(inputs, x)
'''

### VERSION 3.0 ###
"""
import tensorflow as tf
from tensorflow.keras import layers, Model

# Custom PixelShuffle Layer
class PixelShuffle(layers.Layer):
    def __init__(self, scale, **kwargs):
        super(PixelShuffle, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, block_size=self.scale)

# Gaussian Pooling Layer
class GaussianPooling(layers.Layer):
    def __init__(self, pool_size=(2, 2), sigma=1.0, **kwargs):
        super(GaussianPooling, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.sigma = sigma

    def build(self, input_shape):
        kernel_size = (2 * self.sigma + 1, 2 * self.sigma + 1)
        self.kernel = self._gaussian_kernel(kernel_size, self.sigma, input_shape[-1])

    def _gaussian_kernel(self, kernel_size, sigma, channels):
        ax = tf.range(-kernel_size[0] // 2 + 1., kernel_size[0] // 2 + 1.)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[:, :, tf.newaxis, tf.newaxis], [1, 1, channels, 1])
        return kernel

    def call(self, inputs):
        smoothed = tf.nn.depthwise_conv2d(inputs, self.kernel, strides=[1, 1, 1, 1], padding='SAME')
        return tf.nn.avg_pool2d(smoothed, ksize=self.pool_size, strides=self.pool_size, padding='SAME')

# Define Residual Dense Block (RDB)
def residual_dense_block(x, filters, growth_rate=8, layers_in_block=4):
    concat_features = [x]
    for _ in range(layers_in_block):
        x = layers.Conv2D(growth_rate, (3, 3), padding='same')(x)
        x = layers.Activation('relu')(x)
        concat_features.append(x)
        x = layers.Concatenate()(concat_features)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    return x

# Define Residual-in-Residual Dense Block (RRDB)
def rrdb(x, filters, growth_rate=8, res_block=4):
    res = layers.Conv2D(filters, (3, 3), padding='same')(x)
    for _ in range(res_block):
        x = residual_dense_block(x, filters, growth_rate)
    return layers.Add()([x, layers.Lambda(lambda x: x * 0.2)(res)])

# Feature Pyramid Network (FPN) Block
def fpn_block(inputs, filters):
    P3 = layers.Conv2D(filters, (3, 3), padding='same')(inputs)
    P3 = layers.Activation('relu')(P3)
    
    P4 = layers.MaxPooling2D(pool_size=(2, 2))(P3)
    P4 = layers.Conv2D(filters, (3, 3), padding='same')(P4)
    P4 = layers.Activation('relu')(P4)

    P5 = layers.MaxPooling2D(pool_size=(2, 2))(P4)
    P5 = layers.Conv2D(filters, (3, 3), padding='same')(P5)
    P5 = layers.Activation('relu')(P5)
    
    P4_up = layers.UpSampling2D(size=(2, 2))(P5)
    P4 = layers.Add()([P4, P4_up])
    
    P3_up = layers.UpSampling2D(size=(2, 2))(P4)
    P3 = layers.Add()([P3, P3_up])
    
    return P3

def generator(input_shape=(192, 256, 3)):
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv2D(128, (3, 3), padding='same')(inputs)
    x = layers.Activation('relu')(x)
    
    # Multi-Scale Processing Branches
    scale1 = x
    for _ in range(3):  # Increase the number of RRDB blocks for better feature extraction
        scale1 = rrdb(scale1, 128)

    scale2 = GaussianPooling(pool_size=(2, 2))(x)  # Apply Gaussian Pooling here
    for _ in range(3):
        scale2 = rrdb(scale2, 128)
    scale2 = PixelShuffle(scale=2)(scale2)
    
    scale3 = PixelShuffle(scale=2)(x)
    for _ in range(3):
        scale3 = rrdb(scale3, 128)
    scale3 = GaussianPooling(pool_size=(2, 2))(scale3)  # Apply Gaussian Pooling here
    
    # FPN block for feature merging
    fpn_features = fpn_block(multi_scale, 128)

    # Concatenate multi-scale features
    multi_scale = layers.Concatenate()([scale1, scale2, scale3, fpn_features])
    
    # Additional convolutional layers with mixed convolution types
    multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
    multi_scale = layers.Activation('relu')(multi_scale)
    multi_scale = layers.Conv2D(128, (3, 3), padding='same', dilation_rate=2)(multi_scale)  # Dilated convolution
    multi_scale = layers.SeparableConv2D(128, (3, 3), padding='same')(multi_scale)  # Depthwise separable convolution
    
    # Upsampling to the final resolution
    multi_scale = PixelShuffle(scale=2)(multi_scale)
    multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
    multi_scale = layers.Activation('relu')(multi_scale)
    
    multi_scale = PixelShuffle(scale=2)(multi_scale)
    multi_scale = layers.Conv2D(3, (3, 3), padding='same')(multi_scale)
    
    # Final output
    outputs = layers.Conv2D(3, (3, 3), padding='same', activation='tanh')(multi_scale)
    
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
    scale2 = GaussianPooling(pool_size=(2, 2))(x)  # Apply Gaussian Pooling here
    scale2 = res_block(scale2, 32)
    scale2 = PixelShuffle(scale=2)(scale2)
    scale3 = GaussianPooling(pool_size=(4, 4))(x)  # Apply Gaussian Pooling here
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

"""                       
# # Version 1.0
#  import tensorflow as tf
# from tensorflow.keras import layers, Model # type: ignore

# # Custom PixelShuffle Layer
# class PixelShuffle(layers.Layer):
#     def __init__(self, scale, **kwargs):
#         super(PixelShuffle, self).__init__(**kwargs)
#         self.scale = scale

#     def call(self, inputs):
#         return tf.nn.depth_to_space(inputs, block_size=self.scale)

# # Define Residual Dense Block (RDB)
# def residual_dense_block(x, filters, growth_rate=8, layers_in_block=4):
#     concat_features = [x]
#     for _ in range(layers_in_block):
#         x = layers.Conv2D(growth_rate, (3, 3), padding='same')(x)
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

# def generator(input_shape=(192, 256, 3)):
#     inputs = layers.Input(shape=input_shape)
    
#     # Initial convolution
#     x = layers.Conv2D(128, (3, 3), padding='same')(inputs)
#     x = layers.Activation('relu')(x)
    
#     # Multi-Scale Processing Branches
#     scale1 = x
#     for _ in range(3):  # Increase the number of RRDB blocks for better feature extraction
#         scale1 = rrdb(scale1, 128)

#     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
#     for _ in range(3):
#         scale2 = rrdb(scale2, 128)
#     scale2 = PixelShuffle(scale=2)(scale2)
    
#     scale3 = PixelShuffle(scale=2)(x)
#     for _ in range(3):
#         scale3 = rrdb(scale3, 128)
#     scale3 = layers.AveragePooling2D(pool_size=(2, 2))(scale3)
    
#     # Concatenate multi-scale features
#     multi_scale = layers.Concatenate()([scale1, scale2, scale3])
    
#     # Additional convolutional layers
#     multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
#     multi_scale = layers.Activation('relu')(multi_scale)
    
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













## Test 2.0 ## Contaning Gaussian Pooling...
'''

import tensorflow as tf
from tensorflow.keras import layers, Model

# Custom PixelShuffle Layer
class PixelShuffle(layers.Layer):
    def __init__(self, scale, **kwargs):
        super(PixelShuffle, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, block_size=self.scale)

# Gaussian Pooling Layer
class GaussianPooling(layers.Layer):
    def __init__(self, pool_size=(2, 2), sigma=1.0, **kwargs):
        super(GaussianPooling, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.sigma = sigma

    def build(self, input_shape):
        kernel_size = (2 * self.sigma + 1, 2 * self.sigma + 1)
        self.kernel = self._gaussian_kernel(kernel_size, self.sigma, input_shape[-1])

    def _gaussian_kernel(self, kernel_size, sigma, channels):
        ax = tf.range(-kernel_size[0] // 2 + 1., kernel_size[0] // 2 + 1.)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[:, :, tf.newaxis, tf.newaxis], [1, 1, channels, 1])
        return kernel

    def call(self, inputs):
        smoothed = tf.nn.depthwise_conv2d(inputs, self.kernel, strides=[1, 1, 1, 1], padding='SAME')
        return tf.nn.avg_pool2d(smoothed, ksize=self.pool_size, strides=self.pool_size, padding='SAME')

# Define Residual Dense Block (RDB)
def residual_dense_block(x, filters, growth_rate=8, layers_in_block=4):
    concat_features = [x]
    for _ in range(layers_in_block):
        x = layers.Conv2D(growth_rate, (3, 3), padding='same')(x)
        x = layers.Activation('relu')(x)
        concat_features.append(x)
        x = layers.Concatenate()(concat_features)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    return x

# Define Residual-in-Residual Dense Block (RRDB)
def rrdb(x, filters, growth_rate=8, res_block=4):
    res = layers.Conv2D(filters, (3, 3), padding='same')(x)
    for _ in range(res_block):
        x = residual_dense_block(x, filters, growth_rate)
    return layers.Add()([x, layers.Lambda(lambda x: x * 0.2)(res)])

def generator(input_shape=(192, 256, 3)):
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv2D(128, (3, 3), padding='same')(inputs)
    x = layers.Activation('relu')(x)
    
    # Multi-Scale Processing Branches
    scale1 = x
    for _ in range(3):  # Increase the number of RRDB blocks for better feature extraction
        scale1 = rrdb(scale1, 128)

    scale2 = GaussianPooling(pool_size=(2, 2))(x)  # Apply Gaussian Pooling here
    for _ in range(3):
        scale2 = rrdb(scale2, 128)
    scale2 = PixelShuffle(scale=2)(scale2)
    
    scale3 = PixelShuffle(scale=2)(x)
    for _ in range(3):
        scale3 = rrdb(scale3, 128)
    scale3 = GaussianPooling(pool_size=(2, 2))(scale3)  # Apply Gaussian Pooling here
    
    # Concatenate multi-scale features
    multi_scale = layers.Concatenate()([scale1, scale2, scale3])
    
    # Additional convolutional layers
    multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
    multi_scale = layers.Activation('relu')(multi_scale)
    
    # Upsampling to the final resolution
    multi_scale = PixelShuffle(scale=2)(multi_scale)
    multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
    multi_scale = layers.Activation('relu')(multi_scale)
    
    multi_scale = PixelShuffle(scale=2)(multi_scale)
    multi_scale = layers.Conv2D(3, (3, 3), padding='same')(multi_scale)
    
    # Final output
    outputs = layers.Conv2D(3, (3, 3), padding='same', activation='tanh')(multi_scale)
    
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
    scale2 = GaussianPooling(pool_size=(2, 2))(x)  # Apply Gaussian Pooling here
    scale2 = res_block(scale2, 32)
    scale2 = PixelShuffle(scale=2)(scale2)
    scale3 = GaussianPooling(pool_size=(4, 4))(x)  # Apply Gaussian Pooling here
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

'''


##################################################################
# import tensorflow as tf
# from tensorflow.keras import layers, Model # type: ignore


# # Custom PixelShuffle Layer
# class PixelShuffle(layers.Layer):
#     def __init__(self, scale, **kwargs):
#         super(PixelShuffle, self).__init__(**kwargs)
#         self.scale = scale

#     def call(self, inputs):
#         return tf.nn.depth_to_space(inputs, block_size=self.scale)

# # Define Residual Dense Block (RDB)
# def residual_dense_block(x, filters, growth_rate=32, layers_in_block=4):
#     concat_features = [x]
#     for _ in range(layers_in_block):
#         x = layers.Conv2D(growth_rate, (3, 3), padding='same')(x)
#         x = layers.Activation('relu')(x)
#         concat_features.append(x)
#         x = layers.Concatenate()(concat_features)
#     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
#     return x

# # Define Residual-in-Residual Dense Block (RRDB)
# def rrdb(x, filters, growth_rate=32, res_block=4):
#     res = layers.Conv2D(filters, (3, 3), padding='same')(x)
#     for _ in range(res_block):
#         x = residual_dense_block(x, filters, growth_rate)
#     return layers.Add()([x, layers.Lambda(lambda x: x * 0.2)(res)])

# def generator(input_shape=(192, 256, 3)):
#     inputs = layers.Input(shape=input_shape)
    
#     # Initial convolution
#     x = layers.Conv2D(128, (3, 3), padding='same')(inputs)
#     x = layers.Activation('relu')(x)
    
#     # Multi-Scale Processing Branches
#     scale1 = x
#     for _ in range(2):  # Increase the number of RRDB blocks for better feature extraction
#         scale1 = rrdb(scale1, 128)

#     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
#     for _ in range(2):
#         scale2 = rrdb(scale2, 128)
#     scale2 = PixelShuffle(scale=2)(scale2)
    
#     scale3 = PixelShuffle(scale=2)(x)
#     for _ in range(2):
#         scale3 = rrdb(scale3, 128)
#     scale3 = layers.AveragePooling2D(pool_size=(2, 2))(scale3)
    
#     # Concatenate multi-scale features
#     multi_scale = layers.Concatenate()([scale1, scale2, scale3])
    
#     # Additional convolutional layers
#     multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
#     multi_scale = layers.Activation('relu')(multi_scale)
    
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


# # import tensorflow as tf
# # from tensorflow.keras import layers, Model # type: ignore

# # class AttentionGate(layers.Layer):
# #     def __init__(self, filters, **kwargs):
# #         super(AttentionGate, self).__init__(**kwargs)
# #         self.filters = filters
# #         self.conv_f = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)
# #         self.conv_g = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)
# #         self.conv_h = layers.Conv2D(1, (1, 1), padding='same', use_bias=False)
# #         self.activation = layers.Activation('sigmoid')
        
# #     def call(self, x, g):
# #         f = self.conv_f(x)
# #         g = self.conv_g(g)
# #         h = layers.Add()([f, g])
# #         h = self.activation(h)
# #         h = self.conv_h(h)
# #         return layers.Multiply()([x, h])

# # class NoiseInjection(layers.Layer):
# #     def __init__(self, **kwargs):
# #         super(NoiseInjection, self).__init__(**kwargs)
        
# #     def call(self, x, training=None):
# #         if training:
# #             noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=1.0)
# #             return x + noise
# #         return x

# # class SelfAttentionLayer(tf.keras.layers.Layer):
# #     def __init__(self, filters, **kwargs):
# #         super(SelfAttentionLayer, self).__init__(**kwargs)
# #         self.filters = filters
# #         self.query_conv = layers.Conv2D(filters // 8, kernel_size=1)
# #         self.key_conv = layers.Conv2D(filters // 8, kernel_size=1)
# #         self.value_conv = layers.Conv2D(filters, kernel_size=1)
# #         self.gamma = self.add_weight(name="gamma", shape=[1], initializer='zeros', trainable=True)
# #         self.ln = layers.LayerNormalization()

# #     def call(self, x):
# #         batch_size, height, width, channels = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
# #         query = tf.reshape(self.query_conv(x), [batch_size, height * width, self.filters // 8])
# #         key = tf.reshape(self.key_conv(x), [batch_size, height * width, self.filters // 8])
# #         value = tf.reshape(self.value_conv(x), [batch_size, height * width, self.filters])

# #         attention = tf.nn.softmax(tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(tf.cast(self.filters // 8, tf.float32)))
# #         out = tf.matmul(attention, value)
# #         out = tf.reshape(out, [batch_size, height, width, self.filters])
# #         out = self.gamma * out + x
# #         return self.ln(out)
    
# #     def compute_output_shape(self, input_shape):
# #         return input_shape

# # # Define additional context-aware blocks
# # class ContextualAttention(layers.Layer):
# #     def __init__(self, filters, **kwargs):
# #         super(ContextualAttention, self).__init__(**kwargs)
# #         self.filters = filters
# #         self.conv_f = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)
# #         self.conv_g = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)
# #         self.activation = layers.Activation('relu')
        
# #     def call(self, x):
# #         f = self.conv_f(x)
# #         g = self.conv_g(x)
# #         attention = tf.nn.softmax(f + g, axis=-1)
# #         return x * attention
    
# # def se_block(x, filters, ratio=32):
# #     se = layers.GlobalAveragePooling2D()(x)
# #     se = layers.Dense(filters // ratio, activation='relu')(se)
# #     se = layers.Dense(filters, activation='sigmoid')(se)
# #     se = layers.Reshape((1, 1, filters))(se)
# #     x = layers.Multiply()([x, se])
# #     return x

# # def residual_dense_block(x, filters, growth_rate=32, layers_in_block=4):
# #     concat_features = [x]
# #     for _ in range(layers_in_block):
# #         x = layers.Conv2D(growth_rate, (3, 3), padding='same')(x)
# #         x = layers.Activation('relu')(x)
# #         concat_features.append(x)
# #         x = layers.Concatenate()(concat_features)
# #     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
# #     return x

# # def rrdb(x, filters, growth_rate=32, res_block=4):
# #     res = x
# #     for _ in range(res_block):
# #         x = residual_dense_block(x, filters, growth_rate)
# #     return layers.Add()([x, layers.Lambda(lambda x: x * 0.2)(res)])

# # def fpn_block(x, filters):
# #     p1 = layers.Conv2D(filters, (3, 3), padding='same')(x)
# #     p1 = layers.LeakyReLU(alpha=0.2)(p1)
# #     p2 = layers.Conv2D(filters, (3, 3), padding='same', strides=(2, 2))(p1)
# #     p2 = layers.LeakyReLU(alpha=0.2)(p2)
# #     p3 = layers.Conv2D(filters, (3, 3), padding='same', strides=(2, 2))(p2)
# #     p3 = layers.LeakyReLU(alpha=0.2)(p3)
# #     p4 = layers.Conv2D(filters, (3, 3), padding='same', strides=(2, 2))(p3)
# #     p4 = layers.LeakyReLU(alpha=0.2)(p4)

# #     p3_upsampled = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(p4)
# #     p3_upsampled = layers.LeakyReLU(alpha=0.2)(p3_upsampled)
# #     p3 = layers.Add()([p3, p3_upsampled])

# #     p2_upsampled = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(p3)
# #     p2_upsampled = layers.LeakyReLU(alpha=0.2)(p2_upsampled)
# #     p2 = layers.Add()([p2, p2_upsampled])

# #     p1_upsampled = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(p2)
# #     p1_upsampled = layers.LeakyReLU(alpha=0.2)(p1_upsampled)
# #     p1 = layers.Add()([p1, p1_upsampled])

# #     return p1

# # def generator(input_shape=(192, 256, 3)):
# #     inputs = layers.Input(shape=input_shape)
    
# #     # Noise Injection Layer
# #     noisy_inputs = NoiseInjection()(inputs, training=True)

# #     x = layers.Conv2D(128, (3, 3), padding='same')(noisy_inputs)
# #     x = layers.LeakyReLU(alpha=0.2)(x)
# #     # x = se_block(x, 128)  # Adding SE Block

# #     # Multi-Scale Feature Processing with FPN
# #     # Add Contextual Attention blocks
# #     x = ContextualAttention(128)(x)

# #     scale1 = x
# #     for _ in range(4):  # Increase the number of RRDB blocks for better feature extraction
# #         scale1 = rrdb(scale1, 128)

# #     # Scale 2: Half scale
# #     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
# #     for _ in range(4):
# #         scale2 = rrdb(scale2, 128)
# #     scale2 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(scale2)
# #     scale2 = layers.LeakyReLU(alpha=0.2)(scale2)
    
# #     # Scale 3: Double scale 
# #     scale3 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
# #     for _ in range(4):
# #         scale3 = rrdb(scale3, 128)
# #     scale3 = layers.AveragePooling2D(pool_size=(2, 2))(scale3)
    
# #     scale4 = layers.AveragePooling2D(pool_size=(2, 2))(scale3)
# #     scale4 = rrdb(scale4, 128)
# #     scale4 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(scale4)
# #     scale4 = layers.LeakyReLU(alpha=0.2)(scale4)

# #     multi_scale = layers.Concatenate()([scale1, scale2, scale3, scale4])
# #     multi_scale = fpn_block(multi_scale, 128)
    
# #     # Upsampling to the final resolution
# #     multi_scale = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(multi_scale)
# #     multi_scale = layers.LeakyReLU(alpha=0.2)(multi_scale)
# #     multi_scale = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(multi_scale)
# #     multi_scale = layers.LeakyReLU(alpha=0.2)(multi_scale)

# #     outputs = layers.Conv2D(3, (3, 3), padding='same', activation='tanh')(multi_scale)
    
# #     return Model(inputs, outputs)

# # def generator(input_shape=(192, 256, 3)):
# #     inputs = layers.Input(shape=input_shape)
    
# #     # Noise Injection Layer
# #     noisy_inputs = NoiseInjection()(inputs, training=True)

# #     x = layers.Conv2D(128, (3, 3), padding='same')(noisy_inputs)
# #     x = layers.LeakyReLU(alpha=0.2)(x)
# #     # x = se_block(x, 128)  # Adding SE Block

# #     # Adding Residual-in-Residual Dense Blocks with Self-Attention
# #     rrdb_outputs = []
# #     for _ in range(4):
# #         x = rrdb(x, 128)
# #         # x = SelfAttentionLayer(128)(x)
# #         rrdb_outputs.append(x)

# #     # # Attention Gates
# #     # for i in range(len(rrdb_outputs)):
# #     #     if i > 0:
# #     #         x = AttentionGate(64)(x, rrdb_outputs[i])

# #     # Multi-Scale Feature Processing with FPN
# #     scale1 = x
# #     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
# #     scale2 = rrdb(scale2, 128)
# #     scale2 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(scale2)
# #     scale2 = layers.LeakyReLU(alpha=0.2)(scale2)
# #     scale3 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
# #     scale3 = layers.AveragePooling2D(pool_size=(2, 2))(scale3)
# #     scale3 = rrdb(scale3, 128)
# #     scale4 = layers.AveragePooling2D(pool_size=(2, 2))(scale3)
# #     scale4 = rrdb(scale4, 128)
# #     scale4 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(scale4)
# #     scale4 = layers.LeakyReLU(alpha=0.2)(scale4)
    
# #     multi_scale = layers.Concatenate()([scale1, scale2, scale3, scale4])
# #     multi_scale = fpn_block(multi_scale, 128)
    
# #     # Upsampling to the final resolution
# #     multi_scale = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(multi_scale)
# #     multi_scale = layers.LeakyReLU(alpha=0.2)(multi_scale)
# #     multi_scale = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(multi_scale)
# #     multi_scale = layers.LeakyReLU(alpha=0.2)(multi_scale)

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

# #     # Initial Conv Layer
# #     x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
# #     x = layers.LeakyReLU(alpha=0.2)(x)
# #     ## all 32 must be 64. I changed it 1/19
# #     # Multi-Scale Feature Extraction
# #     scale1 = res_block(x, 32)
# #     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
# #     scale2 = res_block(scale2,32)
# #     scale2 = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(scale2)
# #     scale2 = layers.LeakyReLU(alpha=0.2)(scale2)
# #     scale3 = layers.AveragePooling2D(pool_size=(4, 4))(x)
# #     scale3 = res_block(scale3, 32)
# #     scale3 = layers.Conv2DTranspose(32, (3, 3), strides=(4, 4), padding='same')(scale3)
# #     scale3 = layers.LeakyReLU(alpha=0.2)(scale3)

# #     multi_scale = layers.Concatenate()([scale1, scale2, scale3])
# #     x = layers.Conv2D(64, (3, 3), padding='same')(multi_scale)
# #     x = layers.LeakyReLU(alpha=0.2)(x)

# #     # Downsampling Layers
# #     for filters in [64, 128, 256, 512]:
# #         x = layers.Conv2D(filters, (4, 4), strides=(2, 2), padding='same')(x)
# #         x = layers.BatchNormalization()(x)
# #         x = layers.LeakyReLU(alpha=0.2)(x)

# #     # Final Conv Layer for PatchGAN
# #     x = layers.Conv2D(1, (4, 4), padding='same')(x)
    
# #     return Model(inputs, x)


# # '''
# # import tensorflow as tf
# # from tensorflow.keras import layers, Model # type: ignore

# # class AttentionGate(layers.Layer):
# #     def __init__(self, filters, **kwargs):
# #         super(AttentionGate, self).__init__(**kwargs)
# #         self.filters = filters
# #         self.conv_f = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)
# #         self.conv_g = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)
# #         self.conv_h = layers.Conv2D(1, (1, 1), padding='same', use_bias=False)
# #         self.activation = layers.Activation('sigmoid')
        
# #     def call(self, x, g):
# #         f = self.conv_f(x)
# #         g = self.conv_g(g)
# #         h = layers.Add()([f, g])
# #         h = self.activation(h)
# #         h = self.conv_h(h)
# #         return layers.Multiply()([x, h])

# # class NoiseInjection(layers.Layer):
# #     def __init__(self, **kwargs):
# #         super(NoiseInjection, self).__init__(**kwargs)
        
# #     def call(self, x, training=None):
# #         if training:
# #             noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=1.0)
# #             return x + noise
# #         return x

# # class SelfAttentionLayer(tf.keras.layers.Layer):
# #     def __init__(self, filters, **kwargs):
# #         super(SelfAttentionLayer, self).__init__(**kwargs)
# #         self.filters = filters
# #         self.query_conv = layers.Conv2D(filters // 8, kernel_size=1)
# #         self.key_conv = layers.Conv2D(filters // 8, kernel_size=1)
# #         self.value_conv = layers.Conv2D(filters, kernel_size=1)
# #         self.gamma = self.add_weight(name="gamma", shape=[1], initializer='zeros', trainable=True)
# #         self.ln = layers.LayerNormalization()

# #     def call(self, x):
# #         batch_size, height, width, channels = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
# #         query = tf.reshape(self.query_conv(x), [batch_size, height * width, self.filters // 8])
# #         key = tf.reshape(self.key_conv(x), [batch_size, height * width, self.filters // 8])
# #         value = tf.reshape(self.value_conv(x), [batch_size, height * width, self.filters])

# #         attention = tf.nn.softmax(tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(tf.cast(self.filters // 8, tf.float32)))
# #         out = tf.matmul(attention, value)
# #         out = tf.reshape(out, [batch_size, height, width, self.filters])
# #         out = self.gamma * out + x
# #         return self.ln(out)
    
# #     def compute_output_shape(self, input_shape):
# #         return input_shape

# # # Define additional context-aware blocks
# # class ContextualAttention(layers.Layer):
# #     def __init__(self, filters, **kwargs):
# #         super(ContextualAttention, self).__init__(**kwargs)
# #         self.filters = filters
# #         self.conv_f = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)
# #         self.conv_g = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)
# #         self.activation = layers.Activation('relu')
        
# #     def call(self, x):
# #         f = self.conv_f(x)
# #         g = self.conv_g(x)
# #         attention = tf.nn.softmax(f + g, axis=-1)
# #         return x * attention
    
# # def se_block(x, filters, ratio=32):
# #     se = layers.GlobalAveragePooling2D()(x)
# #     se = layers.Dense(filters // ratio, activation='relu')(se)
# #     se = layers.Dense(filters, activation='sigmoid')(se)
# #     se = layers.Reshape((1, 1, filters))(se)
# #     x = layers.Multiply()([x, se])
# #     return x

# # def residual_dense_block(x, filters, growth_rate=32, layers_in_block=4):
# #     concat_features = [x]
# #     for _ in range(layers_in_block):
# #         x = layers.Conv2D(growth_rate, (3, 3), padding='same')(x)
# #         x = layers.Activation('relu')(x)
# #         concat_features.append(x)
# #         x = layers.Concatenate()(concat_features)
# #     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
# #     return x

# # def rrdb(x, filters, growth_rate=32, res_block=4):
# #     res = x
# #     for _ in range(res_block):
# #         x = residual_dense_block(x, filters, growth_rate)
# #     return layers.Add()([x, layers.Lambda(lambda x: x * 0.2)(res)])

# # def fpn_block(x, filters):
# #     p1 = layers.Conv2D(filters, (3, 3), padding='same')(x)
# #     p1 = layers.LeakyReLU(alpha=0.2)(p1)
# #     p2 = layers.Conv2D(filters, (3, 3), padding='same', strides=(2, 2))(p1)
# #     p2 = layers.LeakyReLU(alpha=0.2)(p2)
# #     p3 = layers.Conv2D(filters, (3, 3), padding='same', strides=(2, 2))(p2)
# #     p3 = layers.LeakyReLU(alpha=0.2)(p3)
# #     p4 = layers.Conv2D(filters, (3, 3), padding='same', strides=(2, 2))(p3)
# #     p4 = layers.LeakyReLU(alpha=0.2)(p4)

# #     p3_upsampled = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(p4)
# #     p3_upsampled = layers.LeakyReLU(alpha=0.2)(p3_upsampled)
# #     p3 = layers.Add()([p3, p3_upsampled])

# #     p2_upsampled = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(p3)
# #     p2_upsampled = layers.LeakyReLU(alpha=0.2)(p2_upsampled)
# #     p2 = layers.Add()([p2, p2_upsampled])

# #     p1_upsampled = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(p2)
# #     p1_upsampled = layers.LeakyReLU(alpha=0.2)(p1_upsampled)
# #     p1 = layers.Add()([p1, p1_upsampled])

# #     return p1


# # def generator(input_shape=(192, 256, 3)):
# #     inputs = layers.Input(shape=input_shape)
    
# #     # Noise Injection Layer
# #     noisy_inputs = NoiseInjection()(inputs, training=True)

# #     x = layers.Conv2D(128, (3, 3), padding='same')(noisy_inputs)
# #     x = layers.LeakyReLU(alpha=0.2)(x)
# #     # x = se_block(x, 128)  # Adding SE Block

# #     # Adding Residual-in-Residual Dense Blocks with Self-Attention
# #     rrdb_outputs = []
# #     for _ in range(4):
# #         x = rrdb(x, 128)
# #         # x = SelfAttentionLayer(128)(x)
# #         rrdb_outputs.append(x)

# #     # # Attention Gates
# #     # for i in range(len(rrdb_outputs)):
# #     #     if i > 0:
# #     #         x = AttentionGate(64)(x, rrdb_outputs[i])
# #     # Add Contextual Attention blocks
# #     x = ContextualAttention(128)(x)
# #     # Multi-Scale Feature Processing with FPN
# #     scale1 = x
# #     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
# #     scale2 = rrdb(scale2, 128)
# #     scale2 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(scale2)
# #     scale2 = layers.LeakyReLU(alpha=0.2)(scale2)
# #     scale3 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
# #     scale3 = layers.AveragePooling2D(pool_size=(2, 2))(scale3)
# #     scale3 = rrdb(scale3, 128)
# #     scale4 = layers.AveragePooling2D(pool_size=(2, 2))(scale3)
# #     scale4 = rrdb(scale4, 128)
# #     scale4 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(scale4)
# #     scale4 = layers.LeakyReLU(alpha=0.2)(scale4)
    
# #     multi_scale = layers.Concatenate()([scale1, scale2, scale3, scale4])
# #     multi_scale = fpn_block(multi_scale, 128)
    
# #     # Upsampling to the final resolution
# #     multi_scale = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(multi_scale)
# #     multi_scale = layers.LeakyReLU(alpha=0.2)(multi_scale)
# #     multi_scale = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(multi_scale)
# #     multi_scale = layers.LeakyReLU(alpha=0.2)(multi_scale)

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

# #     # Initial Conv Layer
# #     x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
# #     x = layers.LeakyReLU(alpha=0.2)(x)
# #     ## all 32 must be 64. I changed it 1/19
# #     # Multi-Scale Feature Extraction
# #     scale1 = res_block(x, 32)
# #     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
# #     scale2 = res_block(scale2,32)
# #     scale2 = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(scale2)
# #     scale2 = layers.LeakyReLU(alpha=0.2)(scale2)
# #     scale3 = layers.AveragePooling2D(pool_size=(4, 4))(x)
# #     scale3 = res_block(scale3, 32)
# #     scale3 = layers.Conv2DTranspose(32, (3, 3), strides=(4, 4), padding='same')(scale3)
# #     scale3 = layers.LeakyReLU(alpha=0.2)(scale3)

# #     multi_scale = layers.Concatenate()([scale1, scale2, scale3])
# #     x = layers.Conv2D(64, (3, 3), padding='same')(multi_scale)
# #     x = layers.LeakyReLU(alpha=0.2)(x)

# #     # Downsampling Layers
# #     for filters in [64, 128, 256, 512]:
# #         x = layers.Conv2D(filters, (4, 4), strides=(2, 2), padding='same')(x)
# #         x = layers.BatchNormalization()(x)
# #         x = layers.LeakyReLU(alpha=0.2)(x)

# #     # Final Conv Layer for PatchGAN
# #     x = layers.Conv2D(1, (4, 4), padding='same')(x)
    
# #     return Model(inputs, x)


# # '''



# """
# '''
# ## Relu activation function is used in the generator and discriminator models. ##
# import tensorflow as tf
# from tensorflow.keras import layers, Model # type: ignore

# # Custom PixelShuffle Layer
# class PixelShuffle(layers.Layer):
#     def __init__(self, scale, **kwargs):
#         super(PixelShuffle, self).__init__(**kwargs)
#         self.scale = scale

#     def call(self, inputs):
#         return tf.nn.depth_to_space(inputs, block_size=self.scale)

# # Define Residual Dense Block (RDB)
# def residual_dense_block(x, filters, growth_rate=32, layers_in_block=4):
#     concat_features = [x]
#     for _ in range(layers_in_block):
#         x = layers.Conv2D(growth_rate, (3, 3), padding='same')(x)
#         x = layers.Activation('relu')(x)
#         concat_features.append(x)
#         x = layers.Concatenate()(concat_features)
#     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
#     return x

# # Define Residual-in-Residual Dense Block (RRDB)
# def rrdb(x, filters, growth_rate=32, res_block=4):
#     res = layers.Conv2D(filters, (3, 3), padding='same')(x)
#     for _ in range(res_block):
#         x = residual_dense_block(x, filters, growth_rate)
#     return layers.Add()([x, layers.Lambda(lambda x: x * 0.2)(res)])

# def generator(input_shape=(192, 256, 3)):
#     inputs = layers.Input(shape=input_shape)
    
#     # Initial convolution
#     x = layers.Conv2D(128, (3, 3), padding='same')(inputs)
#     x = layers.Activation('relu')(x)
    
#     # Multi-Scale Processing Branches
#     scale1 = x
#     for _ in range(2):  # Increase the number of RRDB blocks for better feature extraction
#         scale1 = rrdb(scale1, 128)

#     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
#     for _ in range(2):
#         scale2 = rrdb(scale2, 128)
#     scale2 = PixelShuffle(scale=2)(scale2)
    
#     scale3 = PixelShuffle(scale=2)(x)
#     for _ in range(2):
#         scale3 = rrdb(scale3, 128)
#     scale3 = layers.AveragePooling2D(pool_size=(2, 2))(scale3)
    
#     # Concatenate multi-scale features
#     multi_scale = layers.Concatenate()([scale1, scale2, scale3])
    
#     # Additional convolutional layers
#     multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
#     multi_scale = layers.Activation('relu')(multi_scale)
    
#     # Upsampling to the final resolution
#     multi_scale = PixelShuffle(scale=2)(multi_scale)
#     multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
#     multi_scale = layers.Activation('relu')(multi_scale)
    
#     multi_scale = PixelShuffle(scale=2)(multi_scale)
#     multi_scale = layers.Conv2D(3, (3, 3), padding='same')(multi_scale)
    
#     # Final output
#     outputs = layers.Conv2D(3, (3, 3), padding='same', activation='tanh')(multi_scale)
    
#     return Model(inputs, outputs)
# '''

# """
# ## Mesh activation function is used in the generator and discriminator models. ##

# """
# import tensorflow as tf
# from tensorflow.keras import layers, Model # type: ignore

# # Custom PixelShuffle Layer
# class PixelShuffle(layers.Layer):
#     def __init__(self, scale, **kwargs):
#         super(PixelShuffle, self).__init__(**kwargs)
#         self.scale = scale

#     def call(self, inputs):
#         return tf.nn.depth_to_space(inputs, block_size=self.scale)

# # Define Mish Activation Function
# class Mish(layers.Layer):
#     def call(self, inputs):
#         return inputs * tf.math.tanh(tf.math.softplus(inputs))

# # Define Residual Dense Block (RDB)
# def residual_dense_block(x, filters, growth_rate=16, layers_in_block=4):
#     concat_features = [x]
#     for _ in range(layers_in_block):
#         x = layers.Conv2D(growth_rate, (3, 3), padding='same')(x)
#         x = Mish()(x)
#         concat_features.append(x)
#         x = layers.Concatenate()(concat_features)
#     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
#     return x

# # Define Residual-in-Residual Dense Block (RRDB)
# def rrdb(x, filters, growth_rate=16, res_block=4):
#     res = layers.Conv2D(filters, (3, 3), padding='same')(x)
#     for _ in range(res_block):
#         x = residual_dense_block(x, filters, growth_rate)
#     return layers.Add()([x, layers.Lambda(lambda x: x * 0.2)(res)])

# def generator(input_shape=(192, 256, 3)):
#     inputs = layers.Input(shape=input_shape)
    
#     # Initial convolution
#     x = layers.Conv2D(128, (3, 3), padding='same')(inputs)
#     x = Mish()(x)
    
#     # Multi-Scale Processing Branches
#     scale1 = x
#     for _ in range(2):  # Increase the number of RRDB blocks for better feature extraction
#         scale1 = rrdb(scale1, 128)

#     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
#     for _ in range(2):
#         scale2 = rrdb(scale2, 128)
#     scale2 = PixelShuffle(scale=2)(scale2)
    
#     scale3 = PixelShuffle(scale=2)(x)
#     for _ in range(2):
#         scale3 = rrdb(scale3, 128)
#     scale3 = layers.AveragePooling2D(pool_size=(2, 2))(scale3)
    
#     # Concatenate multi-scale features
#     multi_scale = layers.Concatenate()([scale1, scale2, scale3])
    
#     # Additional convolutional layers
#     multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
#     multi_scale = Mish()(multi_scale)
    
#     # Upsampling to the final resolution
#     multi_scale = PixelShuffle(scale=2)(multi_scale)
#     multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
#     multi_scale = Mish()(multi_scale)
    
#     multi_scale = PixelShuffle(scale=2)(multi_scale)
#     multi_scale = layers.Conv2D(3, (3, 3), padding='same')(multi_scale)
    
#     # Final output
#     outputs = layers.Conv2D(3, (3, 3), padding='same', activation='tanh')(multi_scale)
    
#     return Model(inputs, outputs)


# '''
# # # More multi-scale layer model
# import tensorflow as tf
# from tensorflow.keras import layers, Model # type: ignore

# # Custom PixelShuffle Layer
# class PixelShuffle(layers.Layer):
#     def __init__(self, scale, **kwargs):
#         super(PixelShuffle, self).__init__(**kwargs)
#         self.scale = scale

#     def call(self, inputs):
#         return tf.nn.depth_to_space(inputs, block_size=self.scale)

# # Define Residual Dense Block (RDB)
# def residual_dense_block(x, filters, growth_rate=32, layers_in_block=4):
#     concat_features = [x]
#     for _ in range(layers_in_block):
#         x = layers.Conv2D(growth_rate, (3, 3), padding='same')(x)
#         x = layers.Activation('relu')(x)
#         concat_features.append(x)
#         x = layers.Concatenate()(concat_features)
#     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
#     return x

# # Define Residual-in-Residual Dense Block (RRDB)
# def rrdb(x, filters, growth_rate=32, res_block=4):
#     res = layers.Conv2D(filters, (3, 3), padding='same')(x)
#     for _ in range(res_block):
#         x = residual_dense_block(x, filters, growth_rate)
#     return layers.Add()([x, layers.Lambda(lambda x: x * 0.2)(res)])

# def generator(input_shape=(192, 256, 3)):
#     inputs = layers.Input(shape=input_shape)
    
#     # Initial convolution
#     x = layers.Conv2D(128, (3, 3), padding='same')(inputs)
#     x = layers.Activation('relu')(x)
    
#     # Multi-Scale Processing Branches
#     scale1 = x
#     for _ in range(2):  # Number of RRDB blocks for better feature extraction
#         scale1 = rrdb(scale1, 128)

#     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
#     for _ in range(2):
#         scale2 = rrdb(scale2, 128)
#     scale2 = PixelShuffle(scale=2)(scale2)
    
#     scale3 = PixelShuffle(scale=2)(x)
#     for _ in range(2):
#         scale3 = rrdb(scale3, 128)
#     scale3 = layers.AveragePooling2D(pool_size=(2, 2))(scale3)

#     scale4 = PixelShuffle(scale=4)(x)
#     for _ in range(2):
#         scale4 = rrdb(scale4, 128)
#     scale4 = layers.AveragePooling2D(pool_size=(4, 4))(scale4)
    
#     # Concatenate multi-scale features
#     multi_scale = layers.Concatenate()([scale1, scale2, scale3, scale4])
    
#     # Additional convolutional layers
#     multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
#     multi_scale = layers.Activation('relu')(multi_scale)
    
#     # Multiple upsampling blocks
#     multi_scale = PixelShuffle(scale=2)(multi_scale)
#     multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
#     multi_scale = layers.Activation('relu')(multi_scale)

#     multi_scale = PixelShuffle(scale=2)(multi_scale)
#     multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
#     multi_scale = layers.Activation('relu')(multi_scale)
    
#     # Final output
#     outputs = layers.Conv2D(3, (3, 3), padding='same', activation='tanh')(multi_scale)
    
#     return Model(inputs, outputs)
# '''

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


# # import tensorflow as tf
# # from tensorflow.keras import layers, Model # type: ignore

# # class AttentionGate(layers.Layer):
# #     def __init__(self, filters, **kwargs):
# #         super(AttentionGate, self).__init__(**kwargs)
# #         self.filters = filters
# #         self.conv_f = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)
# #         self.conv_g = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)
# #         self.conv_h = layers.Conv2D(1, (1, 1), padding='same', use_bias=False)
# #         self.activation = layers.Activation('sigmoid')
        
# #     def call(self, x, g):
# #         f = self.conv_f(x)
# #         g = self.conv_g(g)
# #         h = layers.Add()([f, g])
# #         h = self.activation(h)
# #         h = self.conv_h(h)
# #         return layers.Multiply()([x, h])

# # class NoiseInjection(layers.Layer):
# #     def __init__(self, **kwargs):
# #         super(NoiseInjection, self).__init__(**kwargs)
        
# #     def call(self, x, training=None):
# #         if training:
# #             noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=1.0)
# #             return x + noise
# #         return x

# # class SelfAttentionLayer(tf.keras.layers.Layer):
# #     def __init__(self, filters, **kwargs):
# #         super(SelfAttentionLayer, self).__init__(**kwargs)
# #         self.filters = filters
# #         self.query_conv = layers.Conv2D(filters // 8, kernel_size=1)
# #         self.key_conv = layers.Conv2D(filters // 8, kernel_size=1)
# #         self.value_conv = layers.Conv2D(filters, kernel_size=1)
# #         self.gamma = self.add_weight(name="gamma", shape=[1], initializer='zeros', trainable=True)
# #         self.ln = layers.LayerNormalization()

# #     def call(self, x):
# #         batch_size, height, width, channels = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
# #         query = tf.reshape(self.query_conv(x), [batch_size, height * width, self.filters // 8])
# #         key = tf.reshape(self.key_conv(x), [batch_size, height * width, self.filters // 8])
# #         value = tf.reshape(self.value_conv(x), [batch_size, height * width, self.filters])

# #         attention = tf.nn.softmax(tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(tf.cast(self.filters // 8, tf.float32)))
# #         out = tf.matmul(attention, value)
# #         out = tf.reshape(out, [batch_size, height, width, self.filters])
# #         out = self.gamma * out + x
# #         return self.ln(out)
    
# #     def compute_output_shape(self, input_shape):
# #         return input_shape

# # # Define additional context-aware blocks
# # class ContextualAttention(layers.Layer):
# #     def __init__(self, filters, **kwargs):
# #         super(ContextualAttention, self).__init__(**kwargs)
# #         self.filters = filters
# #         self.conv_f = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)
# #         self.conv_g = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)
# #         self.activation = layers.Activation('relu')
        
# #     def call(self, x):
# #         f = self.conv_f(x)
# #         g = self.conv_g(x)
# #         attention = tf.nn.softmax(f + g, axis=-1)
# #         return x * attention
    
# # def se_block(x, filters, ratio=32):
# #     se = layers.GlobalAveragePooling2D()(x)
# #     se = layers.Dense(filters // ratio, activation='relu')(se)
# #     se = layers.Dense(filters, activation='sigmoid')(se)
# #     se = layers.Reshape((1, 1, filters))(se)
# #     x = layers.Multiply()([x, se])
# #     return x

# # def residual_dense_block(x, filters, growth_rate=32, layers_in_block=4):
# #     concat_features = [x]
# #     for _ in range(layers_in_block):
# #         x = layers.Conv2D(growth_rate, (3, 3), padding='same')(x)
# #         x = layers.Activation('relu')(x)
# #         concat_features.append(x)
# #         x = layers.Concatenate()(concat_features)
# #     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
# #     return x

# # def rrdb(x, filters, growth_rate=32, res_block=4):
# #     res = x
# #     for _ in range(res_block):
# #         x = residual_dense_block(x, filters, growth_rate)
# #     return layers.Add()([x, layers.Lambda(lambda x: x * 0.2)(res)])

# # def fpn_block(x, filters):
# #     p1 = layers.Conv2D(filters, (3, 3), padding='same')(x)
# #     p1 = layers.LeakyReLU(alpha=0.2)(p1)
# #     p2 = layers.Conv2D(filters, (3, 3), padding='same', strides=(2, 2))(p1)
# #     p2 = layers.LeakyReLU(alpha=0.2)(p2)
# #     p3 = layers.Conv2D(filters, (3, 3), padding='same', strides=(2, 2))(p2)
# #     p3 = layers.LeakyReLU(alpha=0.2)(p3)
# #     p4 = layers.Conv2D(filters, (3, 3), padding='same', strides=(2, 2))(p3)
# #     p4 = layers.LeakyReLU(alpha=0.2)(p4)

# #     p3_upsampled = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(p4)
# #     p3_upsampled = layers.LeakyReLU(alpha=0.2)(p3_upsampled)
# #     p3 = layers.Add()([p3, p3_upsampled])

# #     p2_upsampled = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(p3)
# #     p2_upsampled = layers.LeakyReLU(alpha=0.2)(p2_upsampled)
# #     p2 = layers.Add()([p2, p2_upsampled])

# #     p1_upsampled = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(p2)
# #     p1_upsampled = layers.LeakyReLU(alpha=0.2)(p1_upsampled)
# #     p1 = layers.Add()([p1, p1_upsampled])

# #     return p1

# # def generator(input_shape=(192, 256, 3)):
# #     inputs = layers.Input(shape=input_shape)
    
# #     # Noise Injection Layer
# #     noisy_inputs = NoiseInjection()(inputs, training=True)

# #     x = layers.Conv2D(128, (3, 3), padding='same')(noisy_inputs)
# #     x = layers.LeakyReLU(alpha=0.2)(x)
# #     # x = se_block(x, 128)  # Adding SE Block

# #     # Multi-Scale Feature Processing with FPN
# #     # Add Contextual Attention blocks
# #     x = ContextualAttention(128)(x)

# #     scale1 = x
# #     for _ in range(4):  # Increase the number of RRDB blocks for better feature extraction
# #         scale1 = rrdb(scale1, 128)

# #     # Scale 2: Half scale
# #     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
# #     for _ in range(4):
# #         scale2 = rrdb(scale2, 128)
# #     scale2 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(scale2)
# #     scale2 = layers.LeakyReLU(alpha=0.2)(scale2)
    
# #     # Scale 3: Double scale 
# #     scale3 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
# #     for _ in range(4):
# #         scale3 = rrdb(scale3, 128)
# #     scale3 = layers.AveragePooling2D(pool_size=(2, 2))(scale3)
    
# #     scale4 = layers.AveragePooling2D(pool_size=(2, 2))(scale3)
# #     scale4 = rrdb(scale4, 128)
# #     scale4 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(scale4)
# #     scale4 = layers.LeakyReLU(alpha=0.2)(scale4)

# #     multi_scale = layers.Concatenate()([scale1, scale2, scale3, scale4])
# #     multi_scale = fpn_block(multi_scale, 128)
    
# #     # Upsampling to the final resolution
# #     multi_scale = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(multi_scale)
# #     multi_scale = layers.LeakyReLU(alpha=0.2)(multi_scale)
# #     multi_scale = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(multi_scale)
# #     multi_scale = layers.LeakyReLU(alpha=0.2)(multi_scale)

# #     outputs = layers.Conv2D(3, (3, 3), padding='same', activation='tanh')(multi_scale)
    
# #     return Model(inputs, outputs)

# # def generator(input_shape=(192, 256, 3)):
# #     inputs = layers.Input(shape=input_shape)
    
# #     # Noise Injection Layer
# #     noisy_inputs = NoiseInjection()(inputs, training=True)

# #     x = layers.Conv2D(128, (3, 3), padding='same')(noisy_inputs)
# #     x = layers.LeakyReLU(alpha=0.2)(x)
# #     # x = se_block(x, 128)  # Adding SE Block

# #     # Adding Residual-in-Residual Dense Blocks with Self-Attention
# #     rrdb_outputs = []
# #     for _ in range(4):
# #         x = rrdb(x, 128)
# #         # x = SelfAttentionLayer(128)(x)
# #         rrdb_outputs.append(x)

# #     # # Attention Gates
# #     # for i in range(len(rrdb_outputs)):
# #     #     if i > 0:
# #     #         x = AttentionGate(64)(x, rrdb_outputs[i])

# #     # Multi-Scale Feature Processing with FPN
# #     scale1 = x
# #     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
# #     scale2 = rrdb(scale2, 128)
# #     scale2 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(scale2)
# #     scale2 = layers.LeakyReLU(alpha=0.2)(scale2)
# #     scale3 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
# #     scale3 = layers.AveragePooling2D(pool_size=(2, 2))(scale3)
# #     scale3 = rrdb(scale3, 128)
# #     scale4 = layers.AveragePooling2D(pool_size=(2, 2))(scale3)
# #     scale4 = rrdb(scale4, 128)
# #     scale4 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(scale4)
# #     scale4 = layers.LeakyReLU(alpha=0.2)(scale4)
    
# #     multi_scale = layers.Concatenate()([scale1, scale2, scale3, scale4])
# #     multi_scale = fpn_block(multi_scale, 128)
    
# #     # Upsampling to the final resolution
# #     multi_scale = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(multi_scale)
# #     multi_scale = layers.LeakyReLU(alpha=0.2)(multi_scale)
# #     multi_scale = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(multi_scale)
# #     multi_scale = layers.LeakyReLU(alpha=0.2)(multi_scale)

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

# #     # Initial Conv Layer
# #     x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
# #     x = layers.LeakyReLU(alpha=0.2)(x)
# #     ## all 32 must be 64. I changed it 1/19
# #     # Multi-Scale Feature Extraction
# #     scale1 = res_block(x, 32)
# #     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
# #     scale2 = res_block(scale2,32)
# #     scale2 = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(scale2)
# #     scale2 = layers.LeakyReLU(alpha=0.2)(scale2)
# #     scale3 = layers.AveragePooling2D(pool_size=(4, 4))(x)
# #     scale3 = res_block(scale3, 32)
# #     scale3 = layers.Conv2DTranspose(32, (3, 3), strides=(4, 4), padding='same')(scale3)
# #     scale3 = layers.LeakyReLU(alpha=0.2)(scale3)

# #     multi_scale = layers.Concatenate()([scale1, scale2, scale3])
# #     x = layers.Conv2D(64, (3, 3), padding='same')(multi_scale)
# #     x = layers.LeakyReLU(alpha=0.2)(x)

# #     # Downsampling Layers
# #     for filters in [64, 128, 256, 512]:
# #         x = layers.Conv2D(filters, (4, 4), strides=(2, 2), padding='same')(x)
# #         x = layers.BatchNormalization()(x)
# #         x = layers.LeakyReLU(alpha=0.2)(x)

# #     # Final Conv Layer for PatchGAN
# #     x = layers.Conv2D(1, (4, 4), padding='same')(x)
    
# #     return Model(inputs, x)


# # '''
# # import tensorflow as tf
# # from tensorflow.keras import layers, Model # type: ignore

# # class AttentionGate(layers.Layer):
# #     def __init__(self, filters, **kwargs):
# #         super(AttentionGate, self).__init__(**kwargs)
# #         self.filters = filters
# #         self.conv_f = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)
# #         self.conv_g = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)
# #         self.conv_h = layers.Conv2D(1, (1, 1), padding='same', use_bias=False)
# #         self.activation = layers.Activation('sigmoid')
        
# #     def call(self, x, g):
# #         f = self.conv_f(x)
# #         g = self.conv_g(g)
# #         h = layers.Add()([f, g])
# #         h = self.activation(h)
# #         h = self.conv_h(h)
# #         return layers.Multiply()([x, h])

# # class NoiseInjection(layers.Layer):
# #     def __init__(self, **kwargs):
# #         super(NoiseInjection, self).__init__(**kwargs)
        
# #     def call(self, x, training=None):
# #         if training:
# #             noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=1.0)
# #             return x + noise
# #         return x

# # class SelfAttentionLayer(tf.keras.layers.Layer):
# #     def __init__(self, filters, **kwargs):
# #         super(SelfAttentionLayer, self).__init__(**kwargs)
# #         self.filters = filters
# #         self.query_conv = layers.Conv2D(filters // 8, kernel_size=1)
# #         self.key_conv = layers.Conv2D(filters // 8, kernel_size=1)
# #         self.value_conv = layers.Conv2D(filters, kernel_size=1)
# #         self.gamma = self.add_weight(name="gamma", shape=[1], initializer='zeros', trainable=True)
# #         self.ln = layers.LayerNormalization()

# #     def call(self, x):
# #         batch_size, height, width, channels = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
# #         query = tf.reshape(self.query_conv(x), [batch_size, height * width, self.filters // 8])
# #         key = tf.reshape(self.key_conv(x), [batch_size, height * width, self.filters // 8])
# #         value = tf.reshape(self.value_conv(x), [batch_size, height * width, self.filters])

# #         attention = tf.nn.softmax(tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(tf.cast(self.filters // 8, tf.float32)))
# #         out = tf.matmul(attention, value)
# #         out = tf.reshape(out, [batch_size, height, width, self.filters])
# #         out = self.gamma * out + x
# #         return self.ln(out)
    
# #     def compute_output_shape(self, input_shape):
# #         return input_shape

# # # Define additional context-aware blocks
# # class ContextualAttention(layers.Layer):
# #     def __init__(self, filters, **kwargs):
# #         super(ContextualAttention, self).__init__(**kwargs)
# #         self.filters = filters
# #         self.conv_f = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)
# #         self.conv_g = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)
# #         self.activation = layers.Activation('relu')
        
# #     def call(self, x):
# #         f = self.conv_f(x)
# #         g = self.conv_g(x)
# #         attention = tf.nn.softmax(f + g, axis=-1)
# #         return x * attention
    
# # def se_block(x, filters, ratio=32):
# #     se = layers.GlobalAveragePooling2D()(x)
# #     se = layers.Dense(filters // ratio, activation='relu')(se)
# #     se = layers.Dense(filters, activation='sigmoid')(se)
# #     se = layers.Reshape((1, 1, filters))(se)
# #     x = layers.Multiply()([x, se])
# #     return x

# # def residual_dense_block(x, filters, growth_rate=32, layers_in_block=4):
# #     concat_features = [x]
# #     for _ in range(layers_in_block):
# #         x = layers.Conv2D(growth_rate, (3, 3), padding='same')(x)
# #         x = layers.Activation('relu')(x)
# #         concat_features.append(x)
# #         x = layers.Concatenate()(concat_features)
# #     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
# #     return x

# # def rrdb(x, filters, growth_rate=32, res_block=4):
# #     res = x
# #     for _ in range(res_block):
# #         x = residual_dense_block(x, filters, growth_rate)
# #     return layers.Add()([x, layers.Lambda(lambda x: x * 0.2)(res)])

# # def fpn_block(x, filters):
# #     p1 = layers.Conv2D(filters, (3, 3), padding='same')(x)
# #     p1 = layers.LeakyReLU(alpha=0.2)(p1)
# #     p2 = layers.Conv2D(filters, (3, 3), padding='same', strides=(2, 2))(p1)
# #     p2 = layers.LeakyReLU(alpha=0.2)(p2)
# #     p3 = layers.Conv2D(filters, (3, 3), padding='same', strides=(2, 2))(p2)
# #     p3 = layers.LeakyReLU(alpha=0.2)(p3)
# #     p4 = layers.Conv2D(filters, (3, 3), padding='same', strides=(2, 2))(p3)
# #     p4 = layers.LeakyReLU(alpha=0.2)(p4)

# #     p3_upsampled = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(p4)
# #     p3_upsampled = layers.LeakyReLU(alpha=0.2)(p3_upsampled)
# #     p3 = layers.Add()([p3, p3_upsampled])

# #     p2_upsampled = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(p3)
# #     p2_upsampled = layers.LeakyReLU(alpha=0.2)(p2_upsampled)
# #     p2 = layers.Add()([p2, p2_upsampled])

# #     p1_upsampled = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(p2)
# #     p1_upsampled = layers.LeakyReLU(alpha=0.2)(p1_upsampled)
# #     p1 = layers.Add()([p1, p1_upsampled])

# #     return p1


# # def generator(input_shape=(192, 256, 3)):
# #     inputs = layers.Input(shape=input_shape)
    
# #     # Noise Injection Layer
# #     noisy_inputs = NoiseInjection()(inputs, training=True)

# #     x = layers.Conv2D(128, (3, 3), padding='same')(noisy_inputs)
# #     x = layers.LeakyReLU(alpha=0.2)(x)
# #     # x = se_block(x, 128)  # Adding SE Block

# #     # Adding Residual-in-Residual Dense Blocks with Self-Attention
# #     rrdb_outputs = []
# #     for _ in range(4):
# #         x = rrdb(x, 128)
# #         # x = SelfAttentionLayer(128)(x)
# #         rrdb_outputs.append(x)

# #     # # Attention Gates
# #     # for i in range(len(rrdb_outputs)):
# #     #     if i > 0:
# #     #         x = AttentionGate(64)(x, rrdb_outputs[i])
# #     # Add Contextual Attention blocks
# #     x = ContextualAttention(128)(x)
# #     # Multi-Scale Feature Processing with FPN
# #     scale1 = x
# #     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
# #     scale2 = rrdb(scale2, 128)
# #     scale2 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(scale2)
# #     scale2 = layers.LeakyReLU(alpha=0.2)(scale2)
# #     scale3 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
# #     scale3 = layers.AveragePooling2D(pool_size=(2, 2))(scale3)
# #     scale3 = rrdb(scale3, 128)
# #     scale4 = layers.AveragePooling2D(pool_size=(2, 2))(scale3)
# #     scale4 = rrdb(scale4, 128)
# #     scale4 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(scale4)
# #     scale4 = layers.LeakyReLU(alpha=0.2)(scale4)
    
# #     multi_scale = layers.Concatenate()([scale1, scale2, scale3, scale4])
# #     multi_scale = fpn_block(multi_scale, 128)
    
# #     # Upsampling to the final resolution
# #     multi_scale = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(multi_scale)
# #     multi_scale = layers.LeakyReLU(alpha=0.2)(multi_scale)
# #     multi_scale = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(multi_scale)
# #     multi_scale = layers.LeakyReLU(alpha=0.2)(multi_scale)

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

# #     # Initial Conv Layer
# #     x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
# #     x = layers.LeakyReLU(alpha=0.2)(x)
# #     ## all 32 must be 64. I changed it 1/19
# #     # Multi-Scale Feature Extraction
# #     scale1 = res_block(x, 32)
# #     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
# #     scale2 = res_block(scale2,32)
# #     scale2 = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(scale2)
# #     scale2 = layers.LeakyReLU(alpha=0.2)(scale2)
# #     scale3 = layers.AveragePooling2D(pool_size=(4, 4))(x)
# #     scale3 = res_block(scale3, 32)
# #     scale3 = layers.Conv2DTranspose(32, (3, 3), strides=(4, 4), padding='same')(scale3)
# #     scale3 = layers.LeakyReLU(alpha=0.2)(scale3)

# #     multi_scale = layers.Concatenate()([scale1, scale2, scale3])
# #     x = layers.Conv2D(64, (3, 3), padding='same')(multi_scale)
# #     x = layers.LeakyReLU(alpha=0.2)(x)

# #     # Downsampling Layers
# #     for filters in [64, 128, 256, 512]:
# #         x = layers.Conv2D(filters, (4, 4), strides=(2, 2), padding='same')(x)
# #         x = layers.BatchNormalization()(x)
# #         x = layers.LeakyReLU(alpha=0.2)(x)

# #     # Final Conv Layer for PatchGAN
# #     x = layers.Conv2D(1, (4, 4), padding='same')(x)
    
# #     return Model(inputs, x)


# # '''

# """
# """
# import tensorflow as tf
# from tensorflow.keras import layers, Model # type: ignore

# # Custom PixelShuffle Layer
# class PixelShuffle(layers.Layer):
#     def __init__(self, scale, **kwargs):
#         super(PixelShuffle, self).__init__(**kwargs)
#         self.scale = scale

#     def call(self, inputs):
#         return tf.nn.depth_to_space(inputs, block_size=self.scale)

# # Define Residual Dense Block (RDB)
# def residual_dense_block(x, filters, growth_rate=32, layers_in_block=4):
#     concat_features = [x]
#     for _ in range(layers_in_block):
#         x = layers.Conv2D(growth_rate, (3, 3), padding='same')(x)
#         x = layers.Activation('relu')(x)
#         concat_features.append(x)
#         x = layers.Concatenate()(concat_features)
#     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
#     return x

# # Define Residual-in-Residual Dense Block (RRDB)
# def rrdb(x, filters, growth_rate=32, res_block=4):
#     res = layers.Conv2D(filters, (3, 3), padding='same')(x)
#     for _ in range(res_block):
#         x = residual_dense_block(x, filters, growth_rate)
#     return layers.Add()([x, layers.Lambda(lambda x: x * 0.2)(res)])

# def generator(input_shape=(192, 256, 3)):
#     inputs = layers.Input(shape=input_shape)
    
#     # Initial convolution
#     x = layers.Conv2D(128, (3, 3), padding='same')(inputs)
#     x = layers.Activation('relu')(x)
    
#     # Multi-Scale Processing Branches
#     scale1 = x
#     for _ in range(2):  # Increase the number of RRDB blocks for better feature extraction
#         scale1 = rrdb(scale1, 128)

#     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
#     for _ in range(2):
#         scale2 = rrdb(scale2, 128)
#     scale2 = PixelShuffle(scale=2)(scale2)
    
#     scale3 = PixelShuffle(scale=2)(x)
#     for _ in range(2):
#         scale3 = rrdb(scale3, 128)
#     scale3 = layers.AveragePooling2D(pool_size=(2, 2))(scale3)
    
#     # Concatenate multi-scale features
#     multi_scale = layers.Concatenate()([scale1, scale2, scale3])
    
#     # Additional convolutional layers
#     multi_scale = layers.Conv2D(128, (3, 3), padding='same')(multi_scale)
#     multi_scale = layers.Activation('relu')(multi_scale)
    
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