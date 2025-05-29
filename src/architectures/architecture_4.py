# ## -- Better Model -- ##
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
#     for _ in range(2):
#         x = rrdb(x, 128)
#         # x = SelfAttentionLayer(64)(x)
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
    
#     multi_scale = layers.Concatenate()([scale1, scale2, scale3])
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

#     x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
#     x = layers.LeakyReLU(alpha=0.2)(x)

#     scale1 = res_block(x, 64)
#     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
#     scale2 = res_block(scale2, 64)
#     scale2 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(scale2)
#     scale2 = layers.LeakyReLU(alpha=0.2)(scale2)
#     scale3 = layers.AveragePooling2D(pool_size=(4, 4))(x)
#     scale3 = res_block(scale3, 64)
#     scale3 = layers.Conv2DTranspose(64, (3, 3), strides=(4, 4), padding='same')(scale3)
#     scale3 = layers.LeakyReLU(alpha=0.2)(scale3)

#     multi_scale = layers.Concatenate()([scale1, scale2, scale3])
#     x = layers.Conv2D(64, (3, 3), padding='same')(multi_scale)
#     x = layers.LeakyReLU(alpha=0.2)(x)
    
#     for filters in [64, 128, 256, 512]:
#         x = layers.Conv2D(filters, (4, 4), strides=(2, 2), padding='same')(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.LeakyReLU(negative_slope=0.2)(x)
    
#     x = layers.Conv2D(1, (4, 4), padding='same')(x)
#     return Model(inputs, x)


# # # ## -- Working Architecture -- ## # good results can be gained after tuning
# ## -- Attention Gate with Noise Injection -- ##
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

# # def se_block(x, filters, ratio=16):
# #     se = layers.GlobalAveragePooling2D()(x)
# #     se = layers.Dense(filters // ratio, activation='relu')(se)
# #     se = layers.Dense(filters, activation='sigmoid')(se)
# #     se = layers.Reshape((1, 1, filters))(se)
# #     x = layers.Multiply()([x, se])
# #     return x

# # def residual_dense_block(x, filters, growth_rate=4, layers_in_block=5):
# #     concat_features = [x]
# #     for _ in range(layers_in_block):
# #         x = layers.Conv2D(growth_rate, (3, 3), padding='same')(x)
# #         x = layers.Activation('relu')(x)
# #         concat_features.append(x)
# #         x = layers.Concatenate()(concat_features)
# #     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
# #     return x

# # def rrdb(x, filters, growth_rate=4, res_block=5):
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

# #     x = layers.Conv2D(64, (3, 3), padding='same')(noisy_inputs)
# #     x = layers.LeakyReLU(alpha=0.2)(x)
# #     x = se_block(x, 64)  # Adding SE Block

# #     # Adding Residual-in-Residual Dense Blocks with Self-Attention
# #     rrdb_outputs = []
# #     for _ in range(4):
# #         x = rrdb(x, 64)
# #         # x = SelfAttentionLayer(64)(x)
# #         rrdb_outputs.append(x)

# #     # # Attention Gates
# #     # for i in range(len(rrdb_outputs)):
# #     #     if i > 0:
# #     #         x = AttentionGate(64)(x, rrdb_outputs[i])

# #     # Multi-Scale Feature Processing with FPN
# #     scale1 = x
# #     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
# #     scale2 = rrdb(scale2, 64)
# #     scale2 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(scale2)
# #     scale2 = layers.LeakyReLU(alpha=0.2)(scale2)
# #     scale3 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
# #     scale3 = layers.AveragePooling2D(pool_size=(2, 2))(scale3)
# #     scale3 = rrdb(scale3, 64)
    
# #     multi_scale = layers.Concatenate()([scale1, scale2, scale3])
# #     multi_scale = fpn_block(multi_scale, 64)
    
# #     # Upsampling to the final resolution
# #     multi_scale = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(multi_scale)
# #     multi_scale = layers.LeakyReLU(alpha=0.2)(multi_scale)
# #     multi_scale = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(multi_scale)
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

# #     x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
# #     x = layers.LeakyReLU(alpha=0.2)(x)

# #     scale1 = res_block(x, 64)
# #     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
# #     scale2 = res_block(scale2, 64)
# #     scale2 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(scale2)
# #     scale2 = layers.LeakyReLU(alpha=0.2)(scale2)
# #     scale3 = layers.AveragePooling2D(pool_size=(4, 4))(x)
# #     scale3 = res_block(scale3, 64)
# #     scale3 = layers.Conv2DTranspose(64, (3, 3), strides=(4, 4), padding='same')(scale3)
# #     scale3 = layers.LeakyReLU(alpha=0.2)(scale3)

# #     multi_scale = layers.Concatenate()([scale1, scale2, scale3])
# #     x = layers.Conv2D(64, (3, 3), padding='same')(multi_scale)
# #     x = layers.LeakyReLU(alpha=0.2)(x)
    
# #     for filters in [64, 128, 256, 512]:
# #         x = layers.Conv2D(filters, (4, 4), strides=(2, 2), padding='same')(x)
# #         x = layers.BatchNormalization()(x)
# #         x = layers.LeakyReLU(negative_slope=0.2)(x)
    
# #     x = layers.Conv2D(1, (4, 4), padding='same')(x)
# #     return Model(inputs, x)

# ## -- Self Attention with FPN -- ##
# # import tensorflow as tf
# # from tensorflow.keras import layers, Model # type: ignore

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
# #         batch_size, height, width, channels = x.get_shape().as_list()
# #         query = tf.reshape(self.query_conv(x), [batch_size, -1, height * width])
# #         key = tf.reshape(self.key_conv(x), [batch_size, -1, height * width])
# #         value = tf.reshape(self.value_conv(x), [batch_size, -1, height * width])
        
# #         attention = tf.nn.softmax(tf.matmul(query, key, transpose_b=True) / (self.filters ** 0.5), axis=-1)
# #         out = tf.reshape(tf.matmul(attention, value), [batch_size, height, width, channels])
# #         out = self.gamma * out + x
# #         return self.ln(out)

# # def se_block(x, filters, ratio=16):
# #     se = layers.GlobalAveragePooling2D()(x)
# #     se = layers.Dense(filters // ratio, activation='relu')(se)
# #     se = layers.Dense(filters, activation='sigmoid')(se)
# #     se = layers.Reshape((1, 1, filters))(se)
# #     x = layers.Multiply()([x, se])
# #     return x

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

# # def generator(input_shape=(192, 256, 3)):
# #     inputs = layers.Input(shape=input_shape)
    
# #     x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
# #     x = layers.LeakyReLU(alpha=0.2)(x)
# #     x = se_block(x, 64)  # Adding SE Block

# #     # Adding Residual-in-Residual Dense Blocks with Self-Attention
# #     for _ in range(4):
# #         x = rrdb(x, 64)
# #         x = SelfAttentionLayer(64)(x)

# #     # Multi-Scale Feature Processing with FPN
# #     scale1 = x
# #     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
# #     scale2 = rrdb(scale2, 64)
# #     scale2 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(scale2)
# #     scale2 = layers.LeakyReLU(alpha=0.2)(scale2)
# #     scale3 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
# #     scale3 = layers.AveragePooling2D(pool_size=(2, 2))(scale3)
# #     scale3 = rrdb(scale3, 64)
    
# #     multi_scale = layers.Concatenate()([scale1, scale2, scale3])
# #     multi_scale = fpn_block(multi_scale, 64)
    
# #     # Upsampling to the final resolution
# #     multi_scale = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(multi_scale)
# #     multi_scale = layers.LeakyReLU(alpha=0.2)(multi_scale)
# #     multi_scale = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(multi_scale)
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

# #     x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
# #     x = layers.LeakyReLU(alpha=0.2)(x)

# #     scale1 = res_block(x, 64)
# #     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
# #     scale2 = res_block(scale2, 64)
# #     scale2 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(scale2)
# #     scale2 = layers.LeakyReLU(alpha=0.2)(scale2)
# #     scale3 = layers.AveragePooling2D(pool_size=(4, 4))(x)
# #     scale3 = res_block(scale3, 64)
# #     scale3 = layers.Conv2DTranspose(64, (3, 3), strides=(4, 4), padding='same')(scale3)
# #     scale3 = layers.LeakyReLU(alpha=0.2)(scale3)

# #     multi_scale = layers.Concatenate()([scale1, scale2, scale3])
# #     x = layers.Conv2D(64, (3, 3), padding='same')(multi_scale)
# #     x = layers.LeakyReLU(alpha=0.2)(x)
    
# #     for filters in [64, 128, 256, 512]:
# #         x = layers.Conv2D(filters, (4, 4), strides=(2, 2), padding='same')(x)
# #         x = layers.BatchNormalization()(x)
# #         x = layers.LeakyReLU(negative_slope=0.2)(x)
    
# #     x = layers.Conv2D(1, (4, 4), padding='same')(x)
# #     return Model(inputs, x)


# ## -- Attention  -- ##
# # import tensorflow as tf
# # from tensorflow.keras import layers, Model # type: ignore

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
# #         batch_size, height, width, channels = x.get_shape().as_list()
# #         query = tf.reshape(self.query_conv(x), [batch_size, -1, height * width])
# #         key = tf.reshape(self.key_conv(x), [batch_size, -1, height * width])
# #         value = tf.reshape(self.value_conv(x), [batch_size, -1, height * width])
        
# #         attention = tf.nn.softmax(tf.matmul(query, key, transpose_b=True) / (self.filters ** 0.5), axis=-1)
# #         out = tf.reshape(tf.matmul(attention, value), [batch_size, height, width, channels])
# #         out = self.gamma * out + x
# #         return self.ln(out)

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

# # # Define Residual Dense Block (RDB)
# # def residual_dense_block(x, filters, growth_rate=4, layers_in_block=4):
# #     concat_features = [x]
# #     for _ in range(layers_in_block):
# #         x = layers.Conv2D(growth_rate, (3, 3), padding='same')(x)
# #         x = layers.Activation('relu')(x)
# #         concat_features.append(x)
# #         x = layers.Concatenate()(concat_features)
# #     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
# #     return x

# # def generator(input_shape=(192, 256, 3)):
# #     inputs = layers.Input(shape=input_shape)
    
# #     # Initial convolution
# #     x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
# #     x = layers.LeakyReLU(alpha=0.2)(x)

# #     # Adding Residual Dense Blocks with SE Block and Self-Attention
# #     for _ in range(4):
# #         x = residual_dense_block(x, 64)
# #         x = se_block(x, 64)
# #         x = SelfAttentionLayer(64)(x)
    
# #     # Multi-Scale Feature Processing with FPN
# #     scale1 = x
# #     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
# #     scale2 = residual_dense_block(scale2, 64)
# #     scale2 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(scale2)
# #     scale2 = layers.LeakyReLU(alpha=0.2)(scale2)
# #     scale3 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
# #     scale3 = layers.AveragePooling2D(pool_size=(2, 2))(scale3)
# #     scale3 = residual_dense_block(scale3, 64)
    
# #     multi_scale = layers.Concatenate()([scale1, scale2, scale3])
# #     multi_scale = fpn_block(multi_scale, 64)
    
# #     # Upsampling to the final resolution
# #     multi_scale = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(multi_scale)
# #     multi_scale = layers.LeakyReLU(alpha=0.2)(multi_scale)
# #     multi_scale = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(multi_scale)
# #     multi_scale = layers.LeakyReLU(alpha=0.2)(multi_scale)

# #     outputs = layers.Conv2D(3, (3, 3), padding='same', activation='tanh')(multi_scale)
    
# #     return Model(inputs, outputs)

# # # # Residual Block for Discriminator
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

# #     # Multi-scale processing branches with SE Block and Self-Attention
# #     scale1 = res_block(x, 64)
# #     scale1 = se_block(scale1, 64)
# #     scale1 = SelfAttentionLayer(64)(scale1)
# #     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
# #     scale2 = res_block(scale2, 64)
# #     scale2 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(scale2)
# #     scale2 = layers.LeakyReLU(alpha=0.2)(scale2)
# #     scale3 = layers.AveragePooling2D(pool_size=(4, 4))(x)
# #     scale3 = res_block(scale3, 64)
# #     scale3 = layers.Conv2DTranspose(64, (3, 3), strides=(4, 4), padding='same')(scale3)
# #     scale3 = layers.LeakyReLU(alpha=0.2)(scale3)

# #     multi_scale = layers.Concatenate()([scale1, scale2, scale3])
# #     x = layers.Conv2D(64, (3, 3), padding='same')(multi_scale)
# #     x = layers.LeakyReLU(alpha=0.2)(x)
    
# #     for filters in [64, 128, 256, 512]:
# #         x = layers.Conv2D(filters, (4, 4), strides=(2, 2), padding='same')(x)
# #         x = layers.BatchNormalization()(x)
# #         x = layers.LeakyReLU(negative_slope=0.2)(x)
    
# #     x = layers.Conv2D(1, (4, 4), padding='same')(x)
# #     return Model(inputs, x)
