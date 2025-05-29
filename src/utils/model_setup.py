import tensorflow as tf
from tensorflow.keras.applications import VGG19, EfficientNetB7 # type: ignore
from tensorflow.keras.models import Model   # type: ignore
from tensorflow.keras import layers  # type: ignore
# # def setup_model_and_optimizers(strategy, architecture_module, model_config, train_config):
# #     with strategy.scope():
# #         generator = architecture_module.generator()
# #         discriminator = architecture_module.discriminator()
# #         vgg = VGG19(include_top=False, weights='imagenet', input_shape=tuple(train_config['vgg_input_shape']))
# #         vgg.trainable = False

# #         generator_optimizer = tf.keras.optimizers.Adagrad(learning_rate=train_config['learning_rates']['generator'])
# #         discriminator_optimizer = tf.keras.optimizers.Adagrad(learning_rate=train_config['learning_rates']['discriminator'])

# #     return generator, discriminator, vgg, generator_optimizer, discriminator_optimizer

## --DO NOT DELETE THIS CODE-- ##
# from tensorflow.keras.applications import VGG19 # type: ignore
# from tensorflow.keras.models import Model, load_model # type: ignore
# import os

# def create_flexible_vgg():
#     vgg_base = VGG19(include_top=False, weights='imagenet')
#     vgg_base.trainable = False
#     output_layers = [vgg_base.get_layer(name).output for name in ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']]
#     model = Model(inputs=vgg_base.input, outputs=output_layers)
#     return model

# def setup_model_and_optimizers(strategy, architecture_module, model_config, train_config):

#     initial_lr_gen = train_config['learning_rates']['generator']
#     initial_lr_disc = train_config['learning_rates']['discriminator']

#     lr_schedule_gen = tf.keras.optimizers.schedules.ExponentialDecay(
#         initial_learning_rate=initial_lr_gen,
#         decay_steps=100000,
#         decay_rate=0.96,
#         staircase=True)

#     lr_schedule_disc = tf.keras.optimizers.schedules.ExponentialDecay(
#         initial_learning_rate=initial_lr_disc,
#         decay_steps=100000,
#         decay_rate=0.96,
#         staircase=True)
    
#     # with strategy.scope():
#     #     generator = architecture_module.generator()
#     #     # Load pre-trained generator model
#     #     # generator_path = os.path.join(train_config['model_save_path'], 'best_model.keras')
#     #     # generator = load_model(generator_path, safe_mode=False) # Load the model with safe mode disabled
#     #     discriminator = architecture_module.discriminator()
#     #     # # vgg = VGG19(include_top=False, weights='imagenet', input_shape=tuple(train_config['vgg_input_shape']))
#     #     # vgg = VGG19(include_top=False, weights='imagenet', input_shape=tuple(train_config['vgg_input_shape']))
#     #     # vgg.trainable = False
#     #     flexible_vgg = create_flexible_vgg()
#     #     generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule_gen)
#     #     discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule_disc)

#     # return generator, discriminator, flexible_vgg, generator_optimizer, discriminator_optimizer
    
#     ## Conditional GAN One....
#     with strategy.scope():
#         generator = architecture_module.generator()
#         discriminator = architecture_module.discriminator()
#         efficientnet = EfficientNetB7(include_top=False, weights='imagenet')
#         efficientnet.trainable = False

#         vgg = VGG19(include_top=False, weights='imagenet', input_shape=tuple(train_config['vgg_input_shape']))
#         vgg.trainable = False

#         # Extract features from a specific layer
#         efficientnet = Model(inputs=efficientnet.input, outputs=efficientnet.get_layer('block6e_add').output)
#         # Adjust channels from 384 to 512 to align with the generator
#         efficientnet = Model(inputs=efficientnet.input, outputs=layers.Conv2D(2048, (1, 1), padding='same')(efficientnet.output))

#         generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule_gen)
#         discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule_disc)

#     return generator, discriminator, efficientnet, vgg,  generator_optimizer, discriminator_optimizer

#############################################################################################################################



from tensorflow.keras.applications import VGG19, EfficientNetB7 # type: ignore
from tensorflow.keras.models import Model, load_model   # type: ignore
import tensorflow as tf
from tensorflow.keras import layers # type: ignore
import os


def create_flexible_vgg():
    vgg_base = VGG19(include_top=False, weights='imagenet')
    vgg_base.trainable = False
    output_layers = [vgg_base.get_layer(name).output for name in ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']]
    model = Model(inputs=vgg_base.input, outputs=output_layers)
    return model

def setup_model_and_optimizers(strategy, architecture_module, model_config, train_config):
    initial_lr_gen = train_config['learning_rates']['generator']
    initial_lr_disc = train_config['learning_rates']['discriminator']
    initial_lr_enc = 0.0001  # Added encoder learning rate

    lr_schedule_gen = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr_gen,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)

    lr_schedule_disc = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr_disc,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)

    lr_schedule_enc = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr_enc,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)
     
    with strategy.scope():
        # generator = architecture_module.generator()
        # generator_path = os.path.join(train_config['model_save_path'], 'best_model.keras')
        generator_path ='/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/models/generator_5/best_model.keras'
        generator = load_model(generator_path, safe_mode=False) # Load the model with safe mode disabled
        discriminator = architecture_module.discriminator()
        # encoder = architecture_module.encoder()  # Added encoder initialization

        efficientnet = EfficientNetB7(include_top=False, weights='imagenet')
        efficientnet.trainable = False

        vgg = create_flexible_vgg()

        generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule_gen)
        discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule_disc)
        # encoder_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule_enc)  # Added encoder optimizer

    return generator, discriminator, efficientnet, vgg,  generator_optimizer, discriminator_optimizer
    # return generator, discriminator, encoder, efficientnet, vgg, generator_optimizer, discriminator_optimizer, encoder_optimizer















#############################################################################################################################
# ## Multiple feature extraction layers
#     # with strategy.scope():
#     #     generator = architecture_module.generator()
#     #     discriminator = architecture_module.discriminator()
#     #     vgg = VGG19(include_top=False, weights='imagenet', input_shape=tuple(train_config['vgg_input_shape']))
#     #     vgg.trainable = False

#     #     # Create models to extract features at different scales
#     #     feature_extractor_1 = tf.keras.Model(inputs=vgg.input, outputs=vgg.get_layer('block5_conv3').output)  # (24, 32, 512)
#     #     feature_extractor_2 = tf.keras.Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv4').output)  # (48, 64, 256)
#     #     feature_extractor_3 = tf.keras.Model(inputs=vgg.input, outputs=vgg.get_layer('block1_conv2').output)  # (96, 128, 128)

#     #     generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule_gen)
#     #     discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule_disc)

#     # return generator, discriminator, [feature_extractor_1, feature_extractor_2, feature_extractor_3], generator_optimizer, discriminator_optimizer




'''## For colord images.
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
import tensorflow as tf

def create_flexible_vgg():
    vgg_base = VGG19(include_top=False, weights='imagenet')
    vgg_base.trainable = False
    output_layers = [vgg_base.get_layer(name).output for name in ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']]
    model = Model(inputs=vgg_base.input, outputs=output_layers)
    return model

def setup_model_and_optimizers(strategy, architecture_module, model_config, train_config):

    initial_lr_gen = train_config['learning_rates']['generator']
    initial_lr_disc = train_config['learning_rates']['discriminator']

    lr_schedule_gen = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr_gen,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)

    lr_schedule_disc = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr_disc,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)
    
    with strategy.scope():
        generator = architecture_module.generator(input_shape=(None, None, 3))
        discriminator = architecture_module.discriminator(input_shape=(None, None, 3))
        flexible_vgg = create_flexible_vgg()
        generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule_gen)
        discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule_disc)

    return generator, discriminator, flexible_vgg, generator_optimizer, discriminator_optimizer

def update_vgg_input_shape(flexible_vgg, input_shape):
    input_shape = tuple(input_shape)  # Convert input_shape list to tuple
    print(f"Updating VGG input shape to: {input_shape}")
    
    vgg_base = VGG19(include_top=False, weights='imagenet', input_shape=input_shape)
    vgg_base.trainable = False
    output_layers = [vgg_base.get_layer(name).output for name in ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']]
    flexible_vgg = Model(inputs=vgg_base.input, outputs=output_layers)
    
    print(f"Updated VGG input shape: {flexible_vgg.input.shape}")
    return flexible_vgg

'''
# from tensorflow.keras.applications import VGG19
# from tensorflow.keras.models import Model
# import tensorflow as tf

# def create_flexible_vgg(input_shape):
#     vgg_base = VGG19(include_top=False, weights='imagenet', input_shape=input_shape)
#     vgg_base.trainable = False
#     output_layers = [vgg_base.get_layer(name).output for name in ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']]
#     model = Model(inputs=vgg_base.input, outputs=output_layers)
#     return model

# def setup_model_and_optimizers(strategy, architecture_module, model_config, train_config):

#     initial_lr_gen = train_config['learning_rates']['generator']
#     initial_lr_disc = train_config['learning_rates']['discriminator']

#     lr_schedule_gen = tf.keras.optimizers.schedules.ExponentialDecay(
#         initial_learning_rate=initial_lr_gen,
#         decay_steps=100000,
#         decay_rate=0.96,
#         staircase=True)

#     lr_schedule_disc = tf.keras.optimizers.schedules.ExponentialDecay(
#         initial_learning_rate=initial_lr_disc,
#         decay_steps=100000,
#         decay_rate=0.96,
#         staircase=True)
    
#     with strategy.scope():
#         generator = architecture_module.generator(input_shape=(None, None, 3))
#         discriminator = architecture_module.discriminator(input_shape=(None, None, 3))
#         flexible_vgg = create_flexible_vgg((train_config['vgg_input_shape'][0], train_config['vgg_input_shape'][1], 3))
#         generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule_gen)
#         discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule_disc)

#     return generator, discriminator, flexible_vgg, generator_optimizer, discriminator_optimizer

# def update_vgg_input_shape(flexible_vgg, input_shape):
#     input_shape = tuple(input_shape)  # Convert input_shape list to tuple
#     # print(f"Updating VGG input shape to: {input_shape}")
    
#     vgg_base = VGG19(include_top=False, weights='imagenet', input_shape=input_shape)
#     vgg_base.trainable = False
#     output_layers = [vgg_base.get_layer(name).output for name in ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']]
#     flexible_vgg = Model(inputs=vgg_base.input, outputs=output_layers)
    
#     # print(f"Updated VGG input shape: {flexible_vgg.input.shape}")
#     return flexible_vgg
