import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.applications import VGG19 # type: ignore
from torch.utils.data import DataLoader
from loss_functions.loss_functions import adversarial_loss, total_loss

# def compute_gradient_penalty(discriminator, real_samples, fake_samples):
#     alpha = tf.random.uniform(shape=[real_samples.shape[0], 1, 1, 1], minval=0., maxval=1.)
#     interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
#     with tf.GradientTape() as tape:
#         tape.watch(interpolates)
#         pred_interpolates = discriminator(interpolates)
#     gradients = tape.gradient(pred_interpolates, [interpolates])[0]
#     norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
#     gradient_penalty = tf.reduce_mean((norm - 1.0) ** 2)
#     return gradient_penalty

mse_loss = tf.keras.losses.MeanSquaredError()

def pretrain_step(LR_batch, HR_batch, generator, generator_optimizer):
    with tf.GradientTape() as tape:
        generated_images = generator(LR_batch, training=True)
        loss = mse_loss(HR_batch, generated_images)
    
    gradients = tape.gradient(loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
    
    return loss

def distributed_pretrain_step(LR_batch, HR_batch, generator, generator_optimizer, strategy):
    def step_fn(LR_batch, HR_batch):
        return pretrain_step(LR_batch, HR_batch, generator, generator_optimizer)
    
    per_replica_losses = strategy.run(step_fn, args=(LR_batch, HR_batch))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

''' 
# DO NOT CHANGE THIS...
@tf.function 
def train_step(LR_batch, HR_batch, generator, discriminator, vgg, generator_optimizer, discriminator_optimizer, **kwargs):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(LR_batch, training=True)
        generated_images = (generated_images + 1) / 2.0  # Scale from [-1, 1] to [0, 1]
        real_output = discriminator(HR_batch, training=True)
        fake_output = discriminator(generated_images, training=True)

        total_gen_loss, individual_losses = total_loss(vgg, HR_batch, generated_images, real_output, fake_output, **kwargs)
        disc_loss = adversarial_loss(tf.ones_like(real_output, dtype=tf.float32), real_output) + \
                    adversarial_loss(tf.zeros_like(fake_output, dtype=tf.float32), fake_output)
    
    gradients_of_generator = gen_tape.gradient(total_gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return total_gen_loss, disc_loss, generated_images, individual_losses

@tf.function 
def distributed_train_step(LR_batch, HR_batch, generator, discriminator, vgg, generator_optimizer, discriminator_optimizer, strategy, **kwargs):
    per_replica_results = strategy.run(
        train_step,
        args=(LR_batch, HR_batch, generator, discriminator, vgg, generator_optimizer, discriminator_optimizer),
        kwargs=kwargs
    )

    per_replica_gen_losses, per_replica_disc_losses, per_replica_generated_images, per_replica_individual_losses = per_replica_results

    gen_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_gen_losses, axis=None)
    disc_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_disc_losses, axis=None)
    individual_losses = {key: strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_individual_losses[key], axis=None) for key in per_replica_individual_losses}

    per_replica_generated_images = strategy.experimental_local_results(per_replica_generated_images)[0]

    return gen_loss, disc_loss, per_replica_generated_images, individual_losses


@tf.function
def single_validation_step(LR_batch, HR_batch, generator, discriminator, vgg, **kwargs):
    generated_images = generator(LR_batch, training=False)
    generated_images = (generated_images + 1) / 2.0  # Scale from [-1, 1] to [0, 1]

    real_output = discriminator(HR_batch, training=False)
    fake_output = discriminator(generated_images, training=False)

    total_val_loss, val_individual_losses = total_loss(vgg, HR_batch, generated_images, real_output, fake_output, **kwargs)

    return total_val_loss, val_individual_losses

@tf.function 
def distributed_validation_step(LR_batch, HR_batch, generator, discriminator, vgg, strategy, **kwargs):
    per_replica_results = strategy.run(
        single_validation_step,
        args=(LR_batch, HR_batch, generator, discriminator, vgg),
        kwargs=kwargs
    )

    per_replica_val_losses, per_replica_individual_losses = per_replica_results

    val_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_val_losses, axis=None)
    individual_losses = {key: strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_individual_losses[key], axis=None) for key in per_replica_individual_losses}

    return val_loss, individual_losses
'''
'''##### FOR COLOR DATASET ####

@tf.function 
def train_step(LR_batch, HR_batch, generator, discriminator, vgg, generator_optimizer, discriminator_optimizer, **kwargs):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(LR_batch, training=True)
        generated_images = (generated_images + 1) / 2.0  # Scale from [-1, 1] to [0, 1]
        real_output = discriminator(HR_batch, training=True)
        fake_output = discriminator(generated_images, training=True)

        total_gen_loss, individual_losses = total_loss(vgg, HR_batch, generated_images, real_output, fake_output, **kwargs)
        disc_loss = adversarial_loss(tf.ones_like(real_output, dtype=tf.float32), real_output) + \
                    adversarial_loss(tf.zeros_like(fake_output, dtype=tf.float32), fake_output)
    
    gradients_of_generator = gen_tape.gradient(total_gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return total_gen_loss, disc_loss, generated_images, individual_losses

@tf.function 
def distributed_train_step(LR_batch, HR_batch, generator, discriminator, vgg, generator_optimizer, discriminator_optimizer, strategy, **kwargs):
    per_replica_results = strategy.run(
        train_step,
        args=(LR_batch, HR_batch, generator, discriminator, vgg, generator_optimizer, discriminator_optimizer),
        kwargs=kwargs
    )

    per_replica_gen_losses, per_replica_disc_losses, per_replica_generated_images, per_replica_individual_losses = per_replica_results

    gen_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_gen_losses, axis=None)
    disc_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_disc_losses, axis=None)
    individual_losses = {key: strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_individual_losses[key], axis=None) for key in per_replica_individual_losses}

    per_replica_generated_images = strategy.experimental_local_results(per_replica_generated_images)[0]

    return gen_loss, disc_loss, per_replica_generated_images, individual_losses


@tf.function
def single_validation_step(LR_batch, HR_batch, generator, discriminator, vgg, **kwargs):
    generated_images = generator(LR_batch, training=False)
    generated_images = (generated_images + 1) / 2.0  # Scale from [-1, 1] to [0, 1]

    real_output = discriminator(HR_batch, training=False)
    fake_output = discriminator(generated_images, training=False)

    total_val_loss, val_individual_losses = total_loss(vgg, HR_batch, generated_images, real_output, fake_output, **kwargs)

    return total_val_loss, val_individual_losses

@tf.function 
def distributed_validation_step(LR_batch, HR_batch, generator, discriminator, vgg, strategy, **kwargs):
    per_replica_results = strategy.run(
        single_validation_step,
        args=(LR_batch, HR_batch, generator, discriminator, vgg),
        kwargs=kwargs
    )

    per_replica_val_losses, per_replica_individual_losses = per_replica_results

    val_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_val_losses, axis=None)
    individual_losses = {key: strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_individual_losses[key], axis=None) for key in per_replica_individual_losses}

    return val_loss, individual_losses
'''

"""
## for architecture 5.0

import tensorflow as tf

def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    alpha = tf.random.uniform(shape=[real_samples.shape[0], 1, 1, 1], minval=0., maxval=1.)
    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
    with tf.GradientTape() as tape:
        tape.watch(interpolates)
        pred_interpolates = discriminator(interpolates)
    gradients = tape.gradient(pred_interpolates, [interpolates])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean((norm - 1.0) ** 2)
    return gradient_penalty

@tf.function 
def train_step(LR_batch, HR_batch, generator, discriminator, vgg, feature_extractor, generator_optimizer, discriminator_optimizer, **kwargs):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Ensure correct shape for feature extraction
        HR_batch_resized = tf.image.resize(HR_batch, (1024, 768))  # Resize to match VGG19's expected input size
        extracted_features = feature_extractor(HR_batch_resized, training=False)
        generated_images = generator([LR_batch, extracted_features[-1]], training=True)  # Use the last block's features
        generated_images = (generated_images + 1) / 2.0  # Scale from [-1, 1] to [0, 1]
        real_output = discriminator(HR_batch, training=True)
        fake_output = discriminator(generated_images, training=True)

        total_gen_loss, individual_losses = total_loss(vgg, HR_batch, generated_images, real_output, fake_output, **kwargs)
        disc_loss = adversarial_loss(tf.ones_like(real_output, dtype=tf.float32), real_output) + \
                    adversarial_loss(tf.zeros_like(fake_output, dtype=tf.float32), fake_output)
    
    gradients_of_generator = gen_tape.gradient(total_gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return total_gen_loss, disc_loss, generated_images, individual_losses

@tf.function 
def distributed_train_step(LR_batch, HR_batch, generator, discriminator, vgg, feature_extractor, generator_optimizer, discriminator_optimizer, strategy, **kwargs):
    per_replica_results = strategy.run(
        train_step,
        args=(LR_batch, HR_batch, generator, discriminator, vgg, feature_extractor, generator_optimizer, discriminator_optimizer),
        kwargs=kwargs
    )

    per_replica_gen_losses, per_replica_disc_losses, per_replica_generated_images, per_replica_individual_losses = per_replica_results

    gen_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_gen_losses, axis=None)
    disc_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_disc_losses, axis=None)
    individual_losses = {key: strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_individual_losses[key], axis=None) for key in per_replica_individual_losses}

    per_replica_generated_images = strategy.experimental_local_results(per_replica_generated_images)[0]

    return gen_loss, disc_loss, per_replica_generated_images, individual_losses

@tf.function
def single_validation_step(LR_batch, HR_batch, generator, discriminator, vgg, feature_extractor, **kwargs):
    # Ensure correct shape for feature extraction
    HR_batch_resized = tf.image.resize(HR_batch, (1024, 768))  # Resize to match VGG19's expected input size
    extracted_features = feature_extractor(HR_batch_resized, training=False)
    generated_images = generator([LR_batch, extracted_features[-1]], training=False)  # Use the last block's features
    generated_images = (generated_images + 1) / 2.0  # Scale from [-1, 1] to [0, 1]

    real_output = discriminator(HR_batch, training=False)
    fake_output = discriminator(generated_images, training=False)

    total_val_loss, val_individual_losses = total_loss(vgg, HR_batch, generated_images, real_output, fake_output, **kwargs)

    return total_val_loss, val_individual_losses

@tf.function 
def distributed_validation_step(LR_batch, HR_batch, generator, discriminator, vgg, feature_extractor, strategy, **kwargs):
    per_replica_results = strategy.run(
        single_validation_step,
        args=(LR_batch, HR_batch, generator, discriminator, vgg, feature_extractor),
        kwargs=kwargs
    )

    per_replica_val_losses, per_replica_individual_losses = per_replica_results

    val_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_val_losses, axis=None)
    individual_losses = {key: strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_individual_losses[key], axis=None) for key in per_replica_individual_losses}

    return val_loss, individual_losses



'''
@tf.function 
def train_step(LR_batch, HR_batch, prev_output_batch, generator, discriminator, vgg, generator_optimizer, discriminator_optimizer, **kwargs):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator([LR_batch, prev_output_batch], training=True)
        generated_images = (generated_images + 1) / 2.0  # Scale from [-1, 1] to [0, 1]
        real_output = discriminator(HR_batch, training=True)
        fake_output = discriminator(generated_images, training=True)

        total_gen_loss, individual_losses = total_loss(vgg, HR_batch, generated_images, real_output, fake_output, **kwargs)
        disc_loss = adversarial_loss(tf.ones_like(real_output, dtype=tf.float32), real_output) + \
                    adversarial_loss(tf.zeros_like(fake_output, dtype=tf.float32), fake_output)
    
    gradients_of_generator = gen_tape.gradient(total_gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return total_gen_loss, disc_loss, generated_images, individual_losses

@tf.function 
def distributed_train_step(LR_batch, HR_batch, prev_output_batch, generator, discriminator, vgg, generator_optimizer, discriminator_optimizer, strategy, **kwargs):
    per_replica_results = strategy.run(
        train_step,
        args=(LR_batch, HR_batch, prev_output_batch, generator, discriminator, vgg, generator_optimizer, discriminator_optimizer),
        kwargs=kwargs
    )

    per_replica_gen_losses, per_replica_disc_losses, per_replica_generated_images, per_replica_individual_losses = per_replica_results

    gen_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_gen_losses, axis=None)
    disc_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_disc_losses, axis=None)
    individual_losses = {key: strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_individual_losses[key], axis=None) for key in per_replica_individual_losses}

    per_replica_generated_images = strategy.experimental_local_results(per_replica_generated_images)[0]

    return gen_loss, disc_loss, per_replica_generated_images, individual_losses

'''
"""
#############################################################################################################################
## --DO NOT CHANGE THIS... ##
## for first
# ## for architecture 8.0 CGAN one.
### Working best ####
# '''
# def compute_gradient_penalty(discriminator, real_samples, fake_samples):
#     alpha = tf.random.uniform(shape=[real_samples.shape[0], 1, 1, 1], minval=0., maxval=1.)
#     interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
#     with tf.GradientTape() as tape:
#         tape.watch(interpolates)
#         pred_interpolates = discriminator(interpolates)
#     gradients = tape.gradient(pred_interpolates, [interpolates])[0]
#     norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
#     gradient_penalty = tf.reduce_mean((norm - 1.0) ** 2)
#     return gradient_penalty

# @tf.function 
# def train_step(LR_batch, HR_batch, generator, discriminator, efficientnet,vgg,  generator_optimizer, discriminator_optimizer, **kwargs):
#     # Extract features from HR images using VGG
#     hr_features = efficientnet(HR_batch, training=False)

#     with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
#         generated_images = generator([LR_batch, hr_features], training=True)
#         generated_images = (generated_images + 1) / 2.0  # Scale from [-1, 1] to [0, 1]
#         real_output = discriminator(HR_batch, training=True)
#         fake_output = discriminator(generated_images, training=True)

#         total_gen_loss, individual_losses = total_loss(vgg, HR_batch, generated_images, real_output, fake_output, **kwargs)
#         disc_loss = adversarial_loss(tf.ones_like(real_output, dtype=tf.float32), real_output) + \
#                     adversarial_loss(tf.zeros_like(fake_output, dtype=tf.float32), fake_output)
    
#     gradients_of_generator = gen_tape.gradient(total_gen_loss, generator.trainable_variables)
#     generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

#     gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
#     discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

#     return total_gen_loss, disc_loss, generated_images, individual_losses

# @tf.function 
# def distributed_train_step(LR_batch, HR_batch, generator, discriminator, efficientnet, vgg,  generator_optimizer, discriminator_optimizer, strategy, **kwargs):
#     per_replica_results = strategy.run(
#         train_step,
#         args=(LR_batch, HR_batch, generator, discriminator, efficientnet, vgg,  generator_optimizer, discriminator_optimizer),
#         kwargs=kwargs
#     )

#     per_replica_gen_losses, per_replica_disc_losses, per_replica_generated_images, per_replica_individual_losses = per_replica_results

#     gen_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_gen_losses, axis=None)
#     disc_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_disc_losses, axis=None)
#     individual_losses = {key: strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_individual_losses[key], axis=None) for key in per_replica_individual_losses}

#     per_replica_generated_images = strategy.experimental_local_results(per_replica_generated_images)[0]

#     return gen_loss, disc_loss, per_replica_generated_images, individual_losses


# @tf.function
# def single_validation_step(LR_batch, HR_batch, generator, discriminator, efficientnet, vgg,  **kwargs):
#     # Extract features from HR images using VGG
#     hr_features = efficientnet(HR_batch, training=False)

#     generated_images = generator([LR_batch, hr_features], training=False)
#     generated_images = (generated_images + 1) / 2.0  # Scale from [-1, 1] to [0, 1]

#     real_output = discriminator(HR_batch, training=False)
#     fake_output = discriminator(generated_images, training=False)

#     total_val_loss, val_individual_losses = total_loss(vgg, HR_batch, generated_images, real_output, fake_output, **kwargs)

#     return total_val_loss, val_individual_losses

# @tf.function 
# def distributed_validation_step(LR_batch, HR_batch, generator, discriminator, efficientnet, vgg, strategy, **kwargs):
#     per_replica_results = strategy.run(
#         single_validation_step,
#         args=(LR_batch, HR_batch, generator, discriminator, efficientnet, vgg),
#         kwargs=kwargs
#     )

#     per_replica_val_losses, per_replica_individual_losses = per_replica_results

#     val_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_val_losses, axis=None)
#     individual_losses = {key: strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_individual_losses[key], axis=None) for key in per_replica_individual_losses}

#     return val_loss, individual_losses


#############################################################################################################################



import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError # type: ignore
from loss_functions.loss_functions import adversarial_loss, total_loss
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.applications import VGG19 # type: ignore
from torch.utils.data import DataLoader
from loss_functions.loss_functions import adversarial_loss, total_loss

# def compute_gradient_penalty(discriminator, real_samples, fake_samples):
#     alpha = tf.random.uniform(shape=[real_samples.shape[0], 1, 1, 1], minval=0., maxval=1.)
#     interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
#     with tf.GradientTape() as tape:
#         tape.watch(interpolates)
#         pred_interpolates = discriminator(interpolates)
#     gradients = tape.gradient(pred_interpolates, [interpolates])[0]
#     norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
#     gradient_penalty = tf.reduce_mean((norm - 1.0) ** 2)
#     return gradient_penalty

mse_loss = tf.keras.losses.MeanSquaredError()
mse_loss = MeanSquaredError()

def pretrain_step(LR_batch, HR_batch, generator, generator_optimizer):
    with tf.GradientTape() as tape:
        generated_images = generator(LR_batch, training=True)
        loss = mse_loss(HR_batch, generated_images)
    
    gradients = tape.gradient(loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
    
    return loss

def distributed_pretrain_step(LR_batch, HR_batch, generator, generator_optimizer, strategy):
    def step_fn(LR_batch, HR_batch):
        return pretrain_step(LR_batch, HR_batch, generator, generator_optimizer)
    
    per_replica_losses = strategy.run(step_fn, args=(LR_batch, HR_batch))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

# Pre-training step for the encoder
def pretrain_encoder_step(LR_batch, HR_features_batch, encoder, encoder_optimizer):
    with tf.GradientTape() as tape:
        predicted_features = encoder(LR_batch, training=True)
        feature_loss = mse_loss(HR_features_batch, predicted_features)  # Feature consistency loss
    
    gradients = tape.gradient(feature_loss, encoder.trainable_variables)
    encoder_optimizer.apply_gradients(zip(gradients, encoder.trainable_variables))
    return feature_loss

@tf.function
def distributed_pretrain_encoder_step(LR_batch, HR_features_batch, encoder, encoder_optimizer, strategy):
    def step_fn(LR_batch, HR_features_batch):
        return pretrain_encoder_step(LR_batch, HR_features_batch, encoder, encoder_optimizer)
    
    per_replica_loss = strategy.run(step_fn, args=(LR_batch, HR_features_batch))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)


def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    alpha = tf.random.uniform(shape=[real_samples.shape[0], 1, 1, 1], minval=0., maxval=1.)
    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
    with tf.GradientTape() as tape:
        tape.watch(interpolates)
        pred_interpolates = discriminator(interpolates)
    gradients = tape.gradient(pred_interpolates, [interpolates])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean((norm - 1.0) ** 2)
    return gradient_penalty

@tf.function 
def train_step(LR_batch, HR_batch, generator, discriminator, efficientnet, vgg, generator_optimizer, discriminator_optimizer, **kwargs):
    hr_features = efficientnet(HR_batch, training=False)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator([LR_batch, hr_features], training=True)
        generated_images = (generated_images + 1) / 2.0
        real_output = discriminator(HR_batch, training=True)
        fake_output = discriminator(generated_images, training=True)

        total_gen_loss, individual_losses = total_loss(vgg, HR_batch, generated_images, real_output, fake_output, **kwargs)
        disc_loss = adversarial_loss(tf.ones_like(real_output, dtype=tf.float32), real_output) + \
                    adversarial_loss(tf.zeros_like(fake_output, dtype=tf.float32), fake_output)
    
    gradients_of_generator = gen_tape.gradient(total_gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return total_gen_loss, disc_loss, generated_images, individual_losses

@tf.function 
def distributed_train_step(LR_batch, HR_batch, generator, discriminator, efficientnet, vgg, generator_optimizer, discriminator_optimizer, strategy, **kwargs):
    per_replica_results = strategy.run(
        train_step,
        args=(LR_batch, HR_batch, generator, discriminator, efficientnet, vgg, generator_optimizer, discriminator_optimizer),
        kwargs=kwargs
    )

    per_replica_gen_losses, per_replica_disc_losses, per_replica_generated_images, per_replica_individual_losses = per_replica_results

    gen_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_gen_losses, axis=None)
    disc_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_disc_losses, axis=None)
    individual_losses = {key: strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_individual_losses[key], axis=None) for key in per_replica_individual_losses}

    per_replica_generated_images = strategy.experimental_local_results(per_replica_generated_images)[0]

    return gen_loss, disc_loss, per_replica_generated_images, individual_losses

@tf.function
def single_validation_step(LR_batch, HR_batch, generator, discriminator, efficientnet, vgg, **kwargs):
    hr_features = efficientnet(HR_batch, training=False)

    generated_images = generator([LR_batch, hr_features], training=False)
    generated_images = (generated_images + 1) / 2.0

    real_output = discriminator(HR_batch, training=False)
    fake_output = discriminator(generated_images, training=False)

    total_val_loss, val_individual_losses = total_loss(vgg, HR_batch, generated_images, real_output, fake_output, **kwargs)

    return total_val_loss, val_individual_losses

@tf.function 
def distributed_validation_step(LR_batch, HR_batch, generator, discriminator, efficientnet, vgg, strategy, **kwargs):
    per_replica_results = strategy.run(
        single_validation_step,
        args=(LR_batch, HR_batch, generator, discriminator, efficientnet, vgg),
        kwargs=kwargs
    )

    per_replica_val_losses, per_replica_individual_losses = per_replica_results

    val_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_val_losses, axis=None)
    individual_losses = {key: strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_individual_losses[key], axis=None) for key in per_replica_individual_losses}

    return val_loss, individual_losses






















#############################################################################################################################

# '''

# # ## For second
# import tensorflow as tf

# def compute_gradient_penalty(discriminator, real_samples, fake_samples):
#     alpha = tf.random.uniform(shape=[real_samples.shape[0], 1, 1, 1], minval=0., maxval=1.)
#     interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
#     with tf.GradientTape() as tape:
#         tape.watch(interpolates)
#         pred_interpolates = discriminator(interpolates)
#     gradients = tape.gradient(pred_interpolates, [interpolates])[0]
#     norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
#     gradient_penalty = tf.reduce_mean((norm - 1.0) ** 2)
#     return gradient_penalty

# @tf.function 
# def train_step(LR_batch, HR_batch, generator, discriminator, feature_extractors, generator_optimizer, discriminator_optimizer, **kwargs):
#     # Extract features from HR images using feature extractors at different scales
#     hr_features_1 = feature_extractors[0](HR_batch, training=False)
#     hr_features_2 = feature_extractors[1](HR_batch, training=False)
#     hr_features_3 = feature_extractors[2](HR_batch, training=False)

#     with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
#         generated_images = generator([LR_batch, hr_features_1, hr_features_2, hr_features_3], training=True)
#         generated_images = (generated_images + 1) / 2.0  # Scale from [-1, 1] to [0, 1]
#         real_output = discriminator(HR_batch, training=True)
#         fake_output = discriminator(generated_images, training=True)

#         total_gen_loss, individual_losses = total_loss(feature_extractors, HR_batch, generated_images, real_output, fake_output, **kwargs)
#         disc_loss = adversarial_loss(tf.ones_like(real_output, dtype=tf.float32), real_output) + \
#                     adversarial_loss(tf.zeros_like(fake_output, dtype=tf.float32), fake_output)
    
#     gradients_of_generator = gen_tape.gradient(total_gen_loss, generator.trainable_variables)
#     generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

#     gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
#     discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

#     return total_gen_loss, disc_loss, generated_images, individual_losses

# @tf.function 
# def distributed_train_step(LR_batch, HR_batch, generator, discriminator, feature_extractors, generator_optimizer, discriminator_optimizer, strategy, **kwargs):
#     per_replica_results = strategy.run(
#         train_step,
#         args=(LR_batch, HR_batch, generator, discriminator, feature_extractors, generator_optimizer, discriminator_optimizer),
#         kwargs=kwargs
#     )

#     per_replica_gen_losses, per_replica_disc_losses, per_replica_generated_images, per_replica_individual_losses = per_replica_results

#     gen_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_gen_losses, axis=None)
#     disc_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_disc_losses, axis=None)
#     individual_losses = {key: strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_individual_losses[key], axis=None) for key in per_replica_individual_losses}

#     per_replica_generated_images = strategy.experimental_local_results(per_replica_generated_images)[0]

#     return gen_loss, disc_loss, per_replica_generated_images, individual_losses

# @tf.function
# def single_validation_step(LR_batch, HR_batch, generator, discriminator, feature_extractors, **kwargs):
#     # Extract features from HR images using feature extractors at different scales
#     hr_features_1 = feature_extractors[0](HR_batch, training=False)
#     hr_features_2 = feature_extractors[1](HR_batch, training=False)
#     hr_features_3 = feature_extractors[2](HR_batch, training=False)

#     generated_images = generator([LR_batch, hr_features_1, hr_features_2, hr_features_3], training=False)
#     generated_images = (generated_images + 1) / 2.0  # Scale from [-1, 1] to [0, 1]

#     real_output = discriminator(HR_batch, training=False)
#     fake_output = discriminator(generated_images, training=False)

#     total_val_loss, val_individual_losses = total_loss(feature_extractors, HR_batch, generated_images, real_output, fake_output, **kwargs)

#     return total_val_loss, val_individual_losses

# @tf.function 
# def distributed_validation_step(LR_batch, HR_batch, generator, discriminator, feature_extractors, strategy, **kwargs):
#     per_replica_results = strategy.run(
#         single_validation_step,
#         args=(LR_batch, HR_batch, generator, discriminator, feature_extractors),
#         kwargs=kwargs
#     )

#     per_replica_val_losses, per_replica_individual_losses = per_replica_results

#     val_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_val_losses, axis=None)
#     individual_losses = {key: strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_individual_losses[key], axis=None) for key in per_replica_individual_losses}

#     return val_loss, individual_losses

#######################################################


# import tensorflow as tf

# def compute_gradient_penalty(discriminator, real_samples, fake_samples):
#     alpha = tf.random.uniform(shape=[real_samples.shape[0], 1, 1, 1], minval=0., maxval=1.)
#     interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
#     with tf.GradientTape() as tape:
#         tape.watch(interpolates)
#         pred_interpolates = discriminator(interpolates)
#     gradients = tape.gradient(pred_interpolates, [interpolates])[0]
#     norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
#     gradient_penalty = tf.reduce_mean((norm - 1.0) ** 2)
#     return gradient_penalty

# @tf.function 
# def train_step(LR_batch, HR_batch, generator, discriminator, vgg, generator_optimizer, discriminator_optimizer, **kwargs):
#     # Extract features from HR images using VGG at different layers
#     hr_features_1 = vgg.get_layer('block1_conv2').output(HR_batch, training=False)
#     hr_features_2 = vgg.get_layer('block2_conv2').output(HR_batch, training=False)
#     hr_features_3 = vgg.get_layer('block3_conv4').output(HR_batch, training=False)

#     with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
#         generated_images = generator([LR_batch, hr_features_1, hr_features_2, hr_features_3], training=True)
#         generated_images = (generated_images + 1) / 2.0  # Scale from [-1, 1] to [0, 1]
#         real_output = discriminator(HR_batch, training=True)
#         fake_output = discriminator(generated_images, training=True)

#         total_gen_loss, individual_losses = total_loss(vgg, HR_batch, generated_images, real_output, fake_output, **kwargs)
#         disc_loss = adversarial_loss(tf.ones_like(real_output, dtype=tf.float32), real_output) + \
#                     adversarial_loss(tf.zeros_like(fake_output, dtype=tf.float32), fake_output)
    
#     gradients_of_generator = gen_tape.gradient(total_gen_loss, generator.trainable_variables)
#     generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

#     gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
#     discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

#     return total_gen_loss, disc_loss, generated_images, individual_losses

# @tf.function 
# def distributed_train_step(LR_batch, HR_batch, generator, discriminator, vgg, generator_optimizer, discriminator_optimizer, strategy, **kwargs):
#     per_replica_results = strategy.run(
#         train_step,
#         args=(LR_batch, HR_batch, generator, discriminator, vgg, generator_optimizer, discriminator_optimizer),
#         kwargs=kwargs
#     )

#     per_replica_gen_losses, per_replica_disc_losses, per_replica_generated_images, per_replica_individual_losses = per_replica_results

#     gen_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_gen_losses, axis=None)
#     disc_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_disc_losses, axis=None)
#     individual_losses = {key: strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_individual_losses[key], axis=None) for key in per_replica_individual_losses}

#     per_replica_generated_images = strategy.experimental_local_results(per_replica_generated_images)[0]

#     return gen_loss, disc_loss, per_replica_generated_images, individual_losses

# @tf.function
# def single_validation_step(LR_batch, HR_batch, generator, discriminator, vgg, **kwargs):
#     # Extract features from HR images using VGG at different layers
#     hr_features_1 = vgg.get_layer('block1_conv2').output(HR_batch, training=False)
#     hr_features_2 = vgg.get_layer('block2_conv2').output(HR_batch, training=False)
#     hr_features_3 = vgg.get_layer('block3_conv4').output(HR_batch, training=False)

#     generated_images = generator([LR_batch, hr_features_1, hr_features_2, hr_features_3], training=False)
#     generated_images = (generated_images + 1) / 2.0  # Scale from [-1, 1] to [0, 1]

#     real_output = discriminator(HR_batch, training=False)
#     fake_output = discriminator(generated_images, training=False)

#     total_val_loss, val_individual_losses = total_loss(vgg, HR_batch, generated_images, real_output, fake_output, **kwargs)

#     return total_val_loss, val_individual_losses

# @tf.function 
# def distributed_validation_step(LR_batch, HR_batch, generator, discriminator, vgg, strategy, **kwargs):
#     per_replica_results = strategy.run(
#         single_validation_step,
#         args=(LR_batch, HR_batch, generator, discriminator, vgg),
#         kwargs=kwargs
#     )

#     per_replica_val_losses, per_replica_individual_losses = per_replica_results

#     val_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_val_losses, axis=None)
#     individual_losses = {key: strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_individual_losses[key], axis=None) for key in per_replica_individual_losses}

#     return val_loss, individual_losses
