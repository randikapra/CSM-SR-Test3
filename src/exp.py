# # # # # import pandas as pd # type: ignore
# # # # # import matplotlib.pyplot as plt
# # # # # from utils.config_loader import load_config, load_loss_weights
# # # # # train_config = load_config('/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/code/config/train.json')

# # # # # # Assuming you have saved your losses to a CSV file
# # # # # log_path = train_config['log_path']
# # # # # losses_df = pd.read_csv(f'{log_path}/losses.csv')

# # # # # # Plot the losses
# # # # # plt.figure(figsize=(12, 6))
# # # # # plt.plot(losses_df['epoch'], losses_df['train_total_loss'], label='Train Total Loss')
# # # # # plt.plot(losses_df['epoch'], losses_df['val_total_loss'], label='Validation Total Loss')
# # # # # plt.xlabel('Epoch')
# # # # # plt.ylabel('Loss')
# # # # # plt.legend()
# # # # # plt.title('Training and Validation Losses')

# # # # # # Save the plot to a file
# # # # # plot_save_path = f'{log_path}/losses_plot.png'
# # # # # plt.savefig(plot_save_path)
# # # # # plt.close()

# # # # # print(f'Loss plot saved to {plot_save_path}')


# # # # import tensorflow as tf
# # # # from tensorflow.keras.models import Sequential
# # # # from tensorflow.keras.layers import Dense
# # # # from tensorflow.keras.callbacks import Callback, TensorBoard, ModelCheckpoint

# # # # class CustomReduceLROnPlateau(Callback):
# # # #     def __init__(self, monitor='loss', factor=0.5, patience=1, min_lr=1e-5, **kwargs):
# # # #         super(CustomReduceLROnPlateau, self).__init__(**kwargs)
# # # #         self.monitor = monitor
# # # #         self.factor = factor
# # # #         self.patience = patience
# # # #         self.min_lr = min_lr
# # # #         self.best = float('inf')
# # # #         self.wait = 0
# # # #         self.lr_epsilon = tf.keras.backend.epsilon()

# # # #     def on_epoch_end(self, epoch, logs=None):
# # # #         logs = logs or {}
# # # #         current = logs.get(self.monitor)
        
# # # #         print(f"Epoch {epoch + 1}: Current {self.monitor}: {current}, Best: {self.best}, Wait: {self.wait}")

# # # #         if current is None:
# # # #             print("Current value is None")
# # # #             return

# # # #         if current < self.best:
# # # #             self.best = current
# # # #             self.wait = 0
# # # #         else:
# # # #             self.wait += 1
# # # #             if self.wait >= self.patience:
# # # #                 old_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
# # # #                 if old_lr > self.min_lr + self.lr_epsilon:
# # # #                     new_lr = max(old_lr * self.factor, self.min_lr)
# # # #                     self.model.optimizer.learning_rate.assign(new_lr)
# # # #                     print(f"\nEpoch {epoch + 1}: ReduceLROnPlateau reducing learning rate to {new_lr}.")
# # # #                     print(f"Current Learning Rate: {self.model.optimizer.learning_rate.numpy()}")
# # # #                 self.wait = 0

# # # # # Simple model for testing
# # # # model = Sequential([Dense(10, activation='relu', input_shape=(20,)), Dense(1)])

# # # # model.compile(optimizer='adam', loss='mean_squared_error')

# # # # # Simulate training data
# # # # import numpy as np
# # # # x_train = np.random.randn(100, 20)
# # # # y_train = np.random.randn(100, 1)

# # # # # Define the callback
# # # # callback = CustomReduceLROnPlateau(monitor='loss', factor=0.5, patience=1, min_lr=1e-5)

# # # # # Train the model for a few epochs to allow the loss to improve
# # # # history = model.fit(x_train, y_train, epochs=5, callbacks=[callback])

# # # # # Add noise to the labels to force loss to plateau or increase
# # # # for i in range(len(y_train)):
# # # #     if i % 2 == 0:
# # # #         y_train[i] += np.random.randn() * 0.5  # Adding more noise to force plateau or increase

# # # # # Continue training with the modified data
# # # # for epoch in range(10):
# # # #     model.fit(x_train, y_train, epochs=1, verbose=0)
# # # #     print(f"Epoch {epoch + 1}, Learning Rate: {model.optimizer.learning_rate.numpy()}")

# # # #     # Simulate the end of an epoch and call the callback
# # # #     callback.on_epoch_end(epoch, logs={'loss': model.evaluate(x_train, y_train, verbose=0)})
# # # # """
# # # # import pandas as pd
# # # # import matplotlib.pyplot as plt
# # # # import os

# # # # def plot_loss_curves(csv_path, log_path):
# # # #     # Read the CSV file
# # # #     losses_df = pd.read_csv(csv_path)
    
# # # #     # Extract epochs
# # # #     epochs = losses_df['epoch']
    
# # # #     # Define the loss names
# # # #     loss_names = ['total', 'adv', 'perc', 'grad', 'second_grad', 'struct', 'aux']
    
# # # #     # Plot each loss curve
# # # #     for loss_name in loss_names:
# # # #         plt.figure(figsize=(10, 5))
# # # #         plt.plot(epochs, losses_df[f'train_{loss_name}_loss'], label=f'Train {loss_name.capitalize()} Loss')
# # # #         plt.plot(epochs, losses_df[f'val_{loss_name}_loss'], label=f'Validation {loss_name.capitalize()} Loss')
# # # #         plt.xlabel('Epochs')
# # # #         plt.ylabel('Loss')
# # # #         plt.title(f'{loss_name.capitalize()} Loss Curves')
# # # #         plt.legend()
# # # #         plt.grid(True)
# # # #         if not os.path.exists(log_path):
# # # #             os.makedirs(log_path)
# # # #         plt.savefig(os.path.join(log_path, f'{loss_name}_loss_curves.png'))
# # # #         plt.close()

# # # # # Example usage
# # # # csv_path = '/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/experimental_results/logs6_t4/losses.csv'
# # # # log_path = '/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/experimental_results/logs6_t4'
# # # # plot_loss_curves(csv_path, log_path)
# # # # """
# # # '''
# # # config 1.0
# # # {
# # #   "initial_weights": {
# # #     "adv": 0.5,
# # #     "perc": 2.0,
# # #     "grad": 1.8,
# # #     "second_grad": 0.6,
# # #     "struct": 2.0,
# # #     "aux": 1.5
# # #   },
# # #   "final_weights": {
# # #     "adv": 0.3,
# # #     "perc": 1.5,
# # #     "grad": 1.5,
# # #     "second_grad": 0.5,
# # #     "struct": 2.5,
# # #     "aux": 0.8
# # #   }
# # # }

# # # '''
# # # """
# # # config 2.0
# # # {
# # #   "initial_weights": {
# # #     "adv": 50.0,
# # #     "perc": 2.0,
# # #     "grad": 300.0,
# # #     "second_grad": 150.0,
# # #     "struct": 200.0,
# # #     "aux": 150.0
# # #   },
# # #   "final_weights": {
# # #     "adv": 30.0,
# # #     "perc": 1.5,
# # #     "grad": 150.0,
# # #     "second_grad": 100.0,
# # #     "struct": 250.0,
# # #     "aux": 80.0
# # #   }
# # # }
# # # """
# # # '''
# # # config 3.0

# # # {
# # #   "initial_weights": {
# # #     "adv": 50.0,
# # #     "perc": 2.0,
# # #     "grad": 500.0,
# # #     "second_grad": 150.0,
# # #     "struct": 300.0,
# # #     "aux": 150.0
# # #   },
# # #   "final_weights": {
# # #     "adv": 30.0,
# # #     "perc": 1.5,
# # #     "grad": 350.0,
# # #     "second_grad": 100.0,
# # #     "struct": 350.0,
# # #     "aux": 80.0
# # #   }
# # # }
# # # '''
# # # """
# # # config 4.0

# # # """
# # # ##### 6t_1 #####
# # # '''
# # # # # 66
# # # LPIPS:  0.3146485984325409
# # # SSIM:  0.9171656
# # # PSNR:  33.00963
# # # # # 73
# # # LPIPS:  0.3211759924888611
# # # SSIM:  0.91554785
# # # PSNR:  33.832783
# # # # # 76
# # # LPIPS:  0.3183180093765259
# # # SSIM:  0.91389465
# # # PSNR:  33.881176
# # # # # 77
# # # LPIPS:  0.32944005727767944
# # # SSIM:  0.91525346
# # # PSNR:  34.129562
# # # # # 79


# # # '''
# # # ##### 6t_1 #####
# # # '''


# # # # # 68
# # # LPIPS:  0.282987505197525
# # # SSIM:  0.933271
# # # PSNR:  34.531612
# # # # # 65
# # # LPIPS:  0.3012966215610504
# # # SSIM:  0.9282406
# # # PSNR:  34.277588
# # # # # 69
# # # LPIPS:  0.2852713465690613
# # # SSIM:  0.93560743
# # # PSNR:  34.979225
# # # # # 77

# # # # # 79


# # # '''



# # # ## TEST ##

# # # # def discriminator(input_shape=(768, 1024, 3)):
# # # #     inputs = layers.Input(shape=input_shape)

# # # #     # Initial convolution block
# # # #     x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
# # # #     x = layers.LeakyReLU(alpha=0.2)(x)
# # # #     features1 = x  # Intermediate feature

# # # #     # Multi-scale processing branches
# # # #     scale1 = res_block(x, 32)
# # # #     scale2 = layers.AveragePooling2D(pool_size=(2, 2))(x)
# # # #     scale2 = res_block(scale2, 32)
# # # #     scale2 = PixelShuffle(scale=2)(scale2)
# # # #     features2 = scale2  # Intermediate feature

# # # #     scale3 = layers.AveragePooling2D(pool_size=(4, 4))(x)
# # # #     scale3 = res_block(scale3, 32)
# # # #     scale3 = PixelShuffle(scale=4)(scale3)
# # # #     features3 = scale3  # Intermediate feature
    
# # # #     # Concatenate multi-scale features
# # # #     multi_scale = layers.Concatenate()([scale1, scale2, scale3])

# # # #     # Additional convolutional layers after concatenation
# # # #     x = layers.Conv2D(32, (3, 3), padding='same')(multi_scale)
# # # #     x = layers.LeakyReLU(alpha=0.2)(x)
# # # #     features4 = x  # Intermediate feature
    
# # # #     for filters in [64, 128, 256, 512]:
# # # #         x = layers.Conv2D(filters, (4, 4), strides=(2, 2), padding='same')(x)
# # # #         x = layers.BatchNormalization()(x)
# # # #         x = layers.LeakyReLU(negative_slope=0.2)(x)
    
# # # #     x = layers.Conv2D(1, (4, 4), padding='same')(x)
# # # #     return Model(inputs, [x, features1, features2, features3, features4])

# # # # def feature_matching_loss(discriminator, y_true, y_pred):
# # # #     real_outputs = discriminator(y_true, training=True)
# # # #     fake_outputs = discriminator(y_pred, training=True)
    
# # # #     real_features = real_outputs[1:]  # Skip final output, use intermediate features
# # # #     fake_features = fake_outputs[1:]  # Skip final output, use intermediate features
    
# # # #     loss = 0
# # # #     for real, fake in zip(real_features, fake_features):
# # # #         loss += tf.reduce_mean(tf.square(real - fake))
# # # #     return loss

# # # # def total_loss(vgg, y_true, y_pred, discriminator, discriminator_output_real, discriminator_output_fake, 
# # # #                lambda_adv=1.0, lambda_perceptual=1.0, lambda_gradient=1.0, 
# # # #                lambda_second=1.0, lambda_struct=1.0, lambda_aux=1.0, lambda_feat=1.0):
# # # #     adv_loss = adversarial_loss(discriminator_output_real, tf.ones_like(discriminator_output_real, dtype=tf.float32)) + \
# # # #                adversarial_loss(discriminator_output_fake, tf.zeros_like(discriminator_output_fake, dtype=tf.float32))
# # # #     perc_loss = perceptual_loss(vgg, y_true, y_pred)
# # # #     grad_loss = gradient_loss(y_true, y_pred)
# # # #     second_grad_loss = second_order_gradient_loss(y_true, y_pred)
# # # #     struct_loss = structure_aware_loss(y_true, y_pred)
# # # #     aux_loss = auxiliary_loss(y_true, y_pred)
# # # #     feat_loss = feature_matching_loss(discriminator, y_true, y_pred)

# # # #     total_loss_value = (lambda_adv * adv_loss) + (lambda_perceptual * perc_loss) + (lambda_gradient * grad_loss) + \
# # # #                        (lambda_second * second_grad_loss) + (lambda_struct * struct_loss) + (lambda_aux * aux_loss) + (lambda_feat * feat_loss)
    
# # # #     individual_losses = { 
# # # #         'adv': adv_loss, 
# # # #         'perc': perc_loss, 
# # # #         'grad': grad_loss, 
# # # #         'second_grad': second_grad_loss, 
# # # #         'struct': struct_loss, 
# # # #         'aux': aux_loss,
# # # #         'feat': feat_loss }
    
# # # #     return total_loss_value, individual_losses

# # # # @tf.function 
# # # # def train_step(LR_batch, HR_batch, generator, discriminator, vgg, generator_optimizer, discriminator_optimizer, **kwargs):
# # # #     with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
# # # #         generated_images = generator(LR_batch, training=True)
# # # #         generated_images = (generated_images + 1) / 2.0  # Scale from [-1, 1] to [0, 1]
# # # #         real_output = discriminator(HR_batch, training=True)
# # # #         fake_output = discriminator(generated_images, training=True)

# # # #         total_gen_loss, individual_losses = total_loss(vgg, HR_batch, generated_images, discriminator, real_output, fake_output, **kwargs)
# # # #         disc_loss = adversarial_loss(tf.ones_like(real_output, dtype=tf.float32), real_output) + \
# # # #                     adversarial_loss(tf.zeros_like(fake_output, dtype=tf.float32), fake_output)
    
# # # #     gradients_of_generator = gen_tape.gradient(total_gen_loss, generator.trainable_variables)
# # # #     generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

# # # #     gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
# # # #     discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# # # #     return total_gen_loss, disc_loss, generated_images, individual_losses

# # # # @tf.function 
# # # # def distributed_train_step(LR_batch, HR_batch, generator, discriminator, vgg, generator_optimizer, discriminator_optimizer, strategy, **kwargs):
# # # #     per_replica_results = strategy.run(
# # # #         train_step,
# # # #         args=(LR_batch, HR_batch, generator, discriminator, vgg, generator_optimizer, discriminator_optimizer),
# # # #         kwargs=kwargs
# # # #     )

# # # #     per_replica_gen_losses, per_replica_disc_losses, per_replica_generated_images, per_replica_individual_losses = per_replica_results

# # # #     gen_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_gen_losses, axis=None)
# # # #     disc_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_disc_losses, axis=None)
# # # #     individual_losses = {key: strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_individual_losses[key], axis=None) for key in per_replica_individual_losses}

# # # #     per_replica_generated_images = strategy.experimental_local_results(per_replica_generated_images)[0]

# # # #     return gen_loss, disc_loss, per_replica_generated_images, individual_losses

# # # # import os
# # # # import requests
# # # # import zipfile
# # # # from tqdm import tqdm

# # # # # Define the directory where the datasets will be saved
# # # # data_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/"
# # # # os.makedirs(data_dir, exist_ok=True)

# # # # # URL for the dataset
# # # # url = "https://figshare.com/ndownloader/articles/21586188/versions/1"
# # # # dataset_name = "BSD100_Set5_Set14_Urban100"

# # # # # Function to download and extract the dataset
# # # # def download_and_extract(url, dataset_name):
# # # #     response = requests.get(url, stream=True)
# # # #     file_size = int(response.headers.get('content-length', 0))
# # # #     download_path = os.path.join(data_dir, f"{dataset_name}.zip")
    
# # # #     # Download the dataset
# # # #     with open(download_path, 'wb') as f:
# # # #         for chunk in tqdm(response.iter_content(chunk_size=1024), total=file_size // 1024, unit='KB'):
# # # #             f.write(chunk)
    
# # # #     # Extract the dataset
# # # #     with zipfile.ZipFile(download_path, 'r') as zip_ref:
# # # #         zip_ref.extractall(data_dir)
# # # #     os.remove(download_path)

# # # # # Download and extract the dataset
# # # # download_and_extract(url, dataset_name)
# # # # import os
# # # # import zipfile

# # # # # Define the directory where the datasets are located and where they will be extracted
# # # # data_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/"

# # # # # List of zip files
# # # # zip_files = ["BSD100.zip","Set5.zip","Set14.zip","Urban100.zip"]

# # # # # Function to extract zip files
# # # # def extract_zip_files(zip_files, data_dir):
# # # #     for zip_file in zip_files:
# # # #         zip_path = os.path.join(data_dir, zip_file)
# # # #         if os.path.exists(zip_path):
# # # #             with zipfile.ZipFile(zip_path, 'r') as zip_ref:
# # # #                 zip_ref.extractall(data_dir)
# # # #             os.remove(zip_path)
# # # #         else:
# # # #             print(f"File {zip_file} does not exist.")

# # # # # Extract the zip files
# # # # extract_zip_files(zip_files, data_dir)



# # # # import os
# # # # import shutil

# # # # # Define the source directory
# # # # src_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/Urban100/image_SRF_4"

# # # # # Define the HR and LR directories
# # # # hr_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/Urban100/HR"
# # # # lr_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/Urban100/LR"

# # # # # Create HR and LR directories if they do not exist
# # # # os.makedirs(hr_dir, exist_ok=True)
# # # # os.makedirs(lr_dir, exist_ok=True)

# # # # # Get the list of image files
# # # # image_files = sorted(os.listdir(src_dir))

# # # # # Iterate over the image files and move them to the appropriate directories
# # # # for i, file_name in enumerate(image_files):
# # # #     src_path = os.path.join(src_dir, file_name)
# # # #     if i % 2 == 0:
# # # #         # Move HR images
# # # #         dest_path = os.path.join(hr_dir, file_name)
# # # #     else:
# # # #         # Move LR images
# # # #         dest_path = os.path.join(lr_dir, file_name)
# # # #     shutil.move(src_path, dest_path)

# # # # print("Images moved successfully!")


# # # # import os
# # # # import shutil
# # # # from math import floor

# # # # # Paths for the HR and LR images
# # # # hr_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/Urban100/HR"
# # # # lr_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/Urban100/LR"

# # # # # Output directories
# # # # train_hr_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/Urban100/train/HR"
# # # # train_lr_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/Urban100/train/LR"
# # # # val_hr_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/Urban100/val/HR"
# # # # val_lr_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/Urban100/val/LR"

# # # # # Create output directories if they don't exist
# # # # os.makedirs(train_hr_dir, exist_ok=True)
# # # # os.makedirs(train_lr_dir, exist_ok=True)
# # # # os.makedirs(val_hr_dir, exist_ok=True)
# # # # os.makedirs(val_lr_dir, exist_ok=True)

# # # # # List all HR and LR image files
# # # # hr_files = sorted(os.listdir(hr_dir))
# # # # lr_files = sorted(os.listdir(lr_dir))

# # # # # Ensure the same number of HR and LR images
# # # # assert len(hr_files) == len(lr_files), "Mismatch in number of HR and LR images."

# # # # # Split the data into train and validation sets (80:20)
# # # # total_images = len(hr_files)
# # # # train_size = floor(total_images * 0.8)
# # # # val_size = total_images - train_size

# # # # # Divide HR images
# # # # train_hr_files = hr_files[:train_size]
# # # # val_hr_files = hr_files[train_size:]

# # # # # Divide LR images
# # # # train_lr_files = lr_files[:train_size]
# # # # val_lr_files = lr_files[train_size:]

# # # # # Function to move files to the appropriate directory
# # # # def move_files(files, src_dir, dest_dir):
# # # #     for file_name in files:
# # # #         src_path = os.path.join(src_dir, file_name)
# # # #         dest_path = os.path.join(dest_dir, file_name)
# # # #         shutil.move(src_path, dest_path)

# # # # # Move HR and LR images to the train and validation directories
# # # # move_files(train_hr_files, hr_dir, train_hr_dir)
# # # # move_files(val_hr_files, hr_dir, val_hr_dir)
# # # # move_files(train_lr_files, lr_dir, train_lr_dir)
# # # # move_files(val_lr_files, lr_dir, val_lr_dir)

# # # # print("Dataset split into train and validation sets successfully!")

# # # # import os
# # # # import shutil

# # # # # Define the source directories
# # # # bsd100_hr_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/BSD100/val/HR"
# # # # bsd100_lr_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/BSD100/val/LR"
# # # # urban100_hr_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/Urban100/val/HR"
# # # # urban100_lr_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/Urban100/val/LR"

# # # # # Define the target directories
# # # # target_hr_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/S1/valid_HR"
# # # # target_lr_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/S1/valid_LR"

# # # # # Create target directories if they don't exist
# # # # os.makedirs(target_hr_dir, exist_ok=True)
# # # # os.makedirs(target_lr_dir, exist_ok=True)

# # # # # Function to copy and rename files from source to target directory
# # # # def copy_and_rename_files(source_dirs, target_dir):
# # # #     for src_dir, prefix in source_dirs:
# # # #         files = sorted(os.listdir(src_dir))
# # # #         for file_name in files:
# # # #             src_path = os.path.join(src_dir, file_name)
# # # #             dest_file_name = f"{prefix}_{file_name}"
# # # #             dest_path = os.path.join(target_dir, dest_file_name)
# # # #             shutil.copy(src_path, dest_path)

# # # # # Copy and rename HR images
# # # # copy_and_rename_files([(bsd100_hr_dir, "BSD100"), (urban100_hr_dir, "Urban100")], target_hr_dir)

# # # # # Copy and rename LR images
# # # # copy_and_rename_files([(bsd100_lr_dir, "BSD100"), (urban100_lr_dir, "Urban100")], target_lr_dir)

# # # # print("HR and LR images concatenated and renamed successfully!")

# # # import numpy as np

# # # # Define the data for each column
# # # data = {
# # #     'PSNR': [
# # #         [25.13, 30.489, 31.172, 29.100, 30.016, 33.000],
# # #         [26.98, 30.589, 31.249, 29.000, 30.008, 32.110],
# # #         [25.41, 30.300, 30.249, 30.000, 30.01, 30.506],
# # #         [22.43, 29.000, 28.032, 27.000, 25.000, 29.000],
# # #         [25.44, 30.301, 31.120, 30.000, 30.111, 31.678],
# # #         [26.95, 30.338, 31.138, 32.000, 23.000, 31.600]
# # #     ],
# # #     'SSIM': [
# # #         [0.479, 0.717, 0.791, 0.810, 0.810, 0.820],
# # #         [0.172, 0.738, 0.721, 0.750, 0.760, 0.770],
# # #         [0.203, 0.53, 0.72, 0.600, 0.610, 0.620],
# # #         [0.498, 0.631, 0.739, 0.540, 0.550, 0.760],
# # #         [0.667, 0.771, 0.789, 0.600, 0.610, 0.820],
# # #         [0.698, 0.729, 0.838, 0.640, 0.750, 0.860]
# # #     ],
# # #     'LPIPS': [
# # #         [0.55, 0.362, 0.335, 0.330, 0.310, 0.292],
# # #         [0.59, 0.34, 0.299, 0.350, 0.360, 0.270],
# # #         [0.48, 0.308, 0.299, 0.318, 0.330, 0.310],
# # #         [0.530, 0.437, 0.329, 0.430, 0.440, 0.350],
# # #         [0.40, 0.341, 0.326, 0.430, 0.440, 0.299],
# # #         [0.481, 0.341, 0.337, 0.440, 0.350, 0.306]
# # #     ]
# # # }

# # # # Calculate mean and standard deviation for each column
# # # for metric in data:
# # #     values = np.array(data[metric])
# # #     mean_values = np.mean(values, axis=0)
# # #     std_values = np.std(values, axis=0)
    
# # #     print(f"{metric} Mean: {mean_values}")
# # #     print(f"{metric} Std: {std_values}")
# # #     print()

# # from tensorflow.keras.layers import Input, Concatenate, Conv2D
# # from tensorflow.keras.models import Model

# # # Input tensors
# # input_1 = Input(shape=(64, 64, 3))  # LR image
# # input_2 = Input(shape=(64, 64, 512))  # HR feature maps

# # # Concatenate tensors along the channel axis
# # concatenated = Concatenate(axis=-1)([input_1, input_2])

# # # Further processing
# # x = Conv2D(64, (3, 3), padding='same')(concatenated)
# # x = Conv2D(3, (3, 3), padding='same', activation='tanh')(x)

# # # Model
# # model = Model(inputs=[input_1, input_2], outputs=x)
# # model.summary()


# ## --Image split-- ##
# # import os
# # import shutil
# # import random

# # # Set the paths
# # base_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/Scanning-electron-microscopy-images/SEM-dataset"
# # train_HR_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/A1/train_HR"
# # valid_HR_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/A1/valid_HR"

# # # Create the target directories if they don't exist
# # os.makedirs(train_HR_dir, exist_ok=True)
# # os.makedirs(valid_HR_dir, exist_ok=True)

# # # List all subdirectories
# # subdirs = [os.path.join(base_dir, subdir) for subdir in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, subdir))]

# # for subdir in subdirs:
# #     # Get all image files in the current subdirectory
# #     images = [os.path.join(subdir, img) for img in os.listdir(subdir) if img.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
    
# #     # Shuffle the images to ensure random selection
# #     random.shuffle(images)
    
# #     # Select 15 images for training and 5 images for validation
# #     train_images = images[:15]
# #     valid_images = images[15:20]

# #     # Copy the selected images to the respective directories
# #     for img in train_images:
# #         shutil.copy(img, train_HR_dir)
    
# #     for img in valid_images:
# #         shutil.copy(img, valid_HR_dir)

# # print("Completed copying images to train_HR and valid_HR directories.")

# ## -- Bicubic interpolation-- ##
# import os
# import cv2

# # Set the paths
# train_HR_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/A1/valid_HR"
# train_LR_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/A1/valid_LR3"

# # Create the target directory if it doesn't exist
# os.makedirs(train_LR_dir, exist_ok=True)

# # Define the scaling factor (you can adjust this as needed)
# scale_factor = 0.125  # Reducing the image size to 25% of the original size

# # Go through each image in the train_HR directory
# for img_name in os.listdir(train_HR_dir):
#     img_path = os.path.join(train_HR_dir, img_name)
    
#     if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
#         # Read the high-resolution image
#         hr_image = cv2.imread(img_path)
        
#         # Get the original dimensions
#         height, width = hr_image.shape[:2]
        
#         # Calculate the new dimensions
#         new_height, new_width = int(height * scale_factor), int(width * scale_factor)
        
#         # Perform bicubic interpolation
#         lr_image = cv2.resize(hr_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
#         # Save the low-resolution image
#         lr_image_path = os.path.join(train_LR_dir, img_name)
#         cv2.imwrite(lr_image_path, lr_image)

# print("Completed bicubic interpolation and saving images to train_LR directory.")

# # Crop Images
# import os
# import tensorflow as tf

# # Paths
# hr_image_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/A1/train_HR"
# sr_image_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/experiment/results1"

# hr_crop_save_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/experiment/evaluation_patch/HR384"
# sr_crop_save_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/experiment/evaluation_patch/SR384"

# # Save metrics to CSV
# output_csv = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/experiment/evaluation_patch/image_evaluation384_metrics.csv"

# # Ensure output directories exist
# os.makedirs(hr_crop_save_dir, exist_ok=True)
# os.makedirs(sr_crop_save_dir, exist_ok=True)

# # Function to load an image
# def load_image(image_path):
#     image = tf.io.read_file(image_path)
#     image = tf.image.decode_jpeg(image, channels=3)
#     return image

# # Function to save an image
# def save_image(image, save_path):
#     image = tf.image.convert_image_dtype(image, tf.uint8)  # Convert back to uint8
#     encoded_image = tf.image.encode_jpeg(image)
#     tf.io.write_file(save_path, encoded_image)

# # Function to crop image from the middle
# def crop_center(image, crop_height, crop_width):
#     shape = tf.shape(image)
#     height, width = shape[0], shape[1]
#     offset_height = (height - crop_height) // 2
#     offset_width = (width - crop_width) // 2
#     cropped_image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, crop_height, crop_width)
#     return cropped_image

# # Iterate over SR images
# for sr_image_name in os.listdir(sr_image_dir):
#     sr_image_path = os.path.join(sr_image_dir, sr_image_name)
    
#     # Construct corresponding HR image path
#     true_image_name = "_".join(sr_image_name.split("_")[1:])  # Adjust based on naming convention
#     hr_image_path = os.path.join(hr_image_dir, true_image_name)
    
#     if not os.path.exists(hr_image_path):
#         print(f"HR image not found for {sr_image_name}. Skipping.")
#         continue
    
#     # Load HR and SR images
#     sr_image = load_image(sr_image_path)
#     hr_image = load_image(hr_image_path)
    
#     # Crop both HR and SR images to 64x64 from the center
#     hr_cropped = crop_center(hr_image, 384, 384)
#     sr_cropped = crop_center(sr_image, 384, 384)
    
#     # Save cropped images
#     hr_save_path = os.path.join(hr_crop_save_dir, true_image_name)
#     sr_save_path = os.path.join(sr_crop_save_dir, sr_image_name)
#     save_image(hr_cropped, hr_save_path)
#     save_image(sr_cropped, sr_save_path)
    
#     print(f"Cropped and saved: HR -> {hr_save_path}, SR -> {sr_save_path}")

# print("Cropping and saving completed.")

# # ##Sort FIles
# import pandas as pd

# # Load the CSV file
# csv_file_path = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/experiment/evaluation_patch/1sorted_image_evaluation_metrics.csv"
# data = pd.read_csv(csv_file_path)

# # Sort the data by the PSNR column (second column) in descending order
# sorted_data = data.sort_values(by="Image Name", ascending=False)

# # Save the sorted data back to a new CSV file
# sorted_csv_file_path = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/experiment/evaluation_patch/sorted_image_evaluation_metrics.csv"
# sorted_data.to_csv(sorted_csv_file_path, index=False)

# print(f"Sorted CSV file saved to: {sorted_csv_file_path}")

# ## Remove unmatched images
# import os
# import pandas as pd

# # Paths
# csv_file_path = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/experiment/evaluation_patch/sorted_image_evaluation384_metrics.csv"
# sr_folder = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/experiment/evaluation_patch/SR384"
# hr_folder = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/experiment/evaluation_patch/HR384"

# # Read the CSV file and extract the "Image Name" column
# data = pd.read_csv(csv_file_path)
# image_names_to_keep = set(data["Image Name"].tolist())  # Convert to a set for faster lookup

# # Function to remove unwanted images
# def clean_folder(folder_path, valid_image_names):
#     for image_name in os.listdir(folder_path):
#         if image_name not in valid_image_names:
#             image_path = os.path.join(folder_path, image_name)
#             os.remove(image_path)
#             print(f"Removed: {image_path}")

# # Clean the SR and HR folders
# print("Cleaning SR folder...")
# clean_folder(sr_folder, image_names_to_keep)

# print("Cleaning HR folder...")
# clean_folder(hr_folder, image_names_to_keep)

# print("Image cleaning completed.")

# Error solver
# # import os
# # import shutil

# # # Paths
# # lr_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/A1/train_LR"
# # sem_dataset_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/Scanning-electron-microscopy-images/SEM-dataset"
# # hr_output_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/A1/train_HR"

# # # Ensure the output directory exists
# # os.makedirs(hr_output_dir, exist_ok=True)

# # # Iterate through images in the LR directory
# # for lr_file in os.listdir(lr_dir):
# #     if lr_file.endswith(".jpg"):  # Process only JPG files
# #         matched = False  # Flag to track if match is found

# #         # Check each folder inside the SEM-dataset
# #         for class_folder in os.listdir(sem_dataset_dir):
# #             class_folder_path = os.path.join(sem_dataset_dir, class_folder)

# #             if not os.path.isdir(class_folder_path):  # Skip if not a folder
# #                 continue

# #             # Check if the current LR file exists in this class folder
# #             hr_file_path = os.path.join(class_folder_path, lr_file)
# #             if os.path.exists(hr_file_path):
# #                 # Copy the matching image to the HR output directory
# #                 shutil.copy(hr_file_path, os.path.join(hr_output_dir, lr_file))
# #                 print(f"Copied: {hr_file_path} -> {hr_output_dir}")
# #                 matched = True
# #                 break  # Stop searching other folders once a match is found

# #         if not matched:
# #             print(f"No match found for {lr_file}")

# # print("File matching and copying completed.")


# ## Crop HR only
# import os
# import pandas as pd  # For working with CSV
# import tensorflow as tf

# # Paths
# hr_image_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/A1/train_LR"
# sorted_csv_path = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/experiment/evaluation_patch/sorted_image_evaluation64_metrics.csv"
# hr_crop_save_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/experiment/evaluation_patch/LR64"

# # Ensure the output directory exists
# os.makedirs(hr_crop_save_dir, exist_ok=True)

# # Function to load an image
# def load_image(image_path):
#     image = tf.io.read_file(image_path)
#     image = tf.image.decode_jpeg(image, channels=3)
#     return image

# # Function to save an image
# def save_image(image, save_path):
#     image = tf.image.convert_image_dtype(image, tf.uint8)  # Convert back to uint8
#     encoded_image = tf.image.encode_jpeg(image)
#     tf.io.write_file(save_path, encoded_image)

# # Function to crop image from the middle
# def crop_center(image, crop_height, crop_width):
#     shape = tf.shape(image)
#     height, width = shape[0], shape[1]
#     offset_height = (height - crop_height) // 2
#     offset_width = (width - crop_width) // 2
#     cropped_image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, crop_height, crop_width)
#     return cropped_image

# # Load CSV to get the image names to process
# try:
#     df = pd.read_csv(sorted_csv_path)
#     image_names = df['Image Name'].tolist()  # Replace 'ImageName' with the actual column name in your CSV
# except Exception as e:
#     print(f"Error loading CSV file: {e}")
#     image_names = []

# # Process each image
# for image_name in image_names:
#     # Strip the "1_" prefix from the filename if it exists
#     if image_name.startswith("1_"):
#         hr_image_name = image_name[2:]  # Remove the first two characters
#     else:
#         hr_image_name = image_name

#     # Construct the full HR image path
#     hr_image_path = os.path.join(hr_image_dir, hr_image_name)

#     # Check if the HR image exists
#     if not os.path.exists(hr_image_path):
#         print(f"HR image not found for {hr_image_name}. Skipping.")
#         continue

#     # Load the HR image
#     try:
#         hr_image = load_image(hr_image_path)
#     except Exception as e:
#         print(f"Error loading HR image {hr_image_name}: {e}")
#         continue

#     # Crop the HR image to 384x384 from the center
#     try:
#         hr_cropped = crop_center(hr_image, 16, 16)
#         # Construct save path for the cropped image
#         hr_save_path = os.path.join(hr_crop_save_dir, hr_image_name)
#         save_image(hr_cropped, hr_save_path)
#         print(f"Cropped and saved HR image: {hr_save_path}")
#     except Exception as e:
#         print(f"Error cropping or saving HR image {hr_image_name}: {e}")

# print("Cropping and saving completed.")


# ## Copy bet images
# import os
# import pandas as pd
# import shutil

# # Paths
# hr_image_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/A1/train_LR"
# sorted_csv_path = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/experiment/evaluation_patch/sorted_image_evaluation_metrics.csv"
# hr_output_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/experiment/evaluation_patch/LR"

# # Ensure the output directory exists
# os.makedirs(hr_output_dir, exist_ok=True)

# # Load CSV to get the image names to process
# try:
#     df = pd.read_csv(sorted_csv_path)
#     image_names = df['Image Name'].tolist()  # Replace 'ImageName' with the actual column name in your CSV
# except Exception as e:
#     print(f"Error loading CSV file: {e}")
#     image_names = []

# # Process each image
# for image_name in image_names:
#     # Remove "1_" prefix if it exists
#     if image_name.startswith("1_"):
#         hr_image_name = image_name[2:]  # Ignore the first two characters ("1_")
#     else:
#         hr_image_name = image_name

#     # Construct the full HR image path
#     hr_image_path = os.path.join(hr_image_dir, hr_image_name)

#     # Check if the HR image exists
#     if not os.path.exists(hr_image_path):
#         print(f"HR image not found for {hr_image_name}. Skipping.")
#         continue

#     # Copy the HR image to the output directory
#     try:
#         shutil.copy(hr_image_path, os.path.join(hr_output_dir, hr_image_name))
#         print(f"Copied HR image: {hr_image_path} -> {hr_output_dir}")
#     except Exception as e:
#         print(f"Error copying HR image {hr_image_name}: {e}")

# print("Matching and copying completed.")

'''
import os
import tensorflow as tf

# Paths
hr_image_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/check2/pattern"
hr_crop_save_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/check2/crop1"

# Ensure the output directory exists
os.makedirs(hr_crop_save_dir, exist_ok=True)

# Function to load an image
def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image

# Function to save an image
def save_image(image, save_path):
    image = tf.image.convert_image_dtype(image, tf.uint8)  # Convert back to uint8
    encoded_image = tf.image.encode_jpeg(image)
    tf.io.write_file(save_path, encoded_image)

# Function to crop image from the middle
def crop_center(image, crop_height, crop_width):
    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    offset_height = (height - crop_height) // 2
    offset_width = (width - crop_width) // 2
    cropped_image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, crop_height, crop_width)
    return cropped_image

# # Process all images in the HR image directory
# for image_name in os.listdir(hr_image_dir):
#     hr_image_path = os.path.join(hr_image_dir, image_name)

#     # Ensure it's a file (and not a directory)
#     if not os.path.isfile(hr_image_path):
#         print(f"Skipping non-file: {hr_image_path}")
#         continue

#     # Load the HR image
#     try:
#         hr_image = load_image(hr_image_path)
#     except Exception as e:
#         print(f"Error loading HR image {image_name}: {e}")
#         continue

#     # Crop the HR image to 16x16 from the center
#     try:
#         hr_cropped = crop_center(hr_image, 200, 400)
#         # Construct save path for the cropped image
#         hr_save_path = os.path.join(hr_crop_save_dir, image_name)
#         save_image(hr_cropped, hr_save_path)
#         print(f"Cropped and saved HR image: {hr_save_path}")
#     except Exception as e:
#         print(f"Error cropping or saving HR image {image_name}: {e}")

# print("Cropping and saving completed.")
# Function to crop the image using specified coordinates and dimensions
def crop_from_coordinates(image, start_y, start_x, crop_height, crop_width):
    """
    Crop the image starting from (start_y, start_x) with the specified height and width.
    """
    cropped_image = tf.image.crop_to_bounding_box(image, start_y, start_x, crop_height, crop_width)
    return cropped_image

# Example usage
for image_name in os.listdir(hr_image_dir):
    hr_image_path = os.path.join(hr_image_dir, image_name)

    # Ensure it's a file (and not a directory)
    if not os.path.isfile(hr_image_path):
        print(f"Skipping non-file: {hr_image_path}")
        continue

    # Load the HR image
    try:
        hr_image = load_image(hr_image_path)
    except Exception as e:
        print(f"Error loading HR image {image_name}: {e}")
        continue

    # Specify coordinates and crop dimensions
    start_y = 100  # Top-left corner y-coordinate
    start_x = 300  # Top-left corner x-coordinate
    crop_height = 168  # Desired crop height
    crop_width = 300  # Desired crop width

    # Crop the image from the specified coordinates
    try:
        hr_cropped = crop_from_coordinates(hr_image, start_y, start_x, crop_height, crop_width)
        # Construct save path for the cropped image
        hr_save_path = os.path.join(hr_crop_save_dir, image_name)
        save_image(hr_cropped, hr_save_path)
        print(f"Cropped and saved HR image: {hr_save_path}")
    except Exception as e:
        print(f"Error cropping or saving HR image {image_name}: {e}")

print("Cropping and saving completed.")
'''


# ##Sort FIles
import pandas as pd

# Load the CSV file
csv_file_path = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/experiment/conference/gen7_t5_15d/eval_metrics.csv"
data = pd.read_csv(csv_file_path)

# Sort the data by the PSNR column (second column) in descending order
sorted_data = data.sort_values(by="Image Name", ascending=False)

# Save the sorted data back to a new CSV file
sorted_csv_file_path = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/experiment/conference/gen7_t5_15d/eval_metrics_sorted.csv"
sorted_data.to_csv(sorted_csv_file_path, index=False)

print(f"Sorted CSV file saved to: {sorted_csv_file_path}")