# import os
# import json
# import numpy as np
# import tensorflow as tf
# from torch.utils.data import Dataset, DataLoader
# import cv2

# # Load configuration
# config_path = '/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/code/config/train.json'  # Update this path accordingly
# with open(config_path, 'r') as f:
#     config = json.load(f)

# # # Define the ImageDataset class for loading data directly from disk
# # class ImageDataset(Dataset):
# #     def __init__(self, image_dir, target_size):
# #         """
# #         Initializes the ImageDataset instance.

# #         Args:
# #         - image_dir (str): Directory containing the images.
# #         - target_size (tuple): Target size for the images.
# #         """
# #         self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
# #         self.target_size = (target_size[0], target_size[1]) # Ensure the size is (width, height)

# #     def __len__(self):
# #         """
# #         Returns the number of images in the directory.

# #         Returns:
# #         - int: Number of images.
# #         """
# #         return len(self.image_paths)

# #     def __getitem__(self, idx):
# #         """
# #         Retrieves and preprocesses an image from the directory.

# #         Args:
# #         - idx (int): Index of the image to retrieve.

# #         Returns:
# #         - np.ndarray: Retrieved and preprocessed image.
# #         """
# #         image_path = self.image_paths[idx]
# #         image = cv2.imread(image_path)
# #         if image is None:
# #             raise ValueError(f"Error reading image {image_path}")
# #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
# #         image = cv2.resize(image, self.target_size)  # Resize to target size
# #         # image = image / 255.0  # Normalize to [0, 1]
# #         image = image / 127.5 - 1 # Normalize to [-1, 1] range for tanh activation
# #         return image


# import os
# import cv2
# from torch.utils.data import Dataset

# class PairedImageDataset(Dataset):
#     def __init__(self, lr_image_dir, hr_image_dir, lr_target_size, hr_target_size):
#         self.lr_image_paths = sorted([os.path.join(lr_image_dir, fname) for fname in os.listdir(lr_image_dir)])
#         self.hr_image_paths = sorted([os.path.join(hr_image_dir, fname) for fname in os.listdir(hr_image_dir)])
#         self.lr_target_size = (lr_target_size[0], lr_target_size[1])
#         self.hr_target_size = (hr_target_size[0], hr_target_size[1])

#     def __len__(self):
#         return len(self.lr_image_paths)

#     def __getitem__(self, idx):
#         lr_image_path = self.lr_image_paths[idx]
#         hr_image_path = self.hr_image_paths[idx]
        
#         lr_image = cv2.imread(lr_image_path)
#         hr_image = cv2.imread(hr_image_path)
        
#         if lr_image is None or hr_image is None:
#             raise ValueError(f"Error reading image {lr_image_path} or {hr_image_path}")

#         lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
#         hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
        
#         lr_image = cv2.resize(lr_image, self.lr_target_size)
#         hr_image = cv2.resize(hr_image, self.hr_target_size)
        
#         lr_image = lr_image / 127.5 - 1
#         hr_image = hr_image / 255
        
#         return lr_image, hr_image



# '''
# import os
# import json
# import numpy as np
# import tensorflow as tf
# from torch.utils.data import Dataset, DataLoader
# import cv2

# # Load configuration
# config_path = '/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/code/config/train.json'  # Update this path accordingly
# with open(config_path, 'r') as f:
#     config = json.load(f)

# # # Define the ImageDataset class for loading data directly from disk
# # class ImageDataset(Dataset):
# #     def __init__(self, image_dir, target_size):
# #         """
# #         Initializes the ImageDataset instance.

# #         Args:
# #         - image_dir (str): Directory containing the images.
# #         - target_size (tuple): Target size for the images.
# #         """
# #         self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
# #         self.target_size = (target_size[0], target_size[1]) # Ensure the size is (width, height)

# #     def __len__(self):
# #         """
# #         Returns the number of images in the directory.

# #         Returns:
# #         - int: Number of images.
# #         """
# #         return len(self.image_paths)

# #     def __getitem__(self, idx):
# #         """
# #         Retrieves and preprocesses an image from the directory.

# #         Args:
# #         - idx (int): Index of the image to retrieve.

# #         Returns:
# #         - np.ndarray: Retrieved and preprocessed image.
# #         """
# #         image_path = self.image_paths[idx]
# #         image = cv2.imread(image_path)
# #         if image is None:
# #             raise ValueError(f"Error reading image {image_path}")
# #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
# #         image = cv2.resize(image, self.target_size)  # Resize to target size
# #         # image = image / 255.0  # Normalize to [0, 1]
# #         image = image / 127.5 - 1 # Normalize to [-1, 1] range for tanh activation
# #         return image


# ## -- Ensure -- ##
# # import os
# # import cv2
# # from torch.utils.data import Dataset

# # class PairedImageDataset(Dataset):
# #     def __init__(self, lr_image_dir, hr_image_dir, lr_target_size, hr_target_size):
# #         self.lr_image_paths = sorted([os.path.join(lr_image_dir, fname) for fname in os.listdir(lr_image_dir)])
# #         self.hr_image_paths = sorted([os.path.join(hr_image_dir, fname) for fname in os.listdir(hr_image_dir)])
# #         self.lr_target_size = (lr_target_size[0], lr_target_size[1])
# #         self.hr_target_size = (hr_target_size[0], hr_target_size[1])

# #     def __len__(self):
# #         return len(self.lr_image_paths)

# #     def __getitem__(self, idx):
# #         lr_image_path = self.lr_image_paths[idx]
# #         hr_image_path = self.hr_image_paths[idx]
        
# #         lr_image = cv2.imread(lr_image_path)
# #         hr_image = cv2.imread(hr_image_path)
        
# #         if lr_image is None or hr_image is None:
# #             raise ValueError(f"Error reading image {lr_image_path} or {hr_image_path}")

# #         lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
# #         hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
        
# #         lr_image = cv2.resize(lr_image, self.lr_target_size)
# #         hr_image = cv2.resize(hr_image, self.hr_target_size)
        
# #         lr_image = lr_image / 127.5 - 1
# #         hr_image = hr_image / 255
        
# #         return lr_image, hr_image


# import os
# import cv2
# import tensorflow as tf
# from torch.utils.data import Dataset

# class PairedImageDataset(Dataset):
#     def __init__(self, lr_image_dir, hr_image_dir, lr_target_size, hr_target_size):
#         self.lr_image_paths = sorted([os.path.join(lr_image_dir, fname) for fname in os.listdir(lr_image_dir)])
#         self.hr_image_paths = sorted([os.path.join(hr_image_dir, fname) for fname in os.listdir(hr_image_dir)])
#         self.lr_target_size = (lr_target_size[0], lr_target_size[1])
#         self.hr_target_size = (hr_target_size[0], hr_target_size[1])

#     def __len__(self):
#         return len(self.lr_image_paths)

#     def augment_image(self, image):
#         image = tf.image.random_brightness(image, max_delta=0.2)
#         image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
#         image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
#         image = tf.image.random_hue(image, max_delta=0.1)
#         return image

#     def load_and_preprocess_image(self, image_path, target_size, is_lr=True):
#         image = cv2.imread(image_path)
#         if image is None:
#             raise ValueError(f"Error reading image {image_path}")
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = cv2.resize(image, target_size)
#         if is_lr:
#             image = image / 127.5 - 1  # Normalize LR image to [-1, 1]
#         else:
#             image = image / 255  # Normalize HR image to [0, 1]
#         return image

#     def __getitem__(self, idx):
#         lr_image_path = self.lr_image_paths[idx]
#         hr_image_path = self.hr_image_paths[idx]
        
#         lr_image = self.load_and_preprocess_image(lr_image_path, self.lr_target_size, is_lr=True)
#         hr_image = self.load_and_preprocess_image(hr_image_path, self.hr_target_size, is_lr=False)
        
#         # Apply augmentation to both LR and HR images
#         lr_image = self.augment_image(lr_image)
#         hr_image = self.augment_image(hr_image)
        
#         return lr_image, hr_image


# # # Update the paths and target shapes
# # train_hr_path = config['train_hr_path']
# # train_lr_path = config['train_lr_path']
# # valid_hr_path = config['valid_hr_path']
# # valid_lr_path = config['valid_lr_path']
# # # target_size_hr = (config['target_size_hr'][1], config['target_size_hr'][0])                   # Ensure correct order (width, height)
# # # target_size_lr = (config['target_size_lr'][1], config['target_size_lr'][0])                   # Ensure correct order (width, height)
# # target_size_hr = tuple(config['target_size_hr']) # Ensure correct order (width, height) 
# # target_size_lr = tuple(config['target_size_lr']) # Ensure correct order (width, height)
# # batch_size = config['batch_size']

# # # Initialize the datasets
# # train_hr_dataset = ImageDataset(train_hr_path, target_size_hr)  # Ensure target_size matches preprocessing
# # train_lr_dataset = ImageDataset(train_lr_path, target_size_lr)  # Ensure target_size matches preprocessing
# # valid_hr_dataset = ImageDataset(valid_hr_path, target_size_hr)  # Ensure target_size matches preprocessing
# # valid_lr_dataset = ImageDataset(valid_lr_path, target_size_lr)  # Ensure target_size matches preprocessing

# # # Create DataLoaders
# # train_hr_dataloader = DataLoader(train_hr_dataset, batch_size=batch_size, shuffle=False)
# # train_lr_dataloader = DataLoader(train_lr_dataset, batch_size=batch_size, shuffle=False)
# # valid_hr_dataloader = DataLoader(valid_hr_dataset, batch_size=batch_size, shuffle=False)
# # valid_lr_dataloader = DataLoader(valid_lr_dataset, batch_size=batch_size, shuffle=False)

# # import matplotlib.pyplot as plt

# # def show_sample_images(hr_dataloader, lr_dataloader):
# #     hr_iter = iter(hr_dataloader)
# #     lr_iter = iter(lr_dataloader)
# #     hr_images = next(hr_iter)
# #     lr_images = next(lr_iter)
    
# #     # Plot first image in batch
# #     fig, axs = plt.subplots(1, 2, figsize=(12, 6))
# #     axs[0].imshow(hr_images[0])
# #     axs[0].set_title('High Resolution Image')
# #     axs[1].imshow(lr_images[0])
# #     axs[1].set_title('Low Resolution Image')
# #     plt.show()

# # show_sample_images(train_hr_dataloader, train_lr_dataloader)

# '''



# import os
# import cv2
# import random
# from torch.utils.data import Dataset
# import albumentations as A # type: ignore
# from albumentations.pytorch import ToTensorV2 # type: ignore

# class PairedImageDataset(Dataset):
#     def __init__(self, lr_image_dir, hr_image_dir, lr_target_size, hr_target_size, augmentation=None):
#         self.lr_image_paths = sorted([os.path.join(lr_image_dir, fname) for fname in os.listdir(lr_image_dir)])
#         self.hr_image_paths = sorted([os.path.join(hr_image_dir, fname) for fname in os.listdir(hr_image_dir)])
#         self.lr_target_size = (lr_target_size[0], lr_target_size[1])
#         self.hr_target_size = (hr_target_size[0], hr_target_size[1])
#         self.augmentation = augmentation

#     def __len__(self):
#         return len(self.lr_image_paths)

#     def __getitem__(self, idx):
#         lr_image_path = self.lr_image_paths[idx]
#         hr_image_path = self.hr_image_paths[idx]
        
#         lr_image = cv2.imread(lr_image_path)
#         hr_image = cv2.imread(hr_image_path)
        
#         if lr_image is None or hr_image is None:
#             raise ValueError(f"Error reading image {lr_image_path} or {hr_image_path}")

#         lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
#         hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)

#         # Apply augmentations if any
#         if self.augmentation:
#             augmented = self.augmentation(image=lr_image, image0=hr_image)
#             lr_image = augmented['image']
#             hr_image = augmented['image0']

#         lr_image = cv2.resize(lr_image, self.lr_target_size)
#         hr_image = cv2.resize(hr_image, self.hr_target_size)
        
#         lr_image = lr_image / 127.5 - 1
#         hr_image = hr_image / 255
        
#         return lr_image, hr_image

# # Define comprehensive augmentations using albumentations
# augmentation = A.Compose([
#     # A.RandomCrop(width=128, height=128),               # Random cropping
#     # A.HorizontalFlip(p=0.5),                           # Random horizontal flip
#     # A.Rotate(limit=15, p=0.5),                         # Random rotation
#     A.RandomBrightnessContrast(p=0.2),                 # Random brightness and contrast
#     A.HueSaturationValue(p=0.3),                       # Random changes in hue, saturation, and value
#     A.GaussianBlur(p=0.2),                             # Apply Gaussian blur
#     A.MotionBlur(p=0.2),                               # Apply motion blur
#     A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),  # Random JPEG compression
#     A.RandomGamma(p=0.2),                              # Random gamma adjustments
#     A.RandomShadow(p=0.2),                             # Random shadows
#     ToTensorV2()
# ], additional_targets={'image0': 'image'})             # To ensure the same augmentation is applied to both images



# '''
"""
# Dont Change this...
import os
import cv2
from torch.utils.data import Dataset

class PairedImageDataset(Dataset):
    def __init__(self, lr_image_dir, hr_image_dir, lr_target_size, hr_target_size):
        self.lr_image_paths = sorted([os.path.join(lr_image_dir, fname) for fname in os.listdir(lr_image_dir)])
        self.hr_image_paths = sorted([os.path.join(hr_image_dir, fname) for fname in os.listdir(hr_image_dir)])
        # self.lr_target_size = (lr_target_size[0], lr_target_size[1])
        # self.hr_target_size = (hr_target_size[0], hr_target_size[1])

    def __len__(self):
        return len(self.lr_image_paths)

    def __getitem__(self, idx):
        lr_image_path = self.lr_image_paths[idx]
        hr_image_path = self.hr_image_paths[idx]
        
        lr_image = cv2.imread(lr_image_path)
        hr_image = cv2.imread(hr_image_path)
        
        if lr_image is None or hr_image is None:
            raise ValueError(f"Error reading image {lr_image_path} or {hr_image_path}")

        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
        
        # lr_image = cv2.resize(lr_image, self.lr_target_size)
        # hr_image = cv2.resize(hr_image, self.hr_target_size)
        
        lr_image = lr_image / 127.5 - 1
        hr_image = hr_image / 255
        
        return lr_image, hr_image

"""
import os
import cv2
from torch.utils.data import Dataset
from PIL import Image  # Import the Image module from Pillow
import json

def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height

def update_json_file(config_path, config):
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

class PairedImageDataset(Dataset):
    def __init__(self, lr_image_dir, hr_image_dir, lr_target_size, hr_target_size, train_config):
        self.lr_image_paths = sorted([os.path.join(lr_image_dir, fname) for fname in os.listdir(lr_image_dir)])
        self.hr_image_paths = sorted([os.path.join(hr_image_dir, fname) for fname in os.listdir(hr_image_dir)])
        self.train_config = train_config
        self.config_path = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/code/config/train.json"

    def __len__(self):
        return len(self.lr_image_paths)

    def __getitem__(self, idx):
        lr_image_path = self.lr_image_paths[idx]
        hr_image_path = self.hr_image_paths[idx]
        
        lr_image = cv2.imread(lr_image_path)
        hr_image = cv2.imread(hr_image_path)
        
        if lr_image is None or hr_image is None:
            raise ValueError(f"Error reading image {lr_image_path} or {hr_image_path}")

        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
        
        # Get the dimensions of the current HR image
        hr_width, hr_height = get_image_dimensions(hr_image_path)
        
        # Update the configuration dynamically
        self.train_config['vgg_input_shape'] = [hr_height, hr_width, 3]
        
        # Write the updated configuration back to the JSON file
        update_json_file(self.config_path, self.train_config)
        
        lr_image = lr_image / 127.5 - 1
        hr_image = hr_image / 255
        
        return lr_image, hr_image

#############################################################################################################################


















#############################################################################################################################
# '''


# # Worked but shape missmatched
# import os
# import cv2
# import random
# import numpy as np
# from torch.utils.data import Dataset
# import albumentations as A
# from albumentations.pytorch import ToTensorV2

# class PairedImageDataset(Dataset):
#     def __init__(self, lr_image_dir, hr_image_dir, lr_target_size, hr_target_size):
#         self.lr_image_paths = sorted([os.path.join(lr_image_dir, fname) for fname in os.listdir(lr_image_dir)])
#         self.hr_image_paths = sorted([os.path.join(hr_image_dir, fname) for fname in os.listdir(hr_image_dir)])
#         self.lr_target_size = (lr_target_size[0], lr_target_size[1])
#         self.hr_target_size = (hr_target_size[0], hr_target_size[1])

#         self.transform = A.Compose([
#             A.RandomCrop(width=lr_target_size[1], height=lr_target_size[0]),
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.5),
#             A.RandomRotate90(p=0.5),
#             A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
#             A.ElasticTransform(p=0.5),
#             A.CoarseDropout(max_holes=8, max_height=16, max_width=16, min_holes=1, min_height=1, min_width=1, p=0.5),  # Corrected to use CoarseDropout
#             # A.MixUp(p=0.5),  # MixUp is not a standard Albumentations function, remove or implement manually if needed
#             A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
#             ToTensorV2()
#         ])

#     def __len__(self):
#         return len(self.lr_image_paths)

#     def __getitem__(self, idx):
#         lr_image_path = self.lr_image_paths[idx]
#         hr_image_path = self.hr_image_paths[idx]
        
#         lr_image = cv2.imread(lr_image_path)
#         hr_image = cv2.imread(hr_image_path)
        
#         if lr_image is None or hr_image is None:
#             raise ValueError(f"Error reading image {lr_image_path} or {hr_image_path}")

#         lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
#         hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)

#         # Resize images to the target size before augmentation
#         lr_image = cv2.resize(lr_image, self.lr_target_size)
#         hr_image = cv2.resize(hr_image, self.hr_target_size)

#         # Apply augmentations
#         augmented = self.transform(image=lr_image, mask=hr_image)
#         lr_image = augmented['image'].float() / 127.5 - 1
#         hr_image = augmented['mask'].float() / 255
        
#         return lr_image, hr_image


# import os
# import cv2
# import numpy as np
# from torch.utils.data import Dataset
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import torch

# class PairedImageDataset(Dataset):
#     def __init__(self, lr_image_dir, hr_image_dir, lr_target_size, hr_target_size):
#         self.lr_image_paths = sorted([os.path.join(lr_image_dir, fname) for fname in os.listdir(lr_image_dir)])
#         self.hr_image_paths = sorted([os.path.join(hr_image_dir, fname) for fname in os.listdir(hr_image_dir)])
#         self.lr_target_size = (lr_target_size[0], lr_target_size[1])
#         self.hr_target_size = (hr_target_size[0], hr_target_size[1])

#         # Augmentation pipeline for LR images
#         self.lr_transform = A.Compose([
#             A.Resize(height=lr_target_size[0], width=lr_target_size[1]),
#             A.RandomCrop(height=lr_target_size[0], width=lr_target_size[1]),
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.5),
#             A.RandomRotate90(p=0.5),
#             A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
#             A.ElasticTransform(p=0.5),
#             A.CoarseDropout(max_holes=8, max_height=16, max_width=16, min_holes=1, min_height=1, min_width=1, p=0.5),  # Corrected parameters
#             A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),  # Corrected parameters
#             ToTensorV2()
#         ])

#         # Augmentation pipeline for HR images
#         self.hr_transform = A.Compose([
#             A.Resize(height=hr_target_size[0], width=hr_target_size[1]),
#             A.RandomCrop(height=hr_target_size[0], width=hr_target_size[1]),
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.5),
#             A.RandomRotate90(p=0.5),
#             A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
#             A.ElasticTransform(p=0.5),
#             A.CoarseDropout(max_holes=8, max_height=16, max_width=16, min_holes=1, min_height=1, min_width=1, p=0.5),  # Corrected parameters
#             A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),  # Corrected parameters
#             ToTensorV2()
#         ])

#     def __len__(self):
#         return len(self.lr_image_paths)

#     def __getitem__(self, idx):
#         lr_image_path = self.lr_image_paths[idx]
#         hr_image_path = self.hr_image_paths[idx]
        
#         print(f"\nLoading LR image: {lr_image_path}")
#         print(f"Loading HR image: {hr_image_path}")
        
#         lr_image = cv2.imread(lr_image_path)
#         hr_image = cv2.imread(hr_image_path)
        
#         if lr_image is None or hr_image is None:
#             print(f"Error reading image {lr_image_path} or {hr_image_path}")
#             return None, None

#         print(f"Original LR Image Shape: {lr_image.shape}")
#         print(f"Original HR Image Shape: {hr_image.shape}")
        
#         lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
#         hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)

#         print(f"Converted LR Image Shape: {lr_image.shape}")
#         print(f"Converted HR Image Shape: {hr_image.shape}")

#         # Resize images to the target size before augmentation
#         lr_image = cv2.resize(lr_image, self.lr_target_size)
#         hr_image = cv2.resize(hr_image, self.hr_target_size)

#         print(f"Resized LR Image Shape: {lr_image.shape}")
#         print(f"Resized HR Image Shape: {hr_image.shape}")

#         try:
#             # Apply augmentations separately and ensure NHWC format
#             lr_augmented = self.lr_transform(image=lr_image)['image'].numpy()  # NHWC format
#             hr_augmented = self.hr_transform(image=hr_image)['image'].numpy()  # NHWC format

#             print(f"Augmented LR Image Shape: {lr_augmented.shape}")
#             print(f"Augmented HR Image Shape: {hr_augmented.shape}")

#         except Exception as e:
#             print(f"Augmentation error: {e}")
#             return None, None

#         return lr_augmented, hr_augmented

# # Helper function for debugging
# def collate_fn(batch):
#     batch = list(filter(lambda x: x[0] is not None, batch))
#     return torch.utils.data.dataloader.default_collate(batch)


# import os
# import tensorflow as tf
# from config_loader import load_config, load_loss_weights

# from torch.utils.data import DataLoader
# import importlib
# # from utils.data_loader import PairedImageDataset

# policy = tf.keras.mixed_precision.Policy('mixed_float16')
# tf.keras.mixed_precision.set_global_policy(policy)

# model_config = load_config('/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/code/config/model_config.json')
# train_config = load_config('/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/code/config/train.json')
# loss_weights_config = load_loss_weights('/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/code/config/loss_config.json')


# train_hr_path = train_config['train_hr_path']
# train_lr_path = train_config['train_lr_path']
# valid_hr_path = train_config['valid_hr_path']
# valid_lr_path = train_config['valid_lr_path']
# target_size_hr = tuple(train_config['target_size_hr'])
# target_size_lr = tuple(train_config['target_size_lr'])
# batch_size = train_config['batch_size']

# train_dataset = PairedImageDataset(train_lr_path, train_hr_path, target_size_lr, target_size_hr)
# valid_dataset = PairedImageDataset(valid_lr_path, valid_hr_path, target_size_lr, target_size_hr)

# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
# valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)


# import cv2
# import random
# import numpy as np
# from torch.utils.data import Dataset
# import albumentations as A
# from albumentations.pytorch import ToTensorV2

# class PairedImageDataset(Dataset):
#     def __init__(self, lr_image_dir, hr_image_dir, lr_target_size, hr_target_size):
#         self.lr_image_paths = sorted([os.path.join(lr_image_dir, fname) for fname in os.listdir(lr_image_dir)])
#         self.hr_image_paths = sorted([os.path.join(hr_image_dir, fname) for fname in os.listdir(hr_image_dir)])
#         self.lr_target_size = (lr_target_size[0], lr_target_size[1])
#         self.hr_target_size = (hr_target_size[0], hr_target_size[1])

#         self.transform = A.Compose([
#             A.RandomCrop(width=lr_target_size[1], height=lr_target_size[0]),
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.5),
#             A.RandomRotate90(p=0.5),
#             A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
#             A.ElasticTransform(p=0.5),
#             # A.cutout(num_holes=8, max_h_size=16, max_w_size=16, p=0.5),
#             # A.Cutout(num_holes=8, max_h_size=16, max_w_size=16, p=0.5), # Corrected with appropriate parameters
#             # A.MixUp(p=0.5), # MixUp is not a standard Albumentations function, remove or implement manually if needed,
#             A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
#             ToTensorV2()
#         ])

#     def __len__(self):
#         return len(self.lr_image_paths)

#     def __getitem__(self, idx):
#         lr_image_path = self.lr_image_paths[idx]
#         hr_image_path = self.hr_image_paths[idx]
        
#         lr_image = cv2.imread(lr_image_path)
#         hr_image = cv2.imread(hr_image_path)
        
#         if lr_image is None or hr_image is None:
#             raise ValueError(f"Error reading image {lr_image_path} or {hr_image_path}")

#         lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
#         hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)

#         # Resize images to the target size before augmentation
#         lr_image = cv2.resize(lr_image, self.lr_target_size)
#         hr_image = cv2.resize(hr_image, self.hr_target_size)

#         # Apply augmentations
#         augmented = self.transform(image=lr_image, mask=hr_image)
#         lr_image = augmented['image'].float() / 127.5 - 1
#         hr_image = augmented['mask'].float() / 255
        
#         return lr_image, hr_image


"""
## -- Versoion 1.0 -- ##
import os
import cv2
import random
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class PairedImageDataset(Dataset):
    def __init__(self, lr_image_dir, hr_image_dir, lr_target_size, hr_target_size):
        self.lr_image_paths = sorted([os.path.join(lr_image_dir, fname) for fname in os.listdir(lr_image_dir)])
        self.hr_image_paths = sorted([os.path.join(hr_image_dir, fname) for fname in os.listdir(hr_image_dir)])
        self.lr_target_size = (lr_target_size[0], lr_target_size[1])
        self.hr_target_size = (hr_target_size[0], hr_target_size[1])

        self.transform = A.Compose([
            A.RandomCrop(width=lr_target_size[1], height=lr_target_size[0]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
            A.GaussianNoise(var_limit=(10.0, 50.0), p=0.5),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.lr_image_paths)

    def __getitem__(self, idx):
        lr_image_path = self.lr_image_paths[idx]
        hr_image_path = self.hr_image_paths[idx]
        
        lr_image = cv2.imread(lr_image_path)
        hr_image = cv2.imread(hr_image_path)
        
        if lr_image is None or hr_image is None:
            raise ValueError(f"Error reading image {lr_image_path} or {hr_image_path}")

        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)

        # Apply augmentations
        augmented = self.transform(image=lr_image, mask=hr_image)
        lr_image = augmented['image'].float() / 127.5 - 1
        hr_image = augmented['mask'].float() / 255
        
        return lr_image, hr_image

"""