# # from tensorflow.keras.models import load_model # type: ignore
# # from PIL import Image
# # import numpy as np
# # import os

# # # Load the trained model
# # model_path = '/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/models/generator7_t1_7d/best_model.keras'
# # model = load_model(model_path, safe_mode=False)  # Load the model with safe mode disabled

# # # Load the LR image
# # lr_image_path = '/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/DIV2K/DIV2K_train_LR_bicubic/X4/L9_00fc0a86bd4f02995acdd5b3f63401b9x4.jpg'
# # lr_image = Image.open(lr_image_path)
# # # lr_image = lr_image.resize((192, 256))  # Resize to match your model input shape if necessary
# # lr_image_array = np.array(lr_image) / 255.0  # Normalize the image
# # lr_image_array = np.expand_dims(lr_image_array, axis=0)  # Add batch dimension

# # # Generate the SR image
# # sr_image_array = model.predict(lr_image_array)
# # sr_image_array = np.squeeze(sr_image_array, axis=0)  # Remove batch dimension
# # sr_image_array = (sr_image_array * 255).astype(np.uint8)  # Denormalize the image
# # sr_image = Image.fromarray(sr_image_array)

# # # Save the SR image
# # save_dir = '/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/code/generated_images'
# # if not os.path.exists(save_dir):
# #     os.makedirs(save_dir)
# # sr_image_path = os.path.join(save_dir, 'sr_image.png')
# # sr_image.save(sr_image_path)

# # print(f'Super-Resolution image saved at: {sr_image_path}')


# from tensorflow.keras.models import load_model
# from PIL import Image
# import numpy as np
# import os

# # Load the trained model
# model_path = '/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/models/generator1_t1/best_model.keras'
# model = load_model(model_path, safe_mode=False)  # Load the model with safe mode disabled

# # Load the LR image
# lr_image_path = '/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/DIV2K/DIV2K_train_LR_bicubic/X4/L9_ffaa4e2ca12ad56fe542cbcdb0de6b5bx4.jpg'
# lr_image = Image.open(lr_image_path)
# # lr_image = lr_image.resize((192, 256))  # Resize if necessary
# lr_image_array = np.array(lr_image).astype(np.float32) / 127.5 - 1  # Normalize to [-1, 1]
# lr_image_array = np.expand_dims(lr_image_array, axis=0)  # Add batch dimension

# # Generate the SR image
# sr_image_array = model.predict(lr_image_array)
# sr_image_array = np.squeeze(sr_image_array, axis=0)  # Remove batch dimension
# sr_image_array = (sr_image_array + 1) * 127.5  # Denormalize from [-1, 1] to [0, 255]
# sr_image_array = np.clip(sr_image_array, 0, 255).astype(np.uint8)  # Ensure values are within [0, 255]
# sr_image = Image.fromarray(sr_image_array)

# # Save the SR image
# save_dir = '/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/code/generated_images'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# sr_image_path = os.path.join(save_dir, 'sr_image.png')
# sr_image.save(sr_image_path)

# print(f'Super-Resolution image saved at: {sr_image_path}')


'''import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import keras
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.models import Model

# Load the trained generator model
generator_path = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/models/generator7_t6_15d/best_model.keras"
# Enable unsafe deserialization globally
keras.config.enable_unsafe_deserialization()
generator = tf.keras.models.load_model(generator_path, compile=False)

# Load pre-trained EfficientNet for feature extraction
efficientnet = EfficientNetB7(include_top=False, weights='imagenet')
feature_extractor = Model(inputs=efficientnet.input, outputs=efficientnet.get_layer("block7a_activation").output)

def preprocess_image(lr_image_path):
    """Preprocesses the LR image before passing it to the model."""
    lr_image = cv2.imread(lr_image_path)
    if lr_image is None:
        raise ValueError(f"Could not read image: {lr_image_path}")

    lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    lr_image = lr_image / 127.5 - 1  # Normalize to [-1, 1] (same as training)
    lr_image = np.expand_dims(lr_image, axis=0)  # Add batch dimension
    return lr_image

def postprocess_image(sr_image):
    """Postprocesses the SR image after model inference."""
    sr_image = (sr_image + 1) * 127.5  # Denormalize to [0, 255]
    sr_image = np.clip(sr_image, 0, 255).astype(np.uint8)  # Ensure valid pixel values
    return sr_image

def generate_sr_image(lr_image_path, save_path="sr_output.png"):
    """Generates a super-resolved image using the trained model."""
    lr_image = preprocess_image(lr_image_path)

    # Extract features from LR image using EfficientNet
    hr_features = feature_extractor.predict(lr_image)

    # Run inference
    sr_image = generator.predict([lr_image, hr_features])
    
    sr_image = postprocess_image(sr_image[0])  # Remove batch dimension

    # Convert to PIL Image and save
    sr_image_pil = Image.fromarray(sr_image)
    sr_image_pil.save(save_path, format="PNG")
    print(f"SR image saved at: {save_path}")

# Example usage
lr_image_path = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/DIV2K/DIV2K_train_LR_bicubic/X4/L9_ffaa4e2ca12ad56fe542cbcdb0de6b5bx4.jpg"  # Change this to your LR image path
generate_sr_image(lr_image_path, "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/results/m1/output_sr.png")
'''

# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import cv2
# import numpy as np
# from PIL import Image
# import os
# # Load the pre-trained generator model
# generator = load_model("/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/models/generator7_t6_15d/best_model.keras", safe_mode=False)

# # Load the EfficientNet model (without training)
# efficientnet = tf.keras.applications.EfficientNetB7(include_top=False, weights='imagenet')
# efficientnet.trainable = False

# # Preprocess the LR image
# def preprocess_lr_image(image_path):
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"The file {image_path} does not exist.")
#     lr_image = cv2.imread(image_path)
#     lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
#     lr_image = lr_image / 127.5 - 1  # Normalize LR image to [-1, 1]
#     lr_image = np.expand_dims(lr_image, axis=0)
#     return lr_image

# # Preprocess the HR image
# def preprocess_hr_image(image_path):
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"The file {image_path} does not exist.")
#     hr_image = cv2.imread(image_path)
#     hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
#     hr_image = hr_image / 255  # Normalize HR image to [0, 1]
#     hr_image = np.expand_dims(hr_image, axis=0)
#     return hr_image

# # Extract HR features using EfficientNet
# def extract_hr_features(hr_image):
#     hr_features = efficientnet(hr_image, training=False)
#     return hr_features

# # Generate SR image
# def generate_sr_image(lr_image_path, hr_image_path, output_path):
#     lr_image = preprocess_lr_image(lr_image_path)
#     hr_image = preprocess_hr_image(hr_image_path)
#     hr_features = extract_hr_features(hr_image)
    
#     # Ensure both inputs are tensors
#     lr_image = tf.convert_to_tensor(lr_image, dtype=tf.float32)
#     hr_features = tf.convert_to_tensor(hr_features, dtype=tf.float32)
    
#     sr_image = generator([lr_image, hr_features], training=False)
#     sr_image = (sr_image + 1) / 2.0  # Convert to [0, 1]
#     sr_image = np.squeeze(sr_image, axis=0)  # Remove batch dimension
#     sr_image = (sr_image * 255).astype(np.uint8)
#     sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
#     cv2.imwrite(output_path, sr_image)


## --Working Don't Change-- ##
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import cv2
# import numpy as np
# from PIL import Image
# import os
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import cv2
# import numpy as np
# from PIL import Image

# # Load the pre-trained generator model
# generator = load_model("/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/models/generator7_t1_15d/best_model.keras", safe_mode=False)

# # Load the EfficientNet model
# efficientnet = tf.keras.applications.EfficientNetB7(include_top=False, weights='imagenet')
# efficientnet.trainable = False

# # Preprocess the LR image
# def preprocess_lr_image(image_path):
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"The file {image_path} does not exist.")
#     lr_image = cv2.imread(image_path)
#     lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
#     lr_image = lr_image / 127.5 - 1  # Normalize LR image to [-1, 1]
#     lr_image = np.expand_dims(lr_image, axis=0)
#     return lr_image

# # Preprocess the HR image
# def preprocess_hr_image(image_path):
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"The file {image_path} does not exist.")
#     hr_image = cv2.imread(image_path)
#     hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
#     hr_image = hr_image / 255  # Normalize HR image to [0, 1]
#     hr_image = np.expand_dims(hr_image, axis=0)
#     return hr_image

# # Extract HR features using EfficientNet
# def extract_hr_features(hr_image):
#     hr_features = efficientnet(hr_image, training=False)
#     return hr_features

# # Generate SR image
# def generate_sr_image(lr_image_path, hr_image_path, output_path):
#     lr_image = preprocess_lr_image(lr_image_path)
#     hr_image = preprocess_hr_image(hr_image_path)
#     hr_features = extract_hr_features(hr_image)
    
#     # Ensure both inputs are tensors
#     lr_image = tf.convert_to_tensor(lr_image, dtype=tf.float32)
#     hr_features = tf.convert_to_tensor(hr_features, dtype=tf.float32)
    
#     sr_image = generator([lr_image, hr_features], training=False)
#     sr_image = (sr_image + 1) / 2.0  # Convert to [0, 1]
#     sr_image = np.squeeze(sr_image, axis=0)  # Remove batch dimension
#     sr_image = (sr_image * 255).astype(np.uint8)
#     sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
#     # cv2.imwrite(output_path, sr_image)
#     # Save as JPG
#     sr_image_pil = Image.fromarray(sr_image)
#     sr_image_pil.save(output_path, format='JPEG')

#     print('Completed...')

# # Example usage
# lr_image_path = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/DIV2K/DIV2K_train_LR_bicubic/X4/L9_ffaa4e2ca12ad56fe542cbcdb0de6b5bx4.jpg"
# hr_image_path = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/DIV2K/DIV2K_train_HR/L9_ffaa4e2ca12ad56fe542cbcdb0de6b5b.jpg"
# save_image_path = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/results/m1/output_sr_t1_15d.jpg"
# generate_sr_image(lr_image_path, hr_image_path, save_image_path)

## Working. Don't Change ##
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import cv2
# import numpy as np
# from PIL import Image
# import os
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import cv2
# import numpy as np


# # Preprocess the LR image
# def preprocess_lr_image(image_path):
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"The file {image_path} does not exist.")
#     lr_image = cv2.imread(image_path)
#     lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
#     lr_image = lr_image / 127.5 - 1  # Normalize LR image to [-1, 1]
#     lr_image = np.expand_dims(lr_image, axis=0)
#     return lr_image

# # Preprocess the HR image
# def preprocess_hr_image(image_path):
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"The file {image_path} does not exist.")
#     hr_image = cv2.imread(image_path)
#     hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
#     hr_image = hr_image / 255  # Normalize HR image to [0, 1]
#     hr_image = np.expand_dims(hr_image, axis=0)
#     return hr_image

# # Extract HR features using EfficientNet
# def extract_hr_features(hr_image):
#     hr_features = efficientnet(hr_image, training=False)
#     return hr_features

# # Generate SR image
# def generate_sr_image(lr_image_path, hr_image_path, output_path):
#     lr_image = preprocess_lr_image(lr_image_path)
#     hr_image = preprocess_hr_image(hr_image_path)
#     hr_features = extract_hr_features(hr_image)
    
#     # Ensure both inputs are tensors
#     lr_image = tf.convert_to_tensor(lr_image, dtype=tf.float32)
#     hr_features = tf.convert_to_tensor(hr_features, dtype=tf.float32)
    
#     sr_image = generator([lr_image, hr_features], training=False)
#     sr_image = (sr_image + 1) / 2.0  # Convert to [0, 1]
#     sr_image = np.squeeze(sr_image, axis=0)  # Remove batch dimension
#     sr_image = (sr_image * 255).astype(np.uint8)
#     sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
    
#     # Save as JPG using OpenCV
#     cv2.imwrite(output_path, sr_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])  # Save with quality 95

#     print('Completed...')

# # Load the pre-trained generator model
# generator = load_model("/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/models/generator_4/best_model.keras", safe_mode=False)

# # Load the EfficientNet model
# efficientnet = tf.keras.applications.EfficientNetB7(include_top=False, weights='imagenet')
# efficientnet.trainable = False


# # Example usage
# # lr_image_path = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/DIV2K/DIV2K_train_LR_bicubic/X4/L9_ffaa4e2ca12ad56fe542cbcdb0de6b5bx4.jpg"
# # lr_image_path = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/A1/train_LR/L8_5188d8809aa8745ff313a21fc57c2308.jpg"

# # hr_image_path = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/A1/train_HR/L8_5188d8809aa8745ff313a21fc57c2308.jpg"
# lr_image_path = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/A1/train_LR/L0_1b17b7008bd900a3ac5509d849a4e365.jpg"

# hr_image_path = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/A1/train_HR/L0_1b17b7008bd900a3ac5509d849a4e365.jpg"

# save_image_path = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/experiment/results1/1_L0_1b17b7008bd900a3ac5509d849a4e365.jpg"  # Use .jpg extension
# generate_sr_image(lr_image_path, hr_image_path, save_image_path)

import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
# Preprocess the LR image
def preprocess_lr_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The file {image_path} does not exist.")
    lr_image = cv2.imread(image_path)
    lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
    lr_image = lr_image / 127.5 - 1  # Normalize LR image to [-1, 1]
    lr_image = np.expand_dims(lr_image, axis=0)
    return lr_image

# Preprocess the HR image
def preprocess_hr_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The file {image_path} does not exist.")
    hr_image = cv2.imread(image_path)
    hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
    hr_image = hr_image / 255  # Normalize HR image to [0, 1]
    hr_image = np.expand_dims(hr_image, axis=0)
    return hr_image

# Extract HR features using EfficientNet
def extract_hr_features(hr_image):
    hr_features = efficientnet(hr_image, training=False)
    return hr_features

# # Generate SR image
# def generate_sr_image(lr_image_path, hr_image_path, output_path):
#     lr_image = preprocess_lr_image(lr_image_path)
#     hr_image = preprocess_hr_image(hr_image_path)
#     hr_features = extract_hr_features(hr_image)
    
#     # Ensure both inputs are tensors
#     lr_image = tf.convert_to_tensor(lr_image, dtype=tf.float32)
#     hr_features = tf.convert_to_tensor(hr_features, dtype=tf.float32)
    
#     sr_image = generator([lr_image, hr_features], training=False)
#     sr_image = (sr_image + 1) / 2.0  # Convert to [0, 1]
#     sr_image = np.squeeze(sr_image, axis=0)  # Remove batch dimension
#     sr_image = (sr_image * 255).astype(np.uint8)
#     sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
    
#     # Save as JPG using OpenCV
#     cv2.imwrite(output_path, sr_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])  # Save with quality 95
#############

# # Generate SR image
def generate_sr_image(lr_image_path, hr_image_path, output_path):
    lr_image = preprocess_lr_image(lr_image_path)
    hr_image = preprocess_hr_image(hr_image_path)
    hr_features = extract_hr_features(hr_image)
    
    # Ensure both inputs are tensors
    lr_image = tf.convert_to_tensor(lr_image, dtype=tf.float32)
    hr_features = tf.convert_to_tensor(hr_features, dtype=tf.float32)
    
    sr_image = generator([lr_image, hr_features], training=False)
    sr_image = (sr_image + 1) / 2.0  # Convert to [0, 1]
    sr_image = np.squeeze(sr_image, axis=0)  # Remove batch dimension
    sr_image = (sr_image * 255).astype(np.uint8)
    sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(output_path, sr_image)
    # Save as JPG
    sr_image_pil = Image.fromarray(sr_image)
    sr_image_pil.save(output_path, format='JPEG')

    print('Completed...')

# #################
# Load the pre-trained generator model
generator = load_model("/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/models2/generator7_t1_14d/best_model.keras", safe_mode=False)

# Load the EfficientNet model
efficientnet = tf.keras.applications.EfficientNetB7(include_top=False, weights='imagenet')
efficientnet.trainable = False

# Directories
lr_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/A1/train_LR"
hr_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/A1/train_HR"
output_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/experiment/conference/gen7_t1_14d"

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through each file in the LR directory
for lr_file in os.listdir(lr_dir):
    if lr_file.endswith(".jpg"):  # Process only JPG files
        lr_image_path = os.path.join(lr_dir, lr_file)
        hr_image_path = os.path.join(hr_dir, lr_file)  # Corresponding HR image
        output_path = os.path.join(output_dir, f"1_{lr_file}")  # Save SR image with modified name
        
        try:
            generate_sr_image(lr_image_path, hr_image_path, output_path)
            print(f"Generated SR image for {lr_file} and saved to {output_path}")
        except Exception as e:
            print(f"Error processing {lr_file}: {e}")
