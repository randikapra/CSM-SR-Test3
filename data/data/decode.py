import cv2
from dataset_creation.dataset import LMDBDataset # type: ignore
import numpy as np
def decode_and_save_image(lmdb_path, target_shape, save_path):
    dataset = LMDBDataset(lmdb_path, target_shape)
    sample_image = dataset[0]  # Get the first image in the dataset

    # Convert the sample image back to BGR for saving
    sample_image_bgr = cv2.cvtColor((sample_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Save the image using OpenCV at maximum quality (if applicable)
    cv2.imwrite(save_path, sample_image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 2000])
    print(f'Saved decoded image to {save_path} with shape: {sample_image_bgr.shape}, dtype: {sample_image_bgr.dtype}')
