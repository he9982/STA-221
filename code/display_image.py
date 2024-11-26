import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import h5py
from PIL import Image
import io
import joblib
from checksampling import downsampling,checksampling


def display_sample_images(image_paths, labels, num_images = 5):
    plt.figure(figsize=(5,5))

    for x in range(num_images):
        plt.subplot(3,3, x+1)
        image_path = image_paths[x]
        image = load_img(image_path, color_mode = 'grayscale')
        plt.imshow(image, cmap = 'gray')
        plt.title(f"Label: {labels[x]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def convert_image(df, image_folder_path):
    x = []
    y = []

    for index, row in df.iterrows():
        image_name = f"{row['isic_id']}.jpg"
        label = row['target']
        image_path = os.path.join(image_folder_path, image_name)
        image = load_img(image_path, color_mode = 'grayscale', target_size = (128,128))
        image_array = img_to_array(image)
        image_faltten = image_array.flatten()
        x.append(image_faltten)
        y.append(label)
    return x, y



if __name__ == '__main__':
    image_folder_path = '/Users/kuangli/Desktop/UCD/Fall_2024/STA_221/Vision/data/train-image/image'
    dtype_spec = {'column_name_51': str, 'column_name_52': str}  # replace with actual column names

    df = pd.read_csv('Desktop/UCD/Fall_2024/STA_221/Vision/data/train-metadata.csv', dtype=dtype_spec, low_memory = False)
    downsample_df = downsampling(df, num = 3)
    dist = checksampling(downsample_df)
    print(dist)
      # Filter for Label: 1
    downsample_df1 = downsample_df[downsample_df['target'] == 1]
    if not downsample_df1.empty:
        sample_image_paths = [os.path.join(image_folder_path, f"{row['isic_id']}.jpg") for _, row in downsample_df.iterrows()]
        sample_labels = downsample_df['target'].tolist()
        display_sample_images(sample_image_paths, sample_labels, num_images = 9)

    # fileter label = 0
    downsample_df0 = downsample_df[downsample_df['target'] == 0]
    if not downsample_df0.empty:

        downsample_df0 = downsample_df[downsample_df['target'] == 0]
        sample_image_paths_0 = [os.path.join(image_folder_path, f"{row['isic_id']}.jpg") for _, row in downsample_df0.iterrows()]
        sample_labels_0 = downsample_df0['target'].tolist()
        display_sample_images(sample_image_paths_0, sample_labels_0, num_images = 9)

    x,y = convert_image(downsample_df, image_folder_path)
    print(x)