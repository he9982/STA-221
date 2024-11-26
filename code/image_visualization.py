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
from collections import Counter



def display_image(image_paths, labels, num_images = 5):
    plt.figure(figsize=(5,5))

    for x in range(num_images):
        plt.subplot(3,3, x+1)
        image_paths = image_paths[x]
        image = load_img(image_paths)
        plt.imshow(image)
        plt.title(f"Label: {labels[x]}")
        plt.axis('off')
        
def convert_image(df, image_folder_path, label = 1):
    x = []
    y = []
    df = df[df['target'] == label]
    for index, row in df.iterrows():
        image_name = f"{row['isic_id']}.jpg"
        label = row['target']
        image_path = os.path.join(image_folder_path, image_name)
        if not os.path.exists(image_path):
            continue
        image = load_img(image_path, color_mode = 'grayscale', target_size = (128,128))
        image_array = img_to_array(image)
        image_faltten = image_array.flatten()
        x.append(image_faltten)
        y.append(label)
    return x, y

def convert_gan_image(image_folder_path):
    x = []
    y = []

    for image_name in os.listdir(image_folder_path):
        if image_name.endswith('.jpg'):
            image_path = os.path.join(image_folder_path, image_name)
            image = load_img(image_path, color_mode = 'grayscale', target_size=(128, 128))
            label = 1
            image_array = img_to_array(image)
            image_flatten = image_array.flatten()
            x.append(image_flatten)
            y.append(label)
    return x, y  # Ensure this is outside the loop


    
if __name__ == '__main__':
    image_folder_path = '/Users/kuangli/Desktop/UCD/Fall_2024/STA_221/Vision/data/train-image/image'
    image_folder_path_gan = '/Users/kuangli/Desktop/UCD/Fall_2024/STA_221/Vision/data/train_te'
    image_folder_path_test = '/Users/kuangli/Desktop/UCD/Fall_2024/STA_221/Vision/data/test_te'
    dtype_spec = {'column_name_51': str, 'column_name_52': str}  # replace with actual column names
    df = pd.read_csv('Desktop/UCD/Fall_2024/STA_221/Vision/data/train-metadata.csv', dtype=dtype_spec, low_memory = False)
    df1 = df[['isic_id','target']]
    df1 = df1[df1['target'] == 1]
    malignant_id = df1
    train_id = malignant_id[:293]
    train293, label293 = convert_image(train_id, image_folder_path, label = 1)
    #downsample 0 data
    downsample_df = downsampling(df, num = 9)
    # Convert images (real and synthetic)
    x_real0, y_real0 = convert_image(downsample_df, image_folder_path, label=0)
    xgan, ygan = convert_gan_image(image_folder_path_gan)
    # Process testing data for target=1
    x_target1_test, y_target1_test = convert_image(df=downsample_df, image_folder_path=image_folder_path_test, label=1)
    # Split real data for label=0 into training and testing sets
    x_real0_train, x_real0_test, y_real0_train, y_real0_test = train_test_split(
        x_real0, y_real0, test_size=0.2, random_state=117759, stratify=y_real0)
    
    # Combine real training data (label=0) with GAN data (label=1)
    X_train = np.concatenate((x_real0_train, xgan, train293), axis=0)
    y_train = np.concatenate((y_real0_train, ygan, label293), axis=0)

    #combine test
    X_test = np.concatenate((x_real0_test, x_target1_test), axis=0)
    y_test = np.concatenate((y_real0_test, y_target1_test), axis=0)
    # Use only real data for testing
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Debugging: Verify dataset composition
    print(f"Training set distribution: {Counter(y_train)}")
    print(f"Testing set distribution: {Counter(y_test)}")
    print(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")


    joblib.dump((X_train, X_test, y_train, y_test), '/Users/kuangli/Desktop/UCD/Fall_2024/STA_221/Vision/split_data.pkl')

    



