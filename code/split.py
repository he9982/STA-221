from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
from checksampling import downsampling,checksampling
from display_image import convert_image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np


def split_data(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)
    joblib.dump((X_train, X_test, y_train, y_test), '/Users/kuangli/Desktop/UCD/Fall_2024/STA_221/Vision/split_data.pkl')

def load_train():
    X_train, _, y_train, _ = joblib.load('/Users/kuangli/Desktop/UCD/Fall_2024/STA_221/Vision/split_data.pkl')
    return X_train, y_train

def load_test():
    _, X_test, _, y_test = joblib.load('/Users/kuangli/Desktop/UCD/Fall_2024/STA_221/Vision/split_data.pkl')
    return X_test, y_test

if __name__ == '__main__':
    image_folder_path = '/Users/kuangli/Desktop/UCD/Fall_2024/STA_221/Vision/data/train-image/image'
    dtype_spec = {'column_name_51': str, 'column_name_52': str}  # replace with actual column names

    df = pd.read_csv('Desktop/UCD/Fall_2024/STA_221/Vision/data/train-metadata.csv', dtype=dtype_spec, low_memory = False)
    downsample_df = downsampling(df, num = 3)
    x,y = convert_image(downsample_df, image_folder_path)
    x = np.array(x)
    y = np.array(y)
    split_data(x,y)

    # Load train and test data separately
    X_train, y_train = load_train()
    X_test, y_test = load_test()
    
    print('Shape of train:', len(X_train))
    print('Shape of test:', len(X_test))
    print(f'Loaded train data size: {len(X_train)}')