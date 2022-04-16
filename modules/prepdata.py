import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from pathlib import Path
import os.path
import PIL

#Convert folder to dataframe of images' paths & labels
def get_paths_labels(path, allowed_extension="jpg"):
        global Path
        images_dir = Path(path)
        
        filepaths = pd.Series((images_dir.glob(fr'**/*.{allowed_extension}'))).astype(str)
        filepaths.name = "path"
        
        labels = filepaths.str.split("/")[:].str[-2]
        labels.name = "label"

        # Concatenate filepaths and labels
        df = pd.concat([filepaths, labels], axis=1)

        # Get num code for Cetegories labels
        df.label = pd.Categorical(df.label)
        df['code'] = df.label.cat.codes
        
        # Shuffle the DataFrame and reset index
        df = df.sample(frac=1).reset_index(drop = True)
        return df

# Read image
def get_image(path, shape):
    image = cv2.imread(path)
    image = cv2.resize(image, shape)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image = image/255
    #image = cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
    return gray

#concat df with same columns
def get_df(path_arr):
    datalst=[]
    for path in path_arr:
        df = get_paths_labels(path)
        datalst.append(df)
    # Combine both datasets
    #dataset = pd.concat((train_df, sec_df))
    dataset = pd.concat(datalst, ignore_index=True, sort=False)
    return dataset


#load the data with the format needed in the AC-GAN implementation
def set_data(data_col,data_labels,input_rows, input_cols,input_channels):
    
    lst=[]
    for file in data_col:
        arr=get_image(file, (input_rows, input_cols))
        lst.append(arr)
    X=np.stack(lst)
    # normalize and reshape  set
    X = (X.astype(np.float32) - 127.5) / 127.5
    #changes shape from (28,28) to (28,28,1)
    X = np.expand_dims(X, axis=-1)

    y= data_labels.values
  

    return X,y

def load_data(dataset, input_rows, input_cols,input_channels):
    
    train,test = train_test_split(dataset, test_size=0.2, stratify=dataset["label"])
    x_train,y_train = set_data(train["path"], train["code"], input_rows, input_cols, input_channels)
    x_test,y_test = set_data(test["path"], test["code"], input_rows, input_cols,input_channels)
    return (x_train,y_train), (x_test,y_test)
       