import tensorflow as tf
import cv2
import numpy as np

from tensorflow.keras.models import load_model


model=load_model('CNN_model.h5')

from tkinter import filedialog
from tensorflow.keras.preprocessing import image

def browse_file():
    global select_dir
    browse = filedialog.askopenfile()
    try:
        select_dir = (browse.name)
        return (select_dir)
    except:
        return

def file_location():
    return select_dir

run=True


if run is True:

    while True:
        X_pred = []
        browse_file()
        #one_root_img=(os.path.join(train_dir,train_dir_class[0],train_one_data[1]))
        img_location=file_location()
        img=cv2.imread(img_location,cv2.IMREAD_GRAYSCALE)
        new_array=cv2.resize(img,(130,130))
        x=new_array.reshape(-1,130,130,1)
        classes=model.predict(x)
        print(classes[0])
        if classes[0][0] ==1:
            print("1root")
        else:
            print("2 or more root")
