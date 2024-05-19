import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
data_train_path = "C:\Users\iaade\PycharmProjects\Animla classiications\Fruits_Vegetables\train"
data_test_path = "C:\Users\iaade\PycharmProjects\Animla classiications\Fruits_Vegetables\test"
data_train_val = "C:\Users\iaade\PycharmProjects\Animla classiications\Fruits_Vegetables\validation"
image_width = 180
image_height = 180
#bring data set in form of arrays

data_train = tf._tf_uses_legacy_keras.utils.