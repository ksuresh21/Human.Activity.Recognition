import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

# Set up the directories and file paths
base_dir = "data"
raw_dir = os.path.join(base_dir, "raw")
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")
train_csv_path = os.path.join(train_dir, "train.csv")
test_csv_path = os.path.join(test_dir, "test.csv")

class PoseExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, pose_model, body_dict, model_path='./models/pose.tflite'):
        self.pose_model = pose_model
        self.body_dict = body_dict
        self.model_path = model_path
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        _, self.input_dim, _, _ = self.input_details[0]['shape']
        _, self.mp_dim, _, self.ky_pt_num = self.output_details[0]['shape']
    
    def fit(self, x, y=None):
        return self
    
    def extract(self, x):
        feature_array = []
        file_path = True if isinstance(x[0], str) else False
        for img in x:
            # Read the image from file path or numpy array and resize it for model
            image = Image.open(img) if file_path else Image.fromarray(img)
            image = image.resize((self.input_details[0]['shape'][1], self.input_details[0]['shape'][2]), Image.NEAREST)
            image = np.expand_dims(np.asarray(image).astype(self.input_details[0]['dtype'])[:, :, :3], axis=0)
            # Get pose data from the image
            self.interpreter.set_tensor(self.input_details[0]['index'], image)
            self.interpreter.invoke()
            results = self.interpreter.get_tensor(self.output_details[0]['index'])
            # Get feature array from the results
            result = results.reshape(1, self.mp_dim**2, self.ky_pt_num)
            max_indices = np.argmax(result, axis=1)
            coordinates = list(map(lambda x: divmod(x, self.mp_dim), max_indices))
            feature_vector = np.vstack(coordinates).T.reshape(2 * self.ky_pt_num, 1)
            feature_array.append(feature_vector)
        return feature_array
