import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

class predict_class:
    
    def __init__(self) -> None:
        pass

    def predict_model(self,csv_path):
        df = pd.read_csv(csv_path)
        x=df
        x=x.drop('label', axis=1).values
        
        with open('models\classifier_model', 'rb') as f:
            model = pickle.load(f)
        
        activity = model.predict(x)
        
        return activity[0]