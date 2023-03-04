import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

class train_class:
    
    def __init__(self) -> None:
        pass

    def train_model(self,csv_path):
        # Load the pose data from the CSV file
        df = pd.read_csv(csv_path)

        # Split the data into features and labels
        X = df.drop('label', axis=1).values
        y = df['label'].values

        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(X_train, y_train)

        # Evaluate the model
        accuracy = model.score(X_val, y_val)

        # Save the model to disk
        with open('models/classifier_model', 'wb') as f:
            pickle.dump(model, f)

        # Return the model accuracy as a JSON response
        print(accuracy)
        # return jsonify({'accuracy': accuracy})
        return accuracy
