import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class CalsNetRandomForest:
    def __init__(self, criterion='gini', n_estimators=200, max_depth=20, random_state=42, min_samples_leaf=1, 
    min_samples_split=2):
        self.model = model if model else RandomForestClassifier(criterion=criterion, 
                                                                n_estimators=n_estimators, 
                                                                max_depth=max_depth, 
                                                                random_state=random_state,
                                                                min_samples_leaf=min_samples_leaf, 
                                                                min_samples_split=min_samples_split)
    
    def train(self, x_train, y_train):
        """Trains the model with the provided training data."""
        self.model.fit(x_train, y_train)
    
    def predict(self, x_test):
         """Predicts the target values for the given test data later."""
        return self.model.predict(x_test)
    
    def evaluate(self, x_test, y_test):
        """Evaluates the model using accuracy, classification report, and confusion matrix."""
        predictions = self.predict(x_test)
        acc = accuracy_score(y_test, predictions)
        evaluation = {
            "Accuracy": acc,
            "Classification Report": classification_report(y_test, predictions),
            "Confusion Matrix": confusion_matrix(y_test, predictions)
        }
        return evaluation
    
    def save(self, path='ranfor_model.pkl'):
        """Saves the trained model to a specified path."""
        try:
            with open(path, 'wb') as f:
                pickle.dump(self.model, f)
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load(self, path='CL_ranfor_model.pkl'):
        """Loads a model from a specified path."""
        try:
            with open(path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded from {path}")
        except Exception as e:
            print(f"Error loading model: {e}")
