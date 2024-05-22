"""
K-Nearest Neighbors (KNN) Classifier - Naive Implementation (Without using existing KNN libraries)

This script implements a simple K-Nearest Neighbors classifier for predicting online shopping behavior (whether online shopping customers will complete a purchase).
It is based on the programming exercise in "Shopping - CS50's Introduction to Artificial Intelligence with Python" course.
Course link: https://cs50.harvard.edu/college/2024/fall/

Author: Reza Barzegar Nozari;
Date: May 2024
"""

# Import preminary liberaries
import csv
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# Define the month mapping
MONTHS = {
    "Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "June": 5,
    "Jul": 6, "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11
}

def load_data(filename):
    """
    Load shopping data from a CSV file and return evidence and labels.

    Parameters:
    filename (str): The name of the CSV file to read data from.

    Returns:
    tuple: A tuple containing two lists:
        - evidence: A list of lists where each inner list contains feature data.
        - labels: A list of integers where each integer is the label for the corresponding evidence.
    """
    evidence = []
    labels = []

    try:
        with open(filename, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Process the evidence
                evidence.append([
                    int(row["Administrative"]),
                    float(row["Administrative_Duration"]),
                    int(row["Informational"]),
                    float(row["Informational_Duration"]),
                    int(row["ProductRelated"]),
                    float(row["ProductRelated_Duration"]),
                    float(row["BounceRates"]),
                    float(row["ExitRates"]),
                    float(row["PageValues"]),
                    float(row["SpecialDay"]),
                    MONTHS[row["Month"]],
                    int(row["OperatingSystems"]),
                    int(row["Browser"]),
                    int(row["Region"]),
                    int(row["TrafficType"]),
                    1 if row["VisitorType"] == "Returning_Visitor" else 0,
                    1 if row["Weekend"] == "TRUE" else 0
                ])
                # Process the labels
                labels.append(1 if row["Revenue"] == "TRUE" else 0)

    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
    except KeyError as e:
        print(f"Error: Missing expected column {e} in the input file.")
    except ValueError as e:
        print(f"Error: Incorrect data format in the input file: {e}")

    return (evidence, labels)


class KNNClassifier:
    """
    K-Nearest Neighbors (KNN) Classifier.

    This class implements the KNN algorithm for classification tasks without using existing libraries.
    Given a set of training data points with labels, it predicts the
    label of a new data point by considering the labels of its nearest neighbors.

    Parameters:
    k (int): Number of nearest neighbors to consider for classification.
             Default is 1, meaning it considers only the nearest neighbor.

    Attributes:
    X_train (ndarray): Array of shape (n_samples, n_features) containing the training data.
    y_train (ndarray): Array of shape (n_samples,) containing the labels for the training data.
    """
    def __init__(self, k=1):
        """
        Initialize the KNNClassifier.

        Parameters:
        k (int): Must be a positive integer.
        """
        if k < 1:
            raise ValueError("k must be a positive integer")
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """
        Fit the KNN classifier with the training data.

        Parameters:
        X_train (array-like): Training data features.
        y_train (array-like): Training data labels.
        
        Raises:
        ValueError: If the number of training samples and labels are not the same.
        """
        if len(X_train) != len(y_train):
            raise ValueError("The number of training samples and labels must be the same")
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def euclidean_distance(self, a, b):
        """
        Calculate the Euclidean distance between two points.

        Parameters:
        a (array-like): The first point.
        b (array-like): The second point.

        Returns:
        float: The Euclidean distance between the two points.
        """
        return np.sqrt(np.sum((a - b) ** 2))

    def get_neighbors(self, test_point):
        """
        Find the k nearest neighbors for a given test point.

        Parameters:
        test_point (array-like): The test point to find neighbors for.

        Returns:
        list: A list of labels of the k nearest neighbors.
        """
        distances = []
        for i in range(len(self.X_train)):
            dist = self.euclidean_distance(self.X_train[i], test_point)
            distances.append((self.y_train[i], dist))
        distances.sort(key=lambda x: x[1])
        neighbors = distances[:self.k]
        return [neighbor[0] for neighbor in neighbors]

    def _predict(self, test_point):
        """
        Predict the label for a single test point.

        Parameters:
        test_point (array-like): The test point to predict the label for.

        Returns:
        int/float: The predicted label for the test point.
        """
        neighbors = self.get_neighbors(test_point)
        majority_vote = Counter(neighbors).most_common(1)
        return majority_vote[0][0]

    def predict(self, X_test):
        """
        Predict the labels for a set of test data.

        Parameters:
        X_test (array-like): Test data features.

        Returns:
        np.array: The predicted labels for the test data.

        Raises:
        ValueError: If the model has not been trained.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("The model has not been trained yet. Call `fit` before `predict`.")
        predictions = [self._predict(test_point) for test_point in X_test]
        return np.array(predictions)


def evaluate(labels, predictions):
    """
    Evaluate the performance of predictions against true labels.

    Parameters:
    labels (list or np.array): True class labels.
    predictions (list or np.array): Predicted class labels.

    Returns:
    dict: A dictionary containing the following metrics:
        - 'Correct' (int): Number of correct predictions.
        - 'Incorrect' (int): Number of incorrect predictions.
        - 'Sensitivity' (float): True Positive Rate (Recall).
        - 'Specificity' (float): True Negative Rate.
        - 'Precision' (float): Positive Predictive Value.
        - 'F1 Score' (float): Harmonic mean of precision and recall.
        - 'Accuracy' (float): Overall accuracy of predictions.
    """
    # Convert lists to NumPy arrays if necessary
    labels = np.array(labels)
    predictions = np.array(predictions)

    # Calculate true positives, true negatives, false positives, and false negatives
    tp = np.sum((labels == 1) & (predictions == 1))
    tn = np.sum((labels == 0) & (predictions == 0))
    fp = np.sum((labels == 0) & (predictions == 1))
    fn = np.sum((labels == 1) & (predictions == 0))

    # Calculate metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    # Calculate correct and incorrect counts
    corrects = int(tp + tn)
    incorrects = int(fp + fn)

    return {
        'Correct':corrects,
        'Incorrect':incorrects,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'F1 Score': f1_score,
        'Accuracy': accuracy
    }


# Main function
def main():
    # Load data
    filename = "shopping.csv"
    evidence, labels = load_data(filename)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(evidence, labels, test_size=0.4, random_state=2)

    # Create and train the KNN classifier
    model = KNNClassifier(k=1)
    model.fit(X_train, y_train)

    # Make predictions
    predictions = []
    total_samples = len(X_test)
    for sample in tqdm(X_test, desc="Predicting", unit=" samples", ncols=100):
        prediction = model.predict([sample])
        predictions.extend(prediction)  # Extend the list with the predictions

    # Evaluate the model
    metrics = evaluate(y_test, predictions)

    # Print results
    print('Evaluation Result:')
    for metric, value in metrics.items():
        if isinstance(value, int):  # Correct and Incorrect counts
            print(f"{metric}: {value}")
        else:  # Floating point metrics
            print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()

