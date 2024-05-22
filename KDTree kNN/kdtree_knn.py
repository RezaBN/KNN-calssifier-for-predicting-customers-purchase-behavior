"""
K-Nearest Neighbors (KNN) Classifier - KDTree Implementation (Without using existing libraries of KNN and KDTree)

This script implements a simple K-Nearest Neighbors classifier for predicting online shopping behavior (whether online shopping customers will complete a purchase).
It is based on the programming exercise in "Shopping - CS50's Introduction to Artificial Intelligence with Python" course.
https://cs50.harvard.edu/college/2024/fall/

Description:
- KDTree implementation uses a KDTree for efficient nearest neighbor searches.

Author: Reza Barzegar Nozari
Date: May 2024
"""

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


class KDTreeNode:
    """
    Node structure for a KDTree.

    Each node represents a point in the KDTree and contains references
    to its left and right children in the tree.

    Parameters:
    point (list of float): The coordinates of the point represented by the node.
    label (any): The label or class associated with the point.
    left (KDTreeNode): The left child node in the KDTree. Default is None.
    right (KDTreeNode): The right child node in the KDTree. Default is None.
    """
    def __init__(self, point, label, left=None, right=None):
        self.point = point
        self.label = label
        self.left = left
        self.right = right

class KDTree:
    """
    KDTree data structure for efficient k-nearest neighbor search.

    A KDTree is a binary tree structure used for organizing points in a k-dimensional space.
    It facilitates efficient nearest neighbor queries and range searches.

    Parameters:
    points (list of list of float): The points to build the KDTree with.
    labels (list): The labels corresponding to the points.
    depth (int): The current depth in the tree (used for recursion).

    Attributes:
    root (KDTreeNode): The root node of the KDTree.
    """
    def __init__(self, points, labels, depth=0):
        """
        Initializes a KDTree with given points and labels.
        
        Parameters:
        points (list of list of float): The points to build the KDTree with.
        labels (list): The labels corresponding to the points.
        depth (int): The current depth in the tree (used for recursion).
        """
        self.n = len(points)
        if self.n <= 0:
            self.root = None
        else:
            k = len(points[0])  # assumes all points have the same dimension
            axis = depth % k

            sorted_points = sorted(zip(points, labels), key=lambda x: x[0][axis])
            median = self.n // 2

            self.root = KDTreeNode(
                point=sorted_points[median][0],
                label=sorted_points[median][1],
                left=KDTree(
                    points=[x[0] for x in sorted_points[:median]],
                    labels=[x[1] for x in sorted_points[:median]],
                    depth=depth + 1
                ).root,
                right=KDTree(
                    points=[x[0] for x in sorted_points[median+1:]],
                    labels=[x[1] for x in sorted_points[median+1:]],
                    depth=depth + 1
                ).root
            )

    def query(self, point, k=1):
        """
        Find the k nearest neighbors of a given point.

        Parameters:
        point (list of float): The point to find the nearest neighbors for.
        k (int): The number of nearest neighbors to find.

        Returns:
        list: A list of labels of the k nearest neighbors.
        """
        neighbors = []

        def recursive_search(node, depth=0):
            if node is None:
                return

            axis = depth % len(point)
            distance = self.distance_sq(point, node.point)
            if len(neighbors) < k:
                neighbors.append((distance, node))
                neighbors.sort(key=lambda x: x[0])
            elif distance < neighbors[-1][0]:
                neighbors[-1] = (distance, node)
                neighbors.sort(key=lambda x: x[0])

            diff = point[axis] - node.point[axis]
            close, away = (node.left, node.right) if diff < 0 else (node.right, node.left)

            recursive_search(close, depth + 1)
            if diff ** 2 < neighbors[-1][0] or len(neighbors) < k:
                recursive_search(away, depth + 1)

        recursive_search(self.root)
        return [neighbor[1].label for neighbor in neighbors]

    @staticmethod
    def distance_sq(point1, point2):
        """
        Calculate the squared Euclidean distance between two points.

        Parameters:
        point1 (list of float): The first point.
        point2 (list of float): The second point.

        Returns:
        float: The squared Euclidean distance.
        """
        return sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2))

    def closer_distance(self, point, p1, p2):
        """
        Determine which of two nodes is closer to the given point.

        Parameters:
        point (list of float): The point to compare distance to.
        p1 (KDTreeNode): The first KDTreeNode to compare.
        p2 (KDTreeNode): The second KDTreeNode to compare.

        Returns:
        KDTreeNode: The node that is closer to the given point.
        """
        if not p1:
            return p2
        if not p2:
            return p1

        d1 = self.distance_sq(point, p1.point)
        d2 = self.distance_sq(point, p2.point)

        if d1 < d2:
            return p1
        else:
            return p2


class KNNClassifier:
    """
    K-Nearest Neighbors (KNN) classifier.

    Parameters:
    k (int): Number of neighbors to consider.

    Attributes:
    k (int): Number of neighbors to consider.
    tree (KDTree): KDTree data structure for efficient nearest neighbor search.
    """
    def __init__(self, k=1):
        """
        Initialize the KNNClassifier.

        Parameters:
        k (int): Number of neighbors to consider.
        """
        self.k = k
        self.tree = None

    def fit(self, X_train, y_train):
        """
        Fit the KNNClassifier to the training data.

        Parameters:
        X_train (array-like): Training data features.
        y_train (array-like): Training data labels.
        """
        if not X_train or not y_train:
            raise ValueError("Training data cannot be empty.")
        self.tree = KDTree(X_train, y_train)

    def predict(self, X_test):
        """
        Predict the labels for the test data.

        Parameters:
        X_test (array-like): Test data features.

        Returns:
        np.array: Predicted labels for the test data.
        """
        if not self.tree:
            raise ValueError("Model has not been trained. Call fit() before predict().")
        if not X_test:
            raise ValueError("Test data cannot be empty.")
        
        predictions = []
        for point in X_test:
            neighbors = self.tree.query(point, k=self.k)
            majority_vote = Counter(neighbors).most_common(1)[0][0]
            predictions.append(majority_vote)
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

