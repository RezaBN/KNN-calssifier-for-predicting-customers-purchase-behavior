# K-Nearest Neighbors (KNN) Classifier for Predicting Customers Purchase behavior

This repository contains two implementations of a K-Nearest Neighbors (KNN) classifier for predicting online shopping behavior based on the programming exercise in "Shopping - CS50's Introduction to Artificial Intelligence with Python" course.
https://cs50.harvard.edu/college/2024/fall/.

 * The project description is detailed in the file "Shopping - CS50's Introduction to Artificial Intelligence with Python.pdf"

The classifiers are implemented in Python and use different approaches for finding the nearest neighbors:

1. **Naive Implementation**: A straightforward approach using a brute-force method to compute Euclidean distances.
2. **KDTree Implementation**: An optimized approach using a KDTree for efficient nearest neighbor searches.

 ### ***Both models were implemented from scratch without using existing libraries for KNN or KDTree.***
 
## Dataset

The dataset used in this project is `shopping.csv`, which contains information about online shopping sessions. Each session is described by several features, and the goal is to predict whether the user made a purchase (Revenue).
It is provided by Sakar et al. (2018) (https://link.springer.com/article/10.1007%2Fs00521-018-3523-0)

## Getting Started

### Prerequisites

Make sure you have Python installed. You can download it from [python.org](https://www.python.org/).

### Installation

1. Clone this repository to your local machine:
    ```sh
    git clone https://github.com/RezaBN/KNN-classifier-for-predicting-customers-purchase-behavior.git
    cd KNN-classifier-for-predicting-customers-purchase-behavior
    ```

2. (Optional) Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Running the Code

1. Ensure you have the `shopping.csv` dataset in the same directory as the code files.

2. Run the naive implementation:
    ```sh
    python naive_knn.py
    ```

3. Run the KDTree implementation:
    ```sh
    python kdtree_knn.py
    ```

### Naive Implementation

The naive implementation uses a brute-force method to compute Euclidean distances between points and find the nearest neighbors.

- **File**: `naive_knn.py`
- **Class**: `KNNClassifier`
- **Functionality**: Loads data, splits it into training and testing sets, trains the KNN classifier, makes predictions, evaluates the results, and prints performance metrics.

### KDTree Implementation

The KDTree implementation uses a KDTree for efficient nearest neighbor searches, which improves the performance for large datasets.

- **File**: `kdtree_knn.py`
- **Class**: `KDTree`, `KDTreeNode`, `KNNClassifier`
- **Functionality**: Similar to the naive implementation but uses a KDTree for neighbor searches to optimize performance.

## Evaluation Metrics

Both implementations evaluate the model using the following metrics:
- Correct (Number of correct predictions)
- Incorrect (Number of incorrect predictions)
- Sensitivity (True Positive Rate)
- Specificity (True Negative Rate)
- Precision (Positive Predictive Value)
- F1 Score (Harmonic mean of precision and recall)
- Accuracy

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


Feel free to contribute to this project by submitting issues or pull requests. For major changes, please open an issue first to discuss what you would like to change.

---

Happy coding!
