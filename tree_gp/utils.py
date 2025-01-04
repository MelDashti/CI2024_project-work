import numpy as np

def load_dataset(filepath):
    data = np.load(filepath)
    x = data['x']
    y = data['y']
    return x, y

def mean_squared_error(gp_tree, X, y):
    """
    Calculate the mean squared error between the tree's predictions and true values.
    gp_tree: A GPTree instance
    X: a 2D array of shape (num_samples, num_features)
    y: a 1D array of shape (num_samples,)
    """
    predictions = []
    for i in range(len(X)):
        pred = gp_tree.evaluate(X[i])  # Evaluate tree for the i-th sample
        predictions.append(pred)
    predictions = np.array(predictions)
    mse = np.mean((predictions - y)**2)
    return mse


def calculate_mse(predictions, true_values):
   return 100*np.square(predictions-true_values).sum()/ len(true_values)