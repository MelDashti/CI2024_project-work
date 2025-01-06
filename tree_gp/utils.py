import numpy as np

def load_dataset(filepath="data/problem_0.npz", test_split=0.2):
    """
    Load the dataset and split it into training and test sets.
    Returns x_train, y_train, x_test, y_test.
    """
    data = np.load(filepath)
    x = data['x']  # shape: (n_features, n_samples)
    y = data['y']  # shape: (n_samples,)

    n_samples = y.shape[0]
    n_test = int(n_samples * test_split)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    x_train, x_test = x[:, indices[:-n_test]], x[:, indices[-n_test:]]
    y_train, y_test = y[indices[:-n_test]], y[indices[-n_test:]]
    return x_train, y_train, x_test, y_test

def mean_squared_error(gp_tree, X, y):
    """
    Computes MSE using vectorized evaluation.
    X: shape (n_features, n_samples)
    y: shape (n_samples,)
    """
    preds = gp_tree.evaluate_vectorized(X)  # shape (n_samples,)
    return np.mean((preds - y) ** 2)

def calculate_mse(predictions, true_values):
    return 100 * np.square(predictions - true_values).sum() / len(true_values)
