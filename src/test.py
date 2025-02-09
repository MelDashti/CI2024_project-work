import numpy as np

def load_dataset(filepath, test_split=0.2):
    data = np.load(filepath)
    x = data['x']  # shape (n_features, n_samples)
    y = data['y']  # shape (n_samples,)

    n_samples = y.shape[0]
    n_test = int(n_samples * test_split)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    x_train, x_test = x[:, indices[:-n_test]], x[:, indices[-n_test:]]
    y_train, y_test = y[indices[:-n_test]], y[indices[-n_test:]]
    return x_train, y_train, x_test, y_test

def calculate_mse(predictions, true_values):
    return 100 * np.square(predictions - true_values).sum() / len(true_values)

def f0(x: np.ndarray) -> np.ndarray:
    """
    The provided formula implementation
    """
    return (
        (2.9280*x[0] + x[1] + x[2]) * 
        (x[0] * (24.8006 - 5.4439 * np.cos(1.5377 * 
            (-2.0512 * np.cos(2.1086 / x[0]) - 
             2.0512 * np.cos(1.3159 * x[0] + 18.8110) + 
             3.5862) / x[0])) *
        (2.5198*x[0] + x[1] - 0.2847) *
        (x[1] * (-x[0] - x[2]) + 2*x[2] + 1) -
        2.6333 * (x[2]**3 + x[2] * (-4.0960*x[0] + 
            0.7524*x[2] - 4.0960 * np.cos(1.1499 * x[2]) -
            4.0960 * np.cos(1.3159 * np.cos(35.3119 / x[0]) +
            1.3159 * np.cos(1.3159 * x[0] + 0.1497) -
            2.3007) + 33.1863) -
        0.7524 * np.tan(0.1450 / x[0]) + 0.9116) *
        np.exp(x[0]) + np.cos(4.3440 / (x[0] * x[2])) +
        np.cos(1.3159 * x[2]**3 + 0.9901 * x[2] *
            (-5.4439*x[0] + x[2] - 5.4439 * np.cos(1.1499 * x[2]) -
             5.4439 * np.cos(1.3159 * np.cos(34.0852 / x[0]) +
             1.3159 * np.cos(2.3007 + 3.1173 / x[2]) -
             2.3007) + 43.1071) + 0.9901 * x[2] -
        0.9901 * np.tan(0.1450 / x[0]) + 3.5002) +
        24096.6003) * 
        (-7.1636 * np.cos(2.1086 / x[0]) -
         5.4439 * np.cos(4.3440 / (x[0] * x[2])) -
         7.1636 * np.cos(1.3159 * x[0] + 18.8110) -
         5.4439 * np.cos(1.3159 * np.cos(1.7682 / x[0]) -
         2.2335) + 34.3103))
    
def evaluate_formula(x_train: np.ndarray, y_train: np.ndarray, 
                    x_test: np.ndarray, y_test: np.ndarray) -> tuple:
    """
    Evaluate the formula on training and test sets using the provided MSE calculation
    """
    # Calculate predictions
    train_preds = np.array([f0(x_train[:, i]) for i in range(x_train.shape[1])])
    test_preds = np.array([f0(x_test[:, i]) for i in range(x_test.shape[1])])
    
    # Calculate MSE using the provided function
    train_mse = calculate_mse(train_preds, y_train)
    test_mse = calculate_mse(test_preds, y_test)
    
    return train_mse, test_mse, train_preds, test_preds

def main():
    # Set numpy to ignore warnings
    np.seterr(all='ignore')
    
    # Load your dataset
    filepath = "problem_2.npz"  # Replace with your actual data path
    try:
        x_train, y_train, x_test, y_test = load_dataset(filepath)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Evaluate formula
    train_mse, test_mse, train_preds, test_preds = evaluate_formula(
        x_train, y_train, x_test, y_test
    )
    
    print(f"\nResults:")
    print(f"Training MSE: {train_mse:.6f}")
    print(f"Test MSE: {test_mse:.6f}")
    
    # Example predictions
    print("\nExample Predictions (first 5 samples):")
    for i in range(min(5, len(test_preds))):
        pred = test_preds[i]
        actual = y_test[i]
        print(f"Sample {i}: Predicted = {pred:.3f}, Actual = {actual:.3f}")

if __name__ == "__main__":
    main()