import random
import numpy as np
from node import Node
from gp_tree import GPTree
from utils import load_dataset, mean_squared_error
from evolution import run_evolution
from visualize import visualize_tree
from config import SEED, VISUALIZE_BEST_INDIVIDUALS, TRAIN_DATA_PATH

def main():
    # Set random seed for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)

    # Load data
    x_train, y_train, x_test, y_test = load_dataset(TRAIN_DATA_PATH)
    variables_count = x_train.shape[0]

    # Run evolutionary process
    best_individual, best_fitness = run_evolution(x_train, y_train, variables_count)
    
    # Print results
    print("\nBest Solution Found:")
    print("Formula:", str(best_individual.root))
    print("Training MSE:", best_fitness)

    # Evaluate on test set
    test_mse = evaluate_on_test(best_individual, x_test, y_test)
    print("Test MSE:", test_mse)

    # Optional: Print example predictions
    print("\nExample Predictions (first 5 samples):")
    for i in range(min(5, x_test.shape[1])):
        pred = best_individual.evaluate(x_test[:, i])
        actual = y_test[i]
        print(f"Sample {i}: Predicted = {pred:.3f}, Actual = {actual:.3f}")

def evaluate_on_test(gp_tree, X_test, y_test):
    predictions = gp_tree.evaluate_vectorized(X_test)
    mse = ((predictions - y_test) ** 2).mean()
    return mse

if __name__ == "__main__":
    main()