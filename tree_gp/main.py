import random
import numpy as np
from node import Node
from gp_tree import GPTree
from utils import load_dataset, mean_squared_error, simplify_tree
from evolution import run_evolution
from visualize import visualize_tree
from config import SEED, VISUALIZE_BEST_INDIVIDUALS, TRAIN_DATA_PATH

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    x_train, y_train, x_test, y_test = load_dataset(TRAIN_DATA_PATH)
    variables_count = x_train.shape[0]

    best_individual, best_fitness = run_evolution(x_train, y_train, variables_count)

    print("\nBest Solution Found:")
    print("Raw GP Formula:", str(best_individual.root))
    print("Training MSE:", best_fitness)

    # Evaluate on test set
    test_mse = evaluate_on_test(best_individual, x_test, y_test)
    print("Test MSE:", test_mse)

    # Optionally, simplify final formula with Sympy
    simplified_expr_str = simplify_tree(best_individual)
    print("\nSympy-Simplified Expression:")
    print(simplified_expr_str)

    # Example predictions
    print("\nExample Predictions (first 5 samples):")
    for i in range(min(5, x_test.shape[1])):
        pred = best_individual.evaluate(x_test[:, i])
        actual = y_test[i]
        print(f"Sample {i}: Predicted = {pred:.3f}, Actual = {actual:.3f}")


def evaluate_on_test(gp_tree, X_test, y_test):
    preds = gp_tree.evaluate_vectorized(X_test)
    return np.mean((preds - y_test)**2)

if __name__ == "__main__":
    main()
