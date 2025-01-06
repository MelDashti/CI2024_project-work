SEED = 42
POPULATION_SIZE = 50
MAX_GENERATIONS = 30
TOURNAMENT_SIZE = 5
ELITISM_COUNT = 1

# Tree constraints
MAX_DEPTH = 4  # Reduced for smaller trees
MIN_DEPTH = 2

# Genetic operators probabilities
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.8

# Early stopping parameters
PATIENCE = 5  # Number of generations without improvement before stopping
MIN_FITNESS_IMPROVEMENT = 1e-6  # Minimum improvement for it to count

# Data
TRAIN_DATA_PATH = "data/problem_2.npz"

# Visualization
VISUALIZE_BEST_INDIVIDUALS = True
