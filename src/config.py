SEED = 42
POPULATION_SIZE = 300
MAX_GENERATIONS = 180
TOURNAMENT_SIZE = 8
ELITISM_COUNT = 3

# Tree constraints
MAX_DEPTH = 4  # Reduced for smaller trees
MIN_DEPTH = 2

# Genetic operators probabilities
MUTATION_RATE = 0.3
CROSSOVER_RATE = 0.7

# Early stopping parameters
PATIENCE = 7  # Number of generations without improvement before stopping
MIN_FITNESS_IMPROVEMENT = 1e-7  # Minimum improvement for it to count

# Data
TRAIN_DATA_PATH = "data/problem_7.npz"
