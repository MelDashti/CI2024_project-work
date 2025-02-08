# 🧬 Genetic Programming Formula Evolution

A robust implementation of Tree-based Genetic Programming to automatically discover and evolve mathematical formulas that predict outcomes based on numerical datasets. The system works with `.npz` files containing features and target values, employing evolutionary algorithms to minimize Mean Squared Error (MSE).

## 🌟 Key Features

- **Tree-Based Expression Evolution**: Hierarchical representation of mathematical formulas using a sophisticated node system
- **Vectorized Operations**: Optimized for performance using NumPy's vectorized calculations
- **Safe Mathematical Operations**: Robust handling of edge cases (division by zero, log of negatives, etc.)
- **Bloat Control**: Advanced mechanisms to prevent excessive formula complexity
- **Configurable Parameters**: Easily tune the evolutionary process through a configuration file
- **SymPy Integration**: Symbolic simplification for cleaner, more interpretable formulas

## 🗂️ Project Structure

```
project_root/
├── data/                  # Dataset directory
│   ├── problem_0.npz      # Training dataset 0
│   ├── problem_1.npz      # Training dataset 1
│   ├── problem_2.npz      # Training dataset 2
│   └── ...                # Additional datasets
│
├── src/                   # Source code directory
│   ├── __init__.py       # Python package initializer
│   ├── config.py         # Configuration parameters and settings
│   ├── evolution.py      # Core evolutionary algorithms and processes
│   ├── gp_tree.py        # GPTree class implementation
│   ├── main.py           # Entry point and main execution flow
│   ├── node.py           # Node class for tree structure
│   ├── operators.py      # Mathematical operators and safety wrappers
│   ├── population.py     # Population management and genetic operations
│   ├── utils.py          # Utility functions and helper methods
│   └── requirements.txt  # Project dependencies
```

### Core Components

#### 1. Node Class (`src/node.py`)
- Fundamental building block for expression trees
- Handles variables (x_0, x_1), constants, and operators
- Implements vectorized evaluation for efficient computation
- Contains safety mechanisms for numerical edge cases

#### 2. GPTree Class (`src/gp_tree.py`)
- High-level management of complete expressions
- Implements deep copy operations for population management
- Provides both single-sample and vectorized evaluation interfaces
- Supports configurable random tree generation with depth constraints

#### 3. Evolution Engine (`src/evolution.py`)
- Implements the main evolutionary loop
- Manages selection, crossover, and mutation operations
- Handles fitness evaluation and population advancement
- Implements early stopping and convergence checks

#### 4. Population Management (`src/population.py`)
- Handles population initialization and management
- Implements tournament selection
- Manages crossover and mutation operations
- Controls elitism for preserving best solutions

#### 5. Operators (`src/operators.py`)
```python
UNARY_OPERATORS = {
    'sin': np.sin,
    'cos': np.cos,
    'log': safe_log,
    'exp': safe_exp,
    'cube': lambda x: np.clip(x**3, -1e308, 1e308)
}

# Safe operation implementations
def safe_div(x, y):
    return np.where(np.abs(y) < 1e-10, 1.0, x / y)

def safe_log(x):
    return np.log(np.clip(x, 1e-5, None))
```

## ⚙️ Configuration

```python
# src/config.py - Fully configurable parameters
POPULATION_SIZE = 300
MAX_GENERATIONS = 180
TOURNAMENT_SIZE = 8
ELITISM_COUNT = 3
MUTATION_RATE = 0.3
CROSSOVER_RATE = 0.7
MAX_DEPTH = 4
PARSIMONY_COEFF = 0.005  # Controls bloat

# Data paths
TRAIN_DATA_PATH = "data/problem_0.npz"  # Configurable dataset path
```

## 📊 Performance Analysis

Through extensive experimentation and tuning, I identified key performance characteristics:

### Generation Impact Analysis

| Generations | Performance Characteristics |
|------------|---------------------------|
| 80         | - Simple, readable formulas<br>- Moderate MSE values<br>- Quick convergence |
| 140        | - Balanced complexity/accuracy<br>- Good MSE reduction<br>- Moderate formula size |
| 180        | - Highest accuracy<br>- More complex formulas<br>- Optimal for challenging datasets |

### Key Findings
- **Optimal Parameters**: Mutation rate (0.3) and crossover rate (0.7) provide the best balance
- **Selection Pressure**: Tournament size of 8 with elitism count of 3 ensures stable evolution
- **Population Dynamics**: Population size of 300 provides sufficient genetic diversity
- **Convergence**: Most datasets achieve good results within 20-50 generations
- **Complex Cases**: Challenging datasets may require full 180 generations
- **Parsimony and Formula Complexity**: Parsimony penalized large trees, **forcing overly simple formulas** like `x[0]`. Setting **parsimony to 0** for **Datasets 0 and 1** allowed the GP to evolve more expressive formulas while keeping it for complex datasets helped control bloat and improve MSE.


## 🚀 Getting Started

1. Clone the repository:
```bash
git clone https://github.com/meldashti/CI2024_project-work.git
```

2. Navigate to the source directory and install dependencies:
```bash
cd CI2024_project-work/src
pip install -r requirements.txt
```

3. Configure parameters in `src/config.py` based on your needs, including the dataset path

4. Run the evolution:
```bash
python src/main.py
```

## 📋 Requirements

- Python 3.7+
- NumPy
- SymPy

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Areas of particular interest:
- Performance optimizations
- New operator implementations
- Enhanced bloat control mechanisms
- Additional fitness metrics

