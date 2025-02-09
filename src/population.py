import random
import copy
from gp_tree import GPTree
from node import Node
from utils import mean_squared_error
from config import (
    POPULATION_SIZE, MAX_DEPTH, MIN_DEPTH, MUTATION_RATE, 
    CROSSOVER_RATE, TOURNAMENT_SIZE, ELITISM_COUNT
)

# Increase parsimony coefficient to further discourage bloated trees
PARSIMONY_COEFF = 0.007

def initialize_population(variables_count):
    population = []
    depths = [random.randint(MIN_DEPTH, MAX_DEPTH) for _ in range(POPULATION_SIZE)]
    for i in range(POPULATION_SIZE):
        full_method = (i < POPULATION_SIZE // 2)
        tree = GPTree.generate_random(
            depth=0,
            variables_count=variables_count,
            max_depth=depths[i],
            full=full_method
        )
        population.append(tree)
    return population

def evaluate_population(population, X, y):
    """Fitness = MSE + parsimony * node_count."""
    scored_population = []
    for tree in population:
        base_fitness = mean_squared_error(tree, X, y)
        tree_size = count_nodes(tree.root)
        fitness_value = base_fitness + PARSIMONY_COEFF * tree_size
        scored_population.append((tree, fitness_value))
    return scored_population

def count_nodes(node):
    return 1 + sum(count_nodes(child) for child in node.children)

def tournament_selection(scored_population):
    competitors = random.sample(scored_population, TOURNAMENT_SIZE)
    competitors.sort(key=lambda x: x[1])
    return competitors[0][0].copy()

def mutate(tree, variables_count):
    if random.random() < MUTATION_RATE:
        return subtree_mutation(tree, variables_count)
    return tree

def subtree_mutation(gp_tree, variables_count):
    new_tree = gp_tree.copy()
    nodes = get_all_nodes(new_tree.root)
    mutation_node = random.choice(nodes)

    new_subtree = GPTree.generate_random(
        depth=0, variables_count=variables_count, max_depth=3, full=False
    )
    mutation_node.value = new_subtree.root.value
    mutation_node.arity = new_subtree.root.arity
    mutation_node.children = new_subtree.root.children

    return new_tree

def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        return subtree_crossover(parent1, parent2)
    else:
        return parent1.copy(), parent2.copy()

def subtree_crossover(tree1, tree2):
    offspring1 = tree1.copy()
    offspring2 = tree2.copy()

    nodes1 = get_all_nodes(offspring1.root)
    nodes2 = get_all_nodes(offspring2.root)

    crossover_node1 = random.choice(nodes1)
    crossover_node2 = random.choice(nodes2)

    temp_value = crossover_node1.value
    temp_arity = crossover_node1.arity
    temp_children = crossover_node1.children

    crossover_node1.value = crossover_node2.value
    crossover_node1.arity = crossover_node2.arity
    crossover_node1.children = crossover_node2.children

    crossover_node2.value = temp_value
    crossover_node2.arity = temp_arity
    crossover_node2.children = temp_children

    return offspring1, offspring2

def get_all_nodes(node):
    nodes = [node]
    for child in node.children:
        nodes.extend(get_all_nodes(child))
    return nodes