import random
import copy
from gp_tree import GPTree
from node import Node
from utils import mean_squared_error
from config import (POPULATION_SIZE, MAX_DEPTH, MIN_DEPTH, MUTATION_RATE, 
                    CROSSOVER_RATE, TOURNAMENT_SIZE, ELITISM_COUNT)

def initialize_population(variables_count):
    """
    Initialize the population with random individuals.
    Using 'ramped half-and-half': half with full method, half with grow method.
    """
    population = []
    depths = [random.randint(MIN_DEPTH, MAX_DEPTH) for _ in range(POPULATION_SIZE)]
    for i in range(POPULATION_SIZE):
        full_method = (i < POPULATION_SIZE // 2)  # half full, half grow
        tree = GPTree.generate_random(
            depth=0,
            variables_count=variables_count,
            max_depth=depths[i],
            full=full_method
        )
        population.append(tree)
    return population

def evaluate_population(population, X, y):
    """
    Calculate fitness (MSE) for each individual and return as a list of (tree, fitness).
    """
    scored_population = []
    for tree in population:
        fitness_value = mean_squared_error(tree, X, y)
        scored_population.append((tree, fitness_value))
    return scored_population

def tournament_selection(scored_population):
    """
    Tournament selection. Pick random subset, choose best.
    Return the selected individual (as a GPTree).
    """
    competitors = random.sample(scored_population, TOURNAMENT_SIZE)
    competitors.sort(key=lambda x: x[1])  # sort by fitness ascending
    return competitors[0][0].copy()  # best of the tournament

def mutate(tree, variables_count):
    """
    Mutation can be subtree mutation: pick a random node and replace it with a random subtree.
    """
    if random.random() < MUTATION_RATE:
        # We mutate the tree
        return subtree_mutation(tree, variables_count)
    return tree

def subtree_mutation(gp_tree, variables_count):
    """Perform subtree mutation on a random node."""
    new_tree = gp_tree.copy()
    
    # Convert tree to list of nodes (preorder traversal), pick one at random
    nodes = get_all_nodes(new_tree.root)
    mutation_node = random.choice(nodes)

    # Replace the mutation node with a new randomly generated subtree
    new_subtree = GPTree.generate_random(
        depth=0, variables_count=variables_count, max_depth=3, full=False
    )
    mutation_node.value = new_subtree.root.value
    mutation_node.arity = new_subtree.root.arity
    mutation_node.children = new_subtree.root.children

    return new_tree

def crossover(parent1, parent2):
    """
    Subtree crossover: randomly select a node in parent1 and a node in parent2
    and swap their subtrees.
    """
    if random.random() < CROSSOVER_RATE:
        return subtree_crossover(parent1, parent2)
    else:
        return parent1.copy(), parent2.copy()

def subtree_crossover(tree1, tree2):
    """
    Perform subtree crossover on two parent GPTrees.
    """
    offspring1 = tree1.copy()
    offspring2 = tree2.copy()

    # Get nodes
    nodes1 = get_all_nodes(offspring1.root)
    nodes2 = get_all_nodes(offspring2.root)

    # Randomly choose crossover points
    crossover_node1 = random.choice(nodes1)
    crossover_node2 = random.choice(nodes2)

    # Swap
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
    """Preorder traversal to get all nodes in the tree."""
    nodes = [node]
    for child in node.children:
        nodes.extend(get_all_nodes(child))
    return nodes
