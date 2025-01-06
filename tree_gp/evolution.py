import random
from population import (
    initialize_population, 
    evaluate_population, 
    tournament_selection, 
    mutate, 
    crossover
)
from config import (POPULATION_SIZE, ELITISM_COUNT, MAX_GENERATIONS)

def run_evolution(X, y, variables_count):
    """
    Main evolutionary loop.
    Returns the best individual found across generations.
    """
    # Initialize
    population = initialize_population(variables_count)
    best_individual = None
    best_fitness = float('inf')

    for gen in range(MAX_GENERATIONS):
        # Evaluate
        scored_population = evaluate_population(population, X, y)
        scored_population.sort(key=lambda x: x[1])  # sort by fitness

        # Elitism: carry over top performers
        next_generation = [ind[0] for ind in scored_population[:ELITISM_COUNT]]

        # Track best individual
        if scored_population[0][1] < best_fitness:
            best_fitness = scored_population[0][1]
            best_individual = scored_population[0][0].copy()

        # Generate new population
        while len(next_generation) < POPULATION_SIZE:
            # Selection
            parent1 = tournament_selection(scored_population)
            parent2 = tournament_selection(scored_population)

            # Crossover
            child1, child2 = crossover(parent1, parent2)

            # Mutation
            child1 = mutate(child1, variables_count)
            child2 = mutate(child2, variables_count)

            next_generation.append(child1)
            if len(next_generation) < POPULATION_SIZE:
                next_generation.append(child2)

        population = next_generation
        
        print(f"Generation {gen}: Best fitness = {best_fitness}")

    return best_individual, best_fitness
