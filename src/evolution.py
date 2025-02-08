import random
from population import (
    initialize_population, 
    evaluate_population, 
    tournament_selection, 
    mutate, 
    crossover
)
from config import (
    POPULATION_SIZE, ELITISM_COUNT, MAX_GENERATIONS,
    PATIENCE, MIN_FITNESS_IMPROVEMENT
)

def run_evolution(X, y, variables_count):
    population = initialize_population(variables_count)
    best_individual = None
    best_fitness = float('inf')

    patience_counter = 0
    previous_best = best_fitness

    for gen in range(MAX_GENERATIONS):
        scored_population = evaluate_population(population, X, y)
        scored_population.sort(key=lambda x: x[1])  # sort by fitness

        # Elitism
        next_generation = [ind[0] for ind in scored_population[:ELITISM_COUNT]]

        # Track best in this generation
        current_best_fitness = scored_population[0][1]
        current_best_individual = scored_population[0][0].copy()

        # Update global best
        if (previous_best - current_best_fitness) > MIN_FITNESS_IMPROVEMENT:
            best_fitness = current_best_fitness
            best_individual = current_best_individual
            patience_counter = 0
            previous_best = best_fitness
        else:
            patience_counter += 1

        # Fill the rest of next_generation
        while len(next_generation) < POPULATION_SIZE:
            parent1 = tournament_selection(scored_population)
            parent2 = tournament_selection(scored_population)

            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, variables_count)
            child2 = mutate(child2, variables_count)

            next_generation.append(child1)
            if len(next_generation) < POPULATION_SIZE:
                next_generation.append(child2)

        population = next_generation

        print(f"Generation {gen}: Best fitness = {best_fitness:.6f}")

        if patience_counter >= PATIENCE:
            print(f"Early stopping at generation {gen} due to no improvement.")
            break

    return best_individual, best_fitness
