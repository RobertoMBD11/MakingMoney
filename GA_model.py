from typing import Callable, Dict, List
from GA_population import Population, Individual
import random
import numpy as np


class GA_Trainer:
    def __init__(self,
                 param_bins: Dict[str, List[float]],
                 fitness_func: Callable,
                 num_generations: int = 50,
                 sol_per_pop: int = 20,
                 num_parents_mating: int = 10,
                 mutation_probability: float = 0.1,
                 on_generation: Callable = None):
        """
        Entrenador genético adaptado a tu implementación.

        param_bins: bins para discretizar parámetros (necesarios para individuos)
        fitness_func: función de fitness (recibe individuo, data, dinero_invertido)
        """
        self.param_bins = param_bins
        self.fitness_func = fitness_func
        self.num_generations = num_generations
        self.sol_per_pop = sol_per_pop
        self.num_parents_mating = num_parents_mating
        self.mutation_probability = mutation_probability
        self.on_generation = on_generation

        # Inicializar población
        self.population = Population(param_bins, sol_per_pop)
        self.best_individuals = []
        self.avg_fitness_per_gen = []

    def evaluate_fitness(self, data, dinero_invertido):
        fitnesses = []
        for ind in self.population.individuals:
            fit = self.fitness_func(ind, data, dinero_invertido)
            ind.set_fitness(fit)
            fitnesses.append(fit)
        return fitnesses

    def select_mating_pool(self):
        # Elitismo puro
        sorted_pop = sorted(self.population.individuals, key=lambda ind: ind.fitness, reverse=True)
        return sorted_pop[:self.num_parents_mating]

    def crossover(self, parents: List[Individual]):
        children = []
        num_genes = parents[0].num_genes

        while len(children) < (self.sol_per_pop - self.num_parents_mating):
            p1, p2 = random.sample(parents, 2)
            # Crossover: un punto
            point = np.random.randint(1, num_genes - 1)
            child_genes = np.concatenate([p1.genes[:point], p2.genes[point:]])
            child = Individual(self.param_bins, genes_array=child_genes)
            children.append(child)
        return children

    def mutate(self, individual: Individual):
        for i in range(individual.num_genes):
            if np.random.rand() < self.mutation_probability:
                individual.genes[i] = 1 - individual.genes[i]  # bitflip binario

    def run(self, data_train, dinero_invertido, data_test=None):
        for gen in range(self.num_generations):
            # 1. Evaluar fitness en TRAIN
            fitnesses_train = self.evaluate_fitness(data_train, dinero_invertido)

            # 2. Evaluar fitness en TEST (si hay)
            fitnesses_test = None
            if data_test is not None:
                fitnesses_test = [self.fitness_func(ind, data_test, dinero_invertido)
                                for ind in self.population.individuals]

            # 3. Registrar mejor y promedio (solo de entrenamiento)
            best_ind = max(self.population.individuals, key=lambda ind: ind.fitness)
            avg_fitness = np.mean(fitnesses_train)

            self.best_individuals.append(best_ind)
            self.avg_fitness_per_gen.append(avg_fitness)

            # 4. Callback de generación
            if self.on_generation:
                avg_test = np.mean(fitnesses_test) if fitnesses_test is not None else None
                self.on_generation(gen, best_ind, avg_fitness, avg_test)

            # 5. Reproducción
            parents = self.select_mating_pool()
            children = self.crossover(parents)
            for child in children:
                self.mutate(child)

            self.population.individuals = parents + children

        # Recalcular fitness final para asegurar consistencia
        self.evaluate_fitness(data_train, dinero_invertido)

        return max(self.population.individuals, key=lambda ind: ind.fitness)
