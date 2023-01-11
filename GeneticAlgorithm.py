from tqdm import tqdm
import numpy as np
import random as rand


def generate_genotype(n_features: int, feature_size=32) -> str:
    return ''.join([np.random.choice(['0', '1']) for _ in range(n_features * feature_size)])


def f(x):
    return x[0] ** 2 + 2 * x[1] + 1


class Specimen:
    def __init__(self, cost_function, cost_function_args=None, genotype: str = None, n_features: int = 2,
                 feature_size: int = 32, n_breed_points: int = 2, n_mutation_points: int = 2, mut_chance: float = 0.1):

        if genotype is None:
            self.genotype = generate_genotype(n_features, feature_size)
        else:
            self.genotype = genotype
        self.n_features = n_features
        self.cost_function = cost_function
        self.cost_function_args = cost_function_args if cost_function_args is not None else ()
        self.feature_size = feature_size
        self.n_breed_points = n_breed_points
        self.n_mutation_points = n_mutation_points
        self.mut_chance = mut_chance
        self.score = None
        self.phenotype = None

    def calculate_score(self):
        if self.score is None:
            self.score = -self.cost_function(self.phenotype, *self.cost_function_args)
        else:
            return self.score
        return self.score

    def calculate_phenotype(self):
        phenotype = []
        for feature in range(self.n_features):
            phenotype.append(
                int(self.genotype[feature * self.feature_size:feature * self.feature_size + self.feature_size],
                    2) / 750000000)
        self.phenotype = np.array(phenotype)  # does not return anything !!!

    def breed(self, other):
        splitted_self = [bit for bit in self.genotype]
        splitted_other = [bit for bit in other.genotype]
        split_places = [rand.randint(1, len(splitted_self) - 2) for _ in range(self.n_breed_points)]
        split_places.sort()
        genotype_A = []
        genotype_B = []

        for idx in range(self.n_breed_points):
            genotype_A = splitted_self[:split_places[idx]] + splitted_other[split_places[idx]:]
            genotype_B = splitted_other[:split_places[idx]] + splitted_self[split_places[idx]:]

        merged_genotype_A = ''.join(genotype_A)
        merged_genotype_B = ''.join(genotype_B)

        return Specimen(cost_function=self.cost_function, cost_function_args=self.cost_function_args,
                        genotype=merged_genotype_A,
                        n_features=self.n_features, feature_size=self.feature_size, n_breed_points=self.n_breed_points,
                        n_mutation_points=self.n_mutation_points,
                        mut_chance=self.mut_chance), \
               Specimen(cost_function=self.cost_function, cost_function_args=self.cost_function_args,
                        genotype=merged_genotype_B,
                        n_features=self.n_features, feature_size=self.feature_size, n_breed_points=self.n_breed_points,
                        n_mutation_points=self.n_mutation_points,
                        mut_chance=self.mut_chance)

    def mutate(self):
        mutation_list = [idx for idx in range(self.n_mutation_points + 1)]
        weights = []
        splitted = [bit for bit in self.genotype]
        mutated = []

        for i in mutation_list:
            if i == 0:
                weights.append(1 - self.mut_chance)
            else:
                weights.append(self.mut_chance / (len(mutation_list) - 1))

        mutations_left = rand.choices(mutation_list, weights=weights)[0]

        while mutations_left != 0:
            point_mutation = rand.randint(0, len(splitted) - 1)

            if point_mutation not in mutated:
                if splitted[point_mutation] == '0':
                    splitted[point_mutation] = '1'
                    mutations_left = mutations_left - 1
                else:
                    splitted[point_mutation] = '0'
                    mutations_left = mutations_left - 1
                mutated.append(point_mutation)
            else:
                continue

        self.genotype = ''.join(splitted)  # calculate fenotype after

    def __str__(self):
        return str(self.phenotype)

    def __repr__(self):
        return str(self.phenotype)


class GeneticAlgorithm:
    def __init__(self, cost_function, cost_function_args=None, n_specimens: int = 100, breed_chance: float = 0.8,
                 n_breed_points: int = 2, mut_chance: float = 0.1, n_mutation_points: int = 10, n_features: int = 2,
                 feature_size: int = 32, n_jobs: int = 1):

        self.cost_function = cost_function
        self.cost_function_args = cost_function_args
        self.n_specimens = n_specimens
        self.breed_chance = breed_chance
        self.n_breed_points = n_breed_points
        self.mut_chance = mut_chance
        self.n_mutation_points = n_mutation_points
        self.n_features = n_features
        self.feature_size = feature_size
        self.n_jobs = n_jobs
        self.population = [Specimen(self.cost_function, n_features=self.n_features,
                                    feature_size=self.feature_size,
                                    n_breed_points=self.n_breed_points,
                                    n_mutation_points=self.n_mutation_points,
                                    mut_chance=self.mut_chance,
                                    cost_function_args=self.cost_function_args) for _ in range(self.n_specimens)]

    def natural_selection(self, roulette=True):
        for specimen in self.population:
            specimen.calculate_phenotype()
            specimen.calculate_score()

        if roulette:
            n_survivors = (len(self.population) * self.breed_chance)
            score_sum = 0
            weights = []
            new_population = []

            for specimen in self.population:
                score_sum += specimen.score

            for specimen in self.population:
                try:
                    weights.append(specimen.score / score_sum)
                except:
                    weights = None

            while len(new_population) != n_survivors:
                winner = np.random.choice(self.population, replace=False, p=weights)
                new_population.append(winner)

            while len(new_population) != len(self.population):
                new_population.append(rand.choices(new_population))

            np.random.shuffle(new_population)
            self.population = new_population

        else:
            new_population = []
            n_survivors = (len(self.population) * self.breed_chance)
            sorted_population = sorted(self.population, key=lambda x: x.score, reverse=True)

            fittest = sorted_population[:int(n_survivors)]

            while len(new_population) != len(self.population):
                new_population.append(np.random.choice(fittest, replace=True))

            np.random.shuffle(new_population)
            self.population = new_population

    def breed_population(self):
        new_population = []
        for idx in range(0, len(self.population), 2):
            specimen_A = self.population[idx]
            specimen_B = self.population[idx + 1]

            if rand.random() <= self.breed_chance:
                child_A, child_B = specimen_A.breed(specimen_B)
                new_population.append(child_A)
                new_population.append(child_B)
            else:
                new_population.append(specimen_A)
                new_population.append(specimen_B)
        self.population = new_population

    def run(self, n_iter: int = 100, verbose: bool = True):
        title = f""
        pbar = tqdm(range(n_iter), disable=not verbose, desc=title)
        for _ in pbar:

            self.breed_population()

            for specimen in self.population:
                specimen.mutate()

            self.natural_selection(roulette=False)

            for specimen in self.population:
                specimen.calculate_phenotype()
                specimen.calculate_score()

            scores = np.array([specimen.score for specimen in self.population])
            best_score = max(scores)
            title = f"Current lowest error: {-best_score:.4f}"
            pbar.set_description(title)

    def get_best(self):
        scores = np.array([specimen.score for specimen in self.population])
        best_score = max(scores)
        best_phenotype = self.population[np.argmax(scores)].phenotype
        return best_phenotype, -best_score


if __name__ == '__main__':
    genetic = GeneticAlgorithm(cost_function=f, n_specimens=100, n_features=2, feature_size=32,
                               n_breed_points=3, n_mutation_points=2, n_jobs=6,
                               mut_chance=0.1, breed_chance=0.5)

    genetic.run(n_iter=100)
