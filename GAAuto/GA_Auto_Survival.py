from os import cpu_count
import random as rand

from joblib import Parallel, delayed

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split

from timeit import default_timer as timer
from datetime import timedelta

from statistics import stdev

import pandas as pd
import numpy as np

import copy

import sys
import traceback

import warnings

from .Individual import Individual
from sksurv.ensemble import RandomSurvivalForest
from sksurv.tree import SurvivalTree
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class Auto_ml:
    def __init__(
        self,
        population_size=51,
        generation_count=30,
        mutation_prob=0.1,
        crossover_prob=0.9,
        gene_crossover_prob=0.5,
        tournament_size=2,
        internal_fold_count=3,
        log_directory=None,
        random_state=None,
        n_jobs=-1):

        """Initialise the Auto_PU genetic algorithm.

        Parameters
        ----------
        population_size: int, optional (default: 101)
            Number of indiviiduals in the population for each generation.
        generation_count: int, optional (default: 50)
            Number of iterations to run the optimisation algorithm.
        mutation_prob: float, optional (default: 0.1)
            The probability of gene of an individual undergoing mutation.
        crossover_prob: float, optional (default: 0.9)
            The probability of two individuals undergoing crossover.
        gene_crossover_prob: float, optional (default: 0.5)
            The probability of the values of a gene being swapped
            between two individuals.
        tournament_size: int, optional (default: 2)
            The number of individuals randomly sampled for tournament
            selection.
        internal_fold_count: int, optional (default: 5)
            The number of folds for internal cross validation.
        spies: boolean, optional (default: False)
            Whether to allow spy-based individuals.
        log_directory: string, optional (default: None)
            The directory to store log files.
        random_state: int, optional (default: None)
            The random number generator seed. Use this parameter
            for reproducibility.
        n_jobs: int, optional (default: 1)
            Number of CPUs for evaluating individuals in parallel.
        try_next: bool, optional (default: False)
            Indicates whether to use the next fittest individual
            in the population if an error occurs with the fittest.
            Only recommended for debugging.
            If errors occur with fittest individual it is
            recommended to use a higher generation count or
            number of individuals.

        Returns
        -------
        None

        """
        self.population_size = population_size
        self.generation_count = generation_count
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.gene_crossover_prob = gene_crossover_prob
        self.tournament_size = tournament_size
        self.internal_fold_count = internal_fold_count
        self.log_directory = log_directory
        self.random_state = random_state

        if n_jobs == 0:
            raise ValueError("The value of n_jobs cannot be 0.")
        elif n_jobs < 0:
            self.n_jobs = cpu_count() + 1 + n_jobs
        else:
            self.n_jobs = n_jobs

        self.best_config = None
        self.estimator = None
        self.seed = random_state
        self.NumSuccMut = self.NumFailMut = self.NumNeutMut = self.TotNumMut = 0


    def generate_individual(self):
        """Randomly generate an indvidual.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        individual: Individual
            Returns a randomly generated individual.
        """
        # instantiate the individual
        individual = Individual()

        return individual

    def generate_population(self):
        """Randomly generate the population.
        A population of self.population_size will be created and the values
        for the individual genes will be randomly generated with their
        respective methods.

        Parameters
        ----------
        None

        Returns
        -------
        population: array-like {self.population_size, n_genes}
            Returns a randomly generated population.
        """

        # initialise an empty population list
        population = [self.generate_individual() for _ in range(self.population_size)]

        # return completed population
        return population


    def pred(self, X_train, y_train, X_test):
        """Predict whether the instances in the test set are
        positive or negative.

        Parameters
        ----------
        classifier: Object
            The classifier to build and evaluate.
        X_train: array-like {n_samples, n_features}
            Training set feature matrix.
        y_train: array-like {n_samples}
            Training set class labels.
        X_test: array-like {n_samples, n_features}
            Test set feature matrix.

        Returns
        -------
        y_pred: array-like {n_samples}
            Class label predictions.

        """

        # clone the classifier
        clf = clone(classifier)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # fit the classifier to the training set
            clf.fit(X_train, y_train)

        # predict the instances in the test set
        y_pred = clf.predict(X_test)

        # return the predictions
        return y_pred

    def score(self, X_test, y_test):

        # predict the instances in the test set
        c_val = self.estimator.score(X_test, y_test)

        # return the predictions
        return c_val


    def internal_CV(self, individual, X, y):
        """Perform internal cross validation to get a list of
        recalls and precisions.

        Parameters
        ----------
        individual: Individual
            The configuration to be assessed.
        X: array-like {n_samples, n_features}
            Feature matrix.
        y: array-like {n_samples}
            Class values for feature matrix.

        Returns
        -------
        f_measure: float
            The F-measure value achieved by the individual
        """

        rsf = individual.get_rsf()
        tree = individual.get_survival_tree()
        cox = individual.get_cox_model()
        c_val_list = [0.0] * 3

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, stratify=y["uncensored"], random_state=self.seed)


        # Train the models and test their performance

        try:

            # training
            rsf.fit(X_train, y_train)
            c_val_list[0] += rsf.score(X_test, y_test)

            tree.fit(X_train, y_train)
            c_val_list[1] += tree.score(X_test, y_test)

            cox.fit(X_train, y_train)
            c_val_list[2] += cox.score(X_test, y_test)

        except:
            # if the individual generates an error, fitness for
            # individual is 0
            print(e)
            return 0


        try:
            individual.set_best(c_val_list)
            return individual.fitness
        except:
            return 0

    def assess_fitness(self, individual, features, target, assessed):
        """Assess the fitness of an individual on the current training set.
        Individual will be checked against the list of already assessed
        configurations. If present, their recall and precision will be set
        to the values previously calculated for the configuration on this
        training set.

        Parameters
        ----------
        individual: Individual
            The configuration to be assessed.
        features: array-like {n_samples, n_features}
            Feature matrix.
        target: array-like {n_samples}
            Class labels.
        assessed: array-like {n_assessed_configs}
            The previously assessed configurations.

        Returns
        -------
        new_assessed_configs: array-like {n_assessed_configs}
            The new list of assessed configurations.

        """


        config = str(individual)
        # hash_config = hash(config)
        # print(hash_config, "*** ", assessed.keys())

        # Evaluatate the fitness of this config
        try:
            _ = self.internal_CV(individual, features, target)
        except Exception as e:
            # if individual produces an error, fitness
            # of the individual is set to 0
            print(e)
            individual.fitness = -1

        return individual

    def get_avg_fitness(self, population):
        """Get the average recall, precision, and average
        standard deviation for both.

        Parameters
        ----------
        population: array_like {population_size}
            The list of individuals.

        Returns
        -------
        avg_precision: float
            The average precision of the population.
        avg_recall: float
            The average recall of the population.
        avg_f_measure: float
            The average f-measure of the population.
        avg_std_precision: float
            The average standard deviation of precision of the population.
        avg_std_recall: float
            The average standard deviation of recall of the population.
        avg_std_f_measure: float
            The average standard deviation of f-measure of the population.

        """

        # initialise all values as 0
        avg_fit = 0

        # for every individual in the population
        # add values to total
        for individual in population:
            avg_fit += individual.fitness
        # get the average of the values
        avg_fit = (avg_fit / len(population))

        return avg_fit

    def get_fittest(self, population):
        """Get the fittest individual in a population.

        Parameters
        ----------
        population: array-like {population_size}
            The list of individuals.

        Returns
        -------
        fittest_individual: Individual
            The individual with the best fitness in the population.
        f: int
            The number of individuals deemed fitter than the comparison
            by way of f-measure.
        rec: int
            The number of individuals deemed fitter than the comparison
            by way of recall.

        """

        # initialise fittest individual as the first in the population
        fittest_individual = population[0]

        # compare every individual in population
        for indiv in population:

            # if new individual f_measure is higher than fittest individual
            # new individual becomes fittest individual
            if indiv.fitness > fittest_individual.fitness:
                fittest_individual = indiv

        # return the fittest individual and the counters
        return fittest_individual

    def tournament_selection(self, population):
        """Tournament selection.
        Randomly select individuals from the population and
        return the fittest.

        Parameters
        ----------
        population: array-like {population_size}
            The list of individuals.

        Returns
        -------
        fittest_individual: Individual
            The individual with the best fitness in the population.
        f: int
            The number of individuals deemed fitter than
            the comparison by way of f-measure.
        rec: int
            The number of individuals deemed fitter than
            the comparison by way of recall.
        """

        # initialise tournament population as empty list
        tournament = [rand.choice(population) for _ in range(self.tournament_size)]

        # return the fittest individual from the population
        return self.get_fittest(tournament)

    def cross_exe(self, a, b):
        # return b,a if rand.uniform(0, 1) < self.gene_crossover_prob else a,b
        if rand.uniform(0, 1) < self.gene_crossover_prob:
            return b,a
        return a,b

    def swap_genes(self, a, b):
        """Swap each gene of two individuals with a given probability.

        Parameters
        ----------
        new_indiv1: Individal
            First individual for swapping genes.
        new_indiv2: Individual
            Second individual for swapping genes.

        Returns
        -------
        new_indiv1: Individual
            First individual after swapping genes.
        new_indiv2: Individual
            Second individual after swapping genes.

        """

        a.mtry, b.mtry =  self.cross_exe(a.mtry, b.mtry)
        a.d0_forest, b.d0_forest =  self.cross_exe(a.d0_forest, b.d0_forest)
        a.d0_tree, b.d0_tree =  self.cross_exe(a.d0_tree, b.d0_tree)



        # return the new individuals
        return a, b

    def crossover(self, population):
        """Perform crossover on the population.
        Individuals are selected through tournament selection
        and their genes are swapped with a given probability.

        Parameters
        ----------
        population: array-like {population_size}
            The list of individuals.

        Returns
        -------
        new_population: array-like {population_size}
            The list of individuals after undergoing crossover.
        f: int
            The number of individuals deemed to be fitter
            than the comparison by f-measure.
        rec: int
            The number of individuals deemed to be fitter
            than the comparison by recall.

        """

        # empty list to store the modified population
        new_population = []

        # keep performing crossover until the new population
        # is the correct size
        while len(new_population) < self.population_size - 1:

            # select two individuals with tournament selection
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)

            # initialise the new individuals
            offspring1 = copy.deepcopy(parent1)
            offspring2 = copy.deepcopy(parent2)

            # if random number is less than crossover probability,
            # the individuals undergo crossover
            if rand.uniform(0, 1) < self.crossover_prob:
                offspring1, offspring2 = self.swap_genes(parent1, parent2)

            # add the new individuals to the population
            new_population.append(offspring1)
            new_population.append(offspring2)

        # return the new population and the counters
        return new_population


    def mutate_exe(self, x, x_range):
        if rand.uniform(0, 1) < self.mutation_prob:
            return rand.choice(x_range)
        return x

    def mutate(self, population, features, target):
        """Perform mutation on the population.
        Each gene is slightly altered with a given probability.

        Parameters
        ----------
        population: array-like {population_size}
            The list of individuals.

        Returns
        -------
        population: array-like {population_size}
            The list of individuals after undergoing mutation.

        """

        # perform on every individual in population
        for i, indiv in enumerate(population):

            original = copy.deepcopy(indiv)

            indiv.mtry = self.mutate_exe(indiv.mtry, indiv.mtry_range)
            indiv.n_trees = self.mutate_exe(indiv.n_trees, indiv.n_trees_range)
            indiv.min_sample_leaf_forest = self.mutate_exe(indiv.min_sample_leaf_forest, indiv.min_sample_leaf_range)
            indiv.d0_forest = self.mutate_exe(indiv.d0_forest, indiv.d0_range)
            indiv.min_sample_leaf_tree = self.mutate_exe(indiv.min_sample_leaf_tree, indiv.min_sample_leaf_range)
            indiv.d0_tree = self.mutate_exe(indiv.d0_tree, indiv.d0_range)
            indiv.l1_ratio = self.mutate_exe(indiv.l1_ratio, indiv.l1_ratio_range)
            indiv.alpha = self.mutate_exe(indiv.alpha, indiv.alpha_range)

            if indiv != original:
                self.assess_fitness(individual=original, features=features, target=target, assessed=None)
                self.assess_fitness(individual=indiv, features=features, target=target, assessed=None)
                if indiv.fitness > original.fitness:
                    self.NumSuccMut += 1
                elif indiv.fitness < original.fitness:
                    self.NumFailMut += 1
                else:
                    self.NumNeutMut += 1
                self.TotNumMut += 1

        # return the altered population
        return population

    def log_individuals(self, population, current_generation):
        """Save the details of all individuals in population to csv.

        Parameters
        ----------
        population: array-like {population_size}
            The list of individuals.
        current_generation: int
            The current generation number.

        Returns
        -------
        None

        """

        # initialise list for storing the details of all individuals
        # in the population
        individual_details = []

        # for every individual in the population, convert the values to strings
        # and save to individual details list
        for individual in population:

            indiv_detail = individual.get_config()

            individual_details.append(indiv_detail)

        # column names
        col = ["mtry", "n_trees", "min_sample_leaf_forest", "d0_forest", "min_sample_leaf_tree", "d0_tree", "l1_ratio", "alpha", "C-index", "method"]

        # create dataframe
        individuals_df = pd.DataFrame(individual_details, columns=col)
        # print(individuals_df)
        individuals_df.sort_values('C-index', ascending=False, inplace=True)

        try:
            # save to csv
            individuals_df.to_csv(self.log_directory + "/" + str(current_generation+1) +  "_details.csv", index=False)
        except Exception as e:
            print("Could not save file:", self.log_directory + "/details/" + str(current_generation) +  "_individuals.csv")
            print(e)

    def log_generation_stats(self, fittest_individual,
                             avg_f_measure, current_generation, best_fit_list):
        """Save all statistics from a generation to csv file.

        Parameters
        ----------
        fittest_individual: Individual
            The fittest individual from the population.
        avg_f_measure: float
            Average f-measure of all individuals in the population.
        current_generation: int
            The current generation.

        Returns
        -------
        None

        """

        # first column of the combined stats file
        # multiple blank values are needed as all columns
        # must have the same number of values
        stat_names = ["C-index","Gen. found", "", "","", "", "","", "", ""]

        max_value = max(best_fit_list)
        max_index = best_fit_list.index(max_value)
        c_val = str(fittest_individual.fitness)
        best_stats = [c_val, str(max_index), "", "", "", "", "", "", "", ""]

        avg_stats = [str(avg_f_measure),"", "", "", "", "", "", "", "", ""]

        # the final two columns give details of the best configuration
        # these are the configuration component names
        config_names = ["mtry", "n_trees", "min_sample_leaf_forest", "d0_forest", "min_sample_leaf_tree", "d0_tree", "l1_ratio", "alpha", "C-index", "method"]

        # get the details of the best configuration
        best_configuration = fittest_individual.get_config()

        # combine all columns into a single list
        combined_stats = [stat_names, best_stats, avg_stats,
                          config_names, best_configuration]

        # conver to np array so can be transposed in dataframe
        stats = np.array(combined_stats)
        # print(stats)
        # create dataframe
        generation_stats = pd.DataFrame(
            stats.T, columns=["Stat", "Best individual",
                              "Population average", "Configuration", "Best"])

        try:
            # save dataframe as csv
            generation_stats.to_csv(self.log_directory + "/stats.csv", index=False)
        except Exception as e:
            print("Could not save file:", self.log_directory + "/stats/" +
                                    str(current_generation) + "_gen.csv")
            print(e)

    def log_fitness_progress(self, best_fit, average_fit):
        pd.DataFrame({"Gen.": range(self.generation_count), "Best fitness": best_fit, "Average fitness": average_fit}).to_csv(self.log_directory + "/fitness_progress.csv", index=False)

    def log_mutation_counters(self, counters_df):
        counters_df.to_csv(self.log_directory + "/mutation_counters.csv", index=False)

    def fit(self, features, target):
        """Use a genetic algorithm to fit an optimised PU learning
        algorithm to the input data.

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix.
        target: array-like {n_samples}
            List of class labels for prediction.

        Returns
        -------
        self: object
            Returns a fitted Auto_PU object.
        """

        # features = np.array(features)
        # target = np.array(target)

        if self.random_state is not None:
            rand.seed(self.random_state)
            np.random.seed(self.random_state)

        # randomly initialise the population
        population = self.generate_population()
        # initialise current generation to 0
        current_generation = 0

        # list to store all assessed configurations
        # saves computation time by allowing us to not have to
        # assess configurations that have already been assessed
        assessed_configs = {}
        best_fitness_list = []
        avg_fitness_list = []
        counters_df = pd.DataFrame(columns=["MutProb", "NumSuccMut", "NumFailMut", "NumNeutMut", "TotNumMut"])

        # start the evolution process
        while current_generation < self.generation_count:
            # assess the fitness of individuals concurrently
            # the ray.get function calls the assess_fitness function with
            # the parameters specified in remote()
            # this will call the function for all individuals in population

            start_time = timer()

            # change seed every generation
            self.seed += current_generation

            population = \
                Parallel(n_jobs=self.n_jobs,
                         verbose=0)(delayed(self.assess_fitness)
                                     (individual=population[i],
                                      features=features,
                                      target=target, assessed=assessed_configs)
                                     for i in range(len(population)))

            # non-parallel version
            # population = [self.assess_fitness(population[i], features, target, assessed_configs) for i in range(len(population))]
            print("First indv", population[0].c_index_list ,population[0].best_algorithm)


            # calculate average precision and recall
            avg_fit = self.get_avg_fitness(population)

            # get the fittest individual in the population so that it can
            # be preserved without modification
            # also get the number of decisions made by f-measure and recall
            fittest_individual = self.get_fittest(population)

            # save fitness progress
            best_fitness_list.append(fittest_individual.fitness)
            avg_fitness_list.append(avg_fit)
            # Append the values to the dataframe
            counters_df.loc[len(counters_df)] = {"MutProb": self.mutation_prob, "NumSuccMut": self.NumSuccMut, "NumFailMut": self.NumFailMut, "NumNeutMut": self.NumNeutMut,
                                              "TotNumMut": self.TotNumMut}

            if current_generation < self.generation_count-1:
                # display individual log for every generation
                # self.log_individuals(population, current_generation)

                # remove the fittest individual from the population
                population.remove(fittest_individual)

                # perform crossover
                population = self.crossover(population)

                # perform mutation
                population = self.mutate(population, features, target)


                # add the fittest individual back into the population
                population.append(fittest_individual)
            else:
                if self.log_directory is not None:
                    # save the details of all individuals in population
                    # for this generation
                    self.log_individuals(population, current_generation)
                    # save the statistics of this generation to a csv file
                    self.log_generation_stats(fittest_individual, avg_fit, current_generation, best_fitness_list)
                    self.log_fitness_progress(best_fitness_list,avg_fitness_list)
                    self.log_mutation_counters(counters_df)

            end_time = timer()

            print("Generation", current_generation, "complete, time taken:", timedelta(seconds=end_time-start_time))
            # print("Fittest", fittest_individual.c_index_list ,fittest_individual.best_algorithm)

            self.mutation_prob = (self.mutation_prob + (self.NumSuccMut / self.TotNumMut))/2

            # increment generation count
            current_generation += 1



        try:
            fittest_individual = self.get_fittest(population)

            self.best_config = fittest_individual

            # print("Best configuration")
            # self.best_config.print_details()

            population.remove(fittest_individual)

            self.estimator = self.best_config.get_best_estimator()
            self.estimator.fit(features, target)

        except Exception as e:

            print("Evolved individual was unable to be trained on full \
                    training set.")
            print("It is likely that the individual was overfit during \
                    the evolution process.")
            print("Try again with different parameters, such as a \
                    higher number of individuals or generations.")
            print("For debugging, the exception is printed below.")
            print(e)
            print("Traceback:", traceback.format_exc())

        return self

    def update_assessed(self, population, assessed_configs):
        for individual in population:
            config = str(individual)
            # hash_config = hash(config)
            assessed_configs[config] = individual
        return assessed_configs



    def predict(self, features):
        """Perform mutation on the population.
        Each gene is slightly altered with a given probability.

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            The features of the dataset of which to predict the class.

        Returns
        -------
        predictions: ndarray {n_samples}
            The array of predicted classes for all samples in features.

        """

        # features = np.array(features)

        if not self.best_config:
            raise RuntimeError(
                "Auto_PU has not yet been fitted to the data. \
                Please call fit() first."
            )

        try:
            return self.estimator.predict(features)
        except Exception as e:
            print("Error for individual with following configuration")
            self.best_config.print_details()
            print(e)
            print(traceback.format_exc())