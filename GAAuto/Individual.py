import random as rand
import numpy as np

from sksurv.ensemble import RandomSurvivalForest
from sksurv.tree import SurvivalTree
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis


class Individual:
    """Class to store information of an individual."""
    algorithm_dict = {0: 'RSF', 1: 'Tree', 2: 'Cox'}

    def __init__(self,
                 mtry_range=[7,9,11,15,22],
                 n_trees_range=np.arange(300, 700 +1, 100),
                 min_sample_leaf_range=np.arange(10, 20 +1),
                 d0_range=np.arange(1, 4 +1),
                 l1_ratio_range=np.arange(start=0.1, stop=1.0, step=0.1),
                 alpha_range=[0.001, 0.0055, 0.01, 0.055, 0.1],
                 ):

        self.mtry_range = mtry_range
        self.n_trees_range = n_trees_range
        self.min_sample_leaf_range = min_sample_leaf_range
        self.d0_range = d0_range
        self.l1_ratio_range = l1_ratio_range
        self.alpha_range = alpha_range

        self.mtry = rand.choice(mtry_range)
        self.n_trees = rand.choice(n_trees_range)
        self.min_sample_leaf_forest = rand.choice(min_sample_leaf_range)
        self.d0_forest = rand.choice(d0_range)
        self.min_sample_leaf_tree = rand.choice(min_sample_leaf_range)
        self.d0_tree = rand.choice(d0_range)
        self.l1_ratio = rand.choice(l1_ratio_range)
        self.alpha = rand.choice(alpha_range)

        self.c_index_list = []
        self.fitness = 0
        self.best_algorithm = "Nan"


        # the list of classifiers that can be selected
        # c = Methods()
        # self.classifiers = c.classifiers

    def __str__(self):
        return "{} {} {} {} {} {} {:.1f} {}".format(
            self.mtry, self.n_trees, self.min_sample_leaf_forest, self.d0_forest,
            self.min_sample_leaf_tree, self.d0_tree,
            self.l1_ratio, self.alpha
        )

    def get_rsf(self):
        return RandomSurvivalForest(max_features=self.mtry, d0=self.d0_forest, n_estimators=self.n_trees, min_samples_leaf=self.min_sample_leaf_forest, n_jobs=-1, random_state=0)

    def get_survival_tree(self):
        return SurvivalTree(min_samples_leaf=self.min_sample_leaf_tree, d0=self.d0_tree)

    def get_cox_model(self):
        return CoxnetSurvivalAnalysis(l1_ratio=self.l1_ratio, alpha_min_ratio=self.alpha)

    def set_best(self, c_val_list):
        self.c_index_list = c_val_list
        max_value = max(c_val_list)
        max_index = c_val_list.index(max_value)

        self.fitness = max_value
        self.best_algorithm = self.algorithm_dict[max_index]

    def get_best_estimator(self):
        if self.best_algorithm == "RSF":
            return self.get_rsf()
        elif self.best_algorithm == "Tree":
            return self.get_survival_tree()
        elif self.best_algorithm == "Cox":
            return self.get_cox_model()

        return None

    def print_details(self):
        """Print the details of the individual configuration.

        """
        print("Configuration details for individual")
        print("[RSF]:", self.c_index_list[0])
        print("mtry:", self.mtry)
        print("n_trees:", self.n_trees)
        print("min_sample_leaf:", self.min_sample_leaf_forest)
        print("d_0:", self.d0_forest)
        print("[Tree]: ", self.c_index_list[1])
        print("min_sample_leaf:", self.min_sample_leaf_tree)
        print("d_0:", self.d0_tree)
        print("[Cox]: ", self.c_index_list[2])
        print("l1:", self.l1_ratio)
        print("alpha:", self.alpha)
        print("Fitness:", self.fitness)
    
    def get_config(self):
        """Return the components of the individual.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        iteration_count_1_a: int
            The number of iterations to perform in Phase 1A.
        rn_theshold_1_a: float
            The value under which an instance is considered reliably
            negative in Phase 1A.
        classifier_1_a: Object
            The classifier to use in Phase 1A.
        flag_1_b: bool
            Whether to use Phase 1B.
        iteration_count_1_b: int
            The number of iterations to perform in Phase 1B.
        rn_threshold_1_b: float
            The value under which an instance is considered reliably
            negative in Phase 1B.
        classifier_1_b: Object
            The classifier to use in Phase 1B.
        classifier_2: Object
            The classifier to use in Phase 2.
        """

        return [self.mtry, self.n_trees, self.min_sample_leaf_forest, self.d0_forest, self.min_sample_leaf_tree, self.d0_tree, self.l1_ratio, self.alpha, self.fitness, self.best_algorithm]