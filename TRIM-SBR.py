# %%
# import system packages
import os
import pickle
import itertools
import logging
import re
import time
import glob
import inspect

import numpy as np
from sklearn.neighbors import NearestNeighbors

# for handler in _logger.root.handlers[:]:
#    _logger.root.removeHandler(handler)

# setting the _logger format
_logger = logging.getLogger('Testing')
_logger.setLevel(logging.DEBUG)
_logger_ch = logging.StreamHandler()
_logger_ch.setFormatter(logging.Formatter(
    "%(asctime)s:%(levelname)s:%(message)s"))
_logger.addHandler(_logger_ch)

# exported names
__all__ = ['OverSampling',
           'SMOTE',
           'MulticlassOversampling']

class StatisticsMixin:
    """
    Mixin to compute class statistics and determine minority/majority labels
    """

    def class_label_statistics(self, X, y):
        """
        determines class sizes and minority and majority labels
        Args:
            X (np.array): features
            y (np.array): target labels
        """
        unique, counts = np.unique(y, return_counts=True)
        self.class_stats = dict(zip(unique, counts))
        self.min_label = unique[0] if counts[0] < counts[1] else unique[1]
        self.maj_label = unique[1] if counts[0] < counts[1] else unique[0]
        # shorthands
        self.min_label = self.min_label
        self.maj_label = self.maj_label

    def check_enough_min_samples_for_sampling(self, threshold=2):
        if self.class_stats[self.min_label] < threshold:
            m = ("The number of minority samples (%d) is not enough "
                 "for sampling")
            m = m % self.class_stats[self.min_label]
            _logger.warning(self.__class__.__name__ + ": " + m)
            return False
        return True

class RandomStateMixin:
    """
    Mixin to set random state
    """

    def set_random_state(self, random_state):
        """
        sets the random_state member of the object
        Args:
            random_state (int/np.random.RandomState/None): the random state
                                                                initializer
        """

        self._random_state_init = random_state

        if random_state is None:
            self.random_state = np.random
        elif isinstance(random_state, int):
            self.random_state = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        elif random_state is np.random:
            self.random_state = random_state
        else:
            raise ValueError(
                "random state cannot be initialized by " + str(random_state))

class ParameterCheckingMixin:
    """
    Mixin to check if parameters come from a valid range
    """

    def check_in_range(self, x, name, r):
        """
        Check if parameter is in range
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            r (list-like(2)): the lower and upper bound of a range
        Throws:
            ValueError
        """
        if x < r[0] or x > r[1]:
            m = ("Value for parameter %s outside the range [%f,%f] not"
                 " allowed: %f")
            m = m % (name, r[0], r[1], x)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_out_range(self, x, name, r):
        """
        Check if parameter is outside of range
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            r (list-like(2)): the lower and upper bound of a range
        Throws:
            ValueError
        """
        if x >= r[0] and x <= r[1]:
            m = "Value for parameter %s in the range [%f,%f] not allowed: %f"
            m = m % (name, r[0], r[1], x)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_less_or_equal(self, x, name, val):
        """
        Check if parameter is less than or equal to value
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            val (numeric): value to compare to
        Throws:
            ValueError
        """
        if x > val:
            m = "Value for parameter %s greater than %f not allowed: %f > %f"
            m = m % (name, val, x, val)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_less_or_equal_par(self, x, name_x, y, name_y):
        """
        Check if parameter is less than or equal to another parameter
        Args:
            x (numeric): the parameter value
            name_x (str): the parameter name
            y (numeric): the other parameter value
            name_y (str): the other parameter name
        Throws:
            ValueError
        """
        if x > y:
            m = ("Value for parameter %s greater than parameter %s not"
                 " allowed: %f > %f")
            m = m % (name_x, name_y, x, y)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_less(self, x, name, val):
        """
        Check if parameter is less than value
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            val (numeric): value to compare to
        Throws:
            ValueError
        """
        if x >= val:
            m = ("Value for parameter %s greater than or equal to %f"
                 " not allowed: %f >= %f")
            m = m % (name, val, x, val)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_less_par(self, x, name_x, y, name_y):
        """
        Check if parameter is less than another parameter
        Args:
            x (numeric): the parameter value
            name_x (str): the parameter name
            y (numeric): the other parameter value
            name_y (str): the other parameter name
        Throws:
            ValueError
        """
        if x >= y:
            m = ("Value for parameter %s greater than or equal to parameter"
                 " %s not allowed: %f >= %f")
            m = m % (name_x, name_y, x, y)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_greater_or_equal(self, x, name, val):
        """
        Check if parameter is greater than or equal to value
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            val (numeric): value to compare to
        Throws:
            ValueError
        """
        if x < val:
            m = "Value for parameter %s less than %f is not allowed: %f < %f"
            m = m % (name, val, x, val)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_greater_or_equal_par(self, x, name_x, y, name_y):
        """
        Check if parameter is less than or equal to another parameter
        Args:
            x (numeric): the parameter value
            name_x (str): the parameter name
            y (numeric): the other parameter value
            name_y (str): the other parameter name
        Throws:
            ValueError
        """
        if x < y:
            m = ("Value for parameter %s less than parameter %s is not"
                 " allowed: %f < %f")
            m = m % (name_x, name_y, x, y)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_greater(self, x, name, val):
        """
        Check if parameter is greater than value
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            val (numeric): value to compare to
        Throws:
            ValueError
        """
        if x <= val:
            m = ("Value for parameter %s less than or equal to %f not allowed"
                 " %f < %f")
            m = m % (name, val, x, val)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_greater_par(self, x, name_x, y, name_y):
        """
        Check if parameter is greater than or equal to another parameter
        Args:
            x (numeric): the parameter value
            name_x (str): the parameter name
            y (numeric): the other parameter value
            name_y (str): the other parameter name
        Throws:
            ValueError
        """
        if x <= y:
            m = ("Value for parameter %s less than or equal to parameter %s"
                 " not allowed: %f <= %f")
            m = m % (name_x, name_y, x, y)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_equal(self, x, name, val):
        """
        Check if parameter is equal to value
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            val (numeric): value to compare to
        Throws:
            ValueError
        """
        if x == val:
            m = ("Value for parameter %s equal to parameter %f is not allowed:"
                 " %f == %f")
            m = m % (name, val, x, val)
            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_equal_par(self, x, name_x, y, name_y):
        """
        Check if parameter is equal to another parameter
        Args:
            x (numeric): the parameter value
            name_x (str): the parameter name
            y (numeric): the other parameter value
            name_y (str): the other parameter name
        Throws:
            ValueError
        """
        if x == y:
            m = ("Value for parameter %s equal to parameter %s is not "
                 "allowed: %f == %f")
            m = m % (name_x, name_y, x, y)
            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_isin(self, x, name, li):
        """
        Check if parameter is in list
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            li (list): list to check if parameter is in it
        Throws:
            ValueError
        """
        if x not in li:
            m = "Value for parameter %s not in list %s is not allowed: %s"
            m = m % (name, str(li), str(x))
            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_n_jobs(self, x, name):
        """
        Check n_jobs parameter
        Args:
            x (int/None): number of jobs
            name (str): the parameter name
        Throws:
            ValueError
        """
        if not ((x is None)
                or (x is not None and isinstance(x, int) and not x == 0)):
            m = "Value for parameter n_jobs is not allowed: %s" % str(x)
            raise ValueError(self.__class__.__name__ + ": " + m)


class ParameterCombinationsMixin:
    """
    Mixin to generate parameter combinations
    """

    @classmethod
    def generate_parameter_combinations(cls, dictionary, raw):
        """
        Generates reasonable paramter combinations
        Args:
            dictionary (dict): dictionary of paramter ranges
            num (int): maximum number of combinations to generate
        """
        if raw:
            return dictionary
        keys = sorted(list(dictionary.keys()))
        values = [dictionary[k] for k in keys]
        combinations = [dict(zip(keys, p))
                        for p in list(itertools.product(*values))]
        return combinations

class OverSampling(StatisticsMixin,
                   ParameterCheckingMixin,
                   ParameterCombinationsMixin,
                   RandomStateMixin):
    """
    Base class of oversampling methods
    """

    categories = []

    cat_noise_removal = 'NR'
    cat_dim_reduction = 'DR'
    cat_uses_classifier = 'Clas'
    cat_sample_componentwise = 'SCmp'
    cat_sample_ordinary = 'SO'
    cat_sample_copy = 'SCpy'
    cat_memetic = 'M'
    cat_density_estimation = 'DE'
    cat_density_based = 'DB'
    cat_extensive = 'Ex'
    cat_changes_majority = 'CM'
    cat_uses_clustering = 'Clus'
    cat_borderline = 'BL'
    cat_application = 'A'

    def __init__(self):
        pass

    def det_n_to_sample(self, strategy, n_maj, n_min):
        """
        Determines the number of samples to generate
        Args:
            strategy (str/float): if float, the fraction of the difference
                                    of the minority and majority numbers to
                                    generate, like 0.1 means that 10% of the
                                    difference will be generated if str,
                                    like 'min2maj', the minority class will
                                    be upsampled to match the cardinality
                                    of the majority class
        """
        if isinstance(strategy, float) or isinstance(strategy, int):
            return max([0, int((n_maj - n_min)*strategy)])
        else:
            m = "Value %s for parameter strategy is not supported" % strategy
            raise ValueError(self.__class__.__name__ + ": " + m)

    def sample_between_points(self, x, y):
        """
        Sample randomly along the line between two points.
        Args:
            x (np.array): point 1
            y (np.array): point 2
        Returns:
            np.array: the new sample
        """
        return x + (y - x)*self.random_state.random_sample()

    def sample_by_gaussian_jittering(self, x, std):
        """
        Sample by Gaussian jittering
        Args:
            x (np.array): base point
            std (np.array): standard deviation
        Returns:
            np.array: the new sample
        """
        return self.random_state.normal(x, std)

    def sample(self, X, y):
        """
        The samplig function reimplemented in child classes
        Args:
            X (np.matrix): features
            y (np.array): labels
        Returns:
            np.matrix, np.array: sampled X and y
        """
        return X, y

    def fit_resample(self, X, y):
        """
        Alias of the function "sample" for compatibility with imbalanced-learn
        pipelines
        """
        return self.sample(X, y)

    def get_params(self, deep=False):
        """
        Returns the parameters of the object as a dictionary.
        Returns:
            dict: the parameters of the object
        """
        pass

    def descriptor(self):
        """
        Returns:
            str: JSON description of the current sampling object
        """
        return str((self.__class__.__name__, str(self.get_params())))

    def __str__(self):
        return self.descriptor()

class SMOTE(OverSampling):
    """
    References:
        * BibTex::
            @article{smote,
                author={Chawla, N. V. and Bowyer, K. W. and Hall, L. O. and
                            Kegelmeyer, W. P.},
                title={{SMOTE}: synthetic minority over-sampling technique},
                journal={Journal of Artificial Intelligence Research},
                volume={16},
                year={2002},
                pages={321--357}
              }
    """

    categories = [OverSampling.cat_sample_ordinary,
                  OverSampling.cat_extensive]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the SMOTE object
        Args:
            proportion (float): proportion of the difference of n_maj and
                                n_min to sample e.g. 1.0
            means that after sampling the number of minority samples will
                                 be equal to the number of majority samples
            n_neighbors (int): control parameter of the nearest neighbor
                                technique
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()

        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.
        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7]}

        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.
        Args:
            X (np.ndarray): training set
            y (np.array): target labels
        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        # determining the number of samples to generate
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            # _logger.warning(self.__class__.__name__ +
            #                ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]

        # fitting the model
        n_neigh = min([len(X_min), self.n_neighbors+1])
        nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=self.n_jobs)
        nn.fit(X_min)
        dist, ind = nn.kneighbors(X_min)

        if n_to_sample == 0:
            return X.copy(), y.copy()

        # generating samples
        base_indices = self.random_state.choice(list(range(len(X_min))),
                                                n_to_sample)
        neighbor_indices = self.random_state.choice(list(range(1, n_neigh)),
                                                    n_to_sample)

        X_base = X_min[base_indices]
        X_neighbor = X_min[ind[base_indices, neighbor_indices]]

        samples = X_base + np.multiply(self.random_state.rand(n_to_sample,
                                                              1),
                                       X_neighbor - X_base)

        return (np.vstack([X, samples]),
                np.hstack([y, np.hstack([self.min_label]*n_to_sample)]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


###############################################################################

class TRIM_SBR(OverSampling):

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_sample_componentwise,
                  OverSampling.cat_uses_clustering]

    def __init__(self,
                 proportion=1.0,
                 min_precision=0.3,
                 random_state=None):
        """
        Constructor of the sampling object
        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_in_range(min_precision, 'min_precision', [0, 1])

        self.proportion = proportion
        self.min_precision = min_precision

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.
        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'min_precision': [0.3]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def trim(self, y):
        """
        Determines the trim value.
        Args:
            y (np.array): array of target labels
        Returns:
            float: the trim value
        """
        return np.sum(y == self.min_label)**2/len(y)

    def precision(self, y):
        """
        Determines the precision value.
        Args:
            y (np.array): array of target labels
        Returns:
            float: the precision value
        """
        return np.sum(y == self.min_label)/len(y)

    def determine_splitting_point(self, X, y, split_on_border=False):
        """
        Determines the splitting point.
        Args:
            X (np.matrix): a subset of the training data
            y (np.array): an array of target labels
            split_on_border (bool): wether splitting on class borders is
                                    considered
        Returns:
            tuple(int, float), bool: (splitting feature, splitting value),
                                        make the split
        """
        trim_value = self.trim(y)
        d = len(X[0])
        max_t_minus_gain = 0.0
        split = None

        # checking all dimensions of X
        for i in range(d):
            # sort the elements in dimension i
            sorted_X_y = sorted(zip(X[:, i], y), key=lambda pair: pair[0])
            sorted_y = [yy for _, yy in sorted_X_y]

            # number of minority samples on the left
            left_min = 0
            # number of minority samples on the right
            right_min = np.sum(sorted_y == self.min_label)

            # check all possible splitting points sequentiall
            for j in range(0, len(sorted_y)-1):
                if sorted_y[j] == self.min_label:
                    # adjusting the number of minority and majority samples
                    left_min = left_min + 1
                    right_min = right_min - 1
                # checking of we can split on the border and do not split
                # tieing feature values
                if ((split_on_border is False
                     or (split_on_border is True
                         and not sorted_y[j-1] == sorted_y[j]))
                        and sorted_X_y[j][0] != sorted_X_y[j+1][0]):
                    # compute trim value of the left
                    trim_left = left_min**2/(j+1)
                    # compute trim value of the right
                    trim_right = right_min**2/(len(sorted_y) - j - 1)
                    # let's check the gain
                    if max([trim_left, trim_right]) > max_t_minus_gain:
                        max_t_minus_gain = max([trim_left, trim_right])
                        split = (i, sorted_X_y[j][0])
        # return splitting values and the value of the logical condition
        # in line 9
        if split is not None:
            return split, max_t_minus_gain > trim_value
        else:
            return (0, 0), False

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.
        Args:
            X (np.ndarray): training set
            y (np.array): target labels
        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        leafs = [(X, y)]
        candidates = []
        seeds = []

        # executing the trimming
        # loop in line 2 of the paper
        _logger.info(self.__class__.__name__ +
                     ": " + "do the trimming process")
        while len(leafs) > 0 or len(candidates) > 0:
            add_to_leafs = []
            # executing the loop starting in line 3
            for leaf in leafs:
                # the function implements the loop starting in line 6
                # splitting on class border is forced
                split, gain = self.determine_splitting_point(
                    leaf[0], leaf[1], True)
                if len(leaf[0]) == 1:
                    # small leafs with 1 element (no splitting point)
                    # are dropped as noise
                    continue
                else:
                    # condition in line 9
                    if gain:
                        # making the split
                        mask_left = (leaf[0][:, split[0]] <= split[1])
                        X_left = leaf[0][mask_left]
                        y_left = leaf[1][mask_left]
                        mask_right = np.logical_not(mask_left)
                        X_right = leaf[0][mask_right]
                        y_right = leaf[1][mask_right]

                        # condition in line 11
                        if np.sum(y_left == self.min_label) > 0:
                            add_to_leafs.append((X_left, y_left))
                        # condition in line 13
                        if np.sum(y_right == self.min_label) > 0:
                            add_to_leafs.append((X_right, y_right))
                    else:
                        # line 16
                        candidates.append(leaf)
            # we implement line 15 and 18 by replacing the list of leafs by
            # the list of new leafs.
            leafs = add_to_leafs

            # iterating through all candidates (loop starting in line 21)
            for c in candidates:
                # extracting splitting points, this time split on border
                # is not forced
                split, gain = self.determine_splitting_point(c[0], c[1], False)
                if len(c[0]) == 1:
                    # small leafs are dropped as noise
                    continue
                else:
                    # checking condition in line 27
                    if gain:
                        # doing the split
                        mask_left = (c[0][:, split[0]] <= split[1])
                        X_left, y_left = c[0][mask_left], c[1][mask_left]
                        mask_right = np.logical_not(mask_left)
                        X_right, y_right = c[0][mask_right], c[1][mask_right]
                        # checking logic in line 29
                        if np.sum(y_left == self.min_label) > 0:
                            leafs.append((X_left, y_left))
                        # checking logic in line 31
                        if np.sum(y_right == self.min_label) > 0:
                            leafs.append((X_right, y_right))
                    else:
                        # adding candidate to seeds (line 35)
                        seeds.append(c)
            # line 33 and line 36 are implemented by emptying the candidates
            # list
            candidates = []

        # filtering the resulting set
        filtered_seeds = [s for s in seeds if self.precision(
            s[1]) > self.min_precision]

        # handling the situation when no seeds were found
        if len(seeds) == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "no seeds identified")
            return X.copy(), y.copy()

        # fix for bad choice of min_precision
        multiplier = 0.9
        while len(filtered_seeds) == 0:
            filtered_seeds = [s for s in seeds if self.precision(
                s[1]) > self.min_precision*multiplier]
            multiplier = multiplier*0.9
            if multiplier < 0.1:
                _logger.warning(self.__class__.__name__ + ": " +
                                "no clusters passing the filtering")
                return X.copy(), y.copy()

        seeds = filtered_seeds

        X_seed = np.vstack([s[0] for s in seeds])
        y_seed = np.hstack([s[1] for s in seeds])

        # generating samples by ROSE
        X_seed_minH = X_seed[y_seed == self.min_label]
        if len(X_seed_minH) <= 1:
            _logger.warning(self.__class__.__name__ + ": " +
                            "X_seed_minH contains less than 2 samples")
            return X.copy(), y.copy()

        # Estimating the H matrix
        std = np.std(X_seed_minH, axis=0)
        d = len(X[0])
        n = len(X_seed_minH)
        H = std*(4.0/((d + 1)*n))**(1.0/(d + 4))
        # https://stackoverflow.com/questions/55366188/why-do-stat-density-r-ggplot2-and-gaussian-kde-python-scipy-differ
        # https://en.wikipedia.org/wiki/Kernel_density_estimation#A_rule-of-thumb_bandwidth_estimator
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
        

        # do the sampling
        samples = []
        for _ in range(n_to_sample):
            random_idxH = self.random_state.randint(len(X_seed_minH))
            samples.append(self.sample_by_gaussian_jittering(
                X_seed_minH[random_idxH], H))

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'min_precision': self.min_precision,
                'random_state': self._random_state_init}

class MulticlassOversampling(StatisticsMixin):
    """
    Carries out multiclass oversampling
    Example::
        import smote_variants as sv
        import sklearn.datasets as datasets
        dataset= datasets.load_wine()
        oversampler= sv.MulticlassOversampling(sv.distance_SMOTE())
        X_samp, y_samp= oversampler.sample(dataset['data'], dataset['target'])
    """

    def __init__(self,
                 oversampler=SMOTE(random_state=2),
                 strategy="eq_1_vs_many_successive"):
        """
        Constructor of the multiclass oversampling object
        Args:
            oversampler (obj): an oversampling object
            strategy (str/obj): a multiclass oversampling strategy, currently
                                'eq_1_vs_many_successive' or
                                'equalize_1_vs_many'
        """
        self.oversampler = oversampler
        self.strategy = strategy

    def sample_equalize_1_vs_many(self, X, y):
        """
        Does the sample generation by oversampling each minority class to the
        cardinality of the majority class using all original samples in each
        run.
        Args:
            X (np.ndarray): training set
            y (np.array): target labels
        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """

        message = "Running multiclass oversampling with strategy %s"
        message = message % str(self.strategy)
        _logger.info(self.__class__.__name__ + ": " + message)

        if 'proportion' not in self.oversampler.get_params():
            message = ("Multiclass oversampling strategy %s cannot be "
                       "used with oversampling techniques without proportion"
                       " parameter")
            message = message % str(self.strategy)
            raise ValueError(message)

        # extract class label statistics
        self.class_label_statistics(X, y)

        # sort labels by number of samples
        class_labels = self.class_stats.keys()
        class_labels = sorted(class_labels, key=lambda x: -self.class_stats[x])

        majority_class_label = class_labels[0]

        # determining the majority class data
        X_maj = X[y == majority_class_label]

        # dict to store the results
        results = {}
        results[majority_class_label] = X_maj.copy()

        # running oversampling for all minority classes against all oversampled
        # classes
        for i in range(1, len(class_labels)):
            message = "Sampling minority class with label: %d"
            message = message % class_labels[i]
            _logger.info(self.__class__.__name__ + ": " + message)

            # extract current minority class
            minority_class_label = class_labels[i]
            X_min = X[y == minority_class_label]
            X_maj = X[y != minority_class_label]

            # prepare data to pass to oversampling
            X_training = np.vstack([X_maj, X_min])
            y_training = np.hstack(
                [np.repeat(0, len(X_maj)), np.repeat(1, len(X_min))])

            # prepare parameters by properly setting the proportion value
            params = self.oversampler.get_params()

            num_to_generate = self.class_stats[majority_class_label] - \
                self.class_stats[class_labels[i]]
            num_to_gen_to_all = len(X_maj) - self.class_stats[class_labels[i]]

            params['proportion'] = num_to_generate/num_to_gen_to_all

            # instantiating new oversampling object with the proper proportion
            # parameter
            oversampler = self.oversampler.__class__(**params)

            # executing the sampling
            X_samp, y_samp = oversampler.sample(X_training, y_training)

            # registaring the newly oversampled minority class in the output
            # set
            results[class_labels[i]] = X_samp[len(
                X_training):][y_samp[len(X_training):] == 1]

        # constructing the output set
        X_final = results[class_labels[1]]
        y_final = np.repeat(class_labels[1], len(results[class_labels[1]]))

        for i in range(2, len(class_labels)):
            X_final = np.vstack([X_final, results[class_labels[i]]])
            y_new = np.repeat(class_labels[i], len(results[class_labels[i]]))
            y_final = np.hstack([y_final, y_new])

        return np.vstack([X, X_final]), np.hstack([y, y_final])

    def sample_equalize_1_vs_many_successive(self, X, y):
        """
        Does the sample generation by oversampling each minority class
        successively to the cardinality of the majority class,
        incorporating the results of previous oversamplings.
        Args:
            X (np.ndarray): training set
            y (np.array): target labels
        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """

        message = "Running multiclass oversampling with strategy %s"
        message = message % str(self.strategy)
        _logger.info(self.__class__.__name__ + ": " + message)

        if 'proportion' not in self.oversampler.get_params():
            message = ("Multiclass oversampling strategy %s cannot be used"
                       " with oversampling techniques without proportion"
                       " parameter") % str(self.strategy)
            raise ValueError(message)

        # extract class label statistics
        self.class_label_statistics(X, y)

        # sort labels by number of samples
        class_labels = self.class_stats.keys()
        class_labels = sorted(class_labels, key=lambda x: -self.class_stats[x])

        majority_class_label = class_labels[0]

        # determining the majority class data
        X_maj = X[y == majority_class_label]

        # dict to store the results
        results = {}
        results[majority_class_label] = X_maj.copy()

        # running oversampling for all minority classes against all
        # oversampled classes
        for i in range(1, len(class_labels)):
            message = "Sampling minority class with label: %d"
            message = message % class_labels[i]
            _logger.info(self.__class__.__name__ + ": " + message)

            # extract current minority class
            minority_class_label = class_labels[i]
            X_min = X[y == minority_class_label]

            # prepare data to pass to oversampling
            X_training = np.vstack([X_maj, X_min])
            y_training = np.hstack(
                [np.repeat(0, len(X_maj)), np.repeat(1, len(X_min))])

            # prepare parameters by properly setting the proportion value
            params = self.oversampler.get_params()

            n_majority = self.class_stats[majority_class_label]
            n_class_i = self.class_stats[class_labels[i]]
            num_to_generate = n_majority - n_class_i

            num_to_gen_to_all = i * n_majority - n_class_i

            params['proportion'] = num_to_generate/num_to_gen_to_all

            # instantiating new oversampling object with the proper proportion
            # parameter
            oversampler = self.oversampler.__class__(**params)

            # executing the sampling
            X_samp, y_samp = oversampler.sample(X_training, y_training)

            # adding the newly oversampled minority class to the majority data
            X_maj = np.vstack([X_maj, X_samp[y_samp == 1]])

            # registaring the newly oversampled minority class in the output
            # set
            result_mask = y_samp[len(X_training):] == 1
            results[class_labels[i]] = X_samp[len(X_training):][result_mask]

        # constructing the output set
        X_final = results[class_labels[1]]
        y_final = np.repeat(class_labels[1], len(results[class_labels[1]]))

        for i in range(2, len(class_labels)):
            X_final = np.vstack([X_final, results[class_labels[i]]])
            y_new = np.repeat(class_labels[i], len(results[class_labels[i]]))
            y_final = np.hstack([y_final, y_new])

        return np.vstack([X, X_final]), np.hstack([y, y_final])

    def sample(self, X, y):
        """
        Does the sample generation according to the oversampling strategy.
        Args:
            X (np.ndarray): training set
            y (np.array): target labels
        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """

        if self.strategy == "eq_1_vs_many_successive":
            return self.sample_equalize_1_vs_many_successive(X, y)
        elif self.strategy == "equalize_1_vs_many":
            return self.sample_equalize_1_vs_many(X, y)
        else:
            message = "Multiclass oversampling startegy %s not implemented."
            message = message % self.strategy
            raise ValueError(message)

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the multiclass oversampling object
        """
        return {'oversampler': self.oversampler, 'strategy': self.strategy}
