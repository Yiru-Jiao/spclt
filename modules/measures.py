'''
This script defines the measure calculator class and the measure registrator class.
The measure calculator class is used to compute the evaluation measures for the latent space.
The measure registrator class is used to register the evaluation measures.
These classes are reused and adapted from https://github.com/BorgwardtLab/topological-autoencoders
'''

import numpy as np

class MeasureRegistrator():
    k_dependent_measures = {}

    def register(self,):
        def k_dep_fn(measure):
            self.k_dependent_measures[measure.__name__] = measure
            return measure
        return k_dep_fn

    def get_k_dependent_measures(self):
        return self.k_dependent_measures


class MeasureCalculator():
    measures = MeasureRegistrator()

    def __init__(self, dist_mat_X, dist_mat_Z, k_max):
        # dist_mat_X: data, dist_mat_Z: latent, k_max: maximum number of neighbours to consider
        self.k_max = k_max
        self.dist_mat_X = dist_mat_X
        self.dist_mat_Z = dist_mat_Z
        self.neighbours_X, self.ranks_X = self._neighbours_and_ranks(self.dist_mat_X, k_max)
        self.neighbours_Z, self.ranks_Z = self._neighbours_and_ranks(self.dist_mat_Z, k_max)

    @staticmethod
    def _neighbours_and_ranks(distances, k):
        """
        Inputs: 
        - distances,        distance matrix [n times n], 
        - k,                number of nearest neighbours to consider
        Returns:
        - neighbourhood,    contains the sample indices (from 0 to n-1) of kth nearest neighbor of current sample [n times k]
        - ranks,            contains the rank of each sample to each sample [n times n], whereas entry (i,j) gives the rank that sample j has to i (the how many 'closest' neighbour j is to i) 
        """
        # Warning: this is only the ordering of neighbours that we need to
        # extract neighbourhoods below. The ranking comes later!
        indices = np.argsort(distances, axis=-1, kind='stable')

        # Extract neighbourhoods.
        neighbourhood = indices[:, 1:k+1]

        # Convert this into ranks (finally)
        ranks = indices.argsort(axis=-1, kind='stable')
        
        return neighbourhood, ranks

    def get_X_neighbours_and_ranks(self, k):
        return self.neighbours_X[:, :k], self.ranks_X

    def get_Z_neighbours_and_ranks(self, k):
        return self.neighbours_Z[:, :k], self.ranks_Z

    def compute_k_dependent_measures(self, k):
        return {key: fn(self, k) for key, fn in
                self.measures.get_k_dependent_measures().items()}

    def compute_measures_for_ks(self, ks):
        return {
            key: np.array([fn(self, k) for k in ks])
            for key, fn in self.measures.get_k_dependent_measures().items()
        }

    @staticmethod
    def _trustworthiness(X_neighbourhood, X_ranks, Z_neighbourhood,
                         Z_ranks, n, k):
        '''
        Calculates the trustworthiness measure between the data space `X`
        and the latent space `Z`, given a neighbourhood parameter `k` for
        defining the extent of neighbourhoods.
        '''

        result = 0.0

        # Calculate number of neighbours that are in the $k$-neighbourhood
        # of the latent space but not in the $k$-neighbourhood of the data
        # space.

        # # This is the non-vectorized version
        # for row in range(X_ranks.shape[0]):
        #     missing_neighbours = np.setdiff1d(
        #         Z_neighbourhood[row],
        #         X_neighbourhood[row]
        #     )

        #     for neighbour in missing_neighbours:
        #         result += (X_ranks[row, neighbour] - k)

        for row in range(X_ranks.shape[0]):
            missing_neighbours = np.isin(Z_neighbourhood[row], X_neighbourhood[row], invert=True)
            rank_differences = X_ranks[row, Z_neighbourhood[row][missing_neighbours]] - k
            result += np.sum(rank_differences)

        return 1 - 2 / (n * k * (2 * n - 3 * k - 1) ) * result


    @measures.register()
    def trustworthiness(self, k):
        X_neighbourhood, X_ranks = self.get_X_neighbours_and_ranks(k)
        Z_neighbourhood, Z_ranks = self.get_Z_neighbours_and_ranks(k)
        n = self.dist_mat_X.shape[0]
        return self._trustworthiness(X_neighbourhood, X_ranks, Z_neighbourhood,
                                     Z_ranks, n, k)

    @measures.register()
    def continuity(self, k):
        '''
        Calculates the continuity measure between the data space `X` and the
        latent space `Z`, given a neighbourhood parameter `k` for setting up
        the extent of neighbourhoods.

        This is just the 'flipped' variant of the 'trustworthiness' measure.
        '''

        X_neighbourhood, X_ranks = self.get_X_neighbours_and_ranks(k)
        Z_neighbourhood, Z_ranks = self.get_Z_neighbours_and_ranks(k)
        n = self.dist_mat_X.shape[0]
        # Notice that the parameters have to be flipped here.
        return self._trustworthiness(Z_neighbourhood, Z_ranks, X_neighbourhood,
                                     X_ranks, n, k)

    @measures.register()
    def shared_neighbours(self, k):
        '''
        Calculates the neighbourhood loss quality measure between the data
        space `X` and the latent space `Z` for some neighbourhood size $k$
        that has to be pre-defined.
        '''

        X_neighbourhood, _ = self.get_X_neighbours_and_ranks(k)
        Z_neighbourhood, _ = self.get_Z_neighbours_and_ranks(k)

        result = 0.0
        n = self.dist_mat_X.shape[0]

        for row in range(n):
            shared_neighbours = np.intersect1d(
                X_neighbourhood[row],
                Z_neighbourhood[row],
                assume_unique=True
            )

            result += len(shared_neighbours) / k

        return result / n
 
    @measures.register()
    def dist_mrre(self, k):
        '''
        Calculates the mean relative rank error quality metric of the data
        space `X` with respect to the latent space `Z`, subject to its $k$
        nearest neighbours.
        '''

        X_neighbourhood, X_ranks = self.get_X_neighbours_and_ranks(k)
        Z_neighbourhood, Z_ranks = self.get_Z_neighbours_and_ranks(k)

        n = self.dist_mat_X.shape[0]

        # First component goes from the latent space to the data space, i.e.
        # the relative quality of neighbours in `Z`.

        mrre_ZX = 0.0
        for row in range(n):
            for neighbour in Z_neighbourhood[row]:
                rx = X_ranks[row, neighbour]
                rz = Z_ranks[row, neighbour]

                mrre_ZX += abs(rx - rz) / rz

        # Second component goes from the data space to the latent space,
        # i.e. the relative quality of neighbours in `X`.

        mrre_XZ = 0.0
        for row in range(n):
            # Note that this uses a different neighbourhood definition!
            for neighbour in X_neighbourhood[row]:
                rx = X_ranks[row, neighbour]
                rz = Z_ranks[row, neighbour]

                # Note that this uses a different normalisation factor
                mrre_XZ += abs(rx - rz) / rx

        # Normalisation constant
        C = n * sum([abs(2*j - n - 1) / j for j in range(1, k+1)])
        return mrre_ZX / C, mrre_XZ / C

