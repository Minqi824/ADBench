import math

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from scipy.sparse import csr_matrix, triu, find
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.spatial import distance


class RccCluster:
    """
    Computes a clustering following: Robust continuous clustering, (Shaha and Koltunb, 2017).
    The interface is based on the sklearn.cluster module.

    Parameters
    ----------
    k (int) number of neighbors for each sample in X
    measure (string) distance metric, one of 'cosine' or 'euclidean'
    clustering_threshold (float) threshold to assign points together in a cluster. Higher means fewer larger clusters
    eps (float) numerical epsilon used for computation
    verbose (boolean) verbosity
    """

    def __init__(self, k=10, measure='euclidean', clustering_threshold = 1, eps=1e-5, verbose=True):

        self.k = k
        self.measure = measure
        self.clustering_threshold = clustering_threshold
        self.eps = eps
        self.verbose = verbose

        self.labels_ = None
        self.U = None
        self.i = None
        self.j = None
        self.n_samples = None

    def compute_assignment(self, epsilon):
        """
        Assigns points to clusters based on their representative. Two points are part of the same cluster if their
        representative are close enough (their squared euclidean distance is < delta)
        """
        for m in range(1, 100, 1):
            diff = np.sum((self.U[self.i, :] - self.U[self.j, :]) ** 2, axis=1)

            # computing connected components.
            is_conn = np.sqrt(diff) <= self.clustering_threshold * m * epsilon
            #print(m)
            G = scipy.sparse.coo_matrix((np.ones((2 * np.sum(is_conn),)),
                                         (np.concatenate([self.i[is_conn], self.j[is_conn]], axis=0),
                                          np.concatenate([self.j[is_conn], self.i[is_conn]], axis=0))),
                                        shape=[self.n_samples, self.n_samples])

            num_components, labels = connected_components(G, directed=False)
            if num_components <=20:
                # print(m)
                # print(num_components)
                return labels, num_components
                break

        #20210106,error处理(源代码会报错)
        if num_components > 20:
            return labels, num_components


    @staticmethod
    def geman_mcclure(data, mu):
        """
        Geman McClure function. See Bayesian image analysis. An application to single photon emission tomography (1985).

        Parameters
        ----------
        data (array) 2d numpy array of data
        mu (float) scale parameter
        """
        return (mu / (mu + np.sum(data ** 2, axis=1))) ** 2

    def compute_obj(self, X, U, lpq, i, j, lambda_, mu, weights, iter_num):
        """
        Computes the value of the objective function.

        Parameters
        ----------
        X (array) data points, 2d numpy array of shape (n_features, n_clusters)
        U (int) representative points, 2d numpy array of shape (n_features, n_clusters)
        lpq (array) penalty term on the connections
        i (array) first slice of w, used for convenience
        j (array) second slice of w, used for convenience
        lambda_ (float) term balancing the contributions of the losses
        mu (float) scale parameter
        weights (array) weights of the connections
        iter_num (int) current iteration, only used for printing to screen if verbose=True
        """

        # computing the objective as in equation [2]
        data = 0.5 * np.sum(np.sum((X - U) ** 2))
        diff = np.sum((U[i, :] - U[j, :]) ** 2, axis=1)
        smooth = lambda_ * 0.5 * (np.inner(lpq * weights, diff) + mu *
                                  np.inner(weights, (np.sqrt(lpq + self.eps) - 1) ** 2))

        # final objective
        obj = data + smooth
        # if self.verbose:
        #     print(' {} | {} | {} | {}'.format(iter_num, data, smooth, obj))

        return obj

    @staticmethod
    def m_knn(X, k, measure='euclidean'):
        """
        This code is taken from:
        https://bitbucket.org/sohilas/robust-continuous-clustering/src/
        The original terms of the license apply.
        Construct mutual_kNN for large scale dataset

        If j is one of i's closest neighbors and i is also one of j's closest members,
        the edge will appear once with (i,j) where i < j.

        Parameters
        ----------
        X (array) 2d array of data of shape (n_samples, n_dim)
        k (int) number of neighbors for each sample in X
        measure (string) distance metric, one of 'cosine' or 'euclidean'
        """

        samples = X.shape[0]
        batch_size = 1000000
        b = np.arange(k + 1)
        b = tuple(b[1:].ravel())

        z = np.zeros((samples, k))
        weigh = np.zeros_like(z)

        # This loop speeds up the computation by operating in batches
        # This can be parallelized to further utilize CPU/GPU resource

        for x in np.arange(0, samples, batch_size):
            start = x
            end = min(x + batch_size, samples)

            w = distance.cdist(X[start:end], X, measure)

            y = np.argpartition(w, b, axis=1)

            z[start:end, :] = y[:, 1:k + 1]
            weigh[start:end, :] = np.reshape(w[tuple(np.repeat(np.arange(end - start), k)),
                                               tuple(y[:, 1:k + 1].ravel())], (end - start, k))
            del w

        ind = np.repeat(np.arange(samples), k)

        P = csr_matrix((np.ones((samples * k)), (ind.ravel(), z.ravel())), shape=(samples, samples))
        Q = csr_matrix((weigh.ravel(), (ind.ravel(), z.ravel())), shape=(samples, samples))

        Tcsr = minimum_spanning_tree(Q)
        P = P.minimum(P.transpose()) + Tcsr.maximum(Tcsr.transpose())
        P = triu(P, k=1)

        V = np.asarray(find(P)).T
        return V[:, :2].astype(np.int32)

    def run_rcc(self, X, w, max_iter=100, inner_iter=4):
        """
        Main function for computing the clustering.

        Parameters
        ----------
        X (array) 2d array of data of shape (n_samples, n_dim).
        w (array) weights for each edge, as computed by the mutual knn clustering.
        max_iter (int) maximum number of iterations to run the algorithm.
        inner_iter (int) number of inner iterations. 4 works well in most cases.
        """

        X = X.astype(np.float32)  # features stacked as N x D (D is the dimension)

        w = w.astype(np.int32)  # list of edges represented by start and end nodes
        assert w.shape[1] == 2

        # slice w for convenience
        i = w[:, 0]
        j = w[:, 1]

        # initialization
        n_samples, n_features = X.shape

        n_pairs = w.shape[0]

        # precomputing xi
        xi = np.linalg.norm(X, 2)

        # set the weights as given in equation [S1] (supplementary information), making sure to exploit the data
        # sparsity
        R = scipy.sparse.coo_matrix((np.ones((i.shape[0] * 2,)),
                                     (np.concatenate([i, j], axis=0),
                                      np.concatenate([j, i], axis=0))), shape=[n_samples, n_samples])

        # number of connections
        n_conn = np.sum(R, axis=1)

        # make sure to convert back to a numpy array from a numpy matrix, since the output of the sum() operation on a
        # sparse matrix is a numpy matrix
        n_conn = np.asarray(n_conn)

        # equation [S1]
        weights = np.mean(n_conn) / np.sqrt(n_conn[i] * n_conn[j])
        weights = weights[:, 0]  # squueze out the unnecessary dimension

        # initializing the representatives U to have the same value as X
        U = X.copy()

        # initialize lpq to 1, that is all connections are active
        # lpq is a penalty term on the connections
        lpq = np.ones((i.shape[0],))

        # compute delta and mu, see SI for details
        epsilon = np.sqrt(np.sum((X[i, :] - X[j, :]) ** 2 + self.eps, axis=1))

        # Note: suppress low values. This hard coded threshold could lead to issues with very poorly normalized data.
        epsilon[epsilon / np.sqrt(n_features) < 1e-2] = np.max(epsilon)

        epsilon = np.sort(epsilon)

        # compute mu, see section Graduated Nonconvexity in the SI
        mu = 3.0 * epsilon[-1] ** 2

        # take the top 1% of the closest neighbours as a heuristic
        top_samples = np.minimum(250.0, math.ceil(n_pairs * 0.01))

        delta = np.mean(epsilon[:int(top_samples)])
        epsilon = np.mean(epsilon[:int(math.ceil(n_pairs * 0.01))])

        # computation of matrix A = D-R (here D is the diagonal matrix and R is the symmetric matrix), see equation (8)

        R = scipy.sparse.coo_matrix((np.concatenate([weights * lpq, weights * lpq], axis=0),
                                     (np.concatenate([i, j], axis=0), np.concatenate([j, i], axis=0))),
                                    shape=[n_samples, n_samples])

        D = scipy.sparse.coo_matrix((np.squeeze(np.asarray(np.sum(R, axis=1))),
                                     ((range(n_samples), range(n_samples)))),
                                    (n_samples, n_samples))

        # initial computation of lambda (lambda is a reserved keyword in python)
        # note: compute the largest magnitude eigenvalue instead of the matrix norm as it is faster to compute

        eigval = scipy.sparse.linalg.eigs(D - R, k=1, return_eigenvectors=False).real

        # lambda is a reserved keyword in python, so we use lambda_. Calculate lambda as per equation 9.
        lambda_ = xi / eigval[0]

        # if self.verbose:
        #     print('mu = {}, lambda = {}, epsilon = {}, delta = {}'.format(mu, lambda_, epsilon, delta))
        #     print(' Iter | Data \t | Smooth \t | Obj \t')

            # pre-allocate memory for the values of the objective function
        obj = np.zeros((max_iter,))

        inner_iter_count = 0

        # start of optimization phase

        for iter_num in range(1, max_iter):

            # update lpq. Equation 5.
            lpq = self.geman_mcclure(U[i, :] - U[j, :], mu)

            # compute objective. Equation 6.
            obj[iter_num] = self.compute_obj(X, U, lpq, i, j, lambda_, mu, weights, iter_num)

            # update U. Equation 7. For efficiency we form sparse matrices R and D separately, then combine them.

            R = scipy.sparse.coo_matrix((np.concatenate([weights * lpq, weights * lpq], axis=0),
                                         (np.concatenate([i, j], axis=0), np.concatenate([j, i], axis=0))),
                                        shape=[n_samples, n_samples])

            D = scipy.sparse.coo_matrix((np.asarray(np.sum(R, axis=1))[:, 0], ((range(n_samples), range(n_samples)))),
                                        shape=(n_samples, n_samples))

            M = scipy.sparse.eye(n_samples) + lambda_ * (D - R)

            # Solve for U. This could be further optimised through appropriate preconditioning.
            U = scipy.sparse.linalg.spsolve(M, X)

            # check for stopping criteria
            inner_iter_count += 1

            # check for the termination conditions and modulate delta if necessary.
            if (abs(obj[iter_num - 1] - obj[iter_num]) < 1e-1) or inner_iter_count == inner_iter:
                if mu >= delta:
                    mu /= 2.0
                elif inner_iter_count == inner_iter:
                    mu = 0.5 * delta
                else:
                    break

                eigval = scipy.sparse.linalg.eigs(D - R, k=1, return_eigenvectors=False).real
                lambda_ = xi / eigval[0]
                inner_iter_count = 0

        # at the end of the run, assign values to the class members.
        self.U = U.copy()
        self.i = i
        self.j = j
        self.n_samples = n_samples
        C, num_components = self.compute_assignment(epsilon)

        return U, C, num_components

    def fit(self, X):
        """
        Computes the clustering and returns the labels
        Parameters
        ----------
        X (array) numpy array of data to cluster with shape (n_samples, n_features)
        """

        assert type(X) == np.ndarray
        assert len(X.shape) == 2

        # compute the mutual knn graph
        # print(min(X.shape[0], self.k))
        mknn_matrix = self.m_knn(X, min(X.shape[0]-1, self.k), measure=self.measure)

        # perform the RCC clustering
        U, C, num_components = self.run_rcc(X, mknn_matrix)

        # store the class labels in the appropriate class member to match the sklearn.cluster interface
        self.labels_ = C.copy()

        # return the computed labels
        return self.labels_, num_components
