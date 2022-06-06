# -*- coding: utf-8 -*-
import numpy as np
import os
import warnings;

warnings.simplefilter("ignore")

from sklearn.neighbors import KDTree
from sklearn.utils.random import sample_without_replacement

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Layer
from keras.callbacks import ModelCheckpoint

MAX_INT = np.iinfo(np.int32).max


def sqr_euclidean_dist(x, y):
    return K.sum(K.square(x - y), axis=-1)


class tripletRankingLossLayer(Layer):
    """Triplet ranking loss layer Class
    """

    def __init__(self, confidence_margin, **kwargs):
        self.is_placeholder = True
        self.confidence_margin = confidence_margin
        super(tripletRankingLossLayer, self).__init__(**kwargs)

    def rankingLoss(self, input_example, input_positive, input_negative):
        """Return the mean of the triplet ranking loss"""

        positive_distances = sqr_euclidean_dist(input_example, input_positive)
        negative_distances = sqr_euclidean_dist(input_example, input_negative)
        loss = K.mean(K.maximum(0., self.confidence_margin - (negative_distances - positive_distances)))
        return loss

    def call(self, inputs):
        input_example = inputs[0]
        input_positive = inputs[1]
        input_negative = inputs[2]
        loss = self.rankingLoss(input_example, input_positive, input_negative)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return input_example;


class Repen_network:

    def __init__(self, hidden_dim=20, confidence_margin=1000.0):
        self.hidden_dim = hidden_dim
        self.confidence_margin = confidence_margin
        self.input_shape = None
        self.rankingloss = tripletRankingLossLayer(confidence_margin=self.confidence_margin)

    def compile_model(self, input_dim):
        self.input_shape = input_dim
        input_e = Input(shape=(input_dim,), name='input_e')
        input_p = Input(shape=(input_dim,), name='input_p')
        input_n = Input(shape=(input_dim,), name='input_n')

        hidden_layer = Dense(self.hidden_dim, activation='relu', name='hidden_layer')
        hidden_e = hidden_layer(input_e)
        hidden_p = hidden_layer(input_p)
        hidden_n = hidden_layer(input_n)

        output_layer = self.rankingloss([hidden_e, hidden_p, hidden_n])

        self.model = Model(inputs=[input_e, input_p, input_n], outputs=output_layer)
        self.model.compile(optimizer='adadelta',
                           loss=None)

    def get_representation(self):
        representation = Model(inputs=self.model.input[0],
                               outputs=self.model.get_layer('hidden_layer').get_output_at(0))
        return representation


class Trainer:

    def __init__(self, n_epochs=50, batch_size=256,
                 nb_batch=100, random_seed=42,
                 path_model=os.path.join(os.getcwd(), 'baseline', 'REPEN', 'model'),
                 save_suffix=None):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.nb_batch = nb_batch
        self.rng = np.random.RandomState(random_seed)
        self.path_model = path_model
        self.save_suffix = save_suffix

    def batch_generator(self, X, positive_weights,
                        negative_weights, inlier_ids,
                        outlier_ids):
        """batch generator
        """
        rng = np.random.RandomState(self.rng.randint(MAX_INT, size=1))
        counter = 0
        while 1:
            X1, X2, X3 = self.tripletBatchGeneration(X, rng, positive_weights,
                                                     negative_weights,
                                                     inlier_ids, outlier_ids)
            counter += 1
            yield ([np.array(X1), np.array(X2), np.array(X3)], None)
            if (counter > self.nb_batch):
                counter = 0

    def tripletBatchGeneration(self, X, rng,
                               positive_weights, negative_weights,
                               inlier_ids, outlier_ids):
        """batch generation
        """
        # import pdb; pdb.set_trace()
        examples = np.zeros([self.batch_size]).astype('int')
        positives = np.zeros([self.batch_size]).astype('int')
        negatives = np.zeros([self.batch_size]).astype('int')

        examples = rng.choice(inlier_ids, self.batch_size, p=positive_weights)
        positives = rng.choice(inlier_ids, self.batch_size)
        if (len(outlier_ids) == 2) and (type(outlier_ids) == list):
            # import pdb; pdb.set_trace()
            neg_1 = rng.choice(outlier_ids[0], int(self.batch_size / 2),
                               p=negative_weights[0])
            neg_2 = rng.choice(outlier_ids[1], self.batch_size - int(self.batch_size / 2),
                               p=negative_weights[1])
            negatives = np.hstack([neg_1, neg_2])
        else:
            negatives = rng.choice(outlier_ids, self.batch_size, p=negative_weights)
        inds_to_change = np.where(examples == positives)[0]
        while len(inds_to_change) > 0:
            for inds in inds_to_change:
                while examples[inds] == positives[inds]:
                    positives[inds] = rng.choice(len(inlier_ids), 1)
            inds_to_change = np.where(examples == positives)[0]

        examples = X[examples, :]
        positives = X[positives, :]
        negatives = X[negatives, :]
        return examples, positives, negatives;

    def train(self, network, mode, x_train,
              positive_weights, negative_weights,
              inlier_indices, outlier_indices,
              verbose=True):

        network.compile_model(x_train.shape[1])
        model_name = os.path.join(self.path_model, 'REPEN_'+self.save_suffix+'.h5')

        # try:
        #     model_name = self.path_model + mode + "_" + str(outlier_indices.shape[0]) + "_" + \
        #                  str(network.hidden_dim) + "_" + str(self.batch_size)
        # except:
        #     model_name = self.path_model + mode + "_" + str(np.hstack(outlier_indices).shape[0]) + "_" + \
        #                  str(network.hidden_dim) + "_" + str(self.batch_size)
        checkpointer = ModelCheckpoint(model_name, monitor='loss', verbose=0,
                                       save_best_only=True,
                                       save_weights_only=True)

        network.model.fit_generator(self.batch_generator(x_train, positive_weights,
                                                         negative_weights,
                                                         inlier_indices,
                                                         outlier_indices),
                                    steps_per_epoch=self.nb_batch,
                                    epochs=self.n_epochs,
                                    shuffle=False,
                                    callbacks=[checkpointer],
                                    verbose=verbose)
        return network


class repen:
    def __init__(self, n_epochs=50, batch_size=256, n_neighbors=2,
                 nb_batch=100, random_seed=42,
                 path_model=os.path.join(os.getcwd(), 'baseline', 'REPEN', 'model'), save_suffix=None,
                 mode="semi_supervised", known_outliers=10, hidden_dim=20,
                 confidence_margin=1000.0, input_shape=30, output=None, runs=None):

        assert (mode in ["semi_supervised", "unsupervised", "supervised"])
        self.mode = mode
        self.n_neighbors = n_neighbors
        self.known_outliers = known_outliers
        self.Trainer = Trainer(n_epochs, batch_size, nb_batch, random_seed, path_model, save_suffix)
        self.network = Repen_network(hidden_dim, confidence_margin)

    def prepare_data(self, x_train, y_train=None):

        if self.mode == "unsupervised":
            outlier_scores = self.lesinn(x_train, x_train)
            ind_scores = np.argsort(outlier_scores.flatten())

            # import pdb; pdb.set_trace()
            inlier_ids, outlier_ids = ind_scores[:-self.known_outliers:], ind_scores[-self.known_outliers:]

            transforms = np.sum(outlier_scores[inlier_ids]) - outlier_scores[inlier_ids]
            total_weights_p = np.sum(transforms)

            positive_weights = transforms / total_weights_p
            positive_weights = positive_weights.flatten()
            total_weights_n = np.sum(outlier_scores[outlier_ids])
            negative_weights = outlier_scores[outlier_ids] / total_weights_n
            negative_weights = negative_weights.flatten()

        elif self.mode == "semi_supervised":
            outlier_ids = np.where(y_train == 1)[0]

            if outlier_ids.shape[0] < self.known_outliers:
                outlier_scores = self.lesinn(x_train, x_train)
                ind_scores = np.argsort(outlier_scores.flatten())
                ind_scores = [elt for elt in ind_scores if elt not in outlier_ids]
                mn = self.known_outliers - outlier_ids.shape[0]
                to_add_idx = ind_scores[-mn:]

                total_weights_n = np.sum(outlier_scores[to_add_idx])
                negative_weights = outlier_scores[to_add_idx] / total_weights_n
                negative_weights = negative_weights.flatten()

                negative_weights = [negative_weights,
                                    np.ones(outlier_ids.shape[0]) * 1 / outlier_ids.shape[0]]
                outlier_ids = [to_add_idx, outlier_ids]

            # end if

            inlier_ids = np.delete(np.arange(len(x_train)), np.hstack(outlier_ids), axis=0)
            transforms = np.sum(outlier_scores[inlier_ids]) - outlier_scores[inlier_ids]
            total_weights_p = np.sum(transforms)

            positive_weights = transforms / total_weights_p
            positive_weights = positive_weights.flatten()

        else:
            outlier_ids = np.where(y_train == 1)[0]
            inlier_ids = np.delete(np.arange(len(x_train)),
                                   outlier_ids, axis=0)
            if outlier_ids.shape[0] > self.known_outliers:
                mn = outlier_ids.shape[0] - self.known_outliers
                remove_idx = self.Trainer.rng.choice(outlier_ids, mn, replace=False)

                outlier_ids = np.array([elt for elt in outlier_ids if elt not in remove_idx])  ## to optimize

            positive_weights = np.ones(inlier_ids.shape[0]) * (1 / inlier_ids.shape[0])
            negative_weights = np.ones(outlier_ids.shape[0]) * (1 / outlier_ids.shape[0])

        self.inlier_ids = inlier_ids
        self.outlier_ids = outlier_ids
        self.positive_weights = positive_weights
        self.negative_weights = negative_weights

    def lesinn(self, x_train, to_query):
        ensemble_size = 50
        subsample_size = 8
        scores = np.zeros([to_query.shape[0], 1])
        seeds = self.Trainer.rng.randint(MAX_INT, size=ensemble_size)
        for i in range(0, ensemble_size):
            rs = np.random.RandomState(seeds[i])
            sid = sample_without_replacement(n_population=x_train.shape[0],
                                             n_samples=subsample_size,
                                             random_state=rs)
            subsample = x_train[sid]
            kdt = KDTree(subsample, metric='euclidean')
            dists, indices = kdt.query(to_query, k=self.n_neighbors)
            # import pdb; pdb.set_trace()
            dists = np.mean(dists, axis=1)[:, np.newaxis]
            scores += dists
        scores = scores / ensemble_size
        return scores;

    def fit(self, x_train, y_train=None, verbose=False):

        self.x_train = x_train
        self.prepare_data(x_train, y_train)
        self.network = self.Trainer.train(self.network, self.mode, x_train,
                                          self.positive_weights, self.negative_weights,
                                          self.inlier_ids, self.outlier_ids,
                                          verbose=verbose)

    def decision_function(self, x_val):

        representation = self.network.get_representation()
        hidden_features_tr = representation.predict(self.x_train)
        hidden_features_val = representation.predict(x_val)
        scores = self.lesinn(hidden_features_tr, hidden_features_val)
        return scores