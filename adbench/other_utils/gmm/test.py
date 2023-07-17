import numpy as np
import sklearn.mixture
import torch

from gmm import GaussianMixture

import unittest


class CpuCheck(unittest.TestCase):
    """
    Basic tests for CPU.
    """
    def testPredictClasses(self):
        """
        Assert that torch.FloatTensor is handled correctly.
        """
        x = torch.randn(400, 2)
        n_components = np.random.randint(1, 100)

        model = GaussianMixture(n_components, x.size(1))
        model.fit(x)
        y = model.predict(x)

        # check that dimensionality of class memberships is (n)
        self.assertEqual(torch.Tensor(x.size(0)).size(), y.size())


    def testPredictProbabilities(self):
        """
        Assert that torch.FloatTensor is handled correctly when returning class probabilities.
        """
        x = torch.randn(400, 2)
        n_components = np.random.randint(1, 100)

        model = GaussianMixture(n_components, x.size(1))
        model.fit(x)

        # check that y_p has dimensions (n, k)
        y_p = model.predict(x, probs=True)
        self.assertEqual(torch.Tensor(x.size(0), n_components).size(), y_p.size())


    def testEmMatchesDiagSkLearn(self):
        """
        Assert that log-probabilities (E-step) and parameter updates (M-step) approximately match those of sklearn.
        """
        d = 20
        n_components = np.random.randint(1, 100)

        # (n, k, d)
        x = torch.randn(400, 1, d)
        # (n, d)
        x_np = np.squeeze(x.data.numpy())

        var_init = torch.ones(1, n_components, d) - .4

        model = GaussianMixture(n_components, d, var_init=var_init, covariance_type="diag")
        model_sk = sklearn.mixture.GaussianMixture(n_components,
            covariance_type="diag",
            init_params="random",
            means_init=np.squeeze(model.mu.data.numpy()),
            precisions_init=np.squeeze(1. / np.sqrt(var_init.data.numpy())))

        model_sk._initialize_parameters(x_np, np.random.RandomState())
        log_prob_sk = model_sk._estimate_log_prob(x_np)
        log_prob = model._estimate_log_prob(x)

        # Test whether log-probabilities are approximately equal
        np.testing.assert_almost_equal(np.squeeze(log_prob.data.numpy()),
            log_prob_sk,
            decimal=2,
            verbose=True)

        _, log_resp_sk = model_sk._e_step(x_np)
        _, log_resp = model._e_step(x)

        # Test whether E-steps are approximately equal
        np.testing.assert_almost_equal(np.squeeze(log_resp.data.numpy()),
            log_resp_sk,
            decimal=0,
            verbose=True)

        model_sk._m_step(x_np, log_prob_sk)
        pi_sk = model_sk.weights_
        mu_sk = model_sk.means_
        var_sk = model_sk.means_

        pi, mu, var = model._m_step(x, log_prob)

        # Test whether pi ..
        np.testing.assert_almost_equal(np.squeeze(pi.data.numpy()),
            pi_sk,
            decimal=1,
            verbose=True)

        # .. mu ..
        np.testing.assert_almost_equal(np.squeeze(mu.data.numpy()),
            mu_sk,
            decimal=1,
            verbose=True)

        # .. and var are approximately equal
        np.testing.assert_almost_equal(np.squeeze(var.data.numpy()),
            var_sk,
            decimal=1,
            verbose=True)

    def testEmMatchesFullSkLearn(self):
        """
        Assert that log-probabilities (E-step) and parameter updates (M-step) approximately match those of sklearn.
        """
        d = 20
        n_components = np.random.randint(1, 100)

        # (n, k, d)
        x = torch.randn(400, 1, d)
        # (n, d)
        x_np = np.squeeze(x.data.numpy())

        var_init = torch.eye(d,dtype=torch.float64).reshape(1, 1, d, d).repeat(1,n_components,1, 1)

        model = GaussianMixture(n_components, d, init_params="random", var_init=var_init, covariance_type="full")
        model_sk = sklearn.mixture.GaussianMixture(n_components,
            covariance_type="full",
            init_params="random",
            means_init=np.squeeze(model.mu.data.numpy()),
            precisions_init=np.squeeze(np.linalg.inv(var_init)))

        model_sk._initialize_parameters(x_np, np.random.RandomState())
        log_prob_sk = model_sk._estimate_log_prob(x_np)
        log_prob = model._estimate_log_prob(x)

        # Test whether log-probabilities are approximately equal
        np.testing.assert_almost_equal(np.squeeze(log_prob.data.numpy()),
            log_prob_sk,
            decimal=2,
            verbose=True)

        _, log_resp_sk = model_sk._e_step(x_np)
        _, log_resp = model._e_step(x)

        # Test whether E-steps are approximately equal
        np.testing.assert_almost_equal(np.squeeze(log_resp.data.numpy()),
            log_resp_sk,
            decimal=0,
            verbose=True)

        model_sk._m_step(x_np, log_resp_sk)
        pi_sk = model_sk.weights_
        mu_sk = model_sk.means_
        var_sk = model_sk.covariances_

        pi, mu, var = model._m_step(x, log_resp)

        # Test whether pi ..
        np.testing.assert_almost_equal(np.squeeze(pi.data.numpy()),
            pi_sk,
            decimal=1,
            verbose=True)

        # .. mu ..
        np.testing.assert_almost_equal(np.squeeze(mu.data.numpy()),
            mu_sk,
            decimal=1,
            verbose=True)

        # .. and var are approximately equal
        np.testing.assert_almost_equal(np.squeeze(var.data.numpy()),
            var_sk,
            decimal=1,
            verbose=True)


class GpuCheck(unittest.TestCase):
    """
    Basic tests for GPU.
    """
    def testPredictClasses(self):
        """
        Assert that torch.cuda.FloatTensor is handled correctly.
        """
        x = torch.randn(400, 2).cuda()
        n_components = np.random.randint(1, 100)

        model = GaussianMixture(n_components, x.size(1), covariance_type="diag").cuda()
        model.fit(x)
        y = model.predict(x)

        # check that dimensionality of class memberships is (n)
        self.assertEqual(torch.Tensor(x.size(0)).size(), y.size())


    def testPredictProbabilities(self):
        """
        Assert that torch.cuda.FloatTensor is handled correctly when returning class probabilities.
        """
        x = torch.randn(400, 2).cuda()
        n_components = np.random.randint(1, 100)

        model = GaussianMixture(n_components, x.size(1), covariance_type="diag").cuda()
        model.fit(x)

        # check that y_p has dimensions (n, k)
        y_p = model.predict(x, probs=True)
        self.assertEqual(torch.Tensor(x.size(0), n_components).size(), y_p.size())


if __name__ == "__main__":
    unittest.main()
