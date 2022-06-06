import os
import numpy as np
import torch
from torch.autograd import Variable
from myutils import Utils
import matplotlib.pyplot as plt
import seaborn as sns
from torch.distributions import MultivariateNormal, Normal
from torch.distributions.distribution import Distribution

#实例化utils
utils = Utils()

#calculate the GaussianKDE with Pytorch
class GaussianKDE(Distribution):  # 已经检验过与sklearn计算结果一致
    def __init__(self, X, bw, lam=1e-4):
        """
        X : tensor (n, d)
          `n` points with `d` dimensions to which KDE will be fit
        bw : numeric
          bandwidth for Gaussian kernel
        """
        self.X = X
        self.bw = bw
        self.dims = X.shape[-1]
        self.n = X.shape[0]
        self.mvn = MultivariateNormal(loc=torch.zeros(self.dims),
                                      covariance_matrix=torch.eye(self.dims))
        self.lam = lam

    def sample(self, num_samples):
        idxs = (np.random.uniform(0, 1, num_samples) * self.n).astype(int)
        norm = Normal(loc=self.X[idxs], scale=self.bw)
        return norm.sample()

    def score_samples(self, Y, X=None):
        """Returns the kernel density estimates of each point in `Y`.
        Parameters
        ----------
        Y : tensor (m, d)
          `m` points with `d` dimensions for which the probability density will
          be calculated
        X : tensor (n, d), optional
          `n` points with `d` dimensions to which KDE will be fit. Provided to
          allow batch calculations in `log_prob`. By default, `X` is None and
          all points used to initialize KernelDensityEstimator are included.
        Returns
        -------
        log_probs : tensor (m)
          log probability densities for each of the queried points in `Y`
        """
        if X == None:
            X = self.X

        # 注意此处取log当值接近0时会产生正负无穷的数
        # 利用with autograd.detect_anomaly()检测出算法发散的原因在于torch.log变量值接近0,需要探究接近0的原因
        log_probs = torch.log(
            (self.bw ** (-self.dims) *
             torch.exp(self.mvn.log_prob(
                 (X.unsqueeze(1) - Y) / self.bw))).sum(dim=0) / self.n + self.lam)

        return log_probs

    def log_prob(self, Y):
        """Returns the total log probability of one or more points, `Y`, using
        a Multivariate Normal kernel fit to `X` and scaled using `bw`.
        Parameters
        ----------
        Y : tensor (m, d)
          `m` points with `d` dimensions for which the probability density will
          be calculated
        Returns
        -------
        log_prob : numeric
          total log probability density for the queried points, `Y`
        """

        X_chunks = self.X.split(1000)
        Y_chunks = Y.split(1000)

        log_prob = 0

        for x in X_chunks:
            for y in Y_chunks:
                log_prob += self.score_samples(y, x).sum(dim=0)

        return log_prob

def loss_overlap(s_u, s_a, seed, bw_u=None, bw_a=None, x_num=1000, n_u=None, n_a=None,
                 resample=False, pseudo=True, plot=False, pro=False):

    if bw_u is None and bw_a is None:
        d = 1  # one-dimension data

        # Scott's Rule, which requires the data from the normal distribution.
        # This may be inappropriate when the neural network output can be arbitrary distribution
        # bw_u = n_u ** (-1. / (d + 4))
        # bw_a = n_a ** (-1. / (d + 4))

        # Silverman's Rule
        bw_u = (n_u * (d + 2) / 4.) ** (-1. / (d + 4))
        bw_a = (n_a * (d + 2) / 4.) ** (-1. / (d + 4))

    if not resample:
        # we remove the duplicated anomalies, since they may not be helpful for estimating the overall distribution
        unique, inverse = torch.unique(s_a, sorted=True, return_inverse=True, dim=0)
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        perm = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
        s_a = s_a[perm]
    else:
        assert len(s_u) == len(s_a)

    # set seed
    utils.set_seed(seed)

    # reshape
    s_u = s_u.reshape(-1, 1)
    s_a = s_a.reshape(-1, 1)

    # kde_u = GaussianKDE(X=s_u, bw=bw * torch.std(s_u, unbiased=True))
    # kde_a = GaussianKDE(X=s_a, bw=bw * torch.std(s_a, unbiased=True))

    kde_u = GaussianKDE(X=s_u, bw=bw_u)
    kde_a = GaussianKDE(X=s_a, bw=bw_a)

    if pseudo and s_u.size(0) > s_a.size(0):
        # using the fitted KDE to generate pseudo anomaly scores
        # s_a = kde_a.sample(s_u.size(0))

        # generate pseudo anomaly scores for the difference number of unlabeled data and labeled anomalies
        s_a = torch.cat((s_a, kde_a.sample(s_u.size(0) - s_a.size(0))), dim=0)

        if plot:
            sns.distplot(s_a.detach(), color='red', kde=False, bins=50)
            plt.title('Pseudo score of abnormal data')
            plt.show()

        # we observe that refit the KDE with pseudo scores would deterioriate model performance
        # kde_a = GaussianKDE(X=s_a, bw=bw_a)

    xmin = torch.min(torch.min(s_u), torch.min(s_a))
    xmax = torch.max(torch.max(s_u), torch.max(s_a))

    dx = 0.2 * (xmax - xmin)
    xmin -= dx
    xmax += dx

    x = torch.linspace(xmin.detach(), xmax.detach(), x_num)
    kde_u_x = torch.exp(kde_u.score_samples(x.reshape(-1, 1)))
    kde_a_x = torch.exp(kde_a.score_samples(x.reshape(-1, 1)))

    if plot:
        plt.plot(x, kde_u_x.detach(), color='blue')
        plt.plot(x, kde_a_x.detach(), color='red')
        plt.show()

    if pro:
        # find the intersection point (could be multiple points)
        intersection_points_idx = torch.where(torch.diff(torch.sign(kde_a_x - kde_u_x)))[0]
        if intersection_points_idx.size(0) == 1:
            #             print(f'one intersection point')
            c = x[intersection_points_idx]

            x_u, x_a = x.clone(), x.clone()
            x_u[x_u < c] = 0; x_a[x_a > c] = 0
            area_u = torch.trapz(kde_u_x, x_u)
            area_a = torch.trapz(kde_a_x, x_a)


        elif intersection_points_idx.size(0) == 2:
            #             print(f'two intersection points')
            c1 = x[intersection_points_idx[0]]
            c2 = x[intersection_points_idx[1]]

            assert c1 <= c2

            x_u, x_a = x.clone(), x.clone()
            x_u[x_u < c1] = 0; x_a[x_a > c2] = 0
            area_u = torch.trapz(kde_u_x, x_u)
            area_a = torch.trapz(kde_a_x, x_a)

        else:
            # print('The intersection points are more than 2!')
            #             raise NotImplementedError

            c1 = x[intersection_points_idx[0]]
            c2 = x[intersection_points_idx[-1]]

            assert c1 <= c2

            x_u, x_a = x.clone(), x.clone()
            x_u[x_u < c1] = 0; x_a[x_a > c2] = 0
            area_u = torch.trapz(kde_u_x, x_u)
            area_a = torch.trapz(kde_a_x, x_a)

        area = area_u + area_a

    else:
        inters_x = torch.min(kde_u_x, kde_a_x)
        area = torch.trapz(inters_x, x)

    return area

def fit(train_loader, model, optimizer, epochs, print_loss=False, device=None,
        bw_u=None, bw_a=None,
        resample=False, noise=False, pseudo=True,
        X_val_tensor=None, y_val=None, early_stopping=False, tol=5):
    '''
    noise: whether to add Gaussian noise of the output score of labeled anomalies
    bw_u: the bandwidth of unlabeled samples
    bw_a: the bandwidth of labeled anomalies
    early_stopping: whether to use early stopping based on the performance in validation set
    tol: the tolerance for early stopping
    '''

    # margin loss for keeping the order of score between normal samples and anomalies
    ranking_loss = torch.nn.MarginRankingLoss()
    if X_val_tensor is not None:
        score_val_epoch = np.empty([X_val_tensor.size(0), epochs])
    else:
        score_val_epoch = None

    best_metric_val = 0.0
    tol_count = 0

    for epoch in range(epochs):
        model.train()
        for i, data in enumerate(train_loader):

            X, y = data
            X = X.to(device); y = y.to(device)
            X = Variable(X); y = Variable(y)

            # removing duplicate samples
            unique, inverse = torch.unique(X, sorted=True, return_inverse=True, dim=0)
            perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
            inverse, perm = inverse.flip([0]), perm.flip([0])
            perm = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)

            X_unique = X[perm]
            y_unique = y[perm]

            n_u = torch.where(y_unique == 0)[0].size(0)
            n_a = torch.where(y_unique == 1)[0].size(0)

            # clear gradient
            model.zero_grad()

            # loss forward
            # 注意cv中由于batchnorm的存在要一起计算score
            _, score = model(X)

            # 由于vae的计算方式p(z|x),即每个x样本其实有其专属的高斯分布,需要逐个样本(正常样本、异常样本)计算loss并求平均
            idx_u = torch.where(y == 0)[0]
            idx_a = torch.where(y == 1)[0]

            score_u = score[idx_u]
            score_a = score[idx_a]

            if noise: #additionally inject Gaussian noise for improving robustness
                score_a = score_a + torch.zeros_like(score_a).normal_(0.0, 1.0)

            # # loss forward
            # loss_1 = loss_overlap(s_u=score_u, s_a=score_a, seed=utils.unique(epoch, i),
            #                       resample=resample, pseudo=pseudo,
            #                       bw_u=bw_u, bw_a=bw_a, n_u=n_u, n_a=n_a, pro=False)
            #
            # loss_2 = ranking_loss(score_a, score_u, torch.ones_like(score_a))
            # # combine the loss
            # loss = loss_1 + loss_2

            loss = loss_overlap(s_u=score_u, s_a=score_a, seed=utils.unique(epoch, i),
                                resample=resample, pseudo=pseudo,
                                bw_u=bw_u, bw_a=bw_a, n_u=n_u, n_a=n_a, pro=True)

            # loss backward
            loss.backward()
            # parameter update
            optimizer.step()

            if (i % 50 == 0) & print_loss:
                print('[%d/%d] [%d/%d] Loss: %.4f' % (epoch + 1, epochs, i, len(train_loader), loss))

        # storing the network output score in validation set
        if X_val_tensor is not None:
            model.eval()
            with torch.no_grad():
                _, score_val = model(X_val_tensor)
                score_val_epoch[:, epoch] = score_val.detach().numpy()


            # using the validation set for early stopping
            if early_stopping:
                # the metric in validation set
                metric_val = utils.metric(y_true=y_val, y_score=score_val)['aucpr']

                if best_metric_val < metric_val:
                    best_metric_val = metric_val
                    tol_count = 0

                    # save model
                    torch.save(model, os.path.join(os.getcwd(),'baseline','SDAD','model','SDAD.pt'))
                else:
                    tol_count += 1

                if tol_count >= tol:
                    print(f'Early stopping in epoch: {epoch}')
                    break

    return score_val_epoch