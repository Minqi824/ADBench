import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from baseline.DAGMM.forward_step import ComputeLoss

# def eval(model, dataloaders, device, n_gmm):
#     """Testing the DAGMM model"""
#     dataloader_train, dataloader_test = dataloaders
#     model.eval()
#     print('Testing...')
#     compute = ComputeLoss(model, None, None, device, n_gmm)
#     with torch.no_grad():
#         N_samples = 0
#         gamma_sum = 0
#         mu_sum = 0
#         cov_sum = 0
#         # Obtaining the parameters gamma, mu and cov using the trainin (clean) data.
#         for x, _ in dataloader_train:
#             x = x.float().to(device)
#
#             _, _, z, gamma = model(x)
#             phi_batch, mu_batch, cov_batch = compute.compute_params(z, gamma)
#
#             batch_gamma_sum = torch.sum(gamma, dim=0)
#             gamma_sum += batch_gamma_sum
#             mu_sum += mu_batch * batch_gamma_sum.unsqueeze(-1)
#             cov_sum += cov_batch * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1)
#
#             N_samples += x.size(0)
#
#         train_phi = gamma_sum / N_samples
#         train_mu = mu_sum / gamma_sum.unsqueeze(-1)
#         train_cov = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)
#
#         # Obtaining Labels and energy scores for train data
#         energy_train = []
#         labels_train = []
#         for x, y in dataloader_train:
#             x = x.float().to(device)
#
#             _, _, z, gamma = model(x)
#             sample_energy, cov_diag  = compute.compute_energy(z, gamma, phi=train_phi,
#                                                               mu=train_mu, cov=train_cov,
#                                                               sample_mean=False)
#
#             energy_train.append(sample_energy.detach().cpu())
#             labels_train.append(y)
#         energy_train = torch.cat(energy_train).numpy()
#         labels_train = torch.cat(labels_train).numpy()
#
#         # Obtaining Labels and energy scores for test data
#         energy_test = []
#         labels_test = []
#         for x, y in dataloader_test:
#             x = x.float().to(device)
#
#             _, _, z, gamma = model(x)
#             sample_energy, cov_diag  = compute.compute_energy(z, gamma, train_phi,
#                                                               train_mu, train_cov,
#                                                               sample_mean=False)
#
#             energy_test.append(sample_energy.detach().cpu())
#             labels_test.append(y)
#         energy_test = torch.cat(energy_test).numpy()
#         labels_test = torch.cat(labels_test).numpy()
#
#         scores_total = np.concatenate((energy_train, energy_test), axis=0)
#         labels_total = np.concatenate((labels_train, labels_test), axis=0)
#
#     threshold = np.percentile(scores_total, 100 - 20)
#     pred = (energy_test > threshold).astype(int)
#     gt = labels_test.astype(int)
#     precision, recall, f_score, _ = prf(gt, pred, average='binary')
#     print("Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(precision, recall, f_score))
#     print('ROC AUC score: {:.2f}'.format(roc_auc_score(labels_total, scores_total)*100))
#     return labels_total, scores_total


def eval(model, data, device, n_gmm, batch_size):
    """Testing the DAGMM model"""
    X_train = data['X_train']
    X_test = data['X_test']

    dataloader_train = DataLoader(torch.from_numpy(X_train).float(),
                                  batch_size=batch_size, shuffle=False, drop_last=True)
    dataloader_test = DataLoader(torch.from_numpy(X_test).float(),
                                 batch_size=batch_size, shuffle=False, drop_last=False)

    # evaluation mode
    model.eval()
    print('Testing...')
    compute = ComputeLoss(model, None, None, device, n_gmm)

    with torch.no_grad():
        N_samples = 0
        gamma_sum = 0
        mu_sum = 0
        cov_sum = 0

        # Obtaining the parameters gamma, mu and cov using the trainin (clean) data.
        for x in dataloader_train:
            x = x.float().to(device)

            _, _, z, gamma = model(x)
            phi_batch, mu_batch, cov_batch = compute.compute_params(z, gamma)

            batch_gamma_sum = torch.sum(gamma, dim=0)
            gamma_sum += batch_gamma_sum
            mu_sum += mu_batch * batch_gamma_sum.unsqueeze(-1)
            cov_sum += cov_batch * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1)

            N_samples += x.size(0)

        train_phi = gamma_sum / N_samples
        train_mu = mu_sum / gamma_sum.unsqueeze(-1)
        train_cov = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)

        # Obtaining Labels and energy scores for test data
        energy_test = []
        for x in dataloader_test:
            x = x.float().to(device)

            _, _, z, gamma = model(x)
            sample_energy, cov_diag = compute.compute_energy(z, gamma, train_phi,
                                                             train_mu, train_cov,
                                                             sample_mean=False)

            energy_test.append(sample_energy.detach().cpu())

        energy_test = torch.cat(energy_test).numpy() # the output score

    return energy_test