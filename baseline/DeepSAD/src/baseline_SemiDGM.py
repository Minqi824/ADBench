import click
import torch
import logging
import random
import numpy as np

from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from baselines.SemiDGM import SemiDeepGenerativeModel
from datasets.main import load_dataset


################################################################################
# Settings
################################################################################
@click.command()
@click.argument('dataset_name', type=click.Choice(['mnist', 'fmnist', 'cifar10', 'arrhythmia', 'cardio', 'satellite',
                                                   'satimage-2', 'shuttle', 'thyroid']))
@click.argument('net_name', type=click.Choice(['mnist_DGM_M2', 'mnist_DGM_M1M2', 'fmnist_DGM_M2', 'fmnist_DGM_M1M2',
                                               'cifar10_DGM_M2', 'cifar10_DGM_M1M2',
                                               'arrhythmia_DGM_M2', 'cardio_DGM_M2', 'satellite_DGM_M2',
                                               'satimage-2_DGM_M2', 'shuttle_DGM_M2', 'thyroid_DGM_M2']))
@click.argument('xp_path', type=click.Path(exists=True))
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--load_config', type=click.Path(exists=True), default=None,
              help='Config JSON-file path (default: None).')
@click.option('--load_model', type=click.Path(exists=True), default=None,
              help='Model file path (default: None).')
@click.option('--ratio_known_normal', type=float, default=0.0,
              help='Ratio of known (labeled) normal training examples.')
@click.option('--ratio_known_outlier', type=float, default=0.0,
              help='Ratio of known (labeled) anomalous training examples.')
@click.option('--ratio_pollution', type=float, default=0.0,
              help='Pollution ratio of unlabeled training data with unknown (unlabeled) anomalies.')
@click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--seed', type=int, default=-1, help='Set seed. If -1, use randomization.')
@click.option('--optimizer_name', type=click.Choice(['adam']), default='adam',
              help='Name of the optimizer to use for training the Semi-Supervised Deep Generative model.')
@click.option('--lr', type=float, default=0.001,
              help='Initial learning rate for training. Default=0.001')
@click.option('--n_epochs', type=int, default=50, help='Number of epochs to train.')
@click.option('--lr_milestone', type=int, default=0, multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--batch_size', type=int, default=128, help='Batch size for mini-batch training.')
@click.option('--weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter.')
@click.option('--pretrain', type=bool, default=False, help='Pretrain a variational autoencoder.')
@click.option('--vae_optimizer_name', type=click.Choice(['adam']), default='adam',
              help='Name of the optimizer to use for variational autoencoder pretraining.')
@click.option('--vae_lr', type=float, default=0.001,
              help='Initial learning rate for pretraining. Default=0.001')
@click.option('--vae_n_epochs', type=int, default=100, help='Number of epochs to train the variational autoencoder.')
@click.option('--vae_lr_milestone', type=int, default=0, multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--vae_batch_size', type=int, default=128, help='Batch size for variational autoencoder training.')
@click.option('--vae_weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for variational autoencoder.')
@click.option('--num_threads', type=int, default=0,
              help='Number of threads used for parallelizing CPU operations. 0 means that all resources are used.')
@click.option('--n_jobs_dataloader', type=int, default=0,
              help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
@click.option('--normal_class', type=int, default=0,
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')
@click.option('--known_outlier_class', type=int, default=1,
              help='Specify the known outlier class of the dataset for semi-supervised anomaly detection.')
@click.option('--n_known_outlier_classes', type=int, default=0,
              help='Number of known outlier classes.'
                   'If 0, no anomalies are known.'
                   'If 1, outlier class as specified in --known_outlier_class option.'
                   'If > 1, the specified number of outlier classes will be sampled at random.')
def main(dataset_name, net_name, xp_path, data_path, load_config, load_model, ratio_known_normal, ratio_known_outlier,
         ratio_pollution, device, seed, optimizer_name, lr, n_epochs, lr_milestone, batch_size, weight_decay, pretrain,
         vae_optimizer_name, vae_lr, vae_n_epochs, vae_lr_milestone, vae_batch_size, vae_weight_decay,
         num_threads, n_jobs_dataloader, normal_class, known_outlier_class, n_known_outlier_classes):
    """
    Semi-Supervised Deep Generative model (M1+M2 model) from Kingma et al. (2014)

    :arg DATASET_NAME: Name of the dataset to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """

    # Get configuration
    cfg = Config(locals().copy())

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = xp_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print paths
    logger.info('Log file is %s' % log_file)
    logger.info('Data path is %s' % data_path)
    logger.info('Export path is %s' % xp_path)

    # Print experimental setup
    logger.info('Dataset: %s' % dataset_name)
    logger.info('Normal class: %d' % normal_class)
    logger.info('Ratio of labeled normal train samples: %.2f' % ratio_known_normal)
    logger.info('Ratio of labeled anomalous samples: %.2f' % ratio_known_outlier)
    logger.info('Pollution ratio of unlabeled train data: %.2f' % ratio_pollution)
    if n_known_outlier_classes == 1:
        logger.info('Known anomaly class: %d' % known_outlier_class)
    else:
        logger.info('Number of known anomaly classes: %d' % n_known_outlier_classes)
    logger.info('Network: %s' % net_name)

    # If specified, load experiment config from JSON-file
    if load_config:
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)

    # Set seed
    if cfg.settings['seed'] != -1:
        random.seed(cfg.settings['seed'])
        np.random.seed(cfg.settings['seed'])
        torch.manual_seed(cfg.settings['seed'])
        torch.cuda.manual_seed(cfg.settings['seed'])
        torch.backends.cudnn.deterministic = True
        logger.info('Set seed to %d.' % cfg.settings['seed'])

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'
    # Set the number of threads used for parallelizing CPU operations
    if num_threads > 0:
        torch.set_num_threads(num_threads)
    logger.info('Computation device: %s' % device)
    logger.info('Number of threads: %d' % num_threads)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)

    # Load data
    dataset = load_dataset(dataset_name, data_path, normal_class, known_outlier_class, n_known_outlier_classes,
                           ratio_known_normal, ratio_known_outlier, ratio_pollution,
                           random_state=np.random.RandomState(cfg.settings['seed']))
    # Log random sample of known anomaly classes if more than 1 class
    if n_known_outlier_classes > 1:
        logger.info('Known anomaly classes: %s' % (dataset.known_outlier_classes,))

    # Initialize semiDGM model and set neural network phi
    alpha = 0.1 * (1 - ratio_known_normal - ratio_known_outlier) / (ratio_known_normal + ratio_known_outlier)
    semiDGM = SemiDeepGenerativeModel(alpha=alpha)

    # If specified, load model
    if load_model:
        # Initialize networks
        semiDGM.set_vae(net_name)
        semiDGM.set_network(net_name)
        # Load model
        semiDGM.load_model(model_path=load_model)
        logger.info('Loading model from %s.' % load_model)

    logger.info('Pretraining: %s' % pretrain)
    if pretrain:
        # Log pretraining details
        logger.info('Pretraining optimizer: %s' % cfg.settings['vae_optimizer_name'])
        logger.info('Pretraining learning rate: %g' % cfg.settings['vae_lr'])
        logger.info('Pretraining epochs: %d' % cfg.settings['vae_n_epochs'])
        logger.info('Pretraining learning rate scheduler milestones: %s' % (cfg.settings['vae_lr_milestone'],))
        logger.info('Pretraining batch size: %d' % cfg.settings['vae_batch_size'])
        logger.info('Pretraining weight decay: %g' % cfg.settings['vae_weight_decay'])

        # Pretrain model on dataset (via variational autoencoder)
        semiDGM.set_vae(net_name)
        semiDGM.pretrain(dataset,
                         optimizer_name=cfg.settings['vae_optimizer_name'],
                         lr=cfg.settings['vae_lr'],
                         n_epochs=cfg.settings['vae_n_epochs'],
                         lr_milestones=cfg.settings['vae_lr_milestone'],
                         batch_size=cfg.settings['vae_batch_size'],
                         weight_decay=cfg.settings['vae_weight_decay'],
                         device=device,
                         n_jobs_dataloader=n_jobs_dataloader)

        # Save pretraining results
        semiDGM.save_vae_results(export_json=xp_path + '/vae_results.json')

    # Log training details
    logger.info('Training optimizer: %s' % cfg.settings['optimizer_name'])
    logger.info('Training learning rate: %g' % cfg.settings['lr'])
    logger.info('Training epochs: %d' % cfg.settings['n_epochs'])
    logger.info('Training learning rate scheduler milestones: %s' % (cfg.settings['lr_milestone'],))
    logger.info('Training batch size: %d' % cfg.settings['batch_size'])
    logger.info('Training weight decay: %g' % cfg.settings['weight_decay'])

    # Train model on dataset
    semiDGM.set_network(net_name)
    semiDGM.train(dataset,
                  optimizer_name=cfg.settings['optimizer_name'],
                  lr=cfg.settings['lr'],
                  n_epochs=cfg.settings['n_epochs'],
                  lr_milestones=cfg.settings['lr_milestone'],
                  batch_size=cfg.settings['batch_size'],
                  weight_decay=cfg.settings['weight_decay'],
                  device=device,
                  n_jobs_dataloader=n_jobs_dataloader)

    # Test model
    semiDGM.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)

    # Save results, model, and configuration
    semiDGM.save_results(export_json=xp_path + '/results.json')
    semiDGM.save_model(export_model=xp_path + '/model.tar')
    cfg.save_config(export_json=xp_path + '/config.json')

    # Plot most anomalous and most normal test samples
    indices, labels, scores = zip(*semiDGM.results['test_scores'])
    indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
    idx_all_sorted = indices[np.argsort(scores)]  # from lowest to highest score
    idx_normal_sorted = indices[labels == 0][np.argsort(scores[labels == 0])]  # from lowest to highest score

    if dataset_name in ('mnist', 'fmnist', 'cifar10'):

        if dataset_name in ('mnist', 'fmnist'):
            X_all_low = dataset.test_set.data[idx_all_sorted[:32], ...].unsqueeze(1)
            X_all_high = dataset.test_set.data[idx_all_sorted[-32:], ...].unsqueeze(1)
            X_normal_low = dataset.test_set.data[idx_normal_sorted[:32], ...].unsqueeze(1)
            X_normal_high = dataset.test_set.data[idx_normal_sorted[-32:], ...].unsqueeze(1)

        if dataset_name == 'cifar10':
            X_all_low = torch.tensor(np.transpose(dataset.test_set.data[idx_all_sorted[:32], ...], (0,3,1,2)))
            X_all_high = torch.tensor(np.transpose(dataset.test_set.data[idx_all_sorted[-32:], ...], (0,3,1,2)))
            X_normal_low = torch.tensor(np.transpose(dataset.test_set.data[idx_normal_sorted[:32], ...], (0,3,1,2)))
            X_normal_high = torch.tensor(np.transpose(dataset.test_set.data[idx_normal_sorted[-32:], ...], (0,3,1,2)))

        plot_images_grid(X_all_low, export_img=xp_path + '/all_low', padding=2)
        plot_images_grid(X_all_high, export_img=xp_path + '/all_high', padding=2)
        plot_images_grid(X_normal_low, export_img=xp_path + '/normals_low', padding=2)
        plot_images_grid(X_normal_high, export_img=xp_path + '/normals_high', padding=2)


if __name__ == '__main__':
    main()
