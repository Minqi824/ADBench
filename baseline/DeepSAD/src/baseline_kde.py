import click
import torch
import logging
import random
import numpy as np

from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from baselines.kde import KDE
from datasets.main import load_dataset


################################################################################
# Settings
################################################################################
@click.command()
@click.argument('dataset_name', type=click.Choice(['mnist', 'fmnist', 'cifar10', 'arrhythmia', 'cardio', 'satellite',
                                                   'satimage-2', 'shuttle', 'thyroid']))
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
@click.option('--seed', type=int, default=-1, help='Set seed. If -1, use randomization.')
@click.option('--kernel', type=click.Choice(['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']),
              default='gaussian', help='Kernel for the KDE')
@click.option('--grid_search_cv', type=bool, default=True,
              help='Use sklearn GridSearchCV to determine optimal bandwidth')
@click.option('--n_jobs_model', type=int, default=-1, help='Number of jobs for model training.')
@click.option('--hybrid', type=bool, default=False,
              help='Train KDE on features extracted from an autoencoder. If True, load_ae must be specified.')
@click.option('--load_ae', type=click.Path(exists=True), default=None,
              help='Model file path to load autoencoder weights (default: None).')
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
def main(dataset_name, xp_path, data_path, load_config, load_model, ratio_known_normal, ratio_known_outlier,
         ratio_pollution, seed, kernel, grid_search_cv, n_jobs_model, hybrid, load_ae, n_jobs_dataloader, normal_class,
         known_outlier_class, n_known_outlier_classes):
    """
    (Hybrid) KDE for anomaly detection.

    :arg DATASET_NAME: Name of the dataset to load.
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
    logger.info('Log file is %s.' % log_file)
    logger.info('Data path is %s.' % data_path)
    logger.info('Export path is %s.' % xp_path)

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

    # If specified, load experiment config from JSON-file
    if load_config:
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)

    # Print KDE configuration
    logger.info('KDE kernel: %s' % cfg.settings['kernel'])
    logger.info('Use GridSearchCV for bandwidth selection: %s' % cfg.settings['grid_search_cv'])
    logger.info('Number of jobs for model training: %d' % n_jobs_model)
    logger.info('Hybrid model: %s' % cfg.settings['hybrid'])

    # Set seed
    if cfg.settings['seed'] != -1:
        random.seed(cfg.settings['seed'])
        np.random.seed(cfg.settings['seed'])
        torch.manual_seed(cfg.settings['seed'])
        torch.cuda.manual_seed(cfg.settings['seed'])
        torch.backends.cudnn.deterministic = True
        logger.info('Set seed to %d.' % cfg.settings['seed'])

    # Use 'cpu' as device for KDE
    device = 'cpu'
    torch.multiprocessing.set_sharing_strategy('file_system')  # fix multiprocessing issue for ubuntu
    logger.info('Computation device: %s' % device)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)

    # Load data
    dataset = load_dataset(dataset_name, data_path, normal_class, known_outlier_class, n_known_outlier_classes,
                           ratio_known_normal, ratio_known_outlier, ratio_pollution,
                           random_state=np.random.RandomState(cfg.settings['seed']))
    # Log random sample of known anomaly classes if more than 1 class
    if n_known_outlier_classes > 1:
        logger.info('Known anomaly classes: %s' % (dataset.known_outlier_classes,))

    # Initialize KDE model
    kde = KDE(hybrid=cfg.settings['hybrid'], kernel=cfg.settings['kernel'], n_jobs=n_jobs_model,
              seed=cfg.settings['seed'])

    # If specified, load model parameters from already trained model
    if load_model:
        kde.load_model(import_path=load_model, device=device)
        logger.info('Loading model from %s.' % load_model)

    # If specified, load model autoencoder weights for a hybrid approach
    if hybrid and load_ae is not None:
        kde.load_ae(dataset_name, model_path=load_ae)
        logger.info('Loaded pretrained autoencoder for features from %s.' % load_ae)

    # Train model on dataset
    kde.train(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader,
              bandwidth_GridSearchCV=cfg.settings['grid_search_cv'])

    # Test model
    kde.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)

    # Save results and configuration
    kde.save_results(export_json=xp_path + '/results.json')
    cfg.save_config(export_json=xp_path + '/config.json')

    # Plot most anomalous and most normal test samples
    indices, labels, scores = zip(*kde.results['test_scores'])
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
            X_all_low = torch.tensor(np.transpose(dataset.test_set.data[idx_all_sorted[:32], ...], (0, 3, 1, 2)))
            X_all_high = torch.tensor(np.transpose(dataset.test_set.data[idx_all_sorted[-32:], ...], (0, 3, 1, 2)))
            X_normal_low = torch.tensor(np.transpose(dataset.test_set.data[idx_normal_sorted[:32], ...], (0, 3, 1, 2)))
            X_normal_high = torch.tensor(
                np.transpose(dataset.test_set.data[idx_normal_sorted[-32:], ...], (0, 3, 1, 2)))

        plot_images_grid(X_all_low, export_img=xp_path + '/all_low', padding=2)
        plot_images_grid(X_all_high, export_img=xp_path + '/all_high', padding=2)
        plot_images_grid(X_normal_low, export_img=xp_path + '/normals_low', padding=2)
        plot_images_grid(X_normal_high, export_img=xp_path + '/normals_high', padding=2)


if __name__ == '__main__':
    main()
