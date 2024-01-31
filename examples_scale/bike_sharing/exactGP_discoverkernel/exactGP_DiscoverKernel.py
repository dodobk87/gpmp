from __future__ import division
from __future__ import print_function

import logging
import time

import numpy as np
from dask.distributed import Client
from fvgp import GP
from sklearn.model_selection import train_test_split

from gpmp_scale.gp_utils import load_bike_sharing_dataset, normalize, plot_mean_variance

CLASS_NAME = 'exactGP_DiscoverKernel'
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(f"{CLASS_NAME}_log.log"),
                        logging.StreamHandler()
                    ])

from gpmp_scale.exactGP_discoverkernel.exactGP_DiscoverKernel import train, predict, evaluate


# def wendland_anisotropic_gp2Scale_cpu(x1, x2, hps, obj):
#     distance_matrix = np.zeros((len(x1), len(x2)))
#     for i in range(len(x1[0])):
#         distance_matrix += (np.subtract.outer(x1[:, i], x2[:, i]) / hps[1 + i]) ** 2
#     d = np.sqrt(distance_matrix)
#     d[d > 1.] = 1.
#     kernel = hps[0] * (1. - d) ** 8 * (35. * d ** 3 + 25. * d ** 2 + 8. * d + 1.)
#     return kernel
#
#
# def train(train_x, train_y, input_space_dim, hyperparameter_bounds=None, gp2Scale=True, gp_kernel_function=None,
#           gp2Scale_dask_client=None, batchSize=400, max_iter=120, method="global"):
#     logging.info(f'MAX_INTER: {max_iter}, NUM_TRAIN:{train_x.shape[0]}, BATCH_SIZE: {batchSize}')
#     start_time = time.time()
#     if gp2Scale_dask_client is None:
#         gp2Scale_dask_client = Client()
#     if gp_kernel_function is None:
#         gp_kernel_function = wendland_anisotropic_gp2Scale_cpu
#     if hyperparameter_bounds is None:
#         hps_bounds = np.array([[0.1, 10.],  ##signal var of stat kernel
#                                [0.001, 0.05],  ##length scale for stat kernel
#                                [0.001, 0.05],  ##length scale for stat kernel
#                                [0.001, 0.05],  ##length scale for stat kernel
#                                ])
#     else:
#         hps_bounds = hyperparameter_bounds
#     init_hps = np.random.uniform(size=len(hps_bounds), low=hps_bounds[:, 0], high=hps_bounds[:, 1])
#     logging.info("4. Init GP")
#
#     model = GP(input_space_dim, train_x, train_y, init_hps,
#                hyperparameter_bounds=hps_bounds, gp_kernel_function=gp_kernel_function,
#                gp2Scale=gp2Scale, gp2Scale_batch_size=batchSize, info=True, gp2Scale_dask_client=gp2Scale_dask_client)
#
#     logging.info("5. Standard Training")
#     model.train(hyperparameter_bounds=hps_bounds, max_iter=max_iter, method=method)
#     train_time = time.time() - start_time
#     logging.info(f'Training time: {train_time}')
#     return model, train_time
#
#
# def predict(model, test_x):
#     mean1 = model.posterior_mean(test_x)["f(x)"]
#     var1 = model.posterior_covariance(test_x, variance_only=False, add_noise=True)["v(x)"]
#     return mean1, var1
#
#
# def evaluate(test_y, y_mean):
#     mae = np.mean(np.abs(y_mean - test_y))
#     rmse = np.sqrt(np.mean((y_mean - test_y) ** 2))
#     logging.info('Test MAE: {}'.format(mae))
#     logging.info('Test RMSE: {}'.format(rmse))
#     return mae, rmse


def clean_df(df):
    df['season'] = normalize(df['season'])
    df['mnth'] = normalize(df['mnth'])
    df['hr'] = normalize(df['hr'])
    df['weekday'] = normalize(df['weekday'])
    df['weathersit'] = normalize(df['weathersit'])
    df['cnt'] = normalize(df['cnt'])
    df['yr'] = df['yr'].astype(float)
    df['holiday'] = df['holiday'].astype(float)
    df['workingday'] = df['workingday'].astype(float)
    return df


if __name__ == '__main__':
    logging.info(f'----------------------------------------')
    logging.info(f'Training fvGP NoAck')
    df = load_bike_sharing_dataset()
    logging.info(df.describe())

    # Chuan hoa du lieu
    df = df.drop(['dteday', 'casual', 'registered', 'instant'], axis=1)
    df = clean_df(df)

    y_data = df['cnt']
    df = df.drop(['cnt'], axis=1)

    train_y_full = y_data.to_numpy()
    df_array = df.to_numpy()

    # Tach tap train-test
    train_x, test_x, train_y, test_y = train_test_split(df_array, train_y_full, test_size=0.15, random_state=42)
    logging.info(f'train_x.shape = {train_x.shape}, train_y.shape= {train_y.shape}, '
                 f'test_x.shape={test_x.shape}, test_y.shape={test_y.shape}')

    # Khai bao ham:
    NUM_TRAIN = 100000
    BATCH_SIZE = 400
    MAX_ITER = 40
    INPUT_DIM = 12
    METHOD = 'mcmc'

    hps_bounds = np.array([[0.1, 10.],  ##signal var of stat kernel
                           [0.01, 0.05],  ##length scale for stat kernel
                           [0.01, 0.05],  ##length scale for stat kernel
                           [0.01, 0.05],  ##length scale for stat kernel
                           [0.01, 0.05],  ##length scale for stat kernel
                           [0.01, 0.05],  ##length scale for stat kernel
                           [0.01, 0.05],  ##length scale for stat kernel
                           [0.01, 0.05],  ##length scale for stat kernel
                           [0.01, 0.05],  ##length scale for stat kernel
                           [0.01, 0.05],  ##length scale for stat kernel
                           [0.01, 0.05],  ##length scale for stat kernel
                           [0.01, 0.05],  ##length scale for stat kernel
                           [0.01, 0.05],  ##length scale for stat kernel
                           ])
    client = Client()
    # client = Client("tcp://10.61.100.71:8080")
    client.wait_for_workers(1)
    model, train_time = train(train_x, train_y, INPUT_DIM, hyperparameter_bounds=hps_bounds, max_iter=MAX_ITER,
                              gp2Scale_dask_client=client, batchSize=BATCH_SIZE, method=METHOD)

    mean1, var1 = predict(model, test_x.reshape(-1, INPUT_DIM))
    evaluate(test_y, mean1)

    plot_mean_variance(test_y[:100], mean1[:100], saveFile='ExactGP_DiscoverKernel_mean_variance')
