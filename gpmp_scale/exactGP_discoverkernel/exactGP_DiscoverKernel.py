import logging
import time

import numpy as np
from dask.distributed import Client
from fvgp import GP

CLASS_NAME = 'exactGP_DiscoverKernel'
NUM_TRAIN = 20000
BATCH_SIZE = 400
MAX_ITER = 50
INPUT_DIM = 1

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(f"{CLASS_NAME}_log.log"),
                        logging.StreamHandler()
                    ])


def wendland_anisotropic_gp2Scale_cpu(x1, x2, hps, obj):
    distance_matrix = np.zeros((len(x1), len(x2)))
    for i in range(len(x1[0])):
        distance_matrix += (np.subtract.outer(x1[:, i], x2[:, i]) / hps[1 + i]) ** 2
    d = np.sqrt(distance_matrix)
    d[d > 1.] = 1.
    kernel = hps[0] * (1. - d) ** 8 * (35. * d ** 3 + 25. * d ** 2 + 8. * d + 1.)
    return kernel


def train(train_x, train_y, input_space_dim, hyperparameter_bounds=None, gp2Scale=True, gp_kernel_function=None,
          gp2Scale_dask_client=None, batchSize=400, max_iter=120, method="global", info=False):
    logging.info(f'MAX_INTER: {max_iter}, NUM_TRAIN:{train_x.shape[0]}, BATCH_SIZE: {batchSize}')
    start_time = time.time()
    if gp2Scale_dask_client is None:
        gp2Scale_dask_client = Client()
    if gp_kernel_function is None:
        gp_kernel_function = wendland_anisotropic_gp2Scale_cpu
    if hyperparameter_bounds is None:
        hps_bounds = np.array([[0.1, 10.],  ##signal var of stat kernel
                               [0.001, 0.05],  ##length scale for stat kernel
                               [0.001, 0.05],  ##length scale for stat kernel
                               [0.001, 0.05],  ##length scale for stat kernel
                               ])
    else:
        hps_bounds = hyperparameter_bounds
    init_hps = np.random.uniform(size=len(hps_bounds), low=hps_bounds[:, 0], high=hps_bounds[:, 1])
    logging.info("4. Init GP")

    model = GP(input_space_dim, train_x, train_y, init_hps,
               hyperparameter_bounds=hps_bounds, gp_kernel_function=gp_kernel_function,
               gp2Scale=gp2Scale, gp2Scale_batch_size=batchSize, info=True, gp2Scale_dask_client=gp2Scale_dask_client)

    logging.info("5. Standard Training")
    model.train(hyperparameter_bounds=hps_bounds, max_iter=max_iter, method=method)
    train_time = time.time() - start_time
    logging.info(f'Training time: {train_time}')
    return model, train_time


def predict(model, test_x):
    mean1 = model.posterior_mean(test_x)["f(x)"]
    var1 = model.posterior_covariance(test_x, variance_only=False, add_noise=True)["v(x)"]
    return mean1, var1


def evaluate(test_y, y_mean):
    mae = np.mean(np.abs(y_mean - test_y))
    rmse = np.sqrt(np.mean((y_mean - test_y) ** 2))
    logging.info('Test MAE: {}'.format(mae))
    logging.info('Test RMSE: {}'.format(rmse))
    return mae, rmse

# if __name__ == '__main__':
#     freeze_support()
#     start_time = time.time()
#     logging.info('1. Generate data')
#     train_x, train_y, test_x, test_y = util.load_data(N=20000, N_test=100)
#
#     logging.info('2. Plot data training in chart')
#     plot_data(train_x, train_y, test_x, test_y)
#
#     logging.info('3. Connect Dask Server tcp://10.61.100.71:8080')
#     client = Client("tcp://10.61.100.71:8080")
#     # client = Client()
#     client.wait_for_workers(1)
#
#     hps_bounds = np.array([[0.1, 10.],  ##signal var of stat kernel
#                            [0.001, 0.05],  ##length scale for stat kernel
#                            [0.001, 0.05],  ##length scale for stat kernel
#                            [0.001, 0.05],  ##length scale for stat kernel
#                            ])
#     init_hps = np.random.uniform(size=len(hps_bounds), low=hps_bounds[:, 0], high=hps_bounds[:, 1])
#     logging.info("4. Init GP")
#
#     my_gp1 = GP(INPUT_DIM, train_x, train_y, init_hps,
#                 hyperparameter_bounds=hps_bounds, gp_kernel_function=wendland_anisotropic_gp2Scale_cpu,
#                 gp2Scale=True, gp2Scale_batch_size=BATCH_SIZE, info=True, gp2Scale_dask_client=client)
#
#     logging.info("5. Standard Training")
#     my_gp1.train(hyperparameter_bounds=hps_bounds, max_iter=MAX_ITER)
#
#     mean1 = my_gp1.posterior_mean(test_x.reshape(-1, 1))["f(x)"]
#     var1 = my_gp1.posterior_covariance(test_x.reshape(-1, 1), variance_only=False, add_noise=True)["v(x)"]
#
#     logging.info('6. Plot Result in chart')
#
#     util.plot_output_model(train_x, train_y, test_x, test_y, mean1.reshape(-1, 1), var1.reshape(-1, 1),
#                            file_name=f'{CLASS_NAME}_result.png')
#
#     ##looking at some validation metrics
#     test_x = test_x.reshape(-1, 1)
#     logging.info(my_gp1.rmse(test_x, test_y))
#     logging.info(f'MAX_INTER: {MAX_ITER}, NUM_TRAIN:{NUM_TRAIN}, BATCH_SIZE: {BATCH_SIZE}')
#     logging.info('Test MAE: {}'.format(np.mean(np.abs(mean1.reshape(-1) - test_y.reshape(-1)))))
#     logging.info('Test RMSE: {}'.format(np.sqrt(np.mean((mean1.reshape(-1) - test_y.reshape(-1)) ** 2))))
#     logging.info(f'Time Process: {time.time() - start_time}')
#     logging.info('------------------------------------------')
