from __future__ import division
from __future__ import print_function

import logging

import matplotlib.pyplot as plt
import numpy as np
from dask.distributed import Client
from sklearn.model_selection import train_test_split

from gpmp_scale.exactGP_discoverkernel.exactGP_DiscoverKernel import train, predict, evaluate
from gpmp_scale.gp_utils import load_climate_dataset, load_climate_dataset_2d, normalize, plot_mean_variance

CLASS_NAME = 'exactGP_DiscoverKernel'
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(f"{CLASS_NAME}_log.log"),
                        logging.StreamHandler()
                    ])


def plot_climate_map(x_data, y_data):
    index = np.random.choice(y_data.shape[0], 4000, replace=False)
    stations_map = x_data[index]
    y_map = y_data[index]

    fig, ax = plt.subplots(figsize=(15, 6))
    scatter = ax.scatter(stations_map[:, 0], stations_map[:, 1], c=y_map)
    legend1 = ax.legend(*scatter.legend_elements(num=6),
                        loc="lower right", title="℃ Degree")
    ax.set_ylabel('Latitude')
    ax.set_xlabel('Longitude')
    ax.add_artist(legend1)
    plt.savefig('climate_data.png')


if __name__ == '__main__':
    logging.info(f'----------------------------------------')
    logging.info(f'Training exactGP_DiscoverKernel')
    x_data, y_data = load_climate_dataset_2d()
    # x_data, y_data = load_climate_dataset()  ## Open comment when using 3 dimension dataset

    x_data[:, 0] = normalize(x_data[:, 0])
    x_data[:, 1] = normalize(x_data[:, 1])
    # x_data[:, 2] = normalize(x_data[:, 2]) ## Open comment when using 3 dimension dataset
    y_data = normalize(y_data)

    train_x, test_x, train_y, test_y = train_test_split(x_data, y_data, train_size=0.0004, test_size=0.00002,
                                                        random_state=42)
    print(f'train_x.shape={train_x.shape}, test_x.shape={test_x.shape}, '
          f'train_y.shape={train_y.shape}, test_y.shape={test_y.shape}')

    # Khai bao ham:
    NUM_TRAIN = 20000
    BATCH_SIZE = 1000
    MAX_ITER = 10
    INPUT_DIM = 2

    plot_climate_map(x_data, y_data)

    client = Client()
    # client = Client("tcp://10.61.100.71:8080")
    client.wait_for_workers(1)
    model, train_time = train(train_x, train_y, INPUT_DIM, max_iter=MAX_ITER, gp2Scale_dask_client=client,
                              batchSize=BATCH_SIZE, method='mcmc')

    mean1, var1 = predict(model, test_x.reshape(-1, INPUT_DIM))
    evaluate(test_y, mean1)

    plot_mean_variance(test_y[:100], mean1[:100], saveFile='Climate_ExactGP_DiscoverKernel_mean_variance')
