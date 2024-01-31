from __future__ import division
from __future__ import print_function

import logging

import numpy as np
from dask.distributed import Client
from sklearn.model_selection import train_test_split

from gpmp_scale.exactGP_discoverkernel.exactGP_DiscoverKernel import train, predict, evaluate
from gpmp_scale.gp_utils import load_3d_road_dataset, normalize, plot_mean_variance

CLASS_NAME = 'exactGP_DiscoverKernel'
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(f"{CLASS_NAME}_log.log"),
                        logging.StreamHandler()
                    ])


def clean_df(df):
    df['LONGITUDE'] = normalize(df['LONGITUDE'])
    df['LATITUDE'] = normalize(df['LATITUDE'])
    df['ALTITUDE'] = normalize(df['ALTITUDE'])
    return df


if __name__ == '__main__':
    logging.info(f'----------------------------------------')
    logging.info(f'Training fvGP NoAck')
    df = load_3d_road_dataset()
    print(df.describe())

    # Chuan hoa du lieu
    df = clean_df(df)

    y_data = df['ALTITUDE']
    df = df.drop(['ALTITUDE', 'OSM_ID'], axis=1)
    train_y_full = y_data.to_numpy()
    df_array = df.to_numpy()

    # Tach tap train-test
    train_x, test_x, train_y, test_y = train_test_split(df_array, train_y_full, train_size=0.046, test_size=0.0023,
                                                        random_state=42)
    logging.info(f'train_x.shape = {train_x.shape}, train_y.shape= {train_y.shape}, '
                 f'test_x.shape={test_x.shape}, test_y.shape={test_y.shape}')

    # Khai bao ham:
    NUM_TRAIN = 100000
    BATCH_SIZE = 1000
    MAX_ITER = 10
    INPUT_DIM = 2

    hps_bounds = np.array([[0.1, 10.],  ##signal var of stat kernel
                           [0.001, 0.05],  ##length scale for stat kernel
                           [0.001, 0.05],  ##length scale for stat kernel
                           [0.001, 0.05],  ##length scale for stat kernel
                           ])
    client = Client()
    # client = Client("tcp://10.61.100.71:8080")
    client.wait_for_workers(1)
    model, train_time = train(train_x, train_y, INPUT_DIM, max_iter=MAX_ITER, gp2Scale_dask_client=client,
                              batchSize=BATCH_SIZE, method='mcmc', info=True)

    mean1, var1 = predict(model, test_x.reshape(-1, INPUT_DIM))
    evaluate(test_y, mean1)

    plot_mean_variance(test_y[:100], mean1[:100], saveFile='3D_Road_ExactGP_DiscoverKernel_mean_variance')
