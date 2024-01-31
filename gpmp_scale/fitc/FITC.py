from __future__ import division
from __future__ import print_function

import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import pyGPs
import gpmp_scale.gp_utils as util

CLASS_NAME = 'FITC'
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(f"{CLASS_NAME}_log.log"),
                        logging.StreamHandler()
                    ])


def plot_result(train_x, train_y, test_x, test_y, mean, variance, file_name="result.png"):
    plt.figure(figsize=(16, 10))
    plt.plot(test_x, mean, label="posterior mean", linewidth=4)
    plt.plot(test_x, test_y, label="latent function", linewidth=4)
    plt.fill_between(test_x, mean - 3. * np.sqrt(variance), mean + 3. * np.sqrt(variance), alpha=0.5, color="grey",
                     label="var")
    plt.scatter(train_x, train_y, color='black')
    plt.savefig(file_name)
    plt.show()


def train(train_x, train_y, M):
    # default inducing point
    start_time = time.time()
    logging.info('******************************************')
    logging.info('********Training with custom Inducing')
    logging.info('2. Create model FITC')
    model = pyGPs.GPR_FITC()

    logging.info('3. Training GP FITC')
    # You can define inducing points yourself.
    # You can pick some points by hand
    index = np.random.choice(train_x.shape[0], M, replace=False)
    inducing_point = train_x[index].copy()
    # inducing_point = train_x[:M].copy()

    # and specify inducing point when seting prior
    m = pyGPs.mean.Linear(D=train_x.shape[1]) + pyGPs.mean.Const()
    k = pyGPs.cov.RBF()
    model.setPrior(mean=m, kernel=k, inducing_points=inducing_point)

    # The rest is analogous to what we have done before
    model.setData(train_x, train_y)
    model.getPosterior()
    model.optimize()

    # Prediction
    # output means, output variances, latent means, latent variances, log predictive probabilities

    # mean, variance, fmu, fs2, lp = model.predict(test_x)
    time_train = time.time() - start_time
    logging.info(f'Time traininig = {time_train}')
    return model, time_train


def predict(model, test_x):
    logging.info('Inference')
    mean, variance, fmu, fs2, lp = model.predict(test_x)
    return mean, variance


def evaluate(test_y, y_mean):
    mae = np.mean(np.abs(y_mean - test_y))
    rmse = np.sqrt(np.mean((y_mean - test_y) ** 2))
    logging.info(f'MAE: {mae}')
    logging.info(f'RMSE: {rmse}')
    return mae, rmse


if __name__ == "__main__":
    start_time = time.time()
    logging.info('1. Load data for training and testing')
    train_x, train_y, test_x, test_y = util.load_data()

    if train_x.shape[1] == 1:
        # dataset 1 dimension
        util.plot_data(train_x, train_y, test_x, test_y, file_name=f'{CLASS_NAME}_data.png')

    # inducing point
    M = 32
    model, time_train = train(train_x, train_y, M)
    mean, variance = predict(model, test_x)

    logging.info('5. Plot result GP FITC')

    plot_result(train_x.reshape(-1), train_y, test_x.reshape(-1), test_y, mean.reshape(-1), variance.reshape(-1),
                file_name=f"{CLASS_NAME}_custom_result.png")
    logging.info(f'time total:{time_train}')
