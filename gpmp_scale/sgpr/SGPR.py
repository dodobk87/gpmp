import logging
import time

import gpflow
import matplotlib.pyplot as plt
import numpy as np
from gpflow.utilities import print_summary
import gpmp_scale.gp_utils as util

CLASS_NAME = 'SGPR'
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(f"{CLASS_NAME}_log.log"),
                        logging.StreamHandler()
                    ])


def train(train_x, train_y, m):
    start_time = time.time()
    logging.info(f'training with m = {m}')
    inducing_points = train_x[:m, :].copy()  # Initialise inducing locations to the first M inputs in the dataset
    logging.info('3. Training SGPR')
    model = gpflow.models.SGPR(
        (train_x, train_y),
        kernel=gpflow.kernels.SquaredExponential(),
        inducing_variable=inducing_points,
    )
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables)
    print_summary(model)
    times = time.time() - start_time
    logging.info(f'Time Process: {times}')
    return model, times


def predict_y(model, test_x):
    logging.info('Infer dataset')
    y_mean, y_var = model.predict_y(test_x, full_cov=False)
    return y_mean, y_var


def evaluate(test_y, y_mean):
    mae = np.mean(np.abs(y_mean - test_y))
    rmse = np.sqrt(np.mean((y_mean - test_y) ** 2))
    logging.info('Test MAE: {}'.format(mae))
    logging.info('Test RMSE: {}'.format(rmse))
    return mae, rmse


def train_and_test_with_inducing(train_x, train_y, test_x, test_y, m_array):
    m_arr_x = [i for i in range(1, len(m_array) + 1)]
    m_arr = []
    rmse_arr = []
    time_arr = []
    for m in m_array:
        logging.info('============================================')
        model, _time = train(train_x, train_y, m)
        y_mean, y_var = predict_y(model, test_x)
        mae, rmse = evaluate(test_y, y_mean)

        m_arr.append(m)
        rmse_arr.append(rmse)
        time_arr.append(_time)

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=(15, 6))
    ax1.plot(m_arr_x, rmse_arr, 'b')
    ax1.set_xticks(m_arr_x, m_arr)
    ax1.set_xlabel('inducing point (m)')
    ax1.set_ylabel('RMSE')
    ax1.set_title("RMSE SGPR")

    ax2.plot(m_arr_x, time_arr, 'b')
    ax2.set_xticks(m_arr_x, m_arr)
    ax2.set_xlabel('inducing point (m)')
    ax2.set_ylabel('Training time (seconds)')
    ax2.set_title("Training time SGPR")
    plt.savefig('SGPR_rmse_time_training_inducing_point.png')
    plt.show()


if __name__ == "__main__":
    start_time = time.time()
    logging.info(f'------TRANING SGPR')
    logging.info('1. Load data for training and testing')
    train_x, train_y, test_x, test_y = util.load_data(N=20000, N_test=10000)
    logging.info('2. Plot data in chart and save file')

    if train_x.shape[1] == 1:
        util.plot_data(train_x, train_y, test_x, test_y, file_name=f'{CLASS_NAME}_data.png')

    M = 20
    logging.info(f'-------RUNNING with N = {train_x.shape[0]}, M = {M}')
    model, _ = train(train_x, train_y, M)
    y_mean, y_var = predict_y(model, test_x)
    evaluate(test_y, y_mean)

    if train_x.shape[1] == 1:
        # dataset 1 dimension
        util.plot_output_model(train_x, train_y, test_x, test_y, y_mean, y_var, file_name=f'{CLASS_NAME}_result.png')

    m_inducing_array = [5, 10, 15, 20, 25, 32, 48, 50, 64, 96, 128, 256, 512]
    train_and_test_with_inducing(train_x, train_y, test_x, test_y, m_inducing_array)
