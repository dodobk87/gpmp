import itertools
import logging
import time

import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gpflow.ci_utils import reduce_in_tests

import gpmp_scale.gp_utils as util

CLASS_NAME = 'SVGP'
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(f"{CLASS_NAME}_log.log"),
                        logging.StreamHandler()
                    ])


def run_adam(model, iterations, minibatch_size, train_dataset):
    """
    Utility function running the Adam optimizer

    :param model: GPflow model
    :param interations: number of iterations
    """
    # Create an Adam Optimizer action
    logf = []
    train_iter = iter(train_dataset.batch(minibatch_size))
    training_loss = model.training_loss_closure(train_iter, compile=True)
    optimizer = tf.optimizers.Adam()

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)

    for step in range(iterations):
        optimization_step()
        if step % 10 == 0:
            elbo = -training_loss().numpy()
            logf.append(elbo)
    return logf


def minibatch_proportion(N, train_dataset, elbo):
    minibatch_proportions = np.logspace(-2, 0, 10)
    times = []
    objs = []
    for mbp in minibatch_proportions:
        batchsize = int(N * mbp)
        train_iter = iter(train_dataset.batch(batchsize))
        start_time = time.time()
        objs.append([elbo(minibatch) for minibatch in itertools.islice(train_iter, 20)])
        times.append(time.time() - start_time)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(minibatch_proportions, times, "x-")
    ax1.set_xlabel("Minibatch proportion")
    ax1.set_ylabel("Time taken")

    ax2.plot(minibatch_proportions, np.array(objs), "kx")
    ax2.set_xlabel("Minibatch proportion")
    ax2.set_ylabel("ELBO estimates")
    plt.savefig('SVGP_minibatch_proportion.png')


def plot_result(maxiter, logf):
    plt.plot(np.arange(maxiter)[::10], logf)
    plt.xlabel("iteration")
    plt.ylabel("ELBO")
    plt.savefig('SVGP_elbo_interation.png')


def train(train_x, train_y, minibatch_size=100, m_inducing=128, max_iter=20000):
    logging.info(f'Training SVGP with minibatch_size: {minibatch_size}, m_inducing: {m_inducing}, max_iter:{max_iter}')
    start_time = time.time()
    N = train_x.shape[0]
    # Initialise inducing locations to the first M inputs in the dataset
    inducing_points = train_x[:m_inducing, :].copy()
    logging.info('Training SVGP')
    model = gpflow.models.SVGP(
        kernel=gpflow.kernels.SquaredExponential(),
        likelihood=gpflow.likelihoods.Gaussian(),
        inducing_variable=inducing_points, num_data=N
    )
    data = (train_x, train_y)
    elbo = tf.function(model.elbo)
    # TensorFlow re-traces & compiles a `tf.function`-wrapped method at *every* call if the arguments are numpy arrays instead of tf.Tensors. Hence:
    tensor_data = tuple(map(tf.convert_to_tensor, data))
    elbo(tensor_data)  # run it once to trace & compile

    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).repeat().shuffle(N)
    # ground_truth = elbo(tensor_data).numpy()

    minibatch_proportion(N, train_dataset, elbo)

    # We turn off training for inducing point locations
    gpflow.set_trainable(model.inducing_variable, False)
    maxiter = reduce_in_tests(max_iter)
    logf = run_adam(model, maxiter, minibatch_size, train_dataset)

    plot_result(maxiter, logf)

    gpflow.utilities.print_summary(model, "notebook")

    train_time = time.time() - start_time
    logging.info(f'Time Process: {train_time}')
    return model, train_time


def predict_y(model, test_x):
    y_mean, y_var = model.predict_y(test_x, full_cov=False)
    return y_mean, y_var


def evaluate(test_y, y_mean):
    mae = np.mean(np.abs(y_mean - test_y))
    rmse = np.sqrt(np.mean((y_mean - test_y) ** 2))
    logging.info('Test MAE: {}'.format(mae))
    logging.info('Test RMSE: {}'.format(rmse))
    return mae, rmse


if __name__ == "__main__":
    logging.info('Load data for training and testing')
    train_x, train_y, test_x, test_y = util.load_data(N=1000000, N_test=10000)

    logging.info('Plot data in chart and save file')
    util.plot_data(train_x, train_y, test_x, test_y, file_name=f'{CLASS_NAME}_data.png')

    M = 128  # Number of inducing locations
    logging.info(f'training gia tri m={M}, N = {train_x.shape[0]}')

    model = train(train_x, train_y, minibatch_size=100, m_inducing=M)
    y_mean, y_var = predict_y(model, test_x)
    evaluate(test_y, y_mean)

    util.plot_output_model(train_x, train_y, test_x, test_y, y_mean, y_var, file_name=f'{CLASS_NAME}_result.png')
