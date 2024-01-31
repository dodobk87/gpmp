from __future__ import division
from __future__ import print_function

import logging
import time

from gpmp_scale.fitc.FITC import train, predict, evaluate
from gpmp_scale.gp_utils import load_data, plot_data, plot_output_model

CLASS_NAME = 'FITC'
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(f"{CLASS_NAME}_log.log"),
                        logging.StreamHandler()
                    ])

if __name__ == "__main__":
    start_time = time.time()
    logging.info('1. Load data for training and testing')
    train_x, train_y, test_x, test_y = load_data()

    plot_data(train_x, train_y, test_x, test_y, file_name=f'{CLASS_NAME}_data.png')

    # inducing point
    M = 64
    model, time_train = train(train_x, train_y, M)
    mean, variance = predict(model, test_x)
    evaluate(test_y, mean)

    logging.info('5. Plot result GP FITC')

    plot_output_model(train_x, train_y, test_x, test_y, mean, variance, file_name=f"{CLASS_NAME}_M_{M}_result.png",
                      title='FITC')
    logging.info(f'time total:{time_train}')
