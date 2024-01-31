from __future__ import division
from __future__ import print_function

import logging

from sklearn.model_selection import train_test_split

from gpmp_scale.fitc.FITC import train, predict, evaluate
from gpmp_scale.gp_utils import load_climate_dataset, plot_mean_variance, normalize

CLASS_NAME = 'FITC'
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(f"{CLASS_NAME}_log.log"),
                        logging.StreamHandler()
                    ])

if __name__ == '__main__':
    logging.info(f'----------------------------------------')
    logging.info(f'Training FITC voi du lieu Climate Dataset')
    x_data, y_data = load_climate_dataset()
    x_data[:, 0] = normalize(x_data[:, 0])
    x_data[:, 1] = normalize(x_data[:, 1])
    x_data[:, 2] = normalize(x_data[:, 2])
    y_data = normalize(y_data)

    # Tach tap train-test
    train_x, test_x, train_y, test_y = train_test_split(x_data, y_data, train_size=0.0004, test_size=0.0002,
                                                        random_state=42)

    M = 128

    model, _ = train(train_x, train_y, M)
    mean, _ = predict(model, test_x)
    evaluate(test_y, mean)

    plot_mean_variance(test_y[:100], mean[:100], saveFile='FITC_mean_variance')
