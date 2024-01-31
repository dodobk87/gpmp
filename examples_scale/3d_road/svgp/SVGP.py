import logging

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from gpmp_scale.gp_utils import load_3d_road_dataset, plot_mean_variance, normalize
from gpmp_scale.svgp.SVGP import train, predict_y, evaluate

CLASS_NAME = 'SGPR'
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


def plot_result(maxiter, logf):
    plt.plot(np.arange(maxiter)[::10], logf)
    plt.xlabel("iteration")
    plt.ylabel("ELBO")
    plt.savefig('SVGP_elbo_interation.png')


if __name__ == '__main__':
    logging.info(f'----------------------------------------')
    logging.info(f'Training SVGP with 3D Road')
    df = load_3d_road_dataset()
    logging.info(df.describe())

    # Chuan hoa du lieu
    df = clean_df(df)

    y_data = df['ALTITUDE']
    df = df.drop(['ALTITUDE', 'OSM_ID'], axis=1)
    train_y_full = y_data.to_numpy()
    df_array = df.to_numpy()

    # Tach tap train-test
    train_x, test_x, train_y, test_y = train_test_split(df_array, train_y_full, train_size=0.046, test_size=0.023,
                                                        random_state=42)
    logging.info(f'train_x.shape = {train_x.shape}, train_y.shape= {train_y.shape}, '
                 f'test_x.shape={test_x.shape}, test_y.shape={test_y.shape}')
    train_y = train_y.reshape(-1, 1)
    test_y = test_y.reshape(-1, 1)

    M = 128
    model, _ = train(train_x, train_y, minibatch_size=100, m_inducing=M)
    y_mean, y_var = predict_y(model, test_x)
    evaluate(test_y, y_mean)

    plot_mean_variance(test_y[:100], y_mean[:100], saveFile='SVGP_mean_variance_95')
