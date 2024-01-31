from __future__ import division
from __future__ import print_function

import logging

from sklearn.model_selection import train_test_split

from gpmp_scale.fitc.FITC import train, predict, evaluate
from gpmp_scale.gp_utils import load_bike_sharing_dataset, plot_mean_variance, normalize

CLASS_NAME = 'FITC'
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(f"{CLASS_NAME}_log.log"),
                        logging.StreamHandler()
                    ])
M = 128


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
    logging.info(f'Training FITC voi du lieu Bike Sharing')
    df = load_bike_sharing_dataset()
    logging.info(f'Describe du lieu truoc khi chuan hoa:')
    logging.info(df.describe())

    logging.info(f'Xoa cac cot thua dteday, casual, registered, instant')
    df = df.drop(['dteday', 'casual', 'registered', 'instant'], axis=1)
    df = clean_df(df)
    logging.info(f'Describe du lieu sau khi chuan hoa:')
    logging.info(df.describe())

    y_data = df['cnt']
    df = df.drop(['cnt'], axis=1)

    train_y_full = y_data.to_numpy().reshape(-1, 1)
    df_array = df.to_numpy()

    train_x, test_x, train_y, test_y = train_test_split(df_array, train_y_full, test_size=0.33, random_state=42)
    logging.info(f'train_x.shape = {train_x.shape}, train_y.shape= {train_y.shape}, '
                 f'test_x.shape={test_x.shape}, test_y.shape={test_y.shape}')

    model, _ = train(train_x, train_y, M)
    mean, _ = predict(model, test_x)
    evaluate(test_y, mean)

    plot_mean_variance(test_y[:100], mean.reshape(-1)[:100], saveFile='FITC_mean_variance')
