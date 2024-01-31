import logging

from sklearn.model_selection import train_test_split

from gpmp_scale.gp_utils import load_bike_sharing_dataset, plot_mean_variance, normalize
from gpmp_scale.svgp.SVGP import train, predict_y, evaluate

M = 128
CLASS_NAME = 'SGPR'
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(f"{CLASS_NAME}_log.log"),
                        logging.StreamHandler()
                    ])

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
    logging.info(f'Training SGPR with Bike Sharing m={M}')
    df = load_bike_sharing_dataset()
    logging.info(df.describe())

    # Chuan hoa du lieu
    df = df.drop(['dteday', 'casual', 'registered', 'instant'], axis=1)
    df = clean_df(df)

    y_data = df['cnt']
    df = df.drop(['cnt'], axis=1)

    train_y_full = y_data.to_numpy()
    df_array = df.to_numpy()

    # Tach tap train-test
    train_x, test_x, train_y, test_y = train_test_split(df_array, train_y_full, test_size=0.33, random_state=42)
    logging.info(f'train_x.shape = {train_x.shape}, train_y.shape= {train_y.shape}, '
                 f'test_x.shape={test_x.shape}, test_y.shape={test_y.shape}')
    train_y = train_y.reshape(-1, 1)
    test_y = test_y.reshape(-1, 1)

    M = 32
    model, _ = train(train_x, train_y, minibatch_size=100, m_inducing=M)
    y_mean, y_var = predict_y(model, test_x)
    evaluate(test_y, y_mean)

    plot_mean_variance(test_y[:100], y_mean[:100], saveFile='SVGP_mean_variance_95')
