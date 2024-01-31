import logging

from sklearn.model_selection import train_test_split

from gpmp_scale.bbmm.BBMM import BBMM
from gpmp_scale.gp_utils import load_bike_sharing_dataset, normalize, convert_to_Tensor, plot_mean_variance

CLASS_NAME = 'BBMM'
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
    logging.info(f'Training SGPR with Bike Sharing')
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

    train_x, train_y, test_x, test_y = convert_to_Tensor(train_x, train_y, test_x, test_y)

    gp = BBMM(train_x, train_y)
    gp.run(train_x, train_y, test_x, test_y)

    model, _ = gp.train_data(train_x, train_y)
    mean, lower, upper = gp.infer(test_x, test_y)

    plot_mean_variance(test_y.numpy()[:100], mean.numpy()[:100], saveFile='BBMM_mean_variance')
