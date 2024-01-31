import logging

from sklearn.model_selection import train_test_split

from gpmp_scale.bbmm.BBMM import BBMM
from gpmp_scale.gp_utils import load_climate_dataset, normalize, convert_to_Tensor, plot_mean_variance

CLASS_NAME = 'BBMM'
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(f"{CLASS_NAME}_log.log"),
                        logging.StreamHandler()
                    ])

if __name__ == '__main__':
    logging.info(f'----------------------------------------')
    logging.info(f'Training BBMM with Climate')
    x_data, y_data = load_climate_dataset()
    x_data[:, 0] = normalize(x_data[:, 0])
    x_data[:, 1] = normalize(x_data[:, 1])
    x_data[:, 2] = normalize(x_data[:, 2])
    y_data = normalize(y_data)

    # Tach tap train-test
    train_x, test_x, train_y, test_y = train_test_split(x_data, y_data, train_size=0.0004, test_size=0.0002,
                                                        random_state=42)
    logging.info(f'train_x.shape = {train_x.shape}, train_y.shape= {train_y.shape},'
                 f' test_x.shape={test_x.shape}, test_y.shape={test_y.shape}')

    train_x, train_y, test_x, test_y = convert_to_Tensor(train_x, train_y, test_x, test_y)

    gp = BBMM(train_x, train_y)

    model, _ = gp.train_data(train_x, train_y)
    mean, lower, upper = gp.infer(test_x, test_y)

    plot_mean_variance(test_y.numpy()[:100], mean.numpy()[:100], saveFile='BBMM_mean_variance')
