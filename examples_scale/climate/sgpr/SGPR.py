import logging

from sklearn.model_selection import train_test_split
from gpmp_scale.gp_utils import load_climate_dataset, plot_mean_variance, normalize
from gpmp_scale.sgpr.SGPR import train, predict_y, evaluate, train_and_test_with_inducing

INPUT_DIM = 3
CLASS_NAME = 'SGPR'
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(f"{CLASS_NAME}_log.log"),
                        logging.StreamHandler()
                    ])



if __name__ == '__main__':
    logging.info(f'----------------------------------------')
    logging.info(f'Training SGPR with Climate Dataset')
    x_data, y_data = load_climate_dataset()
    x_data[:, 0] = normalize(x_data[:, 0])
    x_data[:, 1] = normalize(x_data[:, 1])
    x_data[:, 2] = normalize(x_data[:, 2])
    y_data = normalize(y_data)

    # Tach tap train-test
    train_x, test_x, train_y, test_y = train_test_split(x_data, y_data, train_size=0.0004, test_size=0.0002,
                                                        random_state=42)
    logging.info(f'train_x.shape = {train_x.shape}, train_y.shape= {train_y.shape}, '
                 f'test_x.shape={test_x.shape}, test_y.shape={test_y.shape}')
    train_y = train_y.reshape(-1, 1)
    test_y = test_y.reshape(-1, 1)

    M = 32
    model, _ = train(train_x, train_y, M)
    y_mean, y_var = predict_y(model, test_x)
    evaluate(test_y, y_mean)

    m_inducing_array = [32, 48, 64, 96]
    train_and_test_with_inducing(train_x, train_y, test_x, test_y, m_inducing_array)

    plot_mean_variance(test_y[:100], y_mean[:100], saveFile='SGPR_mean_variance')
