import logging
import time
from gpmp_scale.gp_utils import load_data, plot_data, plot_output_model
from gpmp_scale.sgpr.SGPR import train, predict_y, evaluate, train_and_test_with_inducing

CLASS_NAME = 'SGPR'
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(f"{CLASS_NAME}_log.log"),
                        logging.StreamHandler()
                    ])

if __name__ == "__main__":
    start_time = time.time()
    logging.info(f'------TRANING SGPR')
    logging.info('1. Load data for training and testing')
    train_x, train_y, test_x, test_y = load_data(N=20000, N_test=10000)
    logging.info('2. Plot data in chart and save file')

    plot_data(train_x, train_y, test_x, test_y, file_name=f'{CLASS_NAME}_data.png')

    M = 20
    logging.info(f'-------RUNNING with N = {train_x.shape[0]}, M = {M}')
    model, _ = train(train_x, train_y, M)
    y_mean, y_var = predict_y(model, test_x)
    evaluate(test_y, y_mean)

    plot_output_model(train_x, train_y, test_x, test_y, y_mean, y_var, file_name=f'{CLASS_NAME}_result.png', title='SGPR')

    m_inducing_array = [5, 10, 15, 20, 25, 32, 48, 50, 64, 96, 128, 256, 512]
    train_and_test_with_inducing(train_x, train_y, test_x, test_y, m_inducing_array)
