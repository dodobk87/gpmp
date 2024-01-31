import logging
from gpmp_scale.gp_utils import load_data, plot_data, plot_output_model
from gpmp_scale.svgp.SVGP import train, predict_y, evaluate

CLASS_NAME = 'SVGP'
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(f"{CLASS_NAME}_log.log"),
                        logging.StreamHandler()
                    ])

if __name__ == "__main__":
    logging.info('Load data for training and testing')
    train_x, train_y, test_x, test_y = load_data(N=20000, N_test=10000)

    logging.info('Plot data in chart and save file')
    plot_data(train_x, train_y, test_x, test_y, file_name=f'{CLASS_NAME}_data.png')

    M = 128  # Number of inducing locations
    logging.info(f'training gia tri m={M}, N = {train_x.shape[0]}')

    model, _ = train(train_x, train_y, minibatch_size=100, m_inducing=M)
    y_mean, y_var = predict_y(model, test_x)
    evaluate(test_y, y_mean)

    plot_output_model(train_x, train_y, test_x, test_y, y_mean, y_var, file_name=f'{CLASS_NAME}_result.png')
