import logging
import time

from dask.distributed import Client

from gpmp_scale.exactGP_discoverkernel.exactGP_DiscoverKernel import train, predict, evaluate
from gpmp_scale.gp_utils import load_data, plot_data, plot_output_model

CLASS_NAME = 'exactGP_DiscoverKernel'
NUM_TRAIN = 20000
BATCH_SIZE = 1000
MAX_ITER = 10
INPUT_DIM = 1

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(f"{CLASS_NAME}_log.log"),
                        logging.StreamHandler()
                    ])

if __name__ == '__main__':
    start_time = time.time()
    logging.info('1. Generate data')
    train_x, train_y, test_x, test_y = load_data(N=20000, N_test=100)

    logging.info('2. Plot data training in chart')
    plot_data(train_x, train_y, test_x, test_y)
    client = Client()
    # client = Client("tcp://10.61.100.71:8080")
    model, train_time = train(train_x, train_y, INPUT_DIM, max_iter=MAX_ITER, gp2Scale_dask_client=client, batchSize=BATCH_SIZE, method='mcmc')

    mean1, var1 = predict(model, test_x.reshape(-1, 1))
    evaluate(test_y.reshape(-1), mean1.reshape(-1))

    logging.info('6. Plot Result in chart')

    plot_output_model(train_x, train_y, test_x, test_y, mean1.reshape(-1, 1), var1.reshape(-1, 1),
                      file_name=f'{CLASS_NAME}_result.png')

    logging.info(f'Time Process: {train_time}')
