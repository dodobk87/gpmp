import logging
import time
from abc import abstractmethod

import gpytorch
import numpy as np
from matplotlib import pyplot as plt


class BaseGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, clazz):
        super(BaseGP, self).__init__(train_x, train_y, likelihood)
        self.clazz = clazz
        logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s',
                            handlers=[
                                logging.FileHandler(f"{self.clazz}_log.log"),
                                logging.StreamHandler()
                            ])

    @abstractmethod
    def train_data(self, train_x, train_y, lr=0.1, training_iter=2):
        pass

    @abstractmethod
    def infer(self, test_x, test_y):
        pass

    def run(self, train_x, train_y, test_x, test_y):
        start_time = time.time()
        logging.info('1. Init data training')
        logging.info(f'2. DataSet for training: x.shape = {train_x.shape}, y.shape= {train_y.shape}')
        image_name = f'{self.clazz}_data.png'
        if len(train_x.shape) == 1 or train_x.shape[1] == 1:
            # dataset 1 dimension
            logging.info(f"3.save image chart data for training to image:{image_name}")
            self.plot_data(train_x, train_y, test_x, test_y, image_name=image_name)

        model, _ = self.train_data(train_x, train_y)
        mean, lower, upper = self.infer(test_x, test_y)
        if len(train_x.shape) == 1 or train_x.shape[1] == 1:
            # dataset 1 dimension
            logging.info(f'7.Plot result chart:')
            self.plot_result(train_x, train_y, test_x, test_y, mean, lower, upper, file_name=f'{self.clazz}_result.png')
        logging.info(f'8.Time process: {time.time() - start_time}')

    def plot_data(self, x_data, y_data, test_x, test_y, image_name):
        plt.figure(figsize=(15, 5))
        plt.xticks([0., 0.5, 1.0])
        plt.yticks([-2, -1, 0., 1])
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.plot(test_x, test_y, color='orange', linewidth=4)
        plt.scatter(x_data, y_data, color='black')
        plt.savefig(f'{image_name}')

    def plot_result(self, x_data, y_data, test_x, test_y, mean, lower, upper, file_name='result.png', title=None):
        # Initalize plot
        plt.figure(figsize=(15, 5))
        x_train = x_data.detach().numpy()
        y_train = y_data.detach().numpy()
        print('Get 100 point only to show in chart')
        index = np.random.choice(y_data.shape[0], 100, replace=False)
        x_train = x_train[index]
        y_train = y_train[index]
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xticks([0., 0.5, 1.0])
        plt.yticks([-2, -1, 0., 1])
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        if title:
            plt.title(title)
        plt.plot(x_train, y_train, 'k*', linewidth=4)
        # Plot predictive means as blue line
        plt.plot(test_x.detach().numpy(), mean.detach().numpy(), 'b', linewidth=4)

        # Plot confidence bounds as lightly shaded region
        plt.fill_between(test_x.detach().numpy(), lower.detach().numpy(), upper.detach().numpy(), alpha=0.5,
                         color="grey", label="var")
        plt.plot(test_x, test_y, color='orange', linewidth=2)
        plt.legend(['Observed Data', 'Mean', 'Confidence', 'f(x)'])
        plt.savefig(f'{file_name}')
