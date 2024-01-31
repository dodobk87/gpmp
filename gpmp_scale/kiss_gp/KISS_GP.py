import logging
import time

import gpytorch
import torch

import gpmp_scale.gp_utils as util
from gpmp_scale.BaseGP import BaseGP


class KISS_GP(BaseGP):
    def __init__(self, train_x, train_y, clazz='KISS_GP'):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super().__init__(train_x, train_y, likelihood, clazz)

        gpytorch.utils.grid.choose_grid_size(train_x, kronecker_structure=False)
        if len(train_x.shape) > 1:
            DIM = train_x.shape[1]
        else:
            DIM = 1
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.AdditiveStructureKernel(
            gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.GridInterpolationKernel(
                    gpytorch.kernels.RBFKernel(), grid_size=128, num_dims=1
                )
            ), num_dims=DIM
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def train_data(self, train_x, train_y, lr=0.1, training_iter=30):
        start_time = time.time()
        logging.info(f"4. Create GPRegressionModel training GaussianLikelihood")

        print(f'training_iter:{training_iter}')

        logging.info(f"5. Training.....")
        self.train()
        self.likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        print(f'6."Loss" for GPs - the marginal log likelihood')
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        for i in range(training_iter):
            optimizer.zero_grad()
            output = self(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
        logging.info(f'Time process: {time.time() - start_time}')
        time_train = time.time() - start_time
        return self, time_train

    def infer(self, test_x, test_y):
        # Put model & likelihood into eval mode
        self.eval()
        self.likelihood.eval()

        # The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
        # See https://arxiv.org/abs/1803.06058
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            prediction = self.likelihood(self(test_x))
            mean = prediction.mean
            # Get lower and upper predictive bounds
            lower, upper = prediction.confidence_region()
            logging.info('Test MAE: {}'.format(torch.mean(torch.abs(prediction.mean - test_y))))
            logging.info('Test RMSE: {}'.format(torch.sqrt(torch.mean((prediction.mean - test_y) ** 2))))
            logging.info('Test NLL: {}'.format(-prediction.to_data_independent_dist().log_prob(test_y).mean().item()))
        return mean, lower, upper


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = util.generate_data(N=1000000)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gp = KISS_GP(train_x, train_y, likelihood)

    gp.plot_data(train_x, train_y, test_x, test_y, image_name=f'{gp.clazz}_data.png')
    model, _ = gp.train_data(train_x, train_y, training_iter=30)
    mean, lower, upper = gp.infer(test_x, test_y)
    gp.plot_result(train_x, train_y, test_x, test_y, mean, lower, upper, file_name=f'{gp.clazz}_result.png', title='KISS_GP')
