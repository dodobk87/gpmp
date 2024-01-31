import logging
import time
import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal

import gpmp_scale.gp_utils as util
from gpmp_scale.BaseGP import BaseGP


class BBMM(BaseGP):
    def __init__(self, train_x, train_y, clazz='BBMM'):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super().__init__(train_x, train_y, likelihood, clazz)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.keops.MaternKernel(nu=1.5))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def train_data(self, train_x, train_y, lr=0.1, training_iter=2):
        start_time = time.time()
        logging.info(f"4. Create GPRegressionModel training GaussianLikelihood")

        # Find optimal model hyperparameters
        logging.info(f"5. Training.....")
        self.train()
        self.likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        logging.info(f'6."Loss" for GPs - the marginal log likelihood')
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self(train_x)

            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                self.covar_module.base_kernel.lengthscale.item(),
                self.likelihood.noise.item()
            ))
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
    train_x, train_y, test_x, test_y = util.generate_data()
    gp = BBMM(train_x, train_y)

    gp.plot_data(train_x, train_y, test_x, test_y, image_name='BBMM_data.png')
    model, _ = gp.train_data(train_x, train_y)
    mean, lower, upper = gp.infer(test_x, test_y)
    gp.plot_result(train_x, train_y, test_x, test_y, mean, lower, upper, file_name='BBMM_result.png', title='BBMM')
