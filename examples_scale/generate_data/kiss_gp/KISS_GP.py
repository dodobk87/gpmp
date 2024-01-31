import gpytorch

from gpmp_scale.gp_utils import generate_data
from gpmp_scale.kiss_gp.KISS_GP import KISS_GP

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = generate_data(N=20000)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gp = KISS_GP(train_x, train_y, likelihood)

    gp.plot_data(train_x, train_y, test_x, test_y, image_name=f'{gp.clazz}_data.png')
    model, _ = gp.train_data(train_x, train_y, training_iter=30)
    mean, lower, upper = gp.infer(test_x, test_y)
    gp.plot_result(train_x, train_y, test_x, test_y, mean, lower, upper, file_name=f'{gp.clazz}_result.png', title='KISS_GP')
