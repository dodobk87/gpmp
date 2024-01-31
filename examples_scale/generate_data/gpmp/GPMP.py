import math
import time

import matplotlib.pyplot as plt
import numpy as np

import gpmp as gp
import gpmp.num as gnp

# N = 20000
# N_test = 1000

N = 2000
N_test = 100


def f1(x):
    return np.sin(5. * x) + np.cos(10. * x) + (2. * (x - 0.4) ** 2) * np.cos(100. * x)


def load_data():
    train_x = np.random.rand(N)
    train_y = f1(train_x) + (np.random.rand(len(train_x)) - 0.5) * 0.5
    train_x = train_x.reshape(-1, 1)

    test_x = np.linspace(0, 1, N_test)
    test_y = f1(test_x)
    test_x = test_x.reshape(-1, 1)

    return train_x, train_y, test_x, test_y


def constant_mean(x, _):
    return gnp.ones((x.shape[0], 1))


def kernel_ii_or_tt(x, param, pairwise=False):
    p = 2
    sigma2 = gnp.exp(param[0])
    loginvrho = param[1]
    noise_variance = gnp.exp(param[2])

    if pairwise:
        K = sigma2 * gnp.ones((x.shape[0],))
    else:
        K = gnp.scaled_distance(loginvrho, x, x)
        K = sigma2 * gp.kernel.maternp_kernel(p, K) + noise_variance * gnp.eye(K.shape[0])

    return K


def kernel_it(x, y, param, pairwise=False):
    p = 2
    sigma2 = gnp.exp(param[0])
    loginvrho = param[1]

    if pairwise:
        K = gnp.scaled_distance_elementwise(loginvrho, x, y)
    else:
        K = gnp.scaled_distance(loginvrho, x, y)

    K = sigma2 * gp.kernel.maternp_kernel(p, K)
    return K


def kernel(x, y, param, pairwise=False):
    if y is x or y is None:
        return kernel_ii_or_tt(x, param, pairwise)
    else:
        return kernel_it(x, y, param, pairwise)


def main():
    noise_std = 1e-1
    xt, zt, xi, zi = load_data()

    mean = constant_mean
    meanparam = None

    covparam = gnp.array([
        math.log(0.5 ** 2),
        math.log(1 / .7),
        2 * math.log(noise_std)])

    model = gp.core.Model(mean, kernel, meanparam, covparam)

    (zpm, zpv) = model.predict(xt, zt, xi)

    return xt, zt, xi, zi, zpm, zpv


def plot_result(xt, zt, xi, zi, zpm, zpv, chartName="gpmp_result"):
    plt.figure(figsize=(15, 5))
    plt.xticks([0., 0.5, 1.0])
    plt.yticks([-2, -1, 0., 1])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.plot(xt, zt, 'k*', linewidth=1)
    plt.plot(xi, zi, 'b', linewidth=4)
    plt.fill_between(xi.reshape(-1), zpm - 1.96 * np.sqrt(zpv), zpm + 1.96 * np.sqrt(zpv), alpha=0.8, color="g",
                     label="var", linewidth=2)
    plt.scatter(xt, zt, color='black', alpha=0.2)
    plt.savefig(f'{chartName}.png')
    plt.show()


if __name__ == '__main__':
    start_time = time.time()
    xt, zt, xi, zi, zpm, zpv = main()
    plot_result(xt, zt, xi, zi, zpm, zpv)
    mae = np.mean(np.abs(zpm - zi))
    rmse = np.sqrt(np.mean((zpm - zi) ** 2))
    print(f'1. MAE:{mae}')
    print(f'2. RMSE:{rmse}')
    print(f'3. Time: {time.time() - start_time}')
