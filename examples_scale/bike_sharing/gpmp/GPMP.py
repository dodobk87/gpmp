import math
import time

import numpy as np
from sklearn.model_selection import train_test_split

import gpmp as gp
import gpmp.num as gnp
from gpmp_scale.gp_utils import normalize, load_bike_sharing_dataset


def clean_df(df):
    df['season'] = normalize(df['season'])
    df['mnth'] = normalize(df['mnth'])
    df['hr'] = normalize(df['hr'])
    df['weekday'] = normalize(df['weekday'])
    df['weathersit'] = normalize(df['weathersit'])
    df['cnt'] = normalize(df['cnt'])
    df['yr'] = df['yr'].astype(float)
    df['holiday'] = df['holiday'].astype(float)
    df['workingday'] = df['workingday'].astype(float)
    return df


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
    df = load_bike_sharing_dataset()

    df = df.drop(['dteday', 'casual', 'registered', 'instant'], axis=1)
    df = clean_df(df)

    y_data = df['cnt']
    df = df.drop(['cnt'], axis=1)

    train_y_full = y_data.to_numpy().reshape(-1, 1)
    df_array = df.to_numpy()

    xt, xi, zt, zi = train_test_split(df_array, train_y_full, test_size=0.33, random_state=42)

    mean = constant_mean
    meanparam = None

    covparam = gnp.array([
        math.log(0.5 ** 2),
        math.log(1 / .7),
        2 * math.log(noise_std)])

    model = gp.core.Model(mean, kernel, meanparam, covparam)

    (zpm, zpv) = model.predict(xt, zt, xi)

    return xt, zt, xi, zi, zpm, zpv


if __name__ == '__main__':
    start_time = time.time()
    xt, zt, xi, zi, zpm, zpv = main()
    mae = np.mean(np.abs(zpm - zi))
    rmse = np.sqrt(np.mean((zpm - zi) ** 2))
    print(f'1. MAE:{mae}')
    print(f'2. RMSE:{rmse}')
    print(f'3. Time: {time.time() - start_time}')
