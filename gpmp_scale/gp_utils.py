from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

base_path = Path(__file__).parent


def f1(x):
    return torch.sin(5. * x) + torch.cos(10. * x) + (2. * (x - 0.4) ** 2) * torch.cos(100. * x)


def generate_data(N=20000, N_test=10000):
    test_x = torch.linspace(0, 1, N_test)
    test_y = f1(test_x)

    x_data = torch.linspace(0, 1, N)
    y_data = f1(x_data) + (torch.rand(len(x_data)) - 0.5) * 0.5
    return x_data, y_data, test_x, test_y


# Generate dataset numpy
def f1_np(x):
    return np.sin(5. * x) + np.cos(10. * x) + (2. * (x - 0.4) ** 2) * np.cos(100. * x)


def load_data(N=20000, N_test=10000):
    train_x = np.random.rand(N)
    train_y = f1_np(train_x) + (np.random.rand(len(train_x)) - 0.5) * 0.5
    train_x = train_x.reshape(-1, 1)
    train_y = train_y.reshape(-1, 1)

    test_x = np.linspace(0, 1, N_test).reshape(-1, 1)
    test_y = f1_np(test_x)
    return train_x, train_y, test_x, test_y


def plot_result(x_data, y_data, test_x, test_y, mean, lower, upper, file_name='result.png'):
    # Initalize plot
    plt.figure(figsize=(15, 5))
    x_train = x_data.detach().numpy()
    y_train = y_data.detach().numpy()
    print('Get 100 point only to show in chart')
    index = np.random.choice(y_data.shape[0], 100, replace=False)
    x_train = x_train[index]
    y_train = y_train[index]

    plt.xticks([0., 0.5, 1.0])
    plt.yticks([-2, -1, 0., 1])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.plot(x_train, y_train, 'k*', linewidth=4)
    # Plot predictive means as blue line
    plt.plot(test_x.detach().numpy(), mean.detach().numpy(), 'b', linewidth=4)

    # Plot confidence bounds as lightly shaded region
    plt.fill_between(test_x.detach().numpy(), lower.detach().numpy(), upper.detach().numpy(), alpha=0.5,
                     color="grey", label="var")
    plt.plot(test_x, test_y, color='orange', linewidth=4)
    plt.legend(['Observed Data', 'Mean', 'Confidence', 'f(x)'])
    plt.savefig(f'{file_name}')


def plot_output_model(train_x, train_y, test_x, test_y, y_mean, y_var, file_name='result.png', title=None) -> None:
    y_lower = y_mean - 1.96 * np.sqrt(y_var)
    y_upper = y_mean + 1.96 * np.sqrt(y_var)
    index = np.random.choice(train_y.shape[0], 100, replace=False)
    train_x = train_x[index]
    train_y = train_y[index]

    plt.figure(figsize=(15, 5))
    plt.xticks([0., 0.5, 1.0])
    plt.yticks([-2, -1, 0., 1])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    if title:
        plt.title(title)
    plt.plot(train_x, train_y, 'k*', linewidth=4)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.plot(test_x, y_mean, "b", linewidth=4)
    plt.fill_between(test_x[:, 0], y_lower[:, 0], y_upper[:, 0], color="grey", alpha=0.5, label="var")

    plt.plot(test_x, test_y, color='orange', linewidth=2)

    plt.legend(['Observed Data', 'Mean', 'Confidence', 'f(x)'])
    plt.savefig(file_name)
    plt.show()


def plot_data(train_x, train_y, test_x, test_y, file_name='result.png'):
    plt.figure(figsize=(15, 5))
    plt.xticks([0., 0.5, 1.0])
    plt.yticks([-2, -1, 0., 1])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.plot(test_x, test_y, color='orange', linewidth=4)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.scatter(train_x, train_y, color='black')
    plt.savefig(file_name)
    plt.show()


def plot_result2(train_x, train_y, test_x, test_y, mean1, var1, file_name):
    plt.figure(figsize=(16, 10))
    plt.plot(test_x, mean1, label="posterior mean", linewidth=4)
    plt.plot(test_x, test_y, label="latent function", linewidth=4)
    plt.fill_between(test_x, mean1 - 3. * np.sqrt(var1), mean1 + 3. * np.sqrt(var1), alpha=0.5, color="grey",
                     label="var")
    plt.scatter(train_x, train_y, color='black')
    plt.savefig(file_name)
    plt.show()


def normalize(v, scale='MinMaxScale'):
    if 'MinMaxScale' == scale:
        scaled_value = (v - np.min(v)) / (np.max(v) - np.min(v))
    elif 'StandardScaler':
        scaler = StandardScaler()
        scaled_value = scaler.fit_transform(v)
    else:
        scaled_value = (v - np.min(v)) / np.max(v)
    return scaled_value


def load_climate_dataset():
    station_path = (base_path / "dataset/climate/station_coord.npy").resolve()
    data_path = (base_path / "dataset/climate/data.npy").resolve()

    station_locations = np.load(station_path)
    temperatures = np.load(data_path)
    N = len(station_locations) * len(temperatures)
    x_data = np.zeros((N, 3))
    y_data = np.zeros((N))
    count = 0
    for i in range(len(temperatures)):
        for j in range(len(temperatures[0])):
            x_data[count] = np.array([station_locations[j, 0], station_locations[j, 1], float(i)])
            y_data[count] = temperatures[i, j]
            count += 1

    non_nan_indices = np.where(y_data == y_data)  ###nans in data
    x_data = x_data[non_nan_indices]
    y_data = y_data[non_nan_indices]
    return x_data, y_data


def load_climate_dataset_2d():
    station_path = (base_path / "dataset/climate/station_coord.npy").resolve()
    data_path = (base_path / "dataset/climate/data.npy").resolve()

    station_locations = np.load(station_path)
    temperatures = np.load(data_path)
    N = len(station_locations) * len(temperatures)
    x_data = np.zeros((N, 2))
    y_data = np.zeros((N))
    count = 0
    for i in range(len(temperatures)):
        for j in range(len(temperatures[0])):
            x_data[count] = np.array([station_locations[j, 0], station_locations[j, 1]])
            y_data[count] = temperatures[i, j]
            count += 1

    non_nan_indices = np.where(y_data == y_data)  ###nans in data
    x_data = x_data[non_nan_indices]
    y_data = y_data[non_nan_indices]
    return x_data, y_data


def load_bike_sharing_dataset():
    data_path = (base_path / "dataset/bike_sharing/hour.csv").resolve()
    df = pd.read_csv(data_path)
    return df


def load_3d_road_dataset():
    data_path = (base_path / "dataset/3d_road/3D_spatial_network.txt").resolve()
    df = pd.read_csv(data_path, header=None)
    df.columns = ['OSM_ID', 'LONGITUDE', 'LATITUDE', 'ALTITUDE']
    return df


def plot_mean_variance(test_y, y_predict, title='Predictions w/ 95% Confidence', xLabel='Index', yLabel='Y value',
                       ylim=None, saveFile='chart_mean_variance'):
    residuals = sorted([x - y for x, y in zip(y_predict, test_y)])
    valid = pd.DataFrame(test_y)
    valid.columns = ['temp']

    RMSFE = np.sqrt(sum([x ** 2 for x in residuals]) / len(residuals))
    band_size = 1.96 * RMSFE

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(valid.index, valid['temp'], color='#fc7d0b', label='Valid')
    ax.scatter(valid.index, y_predict)
    # ax.fill_between(valid.index, (valid['temp'] - band_size), (valid['temp'] + band_size), color='b', alpha=.1)
    ax.fill_between(valid.index, (y_predict - band_size), (y_predict + band_size), color='b', alpha=.1)
    ax.set_title(title)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    if ylim:
        ax.set_ylim(0, ylim)
    plt.savefig(f'{saveFile}.png')
    plt.show()


def convert_to_Tensor(train_x, train_y, test_x, test_y):
    train_x = torch.Tensor(train_x)
    train_y = torch.Tensor(train_y)
    test_x = torch.Tensor(test_x)
    test_y = torch.Tensor(test_y)
    return train_x, train_y, test_x, test_y
