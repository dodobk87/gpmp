import matplotlib.pyplot as plt

from gpmp_scale.gp_utils import load_bike_sharing_dataset

df = load_bike_sharing_dataset()

def hist(df, col, title, xlabel, ylabel, ylim=None, xticks=None, showLabel=False):
    df_hour = df.groupby(col).hr.agg('count').to_frame('count').reset_index()
    name_h = df_hour[col].to_numpy()
    value_h = df_hour['count'].to_numpy()
    bar = plt.bar(name_h, value_h)
    if xticks:
        plt.xticks(name_h, xticks)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if ylim:
        plt.ylim(0, ylim)
    if showLabel:
        plt.bar_label(bar, padding=3)
    plt.savefig(f'visualize_data/{title}.png')
    plt.show()

if __name__ == '__main__':
    ylim = 800
    hist(df,'hr','Distribution of Hour','Hour','Number of Data', ylim = ylim)

    my_xticks = ['2011','2012']
    ylim = 10000
    hist(df,'yr','Distribution of Year','Year','Number of Data', xticks=my_xticks, ylim = ylim)

    my_xticks = ['springer','summer','fall','winter']
    ylim = 5000
    hist(df,'season','Distribution of Season','Season','Number of Data', xticks=my_xticks, ylim = ylim, showLabel = True)

    ylim = 1600
    hist(df,'mnth','Distribution of Month','Month','Number of Data', ylim = ylim)

    my_xticks = ['Clear','Cloudy','Light Rain','Heavy Rain']
    ylim = 12500
    hist(df,'weathersit','Distribution of Weather','Weather','Number of Data', xticks=my_xticks, ylim = ylim, showLabel = True)

    ylim = 18000
    my_xticks = ['Not Holiday','Holiday']
    hist(df,'holiday','Distribution of Holiday','Holiday','Number of Data', xticks=my_xticks, ylim = ylim)

    ylim = 13000
    my_xticks = ['Not Workingday','Workingday']
    hist(df,'workingday','Distribution of Workingday','Workingday','Number of Data', xticks=my_xticks, ylim = ylim)

    ylim = 2700
    hist(df,'weekday','Distribution of Weekday','Weekday','Number of Data', ylim = ylim)

