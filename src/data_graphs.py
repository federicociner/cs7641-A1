import numpy as np
import pandas as pd
import seaborn as sns
from helpers import get_abspath
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def histogram(labels, dataname, outfile, outpath='plots/datasets'):
    """Generates a histogram of class labels in a given dataset and saves it
    to an output folder in the project directory.

    Args:
        labels (numpy.Array): array containing class labels.
        dataname (str): name of datasets (e.g. winequality).
        outfile (str): name of output file name.
        outpath (str): project folder to save plot file.
    """
    # get number of bins
    bins = len(np.unique(labels))

    # set figure params
    sns.set(font_scale=1.3, rc={'figure.figsize': (8, 8)})

    # create plot and set params
    fig, ax = plt.subplots()
    ax.hist(labels, bins=bins)
    fig.suptitle('Class frequency in ' + dataname)
    ax.set_xlabel('Class')
    ax.set_ylabel('Frequency')

    # save plot
    plt.savefig(get_abspath(outfile, outpath))
    plt.close()


def correlation_matrix(df, outfile, outpath='plots/datasets'):
    """ Generates a correlation matrix of all features in a given dataset

    Args:
        df (pandas.DataFrame): Source dataset.
    """
    # format data
    correlations = df.corr()
    names = list(df.columns.values)

    # set figure params
    sns.set(font_scale=0.8, rc={'figure.figsize': (11, 8)})

    # plot correlation heatmap
    sns.heatmap(correlations,
                annot=True,
                linewidth=0,
                xticklabels=names,
                yticklabels=names)
    plt.xticks(rotation=30)

    # save plot
    plt.savefig(get_abspath(outfile, outpath))
    plt.close()


if __name__ == '__main__':
    # load datasets
    p_wine = get_abspath('winequality.csv', 'data/experiments')
    p_seismic = get_abspath('seismic-bumps.csv', 'data/experiments')
    df_wine = pd.read_csv(p_wine)
    df_seismic = pd.read_csv(p_seismic)

    # generate correlation matrices
    correlation_matrix(df_wine, 'correlation_wine.png')
    correlation_matrix(df_seismic, 'correlation_seismic.png')

    # generate histograms
    histogram(df_wine['class'], 'winequality', 'hist_wine.png')
    histogram(df_seismic['class'], 'seismic-bumps', 'hist_seismic.png')
