from sklearn.metrics import confusion_matrix
from helpers import get_abspath
import matplotlib.pyplot as plt
import numpy as np
import seaborn


def histogram(labels, dataname, outfile, outpath='plots'):
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

    # create plot and set params
    fig, ax = plt.subplots()
    ax.hist(labels, bins=bins)
    fig.suptitle('Class frequency in ' + dataname)
    ax.set_xlabel('Class')
    ax.set_ylabel('Frequency')

    # save plot
    plt.savefig(get_abspath(outfile, outpath))
    plt.close()


def correlation_matrix(df):
    """ Generates a correlation matrix of all features in a given dataset

    """
    # format data
    correlations = df.drop(['class'], axis=1).corr()
    names = list(df.drop(['class'], axis=1).columns.values)

    # add figure
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)

    # set correlation matrix params
    cax = ax.matshow(correlations, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, 13, 1)

    # set tick values
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    plt.xticks(rotation=30)

    plt.show()


def plot_confusion_matrix(y_test, y_pred):

    def font_color(value):
        if value < 100:
            return "black"
        else:
            return "white"

    cm = confusion_matrix(y_test, y_pred)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(cm, cmap=plt.cm.binary,
                    interpolation='nearest', vmin=0, vmax=300)

    # add color bar
    plt.colorbar(res)

    # annotate confusion entries
    width = len(cm)
    height = len(cm[0])

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(cm[x][y]), xy=(y, x), horizontalalignment='center',
                        verticalalignment='center', color=font_color(cm[x][y]))
