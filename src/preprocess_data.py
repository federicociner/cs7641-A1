from scipy.io import arff
from helpers import get_abspath, save_dataset
import pandas as pd


def preprocess_winequality():
    """Cleans and generates wine quality dataset for experiments as a
    CSV file.

    """
    # get file paths
    sdir = 'data/winequality'
    tdir = 'data/experiments'
    wr_file = get_abspath('winequality-red.csv', sdir)
    ww_file = get_abspath('winequality-white.csv', sdir)

    # load as data frame
    wine_red = pd.read_csv(wr_file, sep=';')
    wine_white = pd.read_csv(ww_file, sep=';')

    # encode artifical label to determine if wine is red or not
    wine_red['red'] = 1
    wine_white['red'] = 0

    # sample equal amount of white wine instances as red wine instances to
    # balance dataset
    wine_white_sample = wine_white.sample(wine_red.shape[0], random_state=0)

    # combine datasets and format column names
    df = wine_red.append(wine_white_sample)
    df.columns = ['_'.join(col.split(' ')) for col in df.columns]
    df.rename(columns={'quality': 'class'}, inplace=True)

    # save to CSV
    save_dataset(df, 'winequality.csv', sep=',', subdir=tdir)


def preprocess_seismic():
    """Cleans and generates seismic bumps dataset for experiments as a
    CSV file. Uses one-hot encoding for categorical features.

    """
    # get file path
    sdir = 'data/seismic-bumps'
    tdir = 'data/experiments'
    seismic_file = get_abspath('seismic-bumps.arff', sdir)

    # read arff file and convert to record array
    rawdata = arff.loadarff(seismic_file)
    df = pd.DataFrame(rawdata[0])

    # apply one-hot encoding to categorical features using Pandas get_dummies
    cat_cols = ['seismic', 'seismoacoustic', 'shift', 'ghazard']
    cats = df[cat_cols]
    onehot_cols = pd.get_dummies(cats, prefix=cat_cols)

    # drop original categorical columns and append one-hot encoded columns
    df.drop(columns=cat_cols, inplace=True)
    df = pd.concat((df, onehot_cols), axis=1)

    # save to CSV
    save_dataset(df, 'seismic-bumps.csv', sep=',', subdir=tdir)


if __name__ == '__main__':
    # run preprocessing functions
    preprocess_winequality()
    preprocess_seismic()
