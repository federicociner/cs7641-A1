from __future__ import division
import os
import pickle


def save_dataset(df, filename, sep=',', subdir='data', header=True):
    """Saves Pandas data frame as a CSV file.

    Args:
        df (Pandas.DataFrame): Data frame.
        filename (str): Output file name.
        sep (str): Delimiter.
        subdir (str): Project directory to save output file.
        header (Boolean): Specify inclusion of header.

    """
    tdir = os.path.join(os.getcwd(), os.pardir, subdir, filename)
    df.to_csv(path_or_buf=tdir, sep=sep, header=header, index=False)


def get_abspath(filename, filepath):
    """Gets absolute path of specified file within the project directory. The
    filepath has to be a subdirectory within the main project directory.

    Args:
        filename (str): Name of specified file.
        filepath (str): Subdirectory of file.
    Returns:
        fullpath (str): Absolute filepath.

    """
    p = os.path.abspath(os.path.join(os.curdir, os.pardir))
    fullpath = os.path.join(p, filepath, filename)

    return fullpath


def save_pickled_model(model, filepath):
    """Saves a model as a pickle file.

    Args:
        model (object): sklearn model object.
        filepath (str): Output folder.

    """
    with open(filepath, 'wb+') as model_file:
        pickler = pickle.Pickler(model_file)
        pickler.dump(model)


def load_pickled_model(filepath):
    """Loads a pickled model

    Args:
        filepath (str): Target filepath for pickled model.
    Returns:
        model (object): sklearn model.

    """
    with open(filepath, 'rb+') as model_file:
        model = pickle.load(model_file)

    return model
