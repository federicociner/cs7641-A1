from __future__ import division
import os
import pickle


def save_dataset(df, filename, sep=',', subdir='data'):
    tdir = os.path.join(os.getcwd(), os.pardir, subdir, filename)
    df.to_csv(path_or_buf=tdir, sep=sep, header=True, index=False)


def get_datafile(filename, filepath):
    p = os.path.abspath(os.path.join(os.curdir, os.pardir))
    filepath = os.path.join(p, filepath, filename)

    return filepath


def save_model(model_object, filepath):
    with open(filepath, 'wb+') as model_file:
        pickler = pickle.Pickler(model_file)
        pickler.dump(model_object)


def load_model(filepath):
    with open(filepath, 'rb+') as model_file:
        model_object = pickle.load(model_file)

    return model_object
