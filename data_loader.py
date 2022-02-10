import numpy as np
import pandas as pd
import time
import sys
import math
import pickle
from sklearn.utils import shuffle

import prep

FOLDER_PATH = '/home/student/Data_2_1/'
H5_PATH_PREFIX = '/home/student/Data_2_1/data/'


def run_cluster_calculate_norm_stats():
    H5_FILE_LIST = '/home/student/Data_2_1/file_list.txt'

    now = time.time()

    h5_files = get_h5_file_list(H5_PATH_PREFIX, H5_FILE_LIST)
    print(len(h5_files))
    h5_files = np.random.choice(h5_files, 150)
    data = load_data_chunk(h5_files)

    print("Loading took: ", time.time()-now)
    print("Data shape: ", data.shape)
    sys.stdout.flush()

    now = time.time()
    norm_stats = calculate_normalization_stats(data)
    print("Collecting min max took: ", time.time()-now)
    sys.stdout.flush()

    now = time.time()
    with open("norm_stats.pickle", 'wb') as f:
        pickle.dump(norm_stats, f, pickle.HIGHEST_PROTOCOL)
    print("Pickleing took: ", time.time()-now)
    sys.stdout.flush()


def run_cluster_normalize(set_type=None):
    assert(set_type in ['train', 'test'])
    H5_FILE_LIST = FOLDER_PATH + set_type + '_files.txt'
    OUT_FOLDER = FOLDER_PATH + set_type + '/'

    now = time.time()
    with open("norm_stats.pickle", 'rb') as f:
        norm_stats = pickle.load(f)
    print("Unpickleing took: ", time.time()-now)
    sys.stdout.flush()

    now = time.time()
    h5_files = get_h5_file_list(H5_PATH_PREFIX, H5_FILE_LIST)
    data_chunk = load_data_chunk(h5_files)
    data = normalize_data(data_chunk, norm_stats)
    data = shuffle(data)

    DATA_CHUNK_SIZE = 4000
    num_chunks = int(data.shape[0] / DATA_CHUNK_SIZE)
    rest = data.shape[0] - num_chunks * DATA_CHUNK_SIZE
    current_index = 0
    for i in range(num_chunks):
        size = DATA_CHUNK_SIZE
        if i < rest:
            size += 1
        data_chunk = data[current_index:current_index+size]
        data_chunk.to_hdf(OUT_FOLDER + 'data_chunk_' + "_" + str(i) + '.h5',
                          key='data_chunk', mode='w', complevel=9, complib='zlib')
        current_index += size

    print("Saving took: ", time.time()-now)
    sys.stdout.flush()


def get_h5_file_list(path_prefix, h5_file_list_path):

    with open(h5_file_list_path) as f:
        h5_files = f.readlines()

    h5_files = [x.strip() for x in h5_files]
    h5_files = [x.replace("./", path_prefix) for x in h5_files]
    return h5_files


def load_data_from_file(filename):
    return load_data_chunk([filename])


def load_all_data(path_prefix, h5_file_list_path):
    h5_files = get_h5_file_list(path_prefix, h5_file_list_path)
    return load_data_chunk(h5_files)


def load_data_chunk(h5_files):
    data = None
    n_errors = 0

    for i, filename in enumerate(h5_files):
        if(i % 10) == 9:
            print("Loading file: ", i)
            sys.stdout.flush()
        try:
            if data is None:
                data = pd.read_hdf(filename)
                if data.isnull().values.any():
                    data = None
                    n_errors += 1
                    continue
            else:
                new_data = pd.read_hdf(filename)
                if new_data.isnull().values.any():
                    n_errors += 1
                    continue
                data = pd.concat((data, new_data), sort=False)
        except:
            print('Failed to concat file {}'.format(filename))
            n_errors += 1

    print('{} files could not be concatenated:'.format(n_errors))
    return data


def normalize_data(data, normalization_stats=None):

    if normalization_stats is None:
        normalization_stats = calculate_normalization_stats(data)

    for feature_index, (label, min_value, max_value) in enumerate(normalization_stats):
        for hero_i in range(10):
            true_label = "player_" + str(hero_i) + label
            true_label = true_label.replace("TEAM_SLOT_IDX","000" + str(hero_i % 5))
            true_label = true_label.replace("PLAYER_IDX","000" + str(hero_i))

            if (max_value - min_value) == 0: # does not change, drop it
                data = data.drop(true_label,axis=1)
                if hero_i == 0:
                    print(true_label," is useless!!! It is dropped")

            else:
                # kwargs is weird, if I want to pass the value of the string reather than the name, i  must use the dictionary syntax...
                # reather than this: data = data.assign(true_label=(data[true_label] - min_value) / (max_value - min_value))
                data = data.assign(**{true_label : (data[true_label] - min_value) / (max_value - min_value)})

    return data


def calculate_normalization_stats(data):
    # Calculate min and max based on a fraction of the data
    representative_sample_size = min(10000, data.shape[0])
    take_every_n_th = int(math.floor(float(data.shape[0]) / representative_sample_size))

    # Normalize time
    max_value = data["time"].max()
    min_value = data["time"].min()
    data = data.assign(time=(data["time"] - min_value) / (max_value - min_value))

    # Normalize hero features
    labels = [(i, label) for i, label in enumerate(list(data))]
    hero_labels = [label for i, label in prep.select_features_of_hero(0, labels)
                   if "m_nSelectedHeroID" not in label]
    hero_labels = [label.replace("player_0", "") for label in hero_labels]
    hero_labels = [label.replace("0000", "TEAM_SLOT_IDX")
                   if ("m_vecDataTeam" in label) else label for label in hero_labels]
    hero_labels = [label.replace("0000", "PLAYER_IDX")
                   if ("m_vecPlayerTeamData" in label) else label for label in hero_labels]

    normalization_stats = []

    for label_i, label in enumerate(hero_labels):

        max_value = np.finfo(np.float32).min
        min_value = np.finfo(np.float32).max

        for hero_i in range(10):
            true_label = "player_" + str(hero_i) + label
            true_label = true_label.replace("TEAM_SLOT_IDX", "000" + str(hero_i % 5))
            true_label = true_label.replace("PLAYER_IDX", "000" + str(hero_i))

            max_value = max(max_value, data[true_label][::take_every_n_th].max())
            min_value = min(min_value, data[true_label][::take_every_n_th].min())
        normalization_stats.append((label, max_value, min_value))
    return normalization_stats
