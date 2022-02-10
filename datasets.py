import sys
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from glob import glob

FOLDER_PATH = '/home/student/Data_2_1/'     # path to project directory
NUM_HEROES = 130
DEBUGGING = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)  # Set fixed random number seed


def load_data(h5_files):
    df = None
    for i, filename in enumerate(h5_files):
        if (i % 10) == 9:
            print("Loading file: ", i)
            sys.stdout.flush()
        if df is None:
            df = pd.read_hdf(filename)
        else:
            new_df = pd.read_hdf(filename)
            df = pd.concat((df, new_df), sort=False)
    return df


def add_hero_one_hot(data_df):
    modified_data = data_df

    for hero_i in range(10):
        hero_id_feature_name = "player_" + str(hero_i) + "_m_vecPlayerTeamData.000" + str(hero_i) + ".m_nSelectedHeroID"
        hero_id = data_df[hero_id_feature_name].values[0]
        hero_id_int = int(np.rint(hero_id))

        hero_one_hot = np.zeros(NUM_HEROES)
        hero_one_hot[hero_id_int] = 1

        for i in range(NUM_HEROES):
            feature_name = "player_" + str(hero_i) + "_hero_one_hot_" + str(i)
            modified_data[feature_name] = np.repeat(hero_one_hot[i], data_df.shape[0])
        modified_data = modified_data.drop(columns=hero_id_feature_name)

    return modified_data


class DotaDataset(Dataset):
    def __init__(self, set_type, is_premade=False):
        print('Reading ' + set_type + ' dataset')
        x_fp = FOLDER_PATH + set_type + '_x_tensors_1.pt'
        y_fp = FOLDER_PATH + set_type + '_y_tensors_1.pt'

        if is_premade and not DEBUGGING:
            self.x_data = torch.load(x_fp)
            self.y_data = torch.load(y_fp)
            self.n_samples = self.x_data.shape[0]
            return

        assert (set_type in ['train', 'test', 'validation'])
        dir_path = FOLDER_PATH + set_type + '/'

        h5_files = glob(dir_path + '*.h5')
        if DEBUGGING:
            h5_files = h5_files[0:10]    # debug only take the first 10
        df = load_data(h5_files)
        df = add_hero_one_hot(df)

        xy = df.to_numpy(dtype=np.float32)
        self.n_samples = xy.shape[0]
        self.x_data = torch.from_numpy(np.concatenate((xy[:, :-1311], xy[:, -1300:]), axis=1))
        self.y_data = torch.from_numpy(xy[:, -1310:-1300])  # size [n_samples, 10]

        if not DEBUGGING:
            torch.save(self.x_data, x_fp)
            torch.save(self.y_data, y_fp)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


def labels_to_indices(labels):
    return [i for i, label in labels]


def select_features_of_hero(hero_id, labels):
    hero_id_string = "player_" + str(hero_id) + "_"
    return [(i, label) for i, label in labels if hero_id_string in label]


def get_feature_indices(data, exclude_if_contains_list=None, only_include_list=None):
    example_row = data.sample(n=1, replace=False)

    labels = [(i, label) for i, label in enumerate(list(example_row))]

    if only_include_list is not None:
        filtered_labels = []
        for i, label in labels:
            for include_label in only_include_list:
                if include_label in label:
                    filtered_labels.append((i, label))
        labels = filtered_labels

    if exclude_if_contains_list is not None:
        for exclude_pattern in exclude_if_contains_list:
            labels = [(i, label) for i, label in labels if exclude_pattern not in label]

    hero_feature_indicies = []
    for i in range(10):
        hero_labels = select_features_of_hero(i, labels)
        hero_feature_indicies.append(labels_to_indices(hero_labels))
        hero_feature_indicies[-1].append(0)  # dont forget the time

    return hero_feature_indicies


class Dota2Dataset(Dataset):
    def __init__(self, set_type, is_premade=False):
        print('Reading ' + set_type + ' dataset')
        x_fp = FOLDER_PATH + set_type + '_x_tensors_2.pt'
        y_fp = FOLDER_PATH + set_type + '_y_tensors_2.pt'

        if is_premade:
            print('loading from' + x_fp)
            self.x_data = torch.load(x_fp)
            print('loading from' + y_fp)
            self.y_data = torch.load(y_fp)
            self.n_samples = self.x_data.shape[0]
            return

        assert(set_type in ['train', 'test', 'validation'])
        # file_list_fp = FOLDER_PATH+set_type+'_files.txt'
        dir_path = FOLDER_PATH+set_type+'/'

        h5_files = glob(dir_path+'*.h5')
        if DEBUGGING:
            h5_files = h5_files[0:3]    # debug only take the first 3
        df = load_data(h5_files)
        df = add_hero_one_hot(df)
        feature_indices = get_feature_indices(df)
        xy = df.to_numpy(dtype=np.float32)
        reshaped_data = []

        for xy_i in xy:
            instance_data = []
            for indices in feature_indices:
                instance_data.append(xy_i[indices])
            reshaped_data.append(instance_data)

        self.n_samples = xy.shape[0]
        self.x_data = torch.from_numpy(np.array(reshaped_data, dtype=np.float32))
        print(self.x_data.shape)
        self.y_data = torch.from_numpy(np.array(xy[:, -1310:-1300], dtype=np.float32))  # size [n_samples, 10]
        print(self.y_data.shape)

        if not DEBUGGING:
            print('Saving to' + x_fp)
            torch.save(self.x_data, x_fp)
            print('Saving to' + x_fp)
            torch.save(self.y_data, y_fp)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


class Dota3Dataset(Dataset):
    def __init__(self, set_type, is_premade=False):
        print('Reading ' + set_type + ' dataset')
        x_fp = FOLDER_PATH + set_type + '_x_tensors_3.pt'
        y_fp = FOLDER_PATH + set_type + '_y_tensors_3.pt'

        if is_premade:
            print('loading from' + x_fp)
            self.x_data = torch.load(x_fp)
            print('loading from' + y_fp)
            self.y_data = torch.load(y_fp)
            self.n_samples = self.x_data.shape[0]
            return

        assert(set_type in ['train', 'test', 'validation'])
        # file_list_fp = FOLDER_PATH+set_type+'_files.txt'
        dir_path = FOLDER_PATH+set_type+'/'

        h5_files = glob(dir_path+'*.h5')
        if DEBUGGING:
            h5_files = h5_files[0:3]    # debug only take the first 10
        df = load_data(h5_files)
        df = add_hero_one_hot(df)
        feature_indices = get_feature_indices(df)
        xy = df.to_numpy(dtype=np.float32)
        reshaped_data = []
        label_list = []

        for xy_i in xy:
            instance_data = []
            # print(x.shape)
            for indices in feature_indices:
                instance_data.append(xy_i[indices])
            reshaped_data.append(instance_data)
            y = xy_i[-1310:-1300]
            label_num = int("".join(str(int(i)) for i in y), 2)
            # label = np.zeros(1024, dtype=np.float32)
            # label[label_num] = 1
            label_list.append(label_num)

        self.n_samples = xy.shape[0]
        self.x_data = torch.from_numpy(np.array(reshaped_data, dtype=np.float32))
        self.y_data = torch.from_numpy(np.array(label_list, dtype=np.int_))  # size [n_samples, 10]

        if not DEBUGGING:
            print('Saving to' + x_fp)
            torch.save(self.x_data, x_fp)
            print('Saving to' + x_fp)
            torch.save(self.y_data, y_fp)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


