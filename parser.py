import requests
import urllib
import os
import bz2
import subprocess
import pandas as pd
from time import sleep

import prep
import data_loader


TEMP_DP = '/home/student/Data_2_1/tempo/'
DATA_DP = '/home/student/Data_2_1/data/'


def get_all_match_ids():
    req = requests.get("https://api.opendota.com/api/proMatches?less_than_match_id=6121193105")
    matches_array = req.json()
    m_list = []
    for i in range(0, 30):
        for m in matches_array:
            m_list.append(m['match_id'])
        last_match_id = (matches_array[-1])['match_id']
        req = requests.get("https://api.opendota.com/api/proMatches?less_than_match_id={}".format(last_match_id))
        matches_array = req.json()
    return m_list


def parse_replay(url):
    # Define file and directory paths.
    bz2_fn = url.split('/')[-1]
    bz2_fp = os.path.join(TEMP_DP, bz2_fn)
    temp_fp = os.path.join(TEMP_DP, bz2_fn[:-4])

    # Avoid duplicates.
    data_fn = "{}.csv".format(bz2_fn[:-8])
    data_fp = os.path.join(DATA_DP, data_fn)
    if os.path.exists(data_fp):
        print("File {} already exists. Skipping.".format(data_fn))
        return

    # Download replay and unzip from .bz2 file
    r = urllib.request.urlretrieve(url, bz2_fp)
    with open(temp_fp, 'wb') as new_file, bz2.BZ2File(bz2_fp, 'rb') as file:
        for dat in iter(lambda: file.read(100 * 1024), b''):
            new_file.write(dat)

    # Call java parser
    subprocess.call(['java', '-jar', 'processor.jar', temp_fp, DATA_DP])

    # Delete .bz2 and .dem files
    os.remove(bz2_fp)
    os.remove(temp_fp)


def parse_teamfights(match, url):
    tf_list = match['teamfights']  # get list of teamfights
    filepath = 'tf_data/{}_tf.csv'.format((url.split('/')[-1])[:-8])

    # Initialize dictionary
    tf_dict = {'time': [], 'is_tf': [],
               'player_0': [], 'player_1': [], 'player_2': [], 'player_3': [], 'player_4': [],
               'player_5': [], 'player_6': [], 'player_7': [], 'player_8': [], 'player_9': []}

    # Fill tf_dict with relevant data from tf_list.
    t0 = -100
    for tf in tf_list:
        tf_start = tf['start'] - (tf['start'] % 10)
        tf_end = tf['end']
        t0 += 10
        for t in range(t0, tf_start - 60, 10):
            tf_dict['time'].append(t)
            tf_dict['is_tf'].append(False)
            for i in range(0, 10):
                tf_dict['player_{}'.format(i)].append(0)

        for t in range(tf_start - 60, tf_end, 10):
            tf_dict['time'].append(t)
            tf_dict['is_tf'].append(True)
            p_list = tf['players']
            for i in range(0, 10):
                player = p_list[i]
                was_in_teamfight = (player['damage'] > 100 or player['deaths'] > 0)
                if was_in_teamfight:
                    tf_dict['player_{}'.format(i)].append(1)
                else:
                    tf_dict['player_{}'.format(i)].append(0)
            t0 = t

    duration = match['duration']
    for t in range(t0, duration, 10):
        tf_dict['time'].append(t)
        tf_dict['is_tf'].append(False)
        for i in range(0, 10):
            tf_dict['player_{}'.format(i)].append(0)

    df = pd.DataFrame.from_dict(tf_dict)
    df.to_csv(filepath, index=False)


def parse_all_matches(id_list):
    s_count, f_count = 0, 0
    # Iterate over list of match ids
    for match_id in id_list[0:1000]:  # Change to appropriate range
        try:
            # Get match data including replay url
            _url = "https://api.opendota.com/api/matches/{}".format(match_id)
            new_match = requests.get(_url).json()
            dem_url = new_match['replay_url']

            parse_teamfights(new_match, dem_url)

            parse_replay(dem_url)
            s_count += 1
        except:
            print("Failed to get url on match number {}".format(s_count + f_count))
            f_count += 1

            pass
        sleep(0.8)  # sleep to avoid hitting OpenDota's call limit
        print('So far: {} files parsed with {} failures.'.format(s_count, f_count))
    print(f'Failed to parse {f_count} files')


m_id_list = get_all_match_ids()
file_list = os.listdir(DATA_DP)
parse_all_matches(m_id_list)

count, fail_count = 0, 0
for fn in file_list:
    try:
        filename = fn.split('.')[0]
        match_name = '/home/student/Data_2_1/data/'+filename
        if os.path.exists(match_name+'.h5'):
            print("File {} already exists. Skipping.".format(match_name+'_tf.csv'))
            continue
        print(match_name)
        data = prep.preprocess_data(match_name)
        data.to_hdf(match_name + '.h5', key='data', mode='w', complevel=9, complib='zlib')
        count += 1
    except:
        print("Failed with file {}".format(fn))
        fail_count += 1
        pass
print(f'Failed to process {fail_count} files')

h5_file_list = []
count, fail_count = 0, 0
for fn in file_list:
    if fn[-3:] == '.h5':
        try:
            new_fp = DATA_DP + '/' + fn[:-2] + 'mrg.h5'
            x_data = pd.read_hdf(DATA_DP + '/' + fn)
            y_data = pd.read_csv('/home/student/Data_2_1/tf_data/'+fn[:-3]+'_tf.csv')
            data = x_data.merge(y_data, on='time')
            data.to_hdf(new_fp, key='data', mode='w', complevel=9, complib='zlib')
            print("{}: Merged files of match file {}".format(count, fn))
            h5_file_list.append(new_fp.split('/')[-1])
            count += 1
        except:
            print("Failed with file {}".format(fn))
            fail_count += 1

with open('file_list.txt', 'w') as f:
    for item in h5_file_list:
        f.write('{}\n'.format(item))

del_list = os.listdir(DATA_DP)
for f in del_list:
    if f[-2:] == 'h5' and f[-4] != 'g':
        os.remove(DATA_DP + '/' + f)

data_loader.run_cluster_calculate_norm_stats()
data_loader.run_cluster_normalize('train')
data_loader.run_cluster_normalize('test')
data_loader.run_cluster_normalize('validation')
