import numpy as np
import pandas as pd
import math
import json
from zlib import crc32


def labels_to_indices(labels):
    return [i for i, label in labels]


def select_features_of_hero(hero_id, labels):
    hero_id_string = "player_" + str(hero_id) + "_"
    return [(i, label) for i, label in labels if hero_id_string in label]


def select_features_by_name(name, labels):
    return [(i, label) for i, label in labels if name in label]


def remove_paused_datapoints(data):
    time = np.diff(data["time"].values)
    is_paused = time < 0.0001  # time does not change, game is paused
    data = data.drop(data.index[np.where(is_paused)])
    return data


def add_historical_visibility_features(data):
    times = data.values[:, 0]
    num_datapoints = data.shape[0]

    # estimate timestep
    delta_times = []
    for rand_i in np.random.randint(1, len(times), 300):
        delta_times.append(times[rand_i] - times[rand_i - 1])
    timestep = np.array(delta_times).mean()
    ticks_per_sec = int(math.ceil(1 / timestep))

    for hero_i in range(10):
        feature_name = "player_" + str(hero_i) + "_m_iTaggedAsVisibleByTeam"
        visibilities = data[feature_name].values

        for history_i in range(10):
            # visibility history_i+1 sec ago is a shifted version of visibility, with padeded zeros in the beginning
            new_feature = np.zeros(num_datapoints, dtype=np.float32)
            tick_diff = (history_i + 1) * ticks_per_sec
            new_feature[tick_diff:] = visibilities[:-tick_diff]
            data["player_" + str(hero_i) + "_visibility_history_" + str(history_i)] = new_feature

    return data


def add_tower_position_features(data):
    labels = [(i, label) for i, label in enumerate(list(data))]
    tower_labels = select_features_by_name("Tower_", labels)
    unique_tower_labels = select_features_by_name("m_iHealth", tower_labels)  # this will return one label per tower
    unique_tower_labels = [label.replace("m_iHealth", "") for i, label in unique_tower_labels]

    modified_data = data

    for tower_name in unique_tower_labels:
        cell_x = modified_data[tower_name + "CBodyComponent.m_cellX"].values
        cell_y = modified_data[tower_name + "CBodyComponent.m_cellY"].values
        vec_x = modified_data[tower_name + "CBodyComponent.m_vecX"].values
        vec_y = modified_data[tower_name + "CBodyComponent.m_vecY"].values

        pos_x = cell_x * 256 + vec_x
        pos_y = cell_y * 256 + vec_y

        modified_data[tower_name + "pos_x"] = pos_x.astype(np.float32)
        modified_data[tower_name + "pos_y"] = pos_y.astype(np.float32)

        modified_data = modified_data.drop(tower_name + "CBodyComponent.m_cellX", axis=1)
        modified_data = modified_data.drop(tower_name + "CBodyComponent.m_cellY", axis=1)
        modified_data = modified_data.drop(tower_name + "CBodyComponent.m_vecX", axis=1)
        modified_data = modified_data.drop(tower_name + "CBodyComponent.m_vecY", axis=1)

    return modified_data


def add_position_features(data):
    modified_data = data

    for hero_i in range(10):
        player_prefix = "player_" + str(hero_i) + "_"

        cell_x = modified_data[player_prefix + "CBodyComponent.m_cellX"].values
        cell_y = modified_data[player_prefix + "CBodyComponent.m_cellY"].values
        vec_x = modified_data[player_prefix + "CBodyComponent.m_vecX"].values
        vec_y = modified_data[player_prefix + "CBodyComponent.m_vecY"].values

        pos_x = cell_x * 256 + vec_x  # vec_x overflows at 256
        pos_y = cell_y * 256 + vec_y

        modified_data[player_prefix + "pos_x"] = pos_x.astype(np.float32)
        modified_data[player_prefix + "pos_y"] = pos_y.astype(np.float32)

        modified_data = modified_data.drop(player_prefix + "CBodyComponent.m_cellX", axis=1)
        modified_data = modified_data.drop(player_prefix + "CBodyComponent.m_cellY", axis=1)
        modified_data = modified_data.drop(player_prefix + "CBodyComponent.m_vecX", axis=1)
        modified_data = modified_data.drop(player_prefix + "CBodyComponent.m_vecY", axis=1)

    return modified_data


def add_hero_proximities(data):
    labels = [(i, label) for i, label in enumerate(list(data))]
    labels = select_features_by_name("pos", labels)
    labels = select_features_by_name("player_", labels)
    pos_x_indicies = labels_to_indices(select_features_by_name("pos_x", labels))
    pos_y_indicies = labels_to_indices(select_features_by_name("pos_y", labels))

    pos_x_vals = data.values[:, pos_x_indicies]
    pos_y_vals = data.values[:, pos_y_indicies]

    for hero_i in range(10):
        hero_team = int(hero_i / 5)
        current_ally_i = 0
        current_enemy_i = 0
        player_prefix = "player_" + str(hero_i) + "_"
        for other_hero_i in range(10):
            if other_hero_i == hero_i:
                continue

            feature_name = None
            other_hero_team = int(other_hero_i / 5)
            if hero_team == other_hero_team:
                feature_name = player_prefix + "ally_proximity_" + str(current_ally_i)
                current_ally_i += 1
            else:
                feature_name = player_prefix + "enemy_proximity_" + str(current_enemy_i)
                current_enemy_i += 1

            distances = np.sqrt((pos_x_vals[:, hero_i] - pos_x_vals[:, other_hero_i]) * (
                        pos_x_vals[:, hero_i] - pos_x_vals[:, other_hero_i]) +
                                (pos_y_vals[:, hero_i] - pos_y_vals[:, other_hero_i]) * (
                                            pos_y_vals[:, hero_i] - pos_y_vals[:, other_hero_i]))
            distances = np.minimum(distances,
                                   10000)  # clamp the distances, it does not realy matter if it is so far away
            data[feature_name] = distances

    return data


def add_closest_alive_towers(data):
    labels = [(i, label) for i, label in enumerate(list(data))]
    team_2_tower_lables = select_features_by_name("Tower_2", labels)
    team_3_tower_lables = select_features_by_name("Tower_3", labels)

    team_2_tower_pos_x_labels = select_features_by_name("pos_x", team_2_tower_lables)
    team_2_tower_pos_x_indices = labels_to_indices(team_2_tower_pos_x_labels)
    team_2_tower_pos_y_labels = select_features_by_name("pos_y", team_2_tower_lables)
    team_2_tower_pos_y_indices = labels_to_indices(team_2_tower_pos_y_labels)
    team_2_tower_life_state_labels = select_features_by_name("m_iHealth", team_2_tower_lables)
    team_2_tower_life_state_indicies = labels_to_indices(team_2_tower_life_state_labels)

    team_3_tower_pos_x_labels = select_features_by_name("pos_x", team_3_tower_lables)
    team_3_tower_pos_x_indices = labels_to_indices(team_3_tower_pos_x_labels)
    team_3_tower_pos_y_labels = select_features_by_name("pos_y", team_3_tower_lables)
    team_3_tower_pos_y_indices = labels_to_indices(team_3_tower_pos_y_labels)
    team_3_tower_life_state_labels = select_features_by_name("m_iHealth", team_3_tower_lables)
    team_3_tower_life_state_indices = labels_to_indices(team_3_tower_life_state_labels)

    closest_ally_tower = np.zeros((data.shape[0], 10), dtype=np.float32)
    closest_enemy_tower = np.zeros((data.shape[0], 10), dtype=np.float32)

    for team_iterator in range(2):
        team_index = team_iterator + 2  # first team is team 2, second team is team 3

        ally_tower_pos_x_indices = team_2_tower_pos_x_indices if team_index == 2 else team_3_tower_pos_x_indices
        ally_tower_pos_y_indices = team_2_tower_pos_y_indices if team_index == 2 else team_3_tower_pos_y_indices
        enemy_tower_pos_x_indices = team_3_tower_pos_x_indices if team_index == 2 else team_2_tower_pos_x_indices
        enemy_tower_pos_y_indices = team_3_tower_pos_y_indices if team_index == 2 else team_2_tower_pos_y_indices

        ally_tower_life_state_indices = team_2_tower_life_state_indicies \
            if team_index == 2 else team_3_tower_life_state_indices

        ally_tower_pos_x = data.values[:, ally_tower_pos_x_indices]
        ally_tower_pos_y = data.values[:, ally_tower_pos_y_indices]

        enemy_tower_pos_x = data.values[:, enemy_tower_pos_x_indices]
        enemy_tower_pos_y = data.values[:, enemy_tower_pos_y_indices]

        ally_dead_mask = np.zeros((data.shape[0], 11), dtype=np.uint32)
        ally_dead_mask[:] = data.values[:, ally_tower_life_state_indices] > 0.5

        enemy_dead_mask = np.zeros((data.shape[0], 11), dtype=np.uint32)
        enemy_dead_mask[:] = data.values[:, ally_tower_life_state_indices] > 0.5

        for hero_iterator in range(5):
            hero_index = hero_iterator + 5 * team_iterator

            player_prefix = "player_" + str(hero_index) + "_"
            hero_pos_x = data[player_prefix + "pos_x"].values
            hero_pos_y = data[player_prefix + "pos_y"].values

            ally_tower_distances = np.sqrt(
                (ally_tower_pos_x - hero_pos_x[:, np.newaxis]) * (ally_tower_pos_x - hero_pos_x[:, np.newaxis]) +
                (ally_tower_pos_y - hero_pos_y[:, np.newaxis]) * (ally_tower_pos_y - hero_pos_y[:, np.newaxis]))
            enemy_tower_distances = np.sqrt(
                (enemy_tower_pos_x - hero_pos_x[:, np.newaxis]) * (enemy_tower_pos_x - hero_pos_x[:, np.newaxis]) +
                (enemy_tower_pos_y - hero_pos_y[:, np.newaxis]) * (enemy_tower_pos_y - hero_pos_y[:, np.newaxis]))

            # Make sure dead towers don't effect the the minimum
            ally_tower_distances = ally_tower_distances + ally_dead_mask * 10000000
            enemy_tower_distances = enemy_tower_distances + enemy_dead_mask * 10000000

            # 6000 is around quarter the map length
            closest_ally_tower[:, hero_index] = np.minimum(ally_tower_distances.min(axis=1), 6000)
            closest_enemy_tower[:, hero_index] = np.minimum(enemy_tower_distances.min(axis=1), 6000)

    modified_data = data

    for hero_i in range(10):
        feature_name_prefix = "player_" + str(hero_i) + "_closest_tower_"
        modified_data[feature_name_prefix + "distance_ally"] = closest_ally_tower[:, hero_i]
        modified_data[feature_name_prefix + "distance_enemy"] = closest_enemy_tower[:, hero_i]

    # Delete all tower data
    all_tower_lables = select_features_by_name("Tower_", labels)
    for i, label in all_tower_lables:
        modified_data = modified_data.drop(label, axis=1)

    return modified_data


def add_rate_of_change_features(data):
    labels = [(i, label) for i, label in enumerate(list(data))]

    labels_to_make_diff = []
    diff_feature_name = []

    filtered_labels = select_features_by_name("pos_", labels)
    labels_to_make_diff.extend([label for i, label in filtered_labels])
    diff_feature_name.extend([label.replace("pos_", "speed_") for i, label in filtered_labels])

    filtered_labels = select_features_by_name("_proximity_", labels)
    labels_to_make_diff.extend([label for i, label in filtered_labels])
    diff_feature_name.extend([label.replace("proximity", "delta_proximity") for i, label in filtered_labels])

    filtered_labels = select_features_by_name("closest_tower_distance", labels)
    labels_to_make_diff.extend([label for i, label in filtered_labels])
    diff_feature_name.extend(
        [label.replace("closest_tower_distance", "delta_closest_tower_distance") for i, label in filtered_labels])

    filtered_labels = select_features_by_name("m_iHealth", labels)
    labels_to_make_diff.extend([label for i, label in filtered_labels])
    diff_feature_name.extend([label.replace("m_iHealth", "delta_health") for i, label in filtered_labels])

    for label, new_label in zip(labels_to_make_diff, diff_feature_name):
        data[new_label] = np.insert(np.diff(data[label].values), 0, 0)  # insert a zero in front of the diff

    return data


def bytes_to_float(b):
    return float(crc32(b) & 0xffffffff) / 2 ** 32


def str_to_float(s, encoding="utf-8"):
    return bytes_to_float(s.encode(encoding))


def add_game_name_hash(data, game_name):
    hash_val = str_to_float(game_name)
    data["stat_game_name_hash"] = np.repeat(hash_val, data.shape[0]).astype(np.float32)

    return data


def time_minus90(data):
    # Each game starts at -90 and every tick is in jumps of 10 seconds
    t = -90
    for i in range(len(data['time'])):
        data.iloc[i, 0] = t
        t += 10
    return data


def preprocess_data(game_name):
    data_file_name = game_name + ".csv"
    data = pd.read_csv(data_file_name, dtype=np.float32)
    if data.isnull().values.any() is True:
        return None

    data = remove_paused_datapoints(data)
    data = add_position_features(data)
    data = add_tower_position_features(data)
    data = add_closest_alive_towers(data)
    data = add_hero_proximities(data)
    data = add_rate_of_change_features(data)
    data = add_historical_visibility_features(data)
    data = add_game_name_hash(data, game_name)
    data = time_minus90(data)

    print("Data shape: ", data.shape)
    return data
