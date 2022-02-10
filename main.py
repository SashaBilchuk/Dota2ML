import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import sys
import pickle
from glob import glob
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor

import models
import datasets

FOLDER_PATH = '/home/student/Data_2_1/'     # path to project directory
PREMADE_TENSOR = True      # True if the dataset tensor file was already saved

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)  # Set fixed random number seed


# ----------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------
def convert_id_to_category(data):
    for hero_i in range(10):
        hero_id_feature_name = "player_" + str(hero_i) + "_m_vecPlayerTeamData.000" + str(hero_i) + ".m_nSelectedHeroID"
        data[hero_id_feature_name] = data[hero_id_feature_name].astype('category')
        data["player_" + str(hero_i)] = data["player_" + str(hero_i)].astype('category')
    return data


def test_acc(model, test_instances, targets):
    acc = 0
    pred = model.predict(test_instances)
    pred = np.around(pred)
    for test_y_i in targets:
        acc += accuracy_score(test_y_i, pred) * 0.1
    return acc


def run_baseline_algorithms():
    print('Reading train dataset')
    train_files = glob(FOLDER_PATH + "train/*.h5")
    train_dataset = datasets.load_data(train_files)
    train_dataset = convert_id_to_category(train_dataset)

    print('Reading test dataset')
    test_files = glob(FOLDER_PATH + "train/*.h5")
    test_dataset = datasets.load_data(test_files)
    test_dataset = convert_id_to_category(test_dataset)

    train_x = train_dataset.iloc[:, : 1342]
    train_y = train_dataset.iloc[:, -10:]
    train_y = train_y.iloc[:, :1]

    test_x = test_dataset.iloc[:, : 1342]
    test_y = test_dataset.iloc[:, -10:]
    tests = [test_y.iloc[:, i:i + 1] for i in range(10)]

    # Gradient Boosting Regressor
    model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=5))
    model.fit(train_x, train_y)
    accuracy = test_acc(model, test_x, tests)
    print('MultiOutputRegressor(GradientBoostingRegressor(n_estimators=5))')
    print('accuracy:', accuracy * 100)

    # Decision Tree Regressor
    model = DecisionTreeRegressor(max_depth=5)
    model.fit(train_x, train_y)
    accuracy = test_acc(model, test_x, tests)
    print('DecisionTreeRegressor(max_depth=5)')
    print('accuracy:', accuracy * 100)

    # Ada Boost Regressor
    model = MultiOutputRegressor(AdaBoostRegressor(n_estimators=100))
    model.fit(train_x, train_y)
    accuracy = test_acc(model, test_x, tests)
    print('MultiOutputRegressor(AdaBoostRegressor(n_estimators=100))')
    print('accuracy:', accuracy * 100)

    # Random Forest
    model = RandomForestRegressor(max_depth=4, random_state=0)
    model.fit(train_x, train_y)
    accuracy = test_acc(model, test_x, tests)
    print('RandomForestRegressor(max_depth=4, random_state=0)')
    print('accuracy:', accuracy * 100)


# ----------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------


def simulate(model_type, epoch, batch, hidden_size, learning_rate):
    n_epochs = epoch  # 5
    batch_size = batch  # 300
    hidden_size = hidden_size  # 128
    learning_rate = learning_rate  # 0.001

    n_features = 2632
    n_classes = 10

    model = model_type(input_size=n_features, hidden_size=hidden_size, num_classes=n_classes).to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = datasets.DotaDataset('train', PREMADE_TENSOR)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_dataset = datasets.DotaDataset('test', PREMADE_TENSOR)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    n_total_steps = len(train_loader)
    for epoch in range(n_epochs):
        for idx, (instances, labels) in enumerate(train_loader):
            # print(instances.shape)
            # print(instances)
            instances = instances.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(instances)
            # print(outputs)
            loss = criterion(outputs, labels)

            # backwards
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 100 == 0:
                print(f'epoch {epoch + 1} / {n_epochs}, step {idx + 1}/{n_total_steps}, loss = {loss.item():.4f}')

    # test
    with torch.no_grad():
        total_tp = 0
        total_tn = 0
        total_fp = 0
        total_fn = 0
        n_samples = 0
        for instances, labels in test_loader:
            instances = instances.to(device)
            labels = labels.to(device)
            # print(labels)

            outputs = model(instances)
            outputs = torch.sigmoid(outputs)
            outputs_np = outputs.cpu().detach().numpy()
            fp = ((outputs > 0.5) == (labels < 0.5)).cpu().numpy().astype(np.float32)
            tn = ((outputs < 0.5) == (labels < 0.5)).cpu().numpy().astype(np.float32)
            fn = ((outputs < 0.5) == (labels > 0.5)).cpu().numpy().astype(np.float32)
            tp = ((outputs > 0.5) == (labels > 0.5)).cpu().numpy().astype(np.float32)

            n_samples += labels.shape[0]

            tp_sum = tp.sum().item()
            total_tp += tp_sum
            fp_sum = fp.sum().item()
            total_fp += fp_sum
            tn_sum = tn.sum().item()
            total_tn += tn_sum
            fn_sum = fn.sum().item()
            total_fn += fn_sum

        print(total_tp, total_fp, total_tn, total_fn)
        accuracy = 100.0 * total_tn / (total_tp + total_fp)

        print(f'accuracy per player = {accuracy}')
        return accuracy


def test_hyperparameters(model_type, epochs_list, batch_list, hidden_list, lr_list):
    results_dicts = {}
    for i in range(4):
        epoch = epochs_list[i]
        batch = batch_list[0]
        hidden_size = hidden_list[3]
        learning_rate = lr_list[2]
        key = (epochs_list[i], batch_list[0], hidden_list[3], lr_list[2])
        results_dicts[key] = simulate(model_type, epoch, batch, hidden_size, learning_rate)
    for i in range(4):
        epoch = epochs_list[3]
        batch = batch_list[i]
        hidden_size = hidden_list[3]
        learning_rate = lr_list[2]
        key = (epochs_list[3], batch_list[i], hidden_list[3], lr_list[2])
        results_dicts[key] = simulate(model_type, epoch, batch, hidden_size, learning_rate)
    for i in range(4):
        epoch = epochs_list[3]
        batch = batch_list[0]
        hidden_size = hidden_list[i]
        learning_rate = lr_list[2]
        key = (epochs_list[3], batch_list[0], hidden_list[i], lr_list[2])
        results_dicts[key] = simulate(model_type, epoch, batch, hidden_size, learning_rate)
    for i in range(4):
        epoch = epochs_list[3]
        batch = batch_list[0]
        hidden_size = hidden_list[3]
        learning_rate = lr_list[i]
        key = (epochs_list[3], batch_list[0], hidden_list[3], lr_list[i])
        results_dicts[key] = simulate(model_type, epoch, batch, hidden_size, learning_rate)

    return results_dicts


# ----------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------


def run_base_models():
    simpleFF_results = test_hyperparameters(models.FFNN, [2, 4, 6, 10],
                                            [50, 100, 300, 500],
                                            [32, 64, 128, 256],
                                            [0.01, 0.001, 0.0001, 0.000001])
    print(simpleFF_results)
    pickle.dump(simpleFF_results, open("simpleFF_results.pkl", "wb"))

    MLP_results = test_hyperparameters(models.MLP, [2, 4, 6, 10],
                                       [50, 100, 300, 500],
                                       [32, 64, 128, 256],
                                       [0.01, 0.001, 0.0001, 0.000001])
    print(MLP_results)
    pickle.dump(MLP_results, open("MLP_result.pkl", "wb"))


# Parameters for advanced and creative models
INPUT_SIZE = 264
SEQUENCE_LENGTH = 10
NUM_CLASSES = 1024

HIDDEN_SIZE = 128
NUM_LAYERS = 2
LEARNING_RATE = 0.0001
BATCH_SIZE = 256
NUM_EPOCHS = 10


def run_advanced_model():
    train_dataset = datasets.Dota2Dataset('train', PREMADE_TENSOR)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    test_dataset = datasets.Dota2Dataset('test', PREMADE_TENSOR)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Initializing model")
    model = models.SharedWeightsFFNN(INPUT_SIZE, 10).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting epochs")
    n_total_steps = len(train_loader)
    for epoch in range(NUM_EPOCHS):
        for idx, (instances, labels) in enumerate(train_loader):
            # [batch size, 10, 264]
            instances = instances.reshape(SEQUENCE_LENGTH, instances.shape[0], INPUT_SIZE).to(device)
            labels = labels.to(device)

            outputs = model(instances)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 100 == 0:
                print(f'epoch {epoch + 1} / {NUM_EPOCHS}, step {idx + 1}/{n_total_steps}, loss = {loss.item():.4f}')

    model.eval()

    with torch.no_grad():
        total_tp = 0
        total_tn = 0
        total_fp = 0
        total_fn = 0
        n_samples = 0
        for instances, labels in test_loader:
            instances = instances.reshape(SEQUENCE_LENGTH, instances.shape[0], INPUT_SIZE).to(device)
            labels = labels.to(device)
            outputs = model(instances)
            # outputs = torch.sigmoid(outputs)
            fp = ((outputs > 0.5) == (labels < 0.5)).cpu().numpy().astype(np.float32)
            tn = ((outputs < 0.5) == (labels < 0.5)).cpu().numpy().astype(np.float32)
            fn = ((outputs < 0.5) == (labels > 0.5)).cpu().numpy().astype(np.float32)
            tp = ((outputs > 0.5) == (labels > 0.5)).cpu().numpy().astype(np.float32)

            n_samples += labels.shape[0]

            tp_sum = tp.sum().item()
            total_tp += tp_sum
            fp_sum = fp.sum().item()
            total_fp += fp_sum
            tn_sum = tn.sum().item()
            total_tn += tn_sum
            fn_sum = fn.sum().item()
            total_fn += fn_sum

        print(total_tp, total_fp, total_tn, total_fn)
        accuracy = 100.0 * total_tn / (total_tp + total_fp)

        print(f'accuracy per player = {accuracy}')
        return accuracy


def run_creative_model():
    train_dataset = datasets.Dota3Dataset('train', PREMADE_TENSOR)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    test_dataset = datasets.Dota3Dataset('test', PREMADE_TENSOR)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Initializing model")
    # model = RNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
    # model = RNN_GRU(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
    model = models.RNN_LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES, SEQUENCE_LENGTH).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting epochs")
    n_total_steps = len(train_loader)
    for epoch in range(NUM_EPOCHS):
        for idx, (instances, labels) in enumerate(train_loader):
            # [batch size, 10, 264]
            instances = instances.reshape(-1, SEQUENCE_LENGTH, INPUT_SIZE).to(device)
            labels = labels.to(device)

            outputs = model(instances)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 100 == 0:
                print(f'epoch {epoch + 1} / {NUM_EPOCHS}, step {idx + 1}/{n_total_steps}, loss = {loss.item():.4f}')

    model.eval()
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for instances, labels in test_loader:
            instances = instances.reshape(-1, SEQUENCE_LENGTH, INPUT_SIZE).to(device)
            labels = labels.to(device)
            outputs = model(instances)

            _, predicted = outputs.max(1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum()
        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network on the test instances: {acc} %')
    model.train()


if __name__ == '__main__':
    run_baseline_algorithms()
    run_base_models()
    run_advanced_model()
    run_creative_model()