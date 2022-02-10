import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)   # Set fixed random number seed


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        return self.layers(x)


class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FFNN, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


class SharedWeightsFFNN(torch.nn.Module):
    def __init__(self, num_features_per_hero, num_labels, shared_layer_sizes=None, final_layer_sizes=None):
        super(SharedWeightsFFNN, self).__init__()

        if shared_layer_sizes is None:
            shared_layer_sizes = [128, 64, 32]
        if final_layer_sizes is None:
            final_layer_sizes = [128]

        # Have to use ModuleList because using a plain list fails to populate model.parameters()
        self.shared_layers = nn.ModuleList([])
        self.final_layers = nn.ModuleList([])

        previous_layer_size = num_features_per_hero
        for layer_size in shared_layer_sizes:
            self.shared_layers.append(torch.nn.Linear(in_features=previous_layer_size, out_features=layer_size))
            previous_layer_size = layer_size

        previous_layer_size = 10 * previous_layer_size  # this is the size after the concatenation
        for layer_size in final_layer_sizes:
            self.final_layers.append(torch.nn.Linear(in_features=previous_layer_size, out_features=layer_size))
            previous_layer_size = layer_size

        self.final_layers.append(torch.nn.Linear(in_features=previous_layer_size, out_features=num_labels))

    def forward(self, hero_features):
        vals = []
        for hero_feature in hero_features:
            hero_x = hero_feature
            for shared_layer in self.shared_layers:
                hero_x = torch.relu(shared_layer(hero_x))
            vals.append(hero_x)
        x = torch.cat(vals, dim=1)

        for layer_i, final_layer in enumerate(self.final_layers):
            x = final_layer(x)
            if layer_i < (len(self.final_layers) - 1):  # no ReLU for the last layer
                x = torch.relu(x)
        return x


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


class RNN_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, sequence_length):
        super(RNN_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out


class RNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, sequence_length):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_size * sequence_length, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = out.reshape(out.shape[0], -1)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out

