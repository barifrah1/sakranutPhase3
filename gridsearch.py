from utils import args, loss_function
from data_loader import DataLoader
from torch import nn
from NN import Net, train, predict
import numpy as np


class GridSearch():
    def __init__(self):
        data_loader = DataLoader(args, True)
        # preprocessing
        X, Y, columnsInfo = data_loader.preprocess()
        self.columnsInfo = columnsInfo
        self.num_of_featurs = len(columnsInfo)
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = data_loader.split_train_validation_test(
            X, Y)
        self.nn = None

    def draw_params(self):
        num_layer = np.random.randint(2, 5, size=1)[0]
        batch_size = np.random.randint(256, 50000, size=1)
        lr = np.random.uniform(1e-6, 0.01)
        weight_decay = np.random.uniform(0, lr/1000)
        drop_rand = np.random.uniform(0, 1)
        print('number of layers in the current network', num_layer)
        print('lr', lr)
        print('batch_size', batch_size)
        print('weight_decay', weight_decay)
        print('dropout layer probability', drop_rand)
        return num_layer, lr, batch_size[0], weight_decay, drop_rand

    def exectue_grid_search(self, number_of_iterations):
        print('feature_num', self.num_of_featurs)
        for _ in range(number_of_iterations):
            number_of_layers, lr, batch_size, weight_decay, dropout_prob = self.draw_params()
            net = grid_search_net_build(
                self.num_of_featurs, number_of_layers, dropout_prob)
            self.nn = Net(self.num_of_featurs, net)
            train(self.X_train, self.y_train, self.nn, self.X_val, self.y_val, self.X_test, self.y_test,
                  batch_size, lr, weight_decay,
                  n_epochs=200,
                  criterion=loss_function, is_grid_search=True)


def grid_search_net_build(feature_num, number_of_layers, dropout_prob):
    num_layer = number_of_layers
    drop_rand = dropout_prob
    layer = []
    last = 0
    hidden_size = sorted(np.random.randint(
        2, 35, size=num_layer), reverse=True)
    for x in range(num_layer):
        #hidden_size = np.random.randint(10, 300, size=1)[0]
        print("hidden_size of layer{} is {}.".format(x+1, hidden_size[x]))
        if x == 0:  # index of current layer
            layer.append(nn.Linear(feature_num, hidden_size[x]))
            layer.append(nn.BatchNorm1d(hidden_size[x]))
            layer.append(nn.Dropout(0.5))
            layer.append(nn.ReLU())
            #print("layer{} is {}x{}".format(x+1, feature_num,hidden_size))
            last = hidden_size[x]
            continue
        if x == num_layer-1:
            layer.append(nn.Linear(last, 1))
            #print("layer{} is {}x{}".format(x+1, last,1))
            continue
        layer.append(nn.Linear(last, hidden_size[x]))
        #print("layer{} is {}x{}".format(x+1, last,hidden_size))
        layer.append(nn.ReLU())
        drop_rand=np.random.uniform(0, 1)
        if drop_rand < 0.2:
            drop_size = np.random.uniform(0.2, 0.9)
            layer.append(nn.Dropout(drop_size))
            print("dropout after layer{} with p={}".format(x+1, drop_size))
        last = hidden_size[x]
    layer.append(nn.Sigmoid())
    net = nn.Sequential(*layer)
    return net
