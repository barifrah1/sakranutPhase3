import torch
import NN
from NN import Net
import os


def load_net(feature_num):
    model = NN.Net(feature_num)
    path = os.getcwd()
    path = path + "/net.pt"
    model.load_state_dict(torch.load(path))
    return model


def get_pred(net, x):
    net.eval()
    with torch.no_grad():
        pred = net(x)
    print(pred.shape)
    return pred
