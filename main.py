from data_loader import DataLoader
from utils import args, loss_function
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
import NN
from NN import Net
import torch
import torch.nn as nn
from gridsearch import GridSearch
import os 

GRID_SEARCH_MODE = False
if __name__ == '__main__':
    if(GRID_SEARCH_MODE):
        gridsearch = GridSearch()
        gridsearch.exectue_grid_search(10000)
    else:
        data_loader = DataLoader(args, False)  # False for is_grid_search mode
        # preprocessing
        X, Y, columnsInfo = data_loader.preprocess()
        # split data to train and test
        X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_train_validation_test(X, Y)
        feature_num = len(columnsInfo)
        print('feature_num',feature_num)
        model = NN.Net(feature_num)
        NN.train(X_train, y_train, model, X_val, y_val, X_test, y_test,
                 args["batch_size"], args["lr"], args["weight_decay"],
                 n_epochs=args["n_epochs"],
                 criterion=loss_function)
        path = os.getcwd()
        path=path+ "/net.pt"
        torch.save(model.state_dict(),path)
        # optimizer = nn.BCELoss)
