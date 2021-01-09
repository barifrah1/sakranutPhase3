from data_loader import *
from utils import args, loss_function
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
import NN
from NN import Net
import torch
import torch.nn as nn
from gridsearch import GridSearch
import os
import os.path
from prediction import *
import pickle
import pandas as pd
from project import Project

GRID_SEARCH_MODE = False
if __name__ == '__main__':
    if(GRID_SEARCH_MODE):
        gridsearch = GridSearch()
        gridsearch.exectue_grid_search(10000)
    else:
        if (os.path.isfile('scale_dict.pickle')) == False:
            # False for is_grid_search mode
            data_loader = DataLoader(args, False)
        # preprocessing
            X, Y, columnsInfo = data_loader.preprocess()
        # split data to train and test
            X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_train_validation_test(
                X, Y)
            feature_num = len(columnsInfo)
        else:
            with open('scale_dict.pickle', 'rb') as handle:
                scale_dict = pickle.load(handle)
            with open('option_dict.pickle', 'rb') as handle:
                option_dict = pickle.load(handle)
            with open('category_per_main_cat_dict.pickle', 'rb') as handle:
                category_per_main_cat_dict = pickle.load(handle)
            with open('currency_per_country_dict.pickle', 'rb') as handle:
                currency_per_country = pickle.load(handle)
            empty_data_format = pd.read_pickle("empty_data_format.pkl")
            feature_num = len(empty_data_format.columns)
        print('feature_num', feature_num)
        if (os.path.isfile('net.pt')) == False:
            model = NN.Net(feature_num)
            NN.train(X_train, y_train, model, X_val, y_val, X_test, y_test,
                     args["batch_size"], args["lr"], args["weight_decay"],
                     n_epochs=args["n_epochs"],
                     criterion=loss_function)
            path = os.getcwd()
            path = path + "/net.pt"
            torch.save(model.state_dict(), path)
        # optimizer = nn.BCELoss)
        # [category,main_category,currency,country,goal_level,duration,year_launched,month_launched]
        pro = ['Music', 'main_category_Music', 'USD', 'US', 1891, 20, 2015, 7]
        proj = Project(option_dict, category_per_main_cat_dict,
                       currency_per_country)
        project = random_project_preproesses(
            empty_data_format, proj.getProjectValues(), scale_dict)
        project = torch.from_numpy(project).float()
        print('project.shape', project.shape)
        # learner net starting with random weights
        learner_net = NN.Net(feature_num)
        # ground truth net traind over original dataset
        gt_net = load_net(feature_num)
        pred = get_pred(gt_net, project)
        pred_stupid = get_pred(learner_net, project)
        print(proj.getProjectValues())
        print('prob to be successful project', pred.item())
        print('prob to be successful project_stupid', pred_stupid.item())
        print('epsilon^2 is ', (pred_stupid.item()-pred.item())**2)
