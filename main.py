from data_loader import *
from utils import args, loss_function
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
import NN
from NN import Net, predict
import torch
import torch.nn as nn
from gridsearch import GridSearch
import os
import os.path
from prediction import *
import pickle
import pandas as pd
from project import Project
from args import QArgs
from q_learning import Q_Learning
from utils import plot_loss_graph, plot_auc_graph

if __name__ == '__main__':

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
    # learner net starting with random weights
    learner_net = NN.Net(feature_num)
    # ground truth net traind over original dataset
    gt_net = load_net(feature_num)
    q_args = QArgs()
    print("fdf")

    data_loader = DataLoader(args, False)
    # preprocessing
    X, Y, columnsInfo = data_loader.preprocess()
    # split data to train and test
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_train_validation_test(
        X, Y)
    li = []
    for i in range(q_args.num_of_projects_to_start):
        proj = Project([])
        li.append(proj)

    if(os.path.isfile('Q.pickle')) == True:
        with open('Q.pickle', 'rb') as handle:
            Q = pickle.load(handle)
        q_learn = Q_Learning(li, q_args, gt_net, learner_net,
                             X_test=X_test, y_test=y_test, Q=Q)
    else:
        q_learn = Q_Learning(li, q_args, gt_net, learner_net,
                             X_test=X_test, y_test=y_test, Q=None)
    loss_by_episode, auc_by_episode = q_learn.q_learning_loop()
    plot_loss_graph(loss_by_episode)
    plot_auc_graph(auc_by_episode)
    """q_learn_random = Q_Learning(li, q_args, gt_net, learner_net,
                                X_test=X_test, y_test=y_test,Q=None)
    q_learn_random.q_learning_loop(is_random_policy=True)"""

    auc_gt, _, test_loss_gt, __ = predict(
        X_test, y_test, gt_net, auc_list=None, loss_list=None)
    auc_leraner, _, test_loss_learner, __ = predict(
        X_test, y_test, learner_net, auc_list=None, loss_list=None)
    print("GT net Test_loss : {} and GT net Test auc for : {}".format(
        test_loss_gt, auc_gt))
    print("Learner net Test_loss : {} and Learner net Test auc for : {}".format(
        test_loss_learner, auc_leraner))
