from data_loader import *
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
import os.path
from prediction import *
import pickle
import pandas as pd

GRID_SEARCH_MODE = False
if __name__ == '__main__':
    if(GRID_SEARCH_MODE):
        gridsearch = GridSearch()
        gridsearch.exectue_grid_search(10000)
    else:
        if (os.path.isfile('scale_dict.pickle'))==False:
            data_loader = DataLoader(args, False)  # False for is_grid_search mode
        # preprocessing
            X, Y, columnsInfo = data_loader.preprocess()
        # split data to train and test
            X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_train_validation_test(X, Y)
            feature_num = len(columnsInfo)
        else:
            with open('scale_dict.pickle', 'rb') as handle:
                scale_dict = pickle.load(handle)
            with open('option_dict.pickle', 'rb') as handle:
                option_dict = pickle.load(handle)
            empty_data_format = pd.read_pickle("empty_data_format.pkl")
            feature_num= len(empty_data_format.columns)        
        print('feature_num',feature_num)
        model = NN.Net(feature_num)
        path = os.getcwd()
        path=path+ "/net.pt"
        if (os.path.isfile('net.pt'))==False:
            NN.train(X_train, y_train, model, X_val, y_val, X_test, y_test,
                    args["batch_size"], args["lr"], args["weight_decay"],
                    n_epochs=args["n_epochs"],
                    criterion=loss_function)

            torch.save(model.state_dict(),path)
        # optimizer = nn.BCELoss)
        #[category,main_category,currency,country,goal_level,duration,year_launched,month_launched]
        pro=['Music','main_category_Music','USD','US',1891,20,2015,7]
        project=random_project_preproesses(empty_data_format,pro,scale_dict)
        project=torch.from_numpy(project).float()
        print('project.shape',project.shape)
        net=prepre_net(feature_num)
        pred=get_pred(net,project)
        print('prob to be successful project',pred.item())        

        
        