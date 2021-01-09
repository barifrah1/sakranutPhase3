import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from data_loader import DataLoader
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc
from matplotlib import pyplot as plt
from tqdm import tqdm
from project import Project
from NN import train_small_batch


class Q_Learning():

    # in case building the net for the final model
    def __init__(self, projectsToStartWith: list,  args, gt_net, learner):
        self.Q_func = {}
        self.projects = projectsToStartWith
        self.nextIterProjects = []
        self.args = args
        self.gt = gt_net
        self.learner = learner
        self.features_dict = {'category': 0, 'main_category': 1, 'currency': 2, 'country': 3,
                              'goal_level': 4, 'duration': 5, 'year_launched': 6, 'month_launched': 7}

    def getQvalue(self, state, action):
        return self.Q_func[state][action]

    def setQvalue(self, state, action, value):
        self.Q_func[state][action] = value

    def getNextState(self, state, action):
        next_state = list(state)
        feature_index = self.features_dict[action[0]]
        next_state[feature_index] = action[1]
        return tuple(next_state)

    def getActionsProbs(self, state):
        keys_vector = self.Q_func[state].keys()
        q_values_vector = self.Q_func[state].values()
        if(len(keys_vector) > 0):
            sumOfAllStataValues = sum(q_values_vector)
            probs_vector = q_values_vector/sumOfAllStataValues
            return keys_vector, probs_vector

    def chooseBestAction(self, state):
        actions_vector = self.Q_func[state].keys()
        max_value = -10000000000000000000000000000000000000
        max_action = -1
        for action in actions_vector:
            next_state = self.getNextState(state, action)
            if(next_state not in self.Q_func.keys()):
                self._addStateAndPossibleActionsToQ(Project(next_state))
            if(self.getQvalue(next_state, action) > max_value):
                max_value = self.getQvalue(next_state, action)
                max_action = action
                save_next_state = next_state
        if(max_action != -1):
            return max_action, save_next_state
        else:
            raise Exception(
                f"Sorry,no appropriate action was found for state {state}")

    def reward(self, pred_learner, pred_gt):
        epsilon2 = (pred_learner - pred_gt)**2
        return epsilon2

    def _addStateAndPossibleActionsToQ(self, proj):
        state = proj.getProjectValues()
        if(state not in self.Q_func.keys()):
            self.Q_func[proj.getProjectValues()] = {}
            init = proj.getOptionalActionsTOInitQfunc()
            for f in init.keys():
                for a in init[f].keys():
                    self.setQvalue(state, (f, a), init[f][a])

    def _createTensorOfDataToTrain(self, projects):
        X = torch.tensor([])
        for p in projects:
            X.cat(p[0].getProjectValues())
        self.gt().eval()
        with torch.no_grad():
            y = self.gt(X)
        return X, y

    def _trainLearnerAndGetPreds(self, projects):
        weight_decay = 2.780552870258695e-06
        lr = 0.0015530911275610264
        X, y = self._createTensorOfDataToTrain(projects)
        pred_from_learner = train_small_batch(X, y, self.learner, lr, weight_decay,
                                              1, criterion=nn.BCELoss())
        return pred_from_learner, y  # pred_from_learner, pred_of_gt_net

    def q_learning_loop(self):
        for p in self.projects:
            action = ('duration', list(
                p.getOptionalActions()['duration'].keys())[0])
            self.nextIterProjects.append((p, action))
        for iter in tqdm(range(self.args.num_of_iters)):
            # here we should train the learner_net
            pred_learner, pred_gt = self._trainLearnerAndGetPreds(
                self.nextIterProjects)
            for proj, cur_action in self.nextIterProjects:
                state = proj.getProjectValues()
                # case when we havent visit in this project yet, then defint its q value to uniform distribution
                if(state not in self.Q_func.keys()):
                    self._addStateAndPossibleActionsToQ(proj)
                r = self.reward(pred_learner, pred_gt)
                bestAction, next_state = self.chooseBestAction(state)
                td_error = r + self.args.gamma * \
                    self.getQvalue(next_state, bestAction) - \
                    self.getQvalue(state, cur_action)
                self.setQvalue(state, cur_action, self.getQvalue(
                    state, cur_action)+self.args.eta*td_error)
            self.nextIterProjects = []
            self.nextIterProjects.append((Project(next_state), bestAction))
