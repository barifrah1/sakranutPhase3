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


class Q_Learning():

    # in case building the net for the final model
    def __init__(self, projectsToStartWith: list,  args, gt_net, learner):
        self.Q_func = {}
        self.projects = projectsToStartWith
        self.nextIterProjects = []
        self.args = args
        self.gt = gt_net
        self.learner = learner

    def getQvalue(self, state, action):
        return self.Q_func[state][action]

    def setQvalue(self, state, action, value):
        self.Q_func[state][action] = value

    def getNextState(self, state, action):
        new_state = state
        next_state[action[key]] = action[value]
        return next_state

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
            if(self.getQvalue(next_state, action) > max_value):
                max_value = self.getQvalue(next_state, action)
                max_action = action
        if(max_action != -1):
            return max_action, next_state
        else:
            raise Exception(
                f"Sorry,no appropriate action was found for state {state}")

    def reward(self, proj):
        state = torch.tensor(proj.getNetProjectFormat()).float()
        self.gt.eval()
        self.learner.eval()
        with torch.no_grad():
            pred_gt = self.gt(state).item()
            pred_learner = self.learner(state).item()
        epsilon2 = (pred_gt - pred_learner)**2
        return epsilon2

    def q_learning_loop(self):
        for p in self.projects:
            action = ('duration', list(
                p.getOptionalActions()['duration'].keys())[0])
            self.nextIterProjects.append((p, action))
        for iter in range(self.args.num_of_iters):
            for proj, cur_action in self.nextIterProjects:
                state = proj.getProjectValues()
                # case when we havent visit in this project yet, then defint its q value to uniform distribution
                if(state not in self.Q_func.keys()):
                    self.Q_func[proj.getProjectValues()] = {}
                    init = proj.getOptionalActionsTOInitQfunc()
                    for f in init.keys():
                        for a in init[f].keys():
                            self.setQvalue(state, (f, a), init[f][a])
                r = self.reward(proj)
                bestAction, next_state = self.chooseBestAction(state)
                td_error = r + self.args.gamma * \
                    self.getQvalue(next_state, bestAction) - \
                    self.getQvalue(state, cur_action)
                self.setQvalue(state, cur_action, self.getQvalue(
                    state, cur_action)+self.args.eta*td_error)
            self.nextIterProjects = []
            self.nextIterProjects.append((Project(next_state), bestAction))
