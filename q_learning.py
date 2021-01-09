import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from data_loader import DataLoader
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc
from matplotlib import pyplot as plt
from tqdm import tqdm


class Q_Learning():

    # in case building the net for the final model
    def __init__(self, args, actionsPerFeature):
        self.Q_func = {}
        self.gamma = args.gamma
        self.eta = args.eta
        self.optional_value_per_feature = actionsPerFeature

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

    def q_learning_loop():
