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
from NN import train_small_batch, predict
import random
from copy import deepcopy


class Q_Learning():

    # in case building the net for the final model
    def __init__(self, projectsToStartWith: list,  args, gt_net, learner, X_test=None, y_test=None):
        self.Q_func = {}
        self.projects = projectsToStartWith
        self.nextIterProjects = []
        self.args = args
        self.gt = gt_net
        self.learner = learner
        self.original_learner = learner
        self.X_test = X_test
        self.y_test = y_test
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
        if(action[0] == 'main_category'):
            main_cat = action[1]
            optional_categories = Project.category_per_main_cat_dict[main_cat]
            new_action = random.choice(optional_categories)
            feature_index = self.features_dict['category']
            next_state[feature_index] = new_action
            if(tuple(next_state) not in self.Q_func.keys()):
                self._addStateAndPossibleActionsToQ(Project(next_state))
        if(action[0] == 'country'):
            cou = action[1]
            optional_curr = Project.currency_per_country[cou]
            new_action = random.choice(optional_curr)
            feature_index = self.features_dict['currency']
            next_state[feature_index] = new_action
            if(tuple(next_state) not in self.Q_func.keys()):
                self._addStateAndPossibleActionsToQ(Project(next_state))
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
            if(self.getQvalue(next_state, action) > max_value and next_state != state):
                max_value = self.getQvalue(next_state, action)
                max_action = action
                save_next_state = next_state
        if(max_action != -1):
            return max_action, save_next_state
        else:
            raise Exception(
                f"Sorry,no appropriate action was found for state {state}")

    def chooseBestAction_ver2(self, state):
        actions_vector = self.Q_func[state].keys()
        """max_value = -10000000000000000000000000000000000000
        max_action = -1"""
        weights = []
        for action in actions_vector:
            next_state = self.getNextState(state, action)
            if(next_state not in self.Q_func.keys()):
                self._addStateAndPossibleActionsToQ(Project(next_state))
            weights.append(self.getQvalue(next_state, action))
        draw_action = random.choices(
            list(actions_vector), weights=weights, k=1)[0]
        next_state = self.getNextState(state, draw_action)
        f2 = open("weight_trace.txt", "a")
        f2.write(str(weights)+" "+str(state)+"\n")
        f2.close()

        return draw_action, next_state
        """    if(self.getQvalue(next_state, action) > max_value and next_state != state):
                max_value = self.getQvalue(next_state, action)
                max_action = action
                save_next_state = next_state
        if(max_action != -1):
            return max_action, save_next_state
        else:
            raise Exception(
                f"Sorry,no appropriate action was found for state {state}")
        """

    def _chooseRandomAction(self, state):
        actions_vector = self.Q_func[state].keys()
        chosenRandomAction = random.choice(list(actions_vector))
        next_state = self.getNextState(state, chosenRandomAction)
        if(next_state not in self.Q_func.keys()):
            self._addStateAndPossibleActionsToQ(Project(next_state))
        return chosenRandomAction, next_state

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
        X = torch.empty([0, 187])
        for p in projects:
            X = torch.cat(
                [X, torch.tensor(p[0].getNetProjectFormat()).float()], dim=0)
        self.gt.eval()
        with torch.no_grad():
            y = self.gt(X)
        return X, y

    def _trainLearnerAndGetPreds(self, projects, iter=None):
        weight_decay = 2.780552870258695e-06
        lr = 0.0015530911275610264
        lr = 0.01
        weight_decay = 0
        if(iter != None):
            if(iter == iter % 100):
                lr = lr*0.9
        X, y = self._createTensorOfDataToTrain(projects)
        pred_from_learner = train_small_batch(X, y, self.learner, lr, weight_decay,
                                              1, criterion=nn.BCELoss())
        return pred_from_learner, y  # pred_from_learner, pred_of_gt_net

    def q_learning_loop(self):
        loss_by_episode = []
        auc_by_episode = []
        for episode in range(self.args.num_episodes):
            self.learner = deepcopy(self.original_learner)
            trained_val_auc = -1
            auc_validation = []
            trained_val_loss = -1
            validation_loss_list = []
            for p in self.projects:
                p = Project([])
                action = ('duration', list(
                    p.getOptionalActions()['duration'].keys())[0])
                self.nextIterProjects.append((p, action))
            untrained_test_auc, _, untrained_test_loss, _ = predict(
                self.X_test, self.y_test, self.gt)
            print(
                f"GT net Test loss is: {untrained_test_loss}")
            print(
                f"GT net Test auc is: {untrained_test_auc}")
            untrained_test_auc, _, untrained_test_loss, _ = predict(
                self.X_test, self.y_test, self.learner)
            print(
                f"Learner Net Test loss before training is: {untrained_test_loss}")
            print(
                f"Learner Net Test auc before training is: {untrained_test_auc}")
            for iter in tqdm(range(self.args.num_of_iters)):
                # here we should train the learner_net
                pred_learner, pred_gt = self._trainLearnerAndGetPreds(
                    self.nextIterProjects, iter=iter)
                ind = 0
                saveProjectsToNextIter = []
                """f = open("state_trace.txt", "a")
                f.write(str(self.nextIterProjects[0][0].getProjectValues(
                ))+" "+str(self.nextIterProjects[0][1])+"\n")
                f.close()"""
                for proj, cur_action in self.nextIterProjects:
                    state = proj.getProjectValues()
                    # case when we havent visit in this project yet, then defint its q value to uniform distribution
                    if(state not in self.Q_func.keys()):
                        self._addStateAndPossibleActionsToQ(proj)
                    r = self.reward(pred_learner[ind], pred_gt[ind])

                    if(iter > self.args.iter_to_choose_best_action):
                        sample_random = random.random()
                        if(sample_random < 0.9):
                            bestAction, next_state = self.chooseBestAction(
                                state)
                        else:
                            bestAction, next_state = self._chooseRandomAction(
                                state)
                    else:
                        bestAction, next_state = self._chooseRandomAction(
                            state)
                    td_error = r.item() + self.args.gamma * self.getQvalue(next_state,
                                                                           bestAction) - self.getQvalue(state, cur_action)
                    self.setQvalue(state, cur_action, self.getQvalue(
                        state, cur_action)+self.args.eta*td_error)
                    saveProjectsToNextIter.append(
                        (Project(next_state), bestAction))
                    ind += 1
                if iter % 50 == 0:
                    trained_val_auc, auc_validation, trained_val_loss, validation_loss_list = predict(
                        self.X_test, self.y_test, self.learner, auc_validation, validation_loss_list)
                    print(
                        f"Current validation loss on epoch {iter+1} is: {trained_val_loss} \n Current validation auc is: { trained_val_auc} ")
                self.nextIterProjects = saveProjectsToNextIter

                if(iter % 51 == 0):
                    self.nextIterProjects = []
                    for p in self.projects:
                        p = Project([])
                        action = ('duration', list(
                            p.getOptionalActions()['duration'].keys())[0])
                        self.nextIterProjects.append((p, action))
            trained_val_auc, auc_validation, trained_val_loss, validation_loss_list = predict(
                self.X_test, self.y_test, self.learner, auc_validation, validation_loss_list)
            print(
                f"Final validation loss  is: {trained_val_loss} \n Final validation auc is: { trained_val_auc} ")
            loss_by_episode.append(trained_val_loss)
            auc_by_episode.append(trained_val_auc)
            print(
                "episode {episode+1} : loss: {trained_val_loss}  auc: {trained_val_auc}")
