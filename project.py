import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from data_loader import random_project_preproesses
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle
import pandas as pd
import random


class Project():
    features = ['category', 'main_category', 'currency', 'country',
                'goal_level', 'duration', 'year_launched', 'month_launched']
    features_dict = {'category': 0, 'main_category': 1, 'currency': 2, 'country': 3,
                     'goal_level': 4, 'duration': 5, 'year_launched': 6, 'month_launched': 7}
    CATEGORY = 0
    MAIN_CATEGORY = 1
    CURRENCY = 2
    COUNTRY = 3
    with open('scale_dict.pickle', 'rb') as handle:
        scale_dict = pickle.load(handle)
    with open('option_dict.pickle', 'rb') as handle:
        option_dict = pickle.load(handle)
    with open('category_per_main_cat_dict.pickle', 'rb') as handle:
        category_per_main_cat_dict = pickle.load(handle)
    with open('currency_per_country_dict.pickle', 'rb') as handle:
        currency_per_country = pickle.load(handle)
    empty_data_format = pd.read_pickle("empty_data_format.pkl")

    def __init__(self, project: list):
        self.project = project
        if(project == []):  # in case  random project then create it
            for ind, f in enumerate(Project.features):
                if(ind == Project.CATEGORY):
                    self.project.append(-1)
                elif(ind == Project.CURRENCY):
                    self.project.append(-1)
                else:
                    self.project.append(random.choice(Project.option_dict[f]))
            self.project[Project.CATEGORY] = random.choice(
                Project.category_per_main_cat_dict[self.project[Project.MAIN_CATEGORY]])
            self.project[Project.CURRENCY] = random.choice(
                Project.currency_per_country[self.project[Project.COUNTRY]])
        self.project = tuple(self.project)
        self.actions = self.getOptionalActions()

    def getProjectValues(self):
        return self.project

    def getOptionalActions(self):
        possible_actions = {}
        for ind, f in enumerate(Project.features):
            if(ind == Project.CATEGORY):
                possible_actions[f] = dict(
                    (cat, 1) for cat in Project.category_per_main_cat_dict[self.project[Project.MAIN_CATEGORY]])
            elif(ind == Project.CURRENCY):
                possible_actions[f] = dict(
                    (cur, 1) for cur in Project.currency_per_country[self.project[Project.COUNTRY]])
            else:
                possible_actions[f] = dict(
                    (cur, 1) for cur in Project.option_dict[f])

        return possible_actions

    def getOptionalActionsTOInitQfunc(self):
        init = self.actions
        for f in self.actions.keys():
            count = len(self.actions[f].keys())
            for a in self.actions[f].keys():
                init[f][a] = 1/(8*count)
        return init

    def getNetProjectFormat(self):
        return random_project_preproesses(
            Project.empty_data_format, self.project, Project.scale_dict)
