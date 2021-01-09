import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from data_loader import DataLoader
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc
from matplotlib import pyplot as plt
from tqdm import tqdm
import random


class Project():
    features = ['category', 'main_category', 'currency', 'country',
                'goal_level', 'duration', 'year_launched', 'month_launched']
    MAIN_CATEGORY = 1
    CATEGORY = 0
    COUNTRY = 3
    CURRENCY = 2

    def __init__(self, optional_values, category_per_main_cat_dict, currency_per_country):
        self.project = []
        for ind, f in enumerate(Project.features):
            if(ind == Project.CATEGORY):
                self.project.append(-1)
            elif(ind == Project.CURRENCY):
                self.project.append(-1)
            else:
                self.project.append(random.choice(optional_values[f]))
        self.project[Project.CATEGORY] = random.choice(
            category_per_main_cat_dict[self.project[Project.MAIN_CATEGORY]])
        self.project[Project.CURRENCY] = random.choice(
            currency_per_country[self.project[Project.COUNTRY]])

    def getProjectValues(self):
        return self.project
