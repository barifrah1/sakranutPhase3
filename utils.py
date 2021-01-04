from torch import nn

args = {
    'fileName': 'ks-projects-201801_bar.csv',
    'trainSize': 0.7,
    'batch_size': 13779,
    'n_epochs': 500,
    'weight_decay': 2.780552870258695e-06,
    'lr': 0.0015530911275610264,

}

loss_function = nn.BCELoss()
