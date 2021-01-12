from torch import nn
import matplotlib.pyplot as plt
import pylab
import numpy

args = {
    'fileName': 'ks-projects-201801_bar.csv',
    'trainSize': 0.7,
    'batch_size': 13779,
    'n_epochs': 500,
    'weight_decay': 2.780552870258695e-06,
    'lr': 0.0015530911275610264,

}

loss_function = nn.BCELoss()


def plot_loss_graph(validation_loss_list):
    epoches = list(range(1, len(validation_loss_list)+1))
    # calc the trendline
    z = numpy.polyfit(epoches, validation_loss_list, 1)
    p = numpy.poly1d(z)
    pylab.plot(epoches, p(epoches), "r--")
    # the line equation:
    print("y=%.6fx+(%.6f)" % (z[0], z[1]))
    # plot the graph
    plt.plot(validation_loss_list, 'b', label='validation loss')
    plt.title('Validation loss')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_auc_graph(auc_val_list):
    plt.plot(auc_val_list, 'b', label='validation AUC')
    plt.title('Validation AUC')
    plt.xlabel('Episodes')
    plt.ylabel('AUC')
    plt.legend()
    plt.show()
