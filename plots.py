import matplotlib.pyplot as plt
import pickle
import numpy as np

'''
Plot the accuracies as a function of epochs for train, val and test for CNN
'''
def plot_accuracies(path, title):

    accuracies = pickle.load(open(path, "rb"))
    train_acc = accuracies[0]
    val_acc = accuracies[1]
    test_acc = accuracies[2]

    epochs = np.arange(0, 255, 5)

    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.plot(epochs, test_acc, label="Test Accuracy")
    plt.legend(loc="upper left")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig(title + '.png', bbox_inches='tight')


'''
CNN_ACC_PATH = "CNN_accuracies.p"
plot_accuracies(CNN_ACC_PATH, "LeNet CNN")
'''

def plot_losses(path, title):
    losses = pickle.load(open(path, "rb"))
    train_acc = losses[0]
    val_acc = losses[1]


    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.axvline(77, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.legend(loc="upper left")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.savefig(title + 'losses.png', bbox_inches='tight')
    #plt.show()

CNN_LOSS_PATH = "CNN_early_losses.p"
plot_losses(CNN_LOSS_PATH, "LeNet CNN")
