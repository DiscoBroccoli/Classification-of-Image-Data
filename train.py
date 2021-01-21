import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import CNN
import configurations
import utils
import pickle
import numpy as np
import ResNet

import sys
conf = sys.argv[1]

# Setup GPU usage
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    use_cuda = True
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    use_cuda = False
    print("Running on the CPU")


# Load data
# Chain together image transformations

# Check if we want data augmentation, if yes, we add random flips and crops to the image
# This should help the model to generalize better
transformations = []
if configurations.model[conf]["data_augmentation"]:
    transformations.append(transforms.RandomCrop(32, padding=4))
    transformations.append(transforms.RandomHorizontalFlip())


# ToTensor converts the PILImage into a Torch tensor, the PIlImage has dimensions Height x Width x Channels, tensor has dimensions Channels x Height x Width
# Normalize "standardizes" the input, centering the distribution at 0 with std dev 1. Here it converts ranges [0,1] to [-1,1]
transformations.append(transforms.ToTensor())
transformations.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

if configurations.model[conf]["cutout"]:
    transformations.append(utils.Cutout(n_holes=1, length=8))

transform = transforms.Compose(transformations)

# Do not augment the test set
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the training sets
# The data is shuffled by the trainloader
# Load data with batch size 128 for sgd
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

# We make a train validation split here to evaluate the models
train_set, val_set = torch.utils.data.random_split(trainset, [40000, 10000])

trainloader = torch.utils.data.DataLoader(train_set, batch_size=128,
                                          shuffle=True, num_workers=0)

valloader = torch.utils.data.DataLoader(val_set, batch_size=128,
                                          shuffle=True, num_workers=0)

# Load the test set for evaluation during training
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                          shuffle=True, num_workers=0)



# Definition of the classes in the dataset
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Instance of CNN
# Run with device GPU/CPU
if conf.split("_")[0] == "CNN":
    net = CNN.Net()
elif conf.split("_")[0] == "ResNet":
    net = ResNet.resnet32()

if use_cuda:
    net.cuda()



# Criteria used to calculate the loss between the prediction and the label
criterion = nn.CrossEntropyLoss()

# Stochastic gradient descent optimization with learning rate and momentum
# Model parameters are passed through SGD
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

# Learning rate scheduler, decays the learning rate
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 150], gamma=0.1)

# Go through 200 epochs
num_epochs = 200
train_acc_list = [0]
test_acc_list = [0]
val_acc_list = [0]




# Used to plot losses per epoch
avg_train_losses = []
avg_val_losses = []


early_stopping = utils.EarlyStopping(patience=10, verbose=True)
early_stop_epoch = 200

best_loss = 10
best_acc = 0

# Checkpoint path
PATH = './checkpoints/' + conf + "_net.pth"
BEST_PATH = './checkpoints/' + conf + '_best_net.pth'

for epoch in range(num_epochs):  # loop over the dataset multiple times


    print("Epoch: [%d/%d]" %(epoch + 1, num_epochs))

    # Train model
    train_loss, train_acc = utils.train(net, criterion, optimizer, trainloader, use_cuda, configurations.model[conf]["mixup"])

    print("Training: [Loss %.3f, Accuracy %.3f]" %(train_loss, train_acc))

    # Validate model
    val_loss, val_acc = utils.validate(net, criterion, valloader, use_cuda, configurations.model[conf]["mixup"])

    print("Validation: [Loss %.3f, Accuracy %.3f]" %(val_loss, val_acc))

    # Save checkpoint every 10 epochs if better validation accuracy
    if (epoch + 1) % 10 == 0:
        best_model = False
        if best_acc < val_acc:
            best_acc = val_acc
            best_model = True
        if best_model:
            utils.save_checkpoint(net.state_dict(), BEST_PATH)

    '''
    If Early Stop is enabled, stop training and save the model
    Early stopping happens when the validation loss does not improve over a fixed number of epochs
    '''
    if configurations.model[conf]["early_stop"]:

        early_stopping(val_loss)

        if early_stopping.early_stop:
            early_stop_epoch = epoch + 1
            print("EARLY STOPPING !!!!!!!!!!!!!!!!!!! epoch:", early_stop_epoch)
            break


    # Step in scheduler to adjust learning rate
    scheduler.step()

    # Compute the test, val and train accuracy every 5 epochs
    # This is only to report the accuracy as a function of epochs
    '''
    if (epoch + 1) % 5 == 0:
        train_acc = utils.model_accuracy(net, trainloader, use_cuda)
        test_acc = utils.model_accuracy(net, testloader, use_cuda)
        val_acc = utils.model_accuracy(net, valloader, use_cuda)

        print("Train accuracy: %.3f, Val accuracy: %.3f, Test accuracy: %.3f" %(train_acc, val_acc, test_acc))
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        val_acc_list.append(val_acc)
    '''


print('Finished Training')


torch.save(net.state_dict(), PATH)

# Evaluate the final train and validation accuracy
train_acc = utils.model_accuracy(net, trainloader, use_cuda)
val_acc = utils.model_accuracy(net, valloader, use_cuda)

print("Train Accuracy: %.3f, Validation Accuracy: %.3f" %(train_acc, val_acc))

# Evaluate on best checkpoint
checkpoint = torch.load(BEST_PATH)
best_net = ResNet.resnet32()

if use_cuda:
    best_net.cuda()
best_net.load_state_dict(torch.load(BEST_PATH))
train_acc = utils.model_accuracy(best_net, trainloader, use_cuda)
val_acc = utils.model_accuracy(best_net, valloader, use_cuda)

print("BEST Train Accuracy: %.3f, Validation Accuracy: %.3f" %(train_acc, val_acc))
PICKLE_PATH = "./validation_results/" + conf + "_val.p"
pickle.dump((train_acc, val_acc), open(PICKLE_PATH, "wb"))


# Save the losses for graphing
'''
losses = (avg_train_losses, avg_val_losses)
PICKLE_PATH = conf + "_losses.p"
pickle.dump(losses, open(PICKLE_PATH, "wb" ))
'''





# Save the accuracies for graphing
'''
accuracies = (train_acc_list, val_acc_list, test_acc_list)


PICKLE_PATH = conf + "_accuracies.p"
pickle.dump(accuracies, open(PICKLE_PATH, "wb" ) )
'''

