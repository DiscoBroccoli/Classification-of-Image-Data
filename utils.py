import numpy as np
import torch

'''
Mixup code from https://github.com/facebookresearch/mixup-cifar10
'''
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def model_accuracy(net, dataloader, use_cuda):
    total = 0
    correct = 0

    with torch.no_grad():
        for (images, labels) in dataloader:
            if use_cuda:
                images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

'''
 Early stopping calculator, based on https://github.com/Bjarten/early-stopping-pytorch

'''
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

'''
Cutout code from https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
'''

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img



def train(net, criterion, optimizer, dataloader, use_cuda, mixup, verbose=False):

    total = 0
    correct = 0
    running_loss = 0
    losses = []

    # Setup the model in training mode
    net.train()
    for i, (inputs, labels) in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        # use gpu
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        # Use mixup if needed
        if mixup:
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, 1, use_cuda)
            outputs = net(inputs)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = net(inputs)

            # compute the loss
            loss = criterion(outputs, labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # compute gradients
        loss.backward()

        # update parameters
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)

        if mixup:
            correct += (lam * predicted.eq(labels_a.data).sum().float()
                        + (1 - lam) * predicted.eq(labels_b.data).sum().float())
        else:
            correct += (predicted == labels).sum().item()

        losses.append(loss.item())
        # print every 100 minibatches
        running_loss += loss.item()

        if (i + 1) % 100 == 0:
            if verbose:
                print('[%5d] loss: %.3f, accuracy: %.3f' %
                    (i + 1, running_loss / 2000, (correct / total) * 100))
            running_loss = 0.0

    accuracy = 100 * correct / total

    return np.average(losses), accuracy

def validate(net, criterion, dataloader, use_cuda, mixup):
    net.eval()
    total = 0
    correct = 0
    losses = []
    for inputs, labels in dataloader:
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        # Use mixup if needed
        if mixup:
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, 1, use_cuda)
            outputs = net(inputs)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            # forward pass: compute predicted outputs by passing inputs to the model

            outputs = net(inputs)

            # calculate the loss
            loss = criterion(outputs, labels)

        losses.append(loss.item())

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        if mixup:
            correct += (lam * predicted.eq(labels_a.data).sum().float()
                        + (1 - lam) * predicted.eq(labels_b.data).sum().float())
        else:
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    return np.average(losses), accuracy


def save_checkpoint(state, filename):
    """
    Save the training model
    """
    print("Saving", filename)
    torch.save(state, filename)