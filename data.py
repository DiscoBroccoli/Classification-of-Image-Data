import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import utils
import seaborn as sns

# Load the training sets
# The data is shuffled by the trainloader

transformCutout = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    utils.Cutout(n_holes=1, length=8)
])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transformAugment = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


trainset_normal = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainset_augment = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transformAugment)

trainset_cutout = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transformCutout)

# Definition of the classes in the dataset
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



trainloader_normal = torch.utils.data.DataLoader(trainset_normal, batch_size=128,
                                          shuffle=True, num_workers=0)
trainloader_augment = torch.utils.data.DataLoader(trainset_augment, batch_size=8,
                                          shuffle=True, num_workers=0)
trainloader_cutout = torch.utils.data.DataLoader(trainset_cutout, batch_size=8,
                                          shuffle=True, num_workers=0)




# functions to show an image
def imshow(img1, img2, img3, img4, img5):
    fig, axs = plt.subplots(5)

    img1 = img1 / 2 + 0.5  # unnormalize
    npimg = img1.numpy()
    axs[0].imshow(np.transpose(npimg, (1, 2, 0)))
    axs[0].set_title('No Augmentation')


    img2 = img2 / 2 + 0.5  # unnormalize
    npimg = img2.numpy()
    axs[1].imshow(np.transpose(npimg, (1, 2, 0)))
    axs[1].set_title('Augmentation')


    img3 = img3 / 2 + 0.5  # unnormalize
    npimg = img3.numpy()
    axs[2].imshow(np.transpose(npimg, (1, 2, 0)))
    axs[2].set_title('Cutout')

    img4 = img4 / 2 + 0.5  # unnormalize
    npimg = img4.numpy()
    axs[3].imshow(np.transpose(npimg, (1, 2, 0)))
    axs[3].set_title('Mixup')

    img5 = img5 / 2 + 0.5  # unnormalize
    npimg = img5.numpy()
    axs[4].imshow(np.transpose(npimg, (1, 2, 0)))
    axs[4].set_title('Cutout + Mixup')


    for ax in axs:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig.tight_layout()
    fig.savefig('data.png', bbox_inches='tight')



# get some random training images
dataiter1 = iter(trainloader_normal)
images1, labels1 = dataiter1.next()

dataiter2 = iter(trainloader_augment)
images2, labels2 = dataiter2.next()

dataiter3 = iter(trainloader_cutout)
images3, labels3 = dataiter3.next()

dataiter4 = iter(trainloader_augment)
images4, labels4 = dataiter4.next()

images4, labels_a, labels_b, lam = utils.mixup_data(images4, labels4, 1, False)

dataiter5 = iter(trainloader_augment)
images5, labels5 = dataiter5.next()

images5, labels_a, labels_b, lam = utils.mixup_data(images5, labels5, 1, False)

# show images
#imshow(torchvision.utils.make_grid(images1),torchvision.utils.make_grid(images2),torchvision.utils.make_grid(images3),torchvision.utils.make_grid(images4),torchvision.utils.make_grid(images5) )



# Distribution of classes
label = []
for data in trainset_normal:
    images, labels = data
    label.append(classes[labels])


sns.countplot(label)
plt.savefig('data_dist.png', bbox_inches='tight')