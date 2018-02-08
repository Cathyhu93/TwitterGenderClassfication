import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import pandas as pd

class mymodel(nn.Module):
    def __init__(self):
        super(mymodel, self).__init__()
        self.features = models.vgg16(pretrained=True).features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2),
        )
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, input):
        f = self.features(input)
        f = f.view(f.size(0), -1)
        pred = self.classifier(f)
        return pred


class KaggleDataset(Dataset):
    """Dataset wrapping images and target labels for Kaggle - twitter gender prediction.

    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        PIL transforms
    """

    def __init__(self, img_path,img_ext,csv_path,transform=None):
        tmp_df = pd.read_csv(csv_path)
        self.lb = LabelEncoder()
        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform

        self.X_train,row = tmp_df.iterrows()
        self.y_train = self.lb.fit_transform(tmp_df['gender'])

    def __getitem__(self, index):
        img = Image.open(self.img_path + self.X_train[index]+self.img_ext)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        label = torch.from_numpy(self.y_train[index])
        return img, label

    def __len__(self):
        return len(self.X_train.index)

IMG_PATH = './images_cleaned'
IMG_EXT = '.png'
TXT_PATH = '../gender-classifier-DFE-791531.csv'
transformations = transforms.Compose([transforms.Scale(32),transforms.ToTensor()])

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(, transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)
net = mymodel()
# Define a Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD([{'params':net.my_features.parameters(), 'lr':1e-7},
                        {'params':net.classifier.parameters(), 'lr':1e-3}
                        ], momentum=0.9)

epoch_set = np.repeat(1, 10)
time_set = np.zeros((len(epoch_set)))
trainAccuracy_set = np.zeros(len(epoch_set))
testAccuracy_set = np.zeros(len(epoch_set))

iters = 1
for k in range(len(epoch_set)):
    e = epoch_set[k]
    print "number of iters: ", iters
    start = time.time()
    print('=================Start Training==================')
    for epoch in range(e):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)
            inputs = F.upsample(inputs, size=(224, 224), mode='bilinear')

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    end = time.time()
    time_set[k] = end - start
    print('training time: ', end - start)

    print('=================Start Testing==================')
    # Overall Performance on the whole test dataset
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        images = Variable(images)
        images = F.upsample(images, size=(224, 224), mode='bilinear')
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    # print "correct: ", correct
    #    print "total: ", total
    testAccuracy_set[k] = 100 * float(correct) / float(total)
    print('Testing Accuracy: %f %%' % (
        100 * correct / total))

    # Overall Performance on the whole training dataset
    correct = 0
    total = 0
    for data in trainloader:
        images, labels = data
        images = Variable(images)
        images = F.upsample(images, size=(224, 224), mode='bilinear')
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    # print "correct: ", correct
    #    print "total: ", total
    trainAccuracy_set[k] = 100 * float(correct) / float(total)
    print('Training Accuracy: %f %%' % (
        100 * correct / total))

    iters += 1

epoch_set_ascending = np.arange(1, len(epoch_set) + 1, 1)
plt.plot(epoch_set_ascending, trainAccuracy_set, 'r', label='SGD')
plt.ylabel("Training Accuracy")
plt.xlabel("Number of Epoch")
plt.title("Performance over different models")
plt.show()

PretrainedAlexNet_trainAccuracy = np.copy(trainAccuracy_set)
PretrainedAlexNet_testAccuracy = np.copy(testAccuracy_set)
PretrainedAlexNet_time = np.copy(time_set)

np.savetxt('PretrainedAlexNet_trainAccuracy', PretrainedAlexNet_trainAccuracy, delimiter=',')
np.savetxt('PretrainedAlexNet_testAccuracy', PretrainedAlexNet_testAccuracy, delimiter=',')
np.savetxt('PretrainedAlexNet_time', PretrainedAlexNet_time, delimiter=',')
