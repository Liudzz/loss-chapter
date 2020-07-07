import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import numpy as np
from torch.nn import functional as F
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvNet(nn.Module):
    """LeNet++ as described in the Center Loss paper."""
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 32, 5, stride=1, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.prelu1_2 = nn.PReLU()

        self.conv2_1 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.prelu2_2 = nn.PReLU()

        self.conv3_1 = nn.Conv2d(64, 128, 5, stride=1, padding=2)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.prelu3_2 = nn.PReLU()

        self.fc1 = nn.Linear(128 * 3 * 3, 2)
        self.prelu_fc1 = nn.PReLU()
        self.fc2 = nn.Linear(2, num_classes)

    def forward(self, x):
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x, 2)

        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x, 2)

        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 128 * 3 * 3)
        x = self.prelu_fc1(self.fc1(x))
        y = self.fc2(x)

        return y, x

# hyperparameter setting
EPOCH = 150   # epoch
BATCH_SIZE = 64      # batch_size
LR = 0.001        # learning rate
save_file_name = "CELoss" # name for saving data,most will be used in training data and feature visualization
# dataset preprocess
transform = transforms.ToTensor()
# load train data
if not os.path.exists("data"):
    os.mkdir("data")
trainset = tv.datasets.MNIST(
    root='./data/',
    train=True,
    download=True,
    transform=transform)


trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    )

# load test data
testset = tv.datasets.MNIST(
    root='./data/',
    train=False,
    download=True,
    transform=transform)


testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    )

# set loss function and optimizer
net = ConvNet(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，通常用于多分类问题上
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

def plot_features(features, labels, num_classes, epoch, prefix):
    """Plot features on 2D plane.
    Args:
        features: (num_instances, num_features).
        labels: (num_instances).
    """
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for label_idx in range(num_classes):
        plt.scatter(
            features[labels==label_idx, 0],
            features[labels==label_idx, 1],
            c=colors[label_idx],
            s=1,
        )
    plt.title(str(epoch+1))
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    dirname = save_file_name+"/"+prefix
    if not os.path.exists("cel"):
        os.mkdir("cel")
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    filenames = '/epoch_' + str(epoch+1) + '.png'
    plt.savefig(dirname+filenames, bbox_inches='tight')
    plt.close()

# training
if __name__ == "__main__":
    train_loss = []
    train_acc = []
    for epoch in range(EPOCH):
        batch_counter = 0
        total_loss = 0.0
        since =  time.time()
        all_train_features = []
        all_train_labels = []
        all_test_features = []
        all_test_labels = []
        sum_loss = 0.0
        batch_time = time.time()
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # forward + backward
            outputs, features = net(inputs)
            all_train_features.append(features.data.cpu().numpy())
            all_train_labels.append(labels.data.cpu().numpy())

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print avarage loss every 100 batch
            sum_loss += loss.item()
            if i % 100 == 99:
                batch_counter += 1
                print('[Epoch:%3d,Batch:%3d] loss: %.03f BatchTime: %.1fs'
                      % (epoch + 1, i + 1, sum_loss / 100,time.time()-batch_time))
                with open(save_file_name+'_Log.txt','a')as f:
                    f.write("Epoch:"+str(epoch+1)+" Batch:"+str(i+1)+" loss:"+str(sum_loss / 100)+'\n')
                f.close()
                total_loss += sum_loss / 100
                sum_loss = 0.0
                batch_time = time.time()
        train_loss.append(total_loss/batch_counter)
        all_train_features = np.concatenate(all_train_features, 0)
        all_train_labels = np.concatenate(all_train_labels, 0)
        plot_features(all_train_features, all_train_labels, 10, epoch, prefix='train')
        # test acc after train
        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs,features = net(images)
                all_test_features.append(features.data.cpu().numpy())
                all_test_labels.append(labels.data.cpu().numpy())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('Epoch:%d ,Acc:%.4f%% ,Time:%ds' % (epoch + 1, (100 * correct.item() / total),time.time()-since))
            train_acc.append((correct.item() / total))
            with open(save_file_name+'_Log.txt', 'a')as f:
                f.write("Epoch:" + str(epoch + 1) + " Acc:" + str(100 * correct.item() / total) + '\n')
            f.close()
        all_test_features = np.concatenate(all_test_features, 0)
        all_test_labels = np.concatenate(all_test_labels, 0)
        plot_features(all_test_features, all_test_labels, 10, epoch, prefix='test')
        train_loss = ['{:.4f}'.format(float(i)) for i in train_loss]
    with open(save_file_name+"_record.txt","a") as f:
        f.write("Train Loss:\n\t"+str((train_loss))+"\nTrain Acc:\n\t"+str(train_acc)+"\n")
        f.close()


