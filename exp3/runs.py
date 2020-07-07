import torch
import time
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from FocalLoss import FocalLoss
# whether use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(     #input_size=(1*28*28)
            nn.Conv2d(3, 6, 5, 1, 2), #padding=2
            nn.ReLU(),      #input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),#output_size=(6*14*14)
        )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(6, 16, 5),
        #     nn.ReLU(),      #input_size=(16*10*10)
        #     nn.MaxPool2d(2, 2)  #output_size=(16*5*5)
        # )
        # self.fc1 = nn.Sequential(
        #     nn.Linear(16 * 5 * 5, 120),
        #     nn.ReLU()
        # )
        # self.fc2 = nn.Sequential(
        #     nn.Linear(14*14*6, 60),
        #     nn.ReLU()
        # )
        self.fc3 = nn.Linear(14*14*6, 2)

    # forward
    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)
        # nn.Linear()
        x = x.view(x.size()[0], -1)
        # x = self.fc1(x)
        # x = self.fc2(x)
        x = self.fc3(x)
        return x

# set hyperparameter
lr_record = []
EPOCH = 2  # epoch
LR = 0.001        # learning rate

simple_transform = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# give train and valid path and data and transform them
batch_size = 5
train = ImageFolder('data/train/',simple_transform)
valid = ImageFolder('data/test/',simple_transform)
train_loader = torch.utils.data.DataLoader(dataset = train,batch_size = batch_size,shuffle = True) #
test_loader = torch.utils.data.DataLoader(dataset = valid,batch_size = batch_size,shuffle = False)

net = LeNet()

criterion = FocalLoss(num_class=2,alpha=0.25,gamma=2)
filenames = "BFocalLoss_E"+str(EPOCH)
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

def save_data(filenames,train_loss,train_acc,path = ''):
    filename = path+'/'+filenames+'.txt'
    with open(filename,'a')as f:
        f.write('\t'+'train_acc: '+str(train_acc)+'\n\t'+'train_loss: '+str(train_loss)+'\n')
        f.close()

if __name__ == "__main__":
    train_loss = []
    train_acc = []
    # epoch_counter = 0
    for epoch in range(EPOCH):
        part_features = []
        part_labels = []
        # epoch_counter += 1
        since = time.time()
        sum_loss = 0.0
        loss_counter = 0
        loss_to_record = 0.0
        for i, data in enumerate(train_loader):

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # forward + backward
            outputs = net(inputs)
            part_features.append(outputs.data.cpu().numpy())
            part_labels.append(labels.data.cpu().numpy())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            if (i+1) % 100 == 0:
                loss_counter +=1
                loss_to_record += sum_loss/100
                print('[Epoch:%2d, Batch:%2d] loss: %.03f'
                      % (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0
        if (epoch+1) % 10 == 0:
            train_loss.append(loss_to_record/loss_counter)
        elif epoch in range(0,5):
            train_loss.append(loss_to_record / loss_counter)
        loss_counter = 0
        loss_to_record = 0.0
        # test after train
        # with torch.no_grad():
        count = 0
        pic_nums = 0
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            outputs = torch.nn.functional.softmax(outputs,dim=1)
            try:
                for pic in range(batch_size):
                    pic_nums += 1
                    if outputs[pic,0] > outputs[pic,1] and labels[pic] == 0:
                        count += 1
                    elif outputs[pic,0] < outputs[pic,1] and labels[pic] == 1:
                        count += 1
            except IndexError:
                break
        accurate = count/pic_nums
        count = 0
        pic_nums = 0
        print('epoch : %d acc：%.3f%%--->time:%d s' % (epoch + 1, accurate*100,time.time()-since))
        if (epoch+1) % 10 == 0:
            train_acc.append(str(accurate))
        elif epoch in range(0, 5):
            train_acc.append(str(accurate))
    save_data(filenames="record",train_loss=train_loss,train_acc=train_acc,path=".") #save losses and acces


