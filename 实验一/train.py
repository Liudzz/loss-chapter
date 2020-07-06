import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
import os


since = time.time()
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)
y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)


class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1,1)

    def forward(self,x):
        out = self.linear(x)
        return out


model = LinearRegression()
criterion = nn.SmoothL1Loss()
optimizer = optim.SGD(model.parameters(),lr=0.0001,momentum=0.9, weight_decay=5e-4)
num_epochs = 1000
saving_step = 50
l1_losses = []
l2_losses = []
huber_losses = []
for i in range(1,4):
    if i == 1:
        criterion = nn.L1Loss()
    elif i == 2:
        criterion = nn.MSELoss()
    elif i ==  3:
        criterion = nn.SmoothL1Loss()

    for epoch in range(num_epochs):
        inputs = Variable(x_train)
        target = Variable(y_train)
        out = model(inputs)
        loss = criterion(out,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % saving_step == 0:
            print("*" * 20 + str(epoch) + "*" * 20)
            if i == 1:
                l1_losses.append(loss.item())
            elif i == 2:
                l2_losses.append(loss.item())
            elif i == 3:
                huber_losses.append(loss.item())

            print('Epoch[{:4d}/{:4d}],loss:{:.6f}'.format(epoch+1,
                                                          num_epochs,
                                                          loss.item()))

print('Total Time Costed:{:.4f}'.format(time.time()-since))

if not os.path.exists("record"):
    os.mkdir("record")
if not os.path.exists("img"):
    os.mkdir("img")

# 制作图像并保存
x = range(0,num_epochs,saving_step)
y1 = l1_losses
y2 = l2_losses
y3 = huber_losses
plt.plot(x,y1,'.-',label="L1 Loss",color = "green")
plt.plot(x,y2,'.-',label="L2 Loss",color = "blue")
plt.plot(x,y3,'.-',label="Huber Loss",color = "red")
plt.legend()
plt.title("Train Losses ")
plt.xlabel('epoches')
plt.ylabel('losses')
plt.savefig("img\\"+str(num_epochs)+"_"+str(saving_step)+"_savedloss.jpg")
plt.show()

# 数据保留小数点后四位并保存
l1_losses_pro = ['{:.4f}'.format(i) for i in l1_losses]
l2_losses_pro = ['{:.4f}'.format(i) for i in l2_losses]
huber_losses_pro = ['{:.4f}'.format(i) for i in huber_losses]
with open("record\loss_record.txt",'a') as f:
    f.write("L1_Loss:\n")
    for loss in l1_losses_pro:
        f.write("\t"+str(loss)+"\n")
    f.write("L2_Loss:\n")
    for loss in l2_losses_pro:
        f.write("\t" + str(loss) + "\n")
    f.write("Huber_Loss:\n")
    for loss in l2_losses_pro:
        f.write("\t" + str(loss) + "\n")
    f.close()
