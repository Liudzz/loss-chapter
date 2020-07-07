# train.py
#!/usr/bin/env	python3

import os
import argparse
import torch
import torch.optim as optim
import time
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR
from FocalLoss import FocalLoss
from trainmail import send_mail

def train(epoch2,writer):
    net.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        if epoch2 <= args.warm:
            warmup_scheduler.step()

        since = time.time()

        images = Variable(images)
        labels = Variable(labels)

        labels = labels
        images = images

        if epoch2 <= args.warm:
            optimizer_warm.zero_grad()
        else:
            optimizer_mutistep.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        if epoch2 <= args.warm:
            optimizer_warm.step()
        else:
            optimizer_mutistep.step()

        n_iter = (epoch2 - 1) * len(cifar100_training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        if epoch2 <= args.warm:
            print('Training Epoch:{:3d}[{:5d}/{:5d}] Loss:{:0.4f} LR:{:0.6f} Time:{:2.2f}'.format(
                epoch2,
                batch_index * args.b + len(images),
                len(cifar100_training_loader.dataset),
                loss.item(),
                optimizer_warm.param_groups[0]['lr'],
                time.time()-since,
            ))
        else:
            print('Training Epoch:{:3d}[{:5d}/{:5d}] Loss:{:0.4f} LR:{:0.6f} Time:{:2.2f}'.format(
                epoch2,
                batch_index * args.b + len(images),
                len(cifar100_training_loader.dataset),
                loss.item(),
                optimizer_mutistep.param_groups[0]['lr'],
                time.time() - since,
            ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)
        if epoch2 <= args.warm:
            diary(path,"Step:{:d} Loss:{:0.4f} LR:{:0.6f}".format(n_iter,loss.item(),optimizer_warm.param_groups[0]['lr']))
        else:
            diary(path,"Step:{:d} Loss:{:0.4f} LR:{:0.6f}".format(n_iter, loss.item(), optimizer_mutistep.param_groups[0]['lr']))
        diary(path,"\n")


    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch2)

def eval_training(epoch,writer,content):
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0
    eval_time = time.time()

    for (images, labels) in cifar100_test_loader:
        images = Variable(images)
        labels = Variable(labels)

        images = images
        labels = labels

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f},EvalTime:{:.2f}s'.format(
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset),
        time.time() - eval_time
    ))
    # print()
    #add informations to tensorboard
    writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
    writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)
    diary(path,"Epoch:{:d} Average loss:{:0.4f} Acc:{:0.4f}".format(epoch,
    test_loss / len(cifar100_test_loader.dataset),correct.float() / len(cifar100_test_loader.dataset)))
    diary(path,'\n')
    content += ("Epoch:{:d} Average loss:{:0.4f} Acc:{:0.4f} ".format(epoch, test_loss / len(cifar100_test_loader.dataset),
                                                                      correct.float() / len(cifar100_test_loader.dataset)))
    return correct.float() / len(cifar100_test_loader.dataset),content

def diary(path,data):
    with open(path,'a') as f:
        f.write(str(data))
    f.close

if __name__ == '__main__':

    path = os.path.join("diary",settings.TIME_NOW+'_log.txt')
    content = ""
    subject = ""

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='vgg16',required=True, help='net type')
    parser.add_argument('-gpu', type=bool, default=False, help='use gpu or not')
    parser.add_argument('-w', type=int, default=1, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=2, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    args = parser.parse_args()

    net = get_network(args, use_gpu=args.gpu)
    if not os.path.exists("data"):
        os.mkdir("data")
    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )
    
    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )


    # loss_function = nn.CrossEntropyLoss()
    loss_function = FocalLoss(num_class=100)
    iter_per_epoch = len(cifar100_training_loader)
    optimizer_warm = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer_mutistep = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    warmup_scheduler = WarmUpLR(optimizer_warm, iter_per_epoch * args.warm)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_mutistep, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(os.path.join(settings.LOG_DIR, args.net, settings.TIME_NOW))
    input_tensor = Variable(torch.rand(12, 3, 32, 32))
    # with writer:
    writer.add_graph(net, (input_tensor,))

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')
    if not os.path.exists("diary"):
        os.makedirs("diary")

    best_acc = 0.0
    for epoch1 in range(1, settings.EPOCH):
        subject = "Epoch-" + str(epoch1) + " Train Report"
        batch_time = time.time()
        if epoch1 > args.warm:
            train_scheduler.step()
        diary(path,"Epoch:"+str(epoch1)+":\n")
        train(epoch1,writer)
        acc,content = eval_training(epoch1,writer,content)

        #start to save best performance model after learning rate decay to 0.01 
        if epoch1 > settings.MILESTONES[1] and best_acc < acc:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch1, type='best'))
            best_acc = acc
            continue

        if not epoch1 % settings.SAVE_EPOCH:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch1, type='regular'))
        batch_cost = time.time()-batch_time

        content += ("Batch Time:{:2d}mins{:.2f}s".format(int(int(batch_cost) / 60), int(batch_cost) % 60))
        send_mail(content, subject)
        content = ""
        print("*"*23+"Batch Time:{:2d}mins{:.2f}s".format(int(int(batch_cost)/60),int(batch_cost)%60)+"*"*23)
    writer.close()
