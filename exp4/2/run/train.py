# train.py
#!/usr/bin/env	python3

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from conf import settings
from utils import get_training_dataloader, get_test_dataloader
from trainmail import send_mail
import time
from FocalLoss import FocalLoss

def train(epoch2,writer,net):
    net.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):

        since = time.time()

        images = Variable(images)
        labels = Variable(labels)

        labels = labels
        images = images


        optimizer_mutistep.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer_mutistep.step()
        n_iter = (epoch2 - 1) * len(cifar100_training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)


        print('Training Epoch:{:3d}[{:5d}/{:5d}] Loss:{:0.4f} LR:{:0.6f} Time:{:.2f}'.format(
            epoch2,
            batch_index * args.b + len(images),
            len(cifar100_training_loader.dataset),
            loss.item(),
            optimizer_mutistep.param_groups[0]['lr'],
            time.time() - since,
            ))

        #update training loss for each iteration
        if not n_iter % 20:
            writer.add_scalar('Train/loss', loss.item(), n_iter)
            diary(path,"Step:{:d} Loss:{:0.4f} LR:{:0.6f}".format(n_iter, loss.item(), optimizer_mutistep.param_groups[0]['lr']))
            diary(path,"\n")

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch2)

def eval_training(epoch,writer,net,content):
    net.eval()
    print("."*23+"Eval Starting"+"."*23)
    eval_time = time.time()

    test_loss = 0.0 # cost function error
    correct = 0.0
    # i=0

    for (images, labels) in cifar100_test_loader:
        # print(i)
        # i+=1
        images = Variable(images)
        labels = Variable(labels)

        images = images
        labels = labels

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Eval Time: {:.1f}'.format(
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset),
        time.time()-eval_time
    ))

    train_scheduler.step(correct.float() / len(cifar100_test_loader.dataset))

    writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
    writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)
    diary(path,"Epoch:{:d} Average loss:{:0.4f} Acc:{:0.4f}".format(epoch,
    test_loss / len(cifar100_test_loader.dataset),correct.float() / len(cifar100_test_loader.dataset)))
    diary(path,'\n')
    content += (
        "Epoch:{:d} Average loss:{:0.4f} Acc:{:0.4f} ".format(epoch, test_loss / len(cifar100_test_loader.dataset),
                                                              correct.float() / len(cifar100_test_loader.dataset)))

    return correct.float() / len(cifar100_test_loader.dataset),content

def diary(path,data):
    with open(path,'a') as f:
        f.write(str(data))
    f.close

if __name__ == '__main__':

    path = os.path.join("diary","EfficcientNet"+"_log.txt")
    content = ""
    subject = ""

    parser = argparse.ArgumentParser()
    # parser.add_argument('-net', type=str, default='vgg16',required=True, help='net type')
    parser.add_argument('-gpu', type=bool, default=False, help='use gpu or not')
    parser.add_argument('-w', type=int, default=1, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=8, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('-start_epoch', type=int, default=1, help='epoch to start')
    parser.add_argument('-type', type=str , default="regular", help='type of checkpoint')
    args = parser.parse_args()
    args.net = "EfficientNet"

    torch.hub.list('rwightman/gen-efficientnet-pytorch')
    net = torch.hub.load('rwightman/gen-efficientnet-pytorch','efficientnet_b0',pretrained=True)

    feature = net.classifier.in_features
    net.classifier = nn.Linear(in_features=feature,out_features=100,bias=True)
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists("diary"):
        os.makedirs("diary")

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
        batch_size=4,
        shuffle=False
    )

    # loss_function = nn.CrossEntropyLoss()
    loss_function = FocalLoss(num_class=100)
    iter_per_epoch = len(cifar100_training_loader)
    optimizer_mutistep = optim.Adam(net.parameters(), lr=args.lr,betas=(0.9, 0.999), eps=1e-9)
    train_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_mutistep,mode='max',factor=0.7,verbose=1,min_lr=1e-6,patience=2)
    checkpoint_path1 = os.path.join(settings.CHECKPOINT_PATH, args.net)


    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(os.path.join(settings.LOG_DIR, "EfficientNet"))
    if not os.path.exists(checkpoint_path1):
        os.makedirs(checkpoint_path1)

    checkpoint_path = os.path.join(checkpoint_path1, '{net}-{epoch}-{type}.pth')
    files_to_load = args.net+"-"+str(args.start_epoch)+"-"+args.type+".pth"
    pth_to_load = os.path.join(checkpoint_path1,files_to_load)
    print(pth_to_load)

    if os.path.exists(pth_to_load):
        checkpoint = torch.load(pth_to_load)
        net.load_state_dict(checkpoint['net'])
        optimizer_mutistep.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        # train_scheduler.load_state_dict(checkpoint['train_scheduler'])
        print("Epoch {} Loaded!".format(start_epoch))
    else:
        start_epoch = 0
        print("No Model Saved!")


    best_acc = 0.0
    for epoch1 in range(start_epoch+1, settings.EPOCH+1):
        subject = "Epoch-" + str(epoch1) + " Train Report"
        batch_time = time.time()
        diary(path,"Epoch:"+str(epoch1)+":\n")

        train(epoch1,writer,net)
        acc,content = eval_training(epoch1,writer,net,content)

        state = {'net':net.state_dict(),'optimizer':optimizer_mutistep.state_dict(),'epoch':epoch1}
        if epoch1 > settings.MILESTONES[1] and best_acc < acc:
            torch.save(state, checkpoint_path.format(net=args.net, epoch=epoch1, type='best'))
            best_acc = acc
            continue

        batch_cost = time.time() - batch_time
        content += ("Batch Time:{:2d}mins{:.2f}s\n".format(int(int(batch_cost) / 60), int(batch_cost) % 60))

        if not epoch1 % settings.SAVE_EPOCH:
            torch.save(state, checkpoint_path.format(net=args.net, epoch=epoch1, type='regular'))

        if not epoch1 % 1:
            send_mail(content,subject)
            send_mail(content,subject)
            content=""

        print("*"*23+"Batch Time:{:2d}mins{:.2f}s".format(int(int(batch_cost)/60), int(batch_cost) % 60)+
              "*"*23)

        if not epoch1%1:
            time.sleep(1)

    writer.close()

