# loss-chapter
**该项目为书中对应实验部分代码分享**
>###实验环境
>* Linux Mint 19.1
>* python 3.7.7
>* pytorch-cpu 1.1.0
>* tensorboardX 1.7

>###文件介绍与注意
>实验序号为本章节中对应实验序号，实验四即损失函数对比实验。center loss参考[这里](https://github.com/KaiyangZhou/pytorch-center-loss/blob/master/center_loss.py),
>focal loss参考[这里](https://blog.csdn.net/qq_33278884/article/details/91572173)。<br/>
>losses文件夹中的损失函数可用于实验二三四中，需将其放在对应文件夹下。<br>
>实验数据二所用数据MNIST可由程序自动下载，或者到[官网](http://yann.lecun.com/exdb/mnist/)下载，实验四使用CIFAR100可以在[官网](https://www.cs.toronto.edu/~kriz/cifar.html)下载。<br/>
>实验一参考[这里](https://www.jianshu.com/p/331a995774d8)，每次运行结果略有不同但总体符合趋势，可单独打印显示损失变化趋势。<br/>
>实验二参考代码与特征打印代码参考自网络，出处遗失。提供程序使用交叉熵优化，使用center loss可将文件重新保存更改损失，并打印出每个批次的特征图像，gif_generator.py可将一组图像生成动态图。
>实验三与实验二代码基本相同，我们选用了LeNet网络，训练数据需要自己亲手设计，尽量使两类数据比例尽可能大。<br/>
>实验四主要代码参考[这里](https://www.cs.toronto.edu/~kriz/cifar.html)并做出适应性改变，其中1中包含trainmail可以在一定epoch后发送到指定电子邮箱，但需要进行设置，EfficientNet所用模型来自[这里](https://github.com/rwightman/gen-efficientnet-pytorch)。<br/>

_写作、整理仓促，难免有不恰当之处，欢迎交流！_