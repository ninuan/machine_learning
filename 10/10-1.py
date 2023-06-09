import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

# Hyper Parameters
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False

# Mnist digits dataset
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    # 将下载下来的文件转换成Tensor，且将数值标准化到[0,1]之间
    download=DOWNLOAD_MNIST,
)

# plot one example
# print(train_data.data.size())  # (60000, 28, 28)
# print(train_data.targets.size())  # (60000)
# plt.imshow(train_data.data[1].numpy(), cmap='gray')
# plt.title('%i' % train_data.targets[1])
# plt.show()

# Data Loader for easy mini-batch return in training
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    # shuffle=True表示每个epoch都打乱数据集
)

test_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=False,
)
with torch.no_grad():
    test_x = Variable(torch.unsqueeze(test_data.data, dim=1)).type(torch.FloatTensor)[:2000] / 255.
    # 将数据转换成Variable，然后将数据类型转换成FloatTensor，最后将数据标准化到[0,1]之间
    test_y = test_data.targets[:2000]
# q:no_grad()的作用
# a:在测试集上不需要进行反向传递，所以可以加上这个函数来加快运算速度

# 开始建立CNN网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        '''
        一般来说，卷积网络包括一下内容：
        1、卷积层
        2、神经网络
        3、池化层
        '''
        self.conv1 = nn.Sequential(
            # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # 输入的高度，因为是灰度图，所以是1
                out_channels=16,  # 输出的高度，可以理解为卷积核的个数
                kernel_size=5,  # 卷积核的大小，代表扫描的区域为5*5
                stride=1,  # 卷积核的步长,就是每隔多少步跳一次
                padding=2,  # 如果想要conv2d出来的图片长宽没有变化，就要padding=(kernel_size-1)/2
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(kernel_size=2),  # 池化层，池化核的大小为2*2
            # output shape (16, 14, 14)
        )
        #q:激活层的作用是什么？
        #a:激活层的作用是为了给网络加入一些非线性因素，使得网络可以拟合更加复杂的函数

        self.conv2 = nn.Sequential(
            # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )

        self.out = nn.Linear(32 * 7 * 7, 10)  # 全连接层，输出为10个类别

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # 将三维数据转化为二维数据
        x = x.view(x.size(0), -1)  # (batch_size, 32 * 7 * 7)
        #q:view的参数含义及作用，以及-1的作用
        #a:view的参数是将数据转换成什么样的形式，-1的作用是自动计算该位置的数据
        output = self.out(x)
        return output

cnn = CNN()
# print(cnn)

# 添加优化方法
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
#q:优化方法的作用是什么？
#a:优化方法的作用是为了找到最优的参数，使得损失函数最小
# 指定损失函数使用交叉熵损失函数
loss_fn = nn.CrossEntropyLoss()

# 开始训练
step = 0
for epoch in range(EPOCH):
    # 加载训练数据
    for step,data in enumerate(train_loader):
        x,y = data
        # 分别得到训练数据和标签
        b_x = Variable(x)
        b_y = Variable(y)
        # 将数据转换成Variable

        output = cnn(b_x) # 调用模型预测
        loss = loss_fn(output, b_y) # 计算损失值
        optimizer.zero_grad() # 清空上一步的残余更新参数值
        loss.backward() # 误差反向传播，计算参数更新值
        optimizer.step() # 将参数更新值施加到net的parameters上，梯度下降

        # 每50步打印一次训练结果，输出当下的epoch，loss，accuracy
        if step % 50 == 0:
            test_output = cnn(test_x)
            y_pred = torch.max(test_output, 1)[1].data.squeeze()
            # q:y_pred输出的是什么？
            # a:y_pred输出的是预测的标签
            accuracy = sum(y_pred == test_y) / float(test_y.size(0))

            print('Epoch: ', epoch, '| train loss: %.4f' % loss.item(), '| test accuracy: %.2f' % accuracy)

# 保存模型
# torch.save(cnn.state_dict(), './cnn.pth')
# # 加载模型
# cnn.load_state_dict(torch.load('./cnn.pth'))

# 打印十个测试集的结果
test_output = cnn(test_x[:10])
y_pred = torch.max(test_output, 1)[1].data.numpy().squeeze()
print('predecton Result', y_pred.tolist())
print('Real Result', test_y[:10].tolist())