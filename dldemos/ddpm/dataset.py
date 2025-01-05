import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda, ToTensor


def download_dataset():
    mnist = torchvision.datasets.MNIST(root='./dldemos/ddpm/data/mnist', download=True)#下载手写数字数据集
    print('length of MNIST', len(mnist))#打印数据集的总长度（60000）
    id = 4#选中数据集的第四个样本
    img, label = mnist[id]
    print(img)
    print(label)

    # On computer with monitor
    # img.show()

    img.save('work_dirs/tmp.jpg')#保存图片对象
    tensor = ToTensor()(img)#将图像转化为tensor格式
    print(tensor.shape)
    print(tensor.max())
    print(tensor.min())


def get_dataloader(batch_size: int):#数据预处理流程
    transform = Compose([ToTensor(), Lambda(lambda x: (x - 0.5) * 2)])#将PIL图像转化为Tensor，将像素值从[0,1]转化到[-1,1]
    dataset = torchvision.datasets.MNIST(root='./dldemos/ddpm/data/mnist',
                                         transform=transform)#加载数据集并应用预处理操作
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)#使用DataLoader对数据进行批量加载，shuffle表示支持随机打乱


def get_img_shape():
    return (1, 28, 28)


if __name__ == '__main__':
    import os
    os.makedirs('work_dirs', exist_ok=True)
    download_dataset()#调用下载函数
