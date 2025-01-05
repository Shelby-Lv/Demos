import os
import time

import cv2
import einops
import numpy as np
import torch
import torch.nn as nn

from dldemos.ddpm.dataset import get_dataloader, get_img_shape
from dldemos.ddpm.ddpm_simple import DDPM
from dldemos.ddpm.network import (build_network, convnet_big_cfg,
                                  convnet_medium_cfg, convnet_small_cfg,
                                  unet_1_cfg, unet_res_cfg)

batch_size = 512
n_epochs = 100


def train(ddpm: DDPM, net, device='cuda:3', ckpt_path='dldemos/ddpm/model.pth'):
    print('batch size:', batch_size)
    n_steps = ddpm.n_steps#扩散过程的总步数
    dataloader = get_dataloader(batch_size)#获取数据加载器
    net = net.to(device)#将网络加载到指定设备
    loss_fn = nn.MSELoss()#获取损失函数，此处使用MSE
    optimizer = torch.optim.Adam(net.parameters(), 1e-3)#使用Adam优化器

    tic = time.time()#获取时间
    for e in range(n_epochs):#每一个epoch（总共100个）
        total_loss = 0#累计损失

        for x, _ in dataloader:#x对应图像数据，_对应标签，此处取出图片
            current_batch_size = x.shape[0]#获取当前批次大小
            x = x.to(device)#将数据移动到GPU
            t = torch.randint(0, n_steps, (current_batch_size, )).to(device)#生成随机时间步长，移动到GPU
            eps = torch.randn_like(x).to(device)#生成与x大小相同的随机噪声
            x_t = ddpm.sample_forward(x, t, eps)#正向扩散，生成含噪图像
            eps_theta = net(x_t, t.reshape(current_batch_size, 1))#网络预测噪声
            loss = loss_fn(eps_theta, eps)#计算损失
            optimizer.zero_grad()#梯度清零
            loss.backward()#反向传播
            optimizer.step()#更新模型参数
            total_loss += loss.item() * current_batch_size#累计当前批次的损失
        total_loss /= len(dataloader.dataset)#计算平均损失
        toc = time.time()#获取当前时间
        torch.save(net.state_dict(), ckpt_path)#保存模型
        print(f'epoch {e} loss: {total_loss} elapsed {(toc - tic):.2f}s')
    print('Done')


def sample_imgs(ddpm,#生成图像函数
                net,
                output_path,
                n_sample=81,
                device='cuda:3',
                simple_var=True):
    net = net.to(device)#移动到指定设备
    net = net.eval()#设置网络评估模式
    with torch.no_grad():#禁用梯度计算，加速推理
        shape = (n_sample, *get_img_shape())  # 1, 3, 28, 28 定义生成图像形状
        imgs = ddpm.sample_backward(shape,
                                    net,
                                    device=device,
                                    simple_var=simple_var).detach().cpu()#反向扩散生成图片
        imgs = (imgs + 1) / 2 * 255#像素值从[-1,1]变换到[0,255]
        imgs = imgs.clamp(0, 255)#限制像素范围
        imgs = einops.rearrange(imgs,
                                '(b1 b2) c h w -> (b1 h) (b2 w) c',
                                b1=int(n_sample**0.5))#调整图像布局

        imgs = imgs.numpy().astype(np.uint8)#转化为8位无符号整数（图像格式）

        cv2.imwrite(output_path, imgs)#保存图像


configs = [
    convnet_small_cfg, convnet_medium_cfg, convnet_big_cfg, unet_1_cfg,
    unet_res_cfg
]#定义不同的网络配置

if __name__ == '__main__':
    os.makedirs('work_dirs', exist_ok=True)

    n_steps = 1000
    config_id = 4#使用unet_res_cfg
    device = 'cuda:3'#使用GPU
    model_path = 'dldemos/ddpm/model_unet_res.pth'#模型保存路径

    config = configs[config_id]#选择网络配置
    net = build_network(config, n_steps)#构建网络
    ddpm = DDPM(device, n_steps)#初始化DDPM模型

    train(ddpm, net, device=device, ckpt_path=model_path)#训练模型

    net.load_state_dict(torch.load(model_path))#加载训练好的模型
    sample_imgs(ddpm, net, 'work_dirs/diffusion.jpg', device=device)
