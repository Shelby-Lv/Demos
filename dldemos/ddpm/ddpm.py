import torch


class DDPM():

    def __init__(self,
                 device,
                 n_steps: int,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.02):
        betas = torch.linspace(min_beta, max_beta, n_steps).to(device)#生成一段线性beta序列
        alphas = 1 - betas

        alpha_bars = torch.empty_like(alphas)#计算α累乘
        product = 1
        for i, alpha in enumerate(alphas):
            product *= alpha
            alpha_bars[i] = product

        self.betas = betas
        self.n_steps = n_steps
        self.alphas = alphas
        self.alpha_bars = alpha_bars#保存得到的变量

        alpha_prev = torch.empty_like(alpha_bars)#alpha_prev[i]=alpha_bars[i-1]，在反向扩散时用到
        alpha_prev[1:] = alpha_bars[0:n_steps - 1]
        alpha_prev[0] = 1
        self.coef1 = torch.sqrt(alphas) * (1 - alpha_prev) / (1 - alpha_bars)#反向过程中的两个系数
        self.coef2 = torch.sqrt(alpha_prev) * self.betas / (1 - alpha_bars)

    def sample_forward(self, x, t, eps=None):#前向过程
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)#获取当前时间步t的alpha累乘值
        if eps is None:
            eps = torch.randn_like(x)#随机产生噪声
        res = eps * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar) * x#前向加噪公式
        return res

    def sample_backward(self,#反向过程
                        img_shape,
                        net,#预测噪声的net模型
                        device,
                        simple_var=True,#使用简化的方差
                        clip_x0=True):#是否对x_0进行裁剪
        x = torch.randn(img_shape).to(device)#初始化一个随机噪声图像
        net = net.to(device)
        for t in range(self.n_steps - 1, -1, -1):#从t=n_step-1到t=0，逐步进行反向扩散
            x = self.sample_backward_step(x, t, net, simple_var, clip_x0)
        return x

    def sample_backward_step(self, x_t, t, net, simple_var=True, clip_x0=True):#反向扩散的单步操作

        n = x_t.shape[0]#获取当前维度
        t_tensor = torch.tensor([t] * n,
                                dtype=torch.long).to(x_t.device).unsqueeze(1)#创建当前时间步长的张量
        eps = net(x_t, t_tensor)#神经网络预测噪声

        if t == 0:
            noise = 0
        else:
            if simple_var:
                var = self.betas[t]#使用简易方差
            else:
                var = (1 - self.alpha_bars[t - 1]) / (
                    1 - self.alpha_bars[t]) * self.betas[t]#更复杂的方差公式
            noise = torch.randn_like(x_t)#生成随机噪声
            noise *= torch.sqrt(var)#噪声乘以标准差

        if clip_x0:#如果需要裁剪，先恢复x_0，然后进行裁剪
            x_0 = (x_t - torch.sqrt(1 - self.alpha_bars[t]) *
                   eps) / torch.sqrt(self.alpha_bars[t])
            x_0 = torch.clip(x_0, -1, 1)#裁剪至[-1,1]范围
            mean = self.coef1[t] * x_t + self.coef2[t] * x_0#更新图像
        else:
            mean = (x_t -
                    (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) *
                    eps) / torch.sqrt(self.alphas[t])#直接计算均值
        x_t = mean + noise

        return x_t


def visualize_forward():#可视化前向扩散过程
    import cv2
    import einops
    import numpy as np

    from dldemos.ddpm.dataset import get_dataloader

    n_steps = 100
    device = 'cuda:3'
    dataloader = get_dataloader(5)
    x, _ = next(iter(dataloader))
    x = x.to(device)

    ddpm = DDPM(device, n_steps)
    xts = []#不同步长时间步数的图像
    percents = torch.linspace(0, 0.99, 10)
    for percent in percents:#对于每个时间步，执行前向扩散，并将结果存储起来
        t = torch.tensor([int(n_steps * percent)])#计算对应的时间步长
        t = t.unsqueeze(1)
        x_t = ddpm.sample_forward(x, t)#执行前向扩散
        xts.append(x_t)#加入到xts列表中
    res = torch.stack(xts, 0)
    res = einops.rearrange(res, 'n1 n2 c h w -> (n2 h) (n1 w) c')#堆叠排列
    res = (res.clip(-1, 1) + 1) / 2 * 255
    res = res.cpu().numpy().astype(np.uint8)#存储

    cv2.imwrite('work_dirs/diffusion_forward.jpg', res)


def main():
    visualize_forward()


if __name__ == '__main__':
    main()
