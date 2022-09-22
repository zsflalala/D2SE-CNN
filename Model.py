import megengine as mge
import megengine.module as M
import megengine.functional as F
import math

# D2SE_CNN-Net
class D2SE_CNN(M.Module):
    def __init__(self):
        super(D2SE_CNN, self).__init__()
        # self.lamb = lambda x:( x + M.init.fill_(x,1e-7))
        self.Conv_BN_ReLU1 = M.Sequential(
            M.Conv2d(64, 64, 3, dilation=1, padding=1, stride=1),
            M.BatchNorm2d(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, padding=1),
            M.BatchNorm2d(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, padding=1),
            M.BatchNorm2d(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, padding=1),
            M.BatchNorm2d(64),
            M.ReLU()
        )
        self.Conv_BN_ReLU2 = M.Sequential(
            M.Conv2d(64, 64, 3, padding=1),
            M.BatchNorm2d(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, padding=1),
            M.BatchNorm2d(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, padding=1),
            M.BatchNorm2d(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, padding=1),
            M.BatchNorm2d(64),
            M.ReLU()
        )
        self.Conv_BN_ReLU3 = M.Sequential(
            M.Conv2d(64, 64, 3, padding=1),
            M.BatchNorm2d(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, padding=1),
            M.BatchNorm2d(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, padding=1),
            M.BatchNorm2d(64),
            M.ReLU()
        )
        self.Conv_BN_ReLU = M.Sequential(
            M.Conv2d(64, 64, 3, dilation=1, padding=1),
            M.BatchNorm2d(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, dilation=2, padding=2),
            M.BatchNorm2d(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, dilation=2, padding=2),
            M.BatchNorm2d(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, dilation=2, padding=2),
            M.BatchNorm2d(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, dilation=3, padding=3),
            M.BatchNorm2d(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, dilation=3, padding=3),
            M.BatchNorm2d(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, dilation=2, padding=2),
            M.BatchNorm2d(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, dilation=2, padding=2),
            M.BatchNorm2d(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, dilation=2, padding=2),
            M.BatchNorm2d(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, dilation=1, padding=1),
            M.BatchNorm2d(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, dilation=1, padding=1),
            M.BatchNorm2d(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, dilation=1, padding=1),
            M.BatchNorm2d(64),
            M.ReLU(),
            M.Conv2d(64, 64, 3, dilation=1, padding=1),
            M.BatchNorm2d(64),
            M.ReLU())
        self.dia = M.Sequential(
            M.Conv2d(1, 4, 3, dilation=65, padding=1)

        )
        self.Conv_ReLU_L1 = M.Sequential(
            M.Conv2d(4, 64, 3, padding=1),
            M.ReLU()
        )
        self.Conv_ReLU_L8 = M.Sequential(
            M.Conv2d(64, 4, 3, padding=1)
        )
        self.ConvTrans = M.ConvTranspose2d(4, 4, 4, stride=2, padding=1)
        self.proj = M.Conv2d(1, 1, 5, padding=2, stride=1)
        self.avg_pool = M.AdaptiveAvgPool2d(1)
        self.fc = M.Sequential(
            M.Linear(64, 4, bias=False),
            M.ReLU(),
            M.Linear(4, 64, bias=False),
            M.Sigmoid()
        )
        self.apply(self._init_weights)


    # 初始化各层参数
    def _init_weights(self, m):
        if isinstance(m, M.Conv2d):
            # print("Conv2d")
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            M.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
            # M.init.msra_normal_(m.weight,mode='fan_in')
            if m.bias is not None:
                M.init.zeros_(m.bias)
        if isinstance(m, M.BatchNorm2d):
            # print("Batch")
            M.init.msra_normal_(m.weight, mode='fan_in')
            if m.bias is not None:
                M.init.zeros_(m.bias)
        if isinstance(m, M.LayerNorm):
            # print("layerNorm")
            M.init.zeros_(m.bias)
            M.init.ones_(m.weight)
        if isinstance(m, M.Linear):
            # print("M.linear")
            M.init.normal_(m.weight, std=.02)
            if isinstance(m, M.Linear) and m.bias is not None:
                M.init.zeros_(m.bias)


    def forward(self, x):
        n, c, h, w = x.shape
        # 降采样
        x = x.reshape(n, c, h // 2, 2, w // 2, 2).transpose(0, 1, 3, 5, 2, 4)
        x = x.reshape(n, c * 4, h // 2, w // 2)
        x = self.Conv_ReLU_L1(x)
        # SE 通道注意力
        b, c2, _, _ = x.shape
        y = F.reshape(self.avg_pool(x), (b, c2))  # squeeze操作
        y = F.reshape(self.fc(y), (b, c2, 1, 1))  # FC获取通道注意力权重，是具有全局信息的
        x = x * F.broadcast_to(y, (x.shape))  # 注意力作用每一个通道上
        x = self.Conv_BN_ReLU1(x)
        b, c2, _, _ = x.shape
        y = F.reshape(self.avg_pool(x), (b, c2))  # squeeze操作
        y = F.reshape(self.fc(y), (b, c2, 1, 1))  # FC获取通道注意力权重，是具有全局信息的
        x = x * F.broadcast_to(y, (x.shape))  # 注意力作用每一个通道上
        x = self.Conv_BN_ReLU2(x)
        b, c2, _, _ = x.shape
        y = F.reshape(self.avg_pool(x), (b, c2))  # squeeze操作
        y = F.reshape(self.fc(y), (b, c2, 1, 1))  # FC获取通道注意力权重，是具有全局信息的
        x = x * F.broadcast_to(y, (x.shape))  # 注意力作用每一个通道上
        x = self.Conv_BN_ReLU3(x)
        # x = self.Conv_BN_ReLU(x)
        x = self.Conv_ReLU_L8(x)
        x = x.reshape(n, c, 2, 2, h // 2, w // 2).transpose(0, 1, 4, 2, 5, 3)
        x = x.reshape(n, c, h, w)
        return x
