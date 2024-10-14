import torch


##计算两个向量的余弦相似度
def normalize_vector(x, y):
    x=torch.Tensor(x)
    y=torch.Tensor(y)
    x, y = normalize(x), normalize(y)
    return torch.dot(x, y).tolist()


# 矩阵的归一化处理【l2范数值为1】
def normalize(x, axis=-1):
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


##特征向量a
a = torch.rand(4, 512)

##特征向量b
b = torch.rand(6, 512)

##特征向量进行归一化
a, b = normalize(a), normalize(b)

##矩阵乘法求余弦相似度
cos = 1 - torch.mm(a, b.permute(1, 0))
cos.shape

# 输出
torch.Size([4, 6])
