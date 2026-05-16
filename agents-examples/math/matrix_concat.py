import torch

# 模拟 8 个形状为 (3, 32) 的矩阵
matrix_list = [torch.randn(3, 32) for _ in range(8)]


## 2D Concat ##

# torch.cat 沿着 dim=1 (列) 进行拼接
result = torch.cat(matrix_list, dim=1)

# print(result)
print(result.shape)  # 输出: torch.Size([3, 256])


# torch.cat 沿着 dim=0 (row) 进行拼接
result = torch.cat(matrix_list, dim=0)

# print(result)
print(result.shape)  # 输出: torch.Size([24, 32])


## 3D Stack ##

# 使用 stack 沿着第 0 个维度（新建一个深度维度）进行堆叠
tensor_3d = torch.stack(matrix_list, dim=0)

print("堆叠后的形状:", tensor_3d.shape)
# 输出: 堆叠后的形状: torch.Size([32, 3, 128])