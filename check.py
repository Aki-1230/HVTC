import torch
import torch.nn as nn

class CrossStitchUnit(nn.Module):
    def __init__(self, tasks, num_channels):
        super().__init__()
        self.tasks = tasks
        T = len(tasks)

        # A shape: (T, T, C)
        self.A = nn.Parameter(torch.zeros(T, T, num_channels))

        # 初始化一些不同的值，便于观察
        with torch.no_grad():
            # task0 行
            self.A[0,0] = torch.tensor([2.0, 1.0, -1.0])   # 自己
            self.A[0,1] = torch.tensor([0.5, -0.5, 1.5])   # 来自 task1

            # task1 行
            self.A[1,0] = torch.tensor([-1.0, 0.0, 2.0])   # 来自 task0
            self.A[1,1] = torch.tensor([1.0, 2.0, -0.5])   # 自己

    def forward(self, task_features):
        # Softmax 归一化 (dim=1)
        A_norm = torch.softmax(self.A, dim=1)

        print("=== Softmax 前 A ===")
        print(self.A.detach())

        print("\n=== Softmax 后 A_norm ===")
        print(A_norm.detach())

        outs = {}
        for i, t_i in enumerate(self.tasks):
            out_i = 0
            for j, t_j in enumerate(self.tasks):
                w = A_norm[i, j].view(1, -1, 1, 1)
                out_i = out_i + w * task_features[t_j]
            outs[t_i] = out_i

        return outs


# ================= 运行演示 =================

tasks = ['tran', 'seg']
num_channels = 3
cs = CrossStitchUnit(tasks, num_channels)

# 构造两个任务的简单特征：都是 shape (1, 3, 1, 1)
feat_tran = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])  # (1,3,1,1)
feat_seg  = torch.tensor([[[[4.0]], [[5.0]], [[6.0]]]])  # (1,3,1,1)

outs = cs({
    'tran': feat_tran,
    'seg': feat_seg
})

print("\n=== 输出 tran ===")
print(outs['tran'])

print("\n=== 输出 seg ===")
print(outs['seg'])
