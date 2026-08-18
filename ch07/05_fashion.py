import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from common.load_data import get_fashion_data, get_device

batch_size = 32
lr = 0.01
epochs = 10
device = get_device()

# 1. 读取数据
x_train, x_test, y_train, y_test = get_fashion_data()

import matplotlib.pyplot as plt
# (c, h, w) => (h, w, c)
plt.imshow(x_train[0].permute(1, 2, 0).detach().numpy(), cmap='Grays')
plt.show()

# 2. 构建数据集 和数据加载器
train_ds = TensorDataset(x_train, y_train)
test_ds = TensorDataset(x_test, y_test)

train_dl = DataLoader(train_ds, batch_size=batch_size)
test_dl = DataLoader(test_ds, batch_size=batch_size)

# 3. 定义模式
# model = nn.Sequential(
#     nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
#     nn.Sigmoid(),
#     nn.AvgPool2d(kernel_size=2, stride=2),
#
#     nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
#     nn.Sigmoid(),
#     nn.AvgPool2d(kernel_size=2, stride=2),
#
#     nn.Flatten(),
#
#     nn.Linear(16 * 5 * 5, 120),
#     nn.Sigmoid(),
#
#     nn.Linear(120, 84),
#     nn.Sigmoid(),
#
#     nn.Linear(84, 10),
# )

import torchvision.models as models
model = models.resnet18(weights=None)
model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=3, stride=1, padding=1)
model.maxpool = nn.Identity()
model.fc = nn.Linear(model.fc.in_features, 10)
model.to(device)
#
# x = x_train
# for m in model:
#     x = m(x)
#     print(f"{m.__class__.__name__}: {x.shape}")

# 4. 定义优化器
optimizer = optim.Adam(model.parameters(), lr=lr)

# 5.定义损失函数
loss_fn = nn.CrossEntropyLoss()

# 6. 训练模型
for epoch in range(epochs):

    train_total_loss = 0
    train_acc_num = 0
    model.train()
    for input, target in train_dl:
        input, target = input.to(device), target.to(device)

        y_pred = model(input)

        loss = loss_fn(y_pred, target)

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        train_total_loss += loss.item() * input.shape[0]

        y_pred_class = torch.argmax(y_pred, dim=-1)
        train_acc_num += torch.sum(y_pred_class == target).item()

    tain_loss = train_total_loss / len(train_ds)
    tain_acc = train_acc_num / len(train_ds)

    val_total_loss = 0
    val_acc_num = 0
    model.eval()
    for input, target in test_dl:
        input, target = input.to(device), target.to(device)

        y_pred = model(input)

        loss = loss_fn(y_pred, target)

        val_total_loss += loss.item() * input.shape[0]

        y_pred_class = torch.argmax(y_pred, dim=-1)
        val_acc_num += torch.sum(y_pred_class == target).item()

    val_loss = val_total_loss / len(test_ds)
    val_acc = val_acc_num / len(test_ds)
    # 输出训练损失和训练准确率 验证损失和验证准确率
    print(f" tain_loss:{tain_loss:.6f}, train_acc:{tain_acc:.6f}, val_loss:{val_loss:.6f}, val_acc:{val_acc:.6f}")

