import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

from common.load_data import get_digital_data, get_device

batch_size = 64
epochs = 10
lr = 0.1
device = get_device()

print("device:", device)

# 1. 读入数据
x_train, x_test, y_train, y_test = get_digital_data()

# (N, 784)  => (N, C, H, W)  更改数据shape
x_train = x_train.reshape(-1, 1, 28, 28)
x_test = x_test.reshape(-1, 1, 28, 28)

# 2. 构建数据集和数据加载器
train_ds = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

validate_ds = TensorDataset(x_test, y_test)
validate_loader = DataLoader(validate_ds, batch_size=batch_size, shuffle=True)


# 3. 定义模型
model = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1, stride=1),  # (N, C, H, W)
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),  # 输出减半

    nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),  # 输出减半

    nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1),
    nn.ReLU(),

    # (N, C, H, w)   => (N, *)
    nn.Flatten(),

    nn.Linear(32 * 7 * 7, 128),
    nn.ReLU(),

    nn.Linear(128, 10)
)

model.to(device)  # 把模型放入到gpu

# 5. 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

# 6.训练模型
for epoch in range(epochs):

    train_loss_total = 0
    train_acc_num = 0

    model.train()
    for input, target in train_loader:
        input, target = input.to(device), target.to(device)

        # 5.1 一次前向传播
        y_pred = model(input)
        # 5.2 计算损失
        loss = loss_fn(y_pred, target)
        # 5.3 反向传播计算梯度
        loss.backward()
        # 5.4 更新参数
        optimizer.step()
        # 5.5 清零梯度(叶子结点的梯度)
        optimizer.zero_grad()

        # 计算总损失
        train_loss_total += loss.item() * input.shape[0]

        # 计算预测的类别
        y_pred_class = torch.argmax(model(input), dim=-1)
        # 预测准确的个数
        train_acc_num += (y_pred_class == target).sum().item()

    # 这个epoch的平均损失
    train_loss = train_loss_total / len(train_ds)
    # 准确率
    train_acc = train_acc_num / len(train_ds)

    val_loss_total = 0
    val_acc_num = 0

    model.eval()
    for input, target in validate_loader:
        input, target = input.to(device), target.to(device)

        # 5.1 一次前向传播
        y_pred = model(input)
        # 5.2 计算损失
        loss = loss_fn(y_pred, target)

        # 计算总损失
        val_loss_total += loss.item() * input.shape[0]

        # 计算准确率
        y_pred_class = torch.argmax(model(input), dim=-1)
        # 预测准确的个数
        val_acc_num += (y_pred_class == target).sum().item()

    # 这个epoch的平均损失
    val_loss = val_loss_total / len(validate_ds)
    # 准确率
    val_acc = val_acc_num / len(validate_ds)

    print(
        f"训练{epoch + 1}: 损失:{train_loss:.4f},准确率:{train_acc:.4f} 验证: 损失:{val_loss:.4f},准确率:{val_acc:.4f},")
