import random
import torch
import torch.nn as nn
import torch.optim as optim
from d2l import torch as d2l
import matplotlib.pyplot as plt
import numpy as np

# 1. 生成合成数据
def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声 的合成数据"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)  # 添加高斯噪声
    return X, y.reshape((-1, 1))

# 真实参数
true_w = torch.tensor([2.0, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# 2. 数据可视化
def visualize_data(features, labels):
    """可视化特征与标签的关系"""
    d2l.set_figsize()
    
    # 创建子图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 第一个特征 vs 标签
    axes[0].scatter(features[:, 0].detach().numpy(), 
                   labels.detach().numpy(), 
                   alpha=0.5, s=1)
    axes[0].set_xlabel('特征 1 (x₁)')
    axes[0].set_ylabel('标签 (y)')
    axes[0].set_title('特征 1 与标签的关系')
    
    # 第二个特征 vs 标签
    axes[1].scatter(features[:, 1].detach().numpy(), 
                   labels.detach().numpy(), 
                   alpha=0.5, s=1, color='orange')
    axes[1].set_xlabel('特征 2 (x₂)')
    axes[1].set_ylabel('标签 (y)')
    axes[1].set_title('特征 2 与标签的关系')
    
    plt.tight_layout()
    plt.show()

visualize_data(features, labels)

# 3. 数据迭代器
def data_iter(batch_size, features, labels):
    """随机批次数据迭代器"""
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    
    for i in range(0, num_examples, batch_size):
        batch_indices = indices[i: min(i + batch_size, num_examples)]
        batch_indices = torch.tensor(batch_indices)
        yield features[batch_indices], labels[batch_indices]

# 4. 测试数据迭代器
batch_size = 10
print(f"\n批次大小: {batch_size}")
print("第一个批次的数据:")

for batch_idx, (X_batch, y_batch) in enumerate(data_iter(batch_size, features, labels)):
    print(f"批次 {batch_idx + 1}:")
    print(f"  特征形状: {X_batch.shape}")
    print(f"  标签形状: {y_batch.shape}")
    print(f"  特征样本:\n{X_batch}")
    print(f"  对应标签:\n{y_batch}")
    break  # 只显示第一个批次

# 5. 简单线性回归模型
class LinearRegressionModel(nn.Module):
    """简单的线性回归模型"""
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.linear(x)

# 6. 初始化模型
input_dim = features.shape[1]
model = LinearRegressionModel(input_dim)
print(f"\n模型结构: {model}")
print(f"初始权重: {model.linear.weight.data}")
print(f"初始偏置: {model.linear.bias.data}")

# 7. 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 8. 训练模型
num_epochs = 10
batch_size = 32

print(f"\n开始训练...")
print(f"训练轮数: {num_epochs}")
print(f"批次大小: {batch_size}")

train_losses = []

for epoch in range(num_epochs):
    total_loss = 0
    batch_count = 0
    
    # 遍历所有批次
    for X_batch, y_batch in data_iter(batch_size, features, labels):
        # 前向传播
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
    
    # 计算平均损失
    avg_loss = total_loss / batch_count
    train_losses.append(avg_loss)
    
    # 打印训练进度
    if (epoch + 1) % 2 == 0 or epoch == 0:
        print(f'轮次 [{epoch + 1}/{num_epochs}], 损失: {avg_loss:.6f}')

# 9. 评估训练结果
trained_w = model.linear.weight.data.squeeze()
trained_b = model.linear.bias.data.item()

print(f"\n训练结果:")
print(f"真实权重: w={true_w.tolist()}, b={true_b:.4f}")
print(f"训练权重: w={trained_w.tolist()}, b={trained_b:.4f}")
print(f"权重误差: {torch.abs(true_w - trained_w).tolist()}")
print(f"偏置误差: {abs(true_b - trained_b):.4f}")

# 10. 损失曲线可视化
plt.figure(figsize=(8, 4))
plt.plot(range(1, num_epochs + 1), train_losses, 'b-', linewidth=2)
plt.xlabel('训练轮次')
plt.ylabel('平均损失 (MSE)')
plt.title('训练损失曲线')
plt.grid(True, alpha=0.3)
plt.show()

# 11. 预测示例
print(f"\n预测示例:")
test_indices = [0, 5, 10]  # 测试前几个样本
for idx in test_indices:
    with torch.no_grad():
        prediction = model(features[idx].unsqueeze(0))
    print(f"样本 {idx}:")
    print(f"  真实标签: {labels[idx].item():.4f}")
    print(f"  模型预测: {prediction.item():.4f}")
    print(f"  误差: {abs(labels[idx].item() - prediction.item()):.4f}")