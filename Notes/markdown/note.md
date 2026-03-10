# 深度学习笔记

Based on "LEEDL TUTORIALS" by Dr. Li in Taiwan University

## 第一章: 机器学习基础

机器学习(Machine Learning, ML), 顾名思义, 就是让机器能够学习, 从数据中提取规律, 并进行预测或决策. 深度学习(Deep Learning, DL) 是机器学习的一个分支, 主要关注于使用多层神经网络来进行学习和预测.

在机器学习中, 我们主要是为了解决几个问题:

1. 回归(Regression): 预测连续值, 如房价预测.
2. 分类(Classification): 预测离散类别, 如垃圾邮件分类.
3. 结构化学习(Structured Learning): 机器不仅仅需要给出预测, 给出概率序列, 还要生成结构化的产物, 比如一篇文章, 一段代码, 一张图片等.

### 从案例出发看机器学习

考虑一个例子, 我们想通过过往的信息(比如过去若干天的观看次数)来预测未来的观看次数. 根据最朴素的经验, 从最简单的假设考虑, 我们假设一天的观看次数与前一天有某些关联.

从最简化的思路出发, 我们可以假设未来的观看次数与前一天的观看次数成线性关系, 即:
$$y = wx + b$$
其中, $y$ 是未来的观看次数, $x$ 是前一天的观看次数, $w$ 是权重, $b$ 是偏置.

于是我们就得到了一个最简单的线性回归模型. 训练这个模型, 本质上就是在寻找最合适的 $w$ 和 $b$ 使得模型的预测值尽可能接近真实值. 这就是机器学习中的参数优化问题. 所以我们需要一个衡量标注(criterion)来评估模型的预测效果, 这就是损失函数, 比如均方误差(MSE):

$$L(w, b) = \frac{1}{N} \sum_{i=1}^{N} (y_i - (wx_i + b))^2$$

定义了损失函数后, 我们需要通过迭代来逐步调整 $w$ 和 $b$ 的值, 使得损失函数的值最小化. 这就是优化算法的核心, 最常用的优化算法是梯度下降(Gradient Descent). 在梯度下降中, 我们计算损失函数关于参数的梯度, 然后沿着梯度的反方向更新参数:
$$w := w - \alpha \frac{\partial L}{\partial w}$$
$$b := b - \alpha \frac{\partial L}{\partial b}$$
其中, $\alpha$ 是学习率(learning rate), 控制着每次更新的步长. 通过不断迭代这个过程, 我们可以找到一组参数 $w$ 和 $b$ 使得模型的预测效果最佳.

以上的全部过程就是机器学习的基本流程: 

```Example
定义模型 -> 定义损失函数 -> 选择优化算法 -> 训练模型
```

通过这个流程, 我们可以从数据中学习到规律, 并进行预测.

#### 一个简单的线性回归实现

我们可以写一个简易的python程序来实现这个线性回归模型的训练过程:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
pip install numpy
```

```python
import numpy as np

# Generate synthetic data
def generate_data(N):
    np.random.seed(42) # Set seed for reproducibility
    x = np.random.rand(N) * 10  # Generate random numbers between 0 and 10
    w_true = 2.5
    b_true = 1.0
    noise = np.random.randn(N) * 2  # Add Gaussian noise
    y = w_true * x + b_true + noise  # Generate target values
    return x, y

# Train Linear Regression model using Gradient Descent
def train(x, y, learning_rate=0.01, epochs=1000):
    # Initialize weights and bias to 0 or random values
    w = 0.0
    b = 0.0
    N = len(x)
    
    for epoch in range(epochs):
        # 1. Prediction (Forward pass)
        y_pred = w * x + b
        
        # 2. Compute Loss (Mean Squared Error)
        loss = (1/N) * np.sum((y - y_pred) ** 2)
        
        # 3. Compute Gradients manually
        # Partial derivative of Loss w.r.t w: dL/dw = -2/N * sum(x * (y - y_pred))
        # Partial derivative of Loss w.r.t b: dL/db = -2/N * sum(y - y_pred)
        dw = (-2/N) * np.sum(x * (y - y_pred))
        db = (-2/N) * np.sum(y - y_pred)
        
        # 4. Update parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # Print progress every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}, w = {w:.4f}, b = {b:.4f}")

    return w, b

# Main execution
if __name__ == "__main__":
    # Generate data
    N_samples = 100
    x_data, y_data = generate_data(N_samples)
    
    # Train model
    print("Starting training...")
    w_est, b_est = train(x_data, y_data)
    
    print("\nTraining complete.")
    print(f"Estimated w: {w_est:.4f}, Estimated b: {b_est:.4f}")
    print(f"True w: 2.5, True b: 1.0")
```

#### PyTorch 实现 (自动求导)

如果我们使用 PyTorch，可以利用其自动求导机制 (Autograd) 省去手动计算梯度的步骤：

```python
import torch
import numpy as np

# 1. Prepare Data
# Same data generation pattern, converted to PyTorch tensors
N = 100
# Generate numpy arrays
x_np = np.random.rand(N) * 10
y_np = 2.5 * x_np + 1.0 + np.random.randn(N) * 2

# Convert to PyTorch tensors
x = torch.from_numpy(x_np).float()
y = torch.from_numpy(y_np).float()

# 2. Initialize Parameters
# requires_grad=True tells PyTorch to track operations for gradient calculation
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

learning_rate = 0.01

print("Starting PyTorch training...")
for epoch in range(1000):
    # Forward pass: Compute predicted y
    y_pred = w * x + b
    
    # Compute and print loss
    loss = (y_pred - y).pow(2).mean()
    
    # Backward pass: compute gradients of the loss with respect to model parameters
    loss.backward()
    
    # Update weights
    # We don't want to track this update in the computation graph
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        
        # Manually zero the gradients after updating weights
        w.grad.zero_()
        b.grad.zero_()
        
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}, w = {w.item():.4f}, b = {b.item():.4f}")

print(f"\nResult: w = {w.item():.4f}, b = {b.item():.4f}")
```

在这个程序中, 我们首先生成了一些合成数据, 其中 $x$ 是随机生成的输入数据, $y$ 是根据一个线性关系加上噪声生成的目标值. 然后我们定义了一个训练函数 `train`, 该函数使用梯度下降算法来优化模型参数 $w$ 和 $b$. 在每个epoch中, 我们计算预测值, 计算损失, 计算梯度, 并更新参数. 最后我们打印出训练过程中的损失和参数值, 以及最终的估计参数和真实参数进行比较.

在这个例子中, 因为模型非常简单, 我们可以手动求出梯度并更新参数. 但是在更复杂的模型中, 手动计算梯度会非常麻烦, 这时候我们可以使用像 `PyTorch` 这样的深度学习框架来自动计算梯度, 大大简化了训练过程.

这样的拟合是非常适合于线性关系的数据, 但是大多数工程关系都是非线性的. 我们下面考虑一个相对复杂的情况: 分段线性曲线.

#### 更复杂的策略

对于分段线性曲线, 我们可以使用许多简单的线性函数来逐段拟合. 但是我们很难确定边界在哪里. 这意味着我们需要手动地调整模型的结构, 这就非常麻烦. 所以我们可以使用Hard Sigmoid函数来实现分段线性拟合. Hard Sigmoid函数在输入小于0时输出0, 输入大于1时输出1, 在0到1之间线性变化. 通过调整Hard Sigmoid函数的参数, 我们可以实现不同的分段线性拟合.

![(Hard) Sigmoid Functions](figures\Figure_1.svg)

如上图所示, 橙色的线条就是我们提及的Hard Sigmoid函数. 通过调整参数, 我们可以让它在不同的区间内进行线性拟合. 同时, 为了实现在不同线性区段上的准确拟合, 我们可以使用多个Hard Sigmoid函数的组合来实现更复杂的分段线性拟合.

但是有时候, 用Hard sigmoid函数来拟合分段线性曲线可能会有一些问题, 比如在边界处的拟合效果不佳. 这时候我们可以使用ReLU函数来代替Hard Sigmoid函数. ReLU函数在输入小于0时输出0, 在输入大于0时输出输入值本身. 通过调整ReLU函数的参数, 我们也可以实现不同的分段线性拟合.

很多时候, 我们也可以使用sigmoid函数来拟合分段线性曲线. Sigmoid函数在输入小于0时输出接近0的值, 在输入大于0时输出接近1的值, 在0附近线性变化. 通过调整sigmoid函数的参数, 我们也可以实现不同的分段线性拟合.

Hard Sigmoid和sigmoid, ReLU函数的定义如下:
$$\text{HardSigmoid}(x) = \begin{cases} 
y_1 & \text{if } x < x_1 \\
y_2 & \text{if } x > x_2 \\
y_1 + (y_2 - y_1)\dfrac{x - x_1}{x_2 - x_1} & \text{otherwise}
\end{cases}$$

$$\text{Sigmoid}(x) = \frac{1}{1 + e^{-x}}$$

$$\text{ReLU}(x) = \begin{cases}
0 & \text{if } x < 0 \\
x & \text{if } x \geq 0
\end{cases}$$

但是我们事实上只会关注带有斜率的那段, 所以我们可以把一个Hard Sigmoid函数写成:

$$
\text{HardSigmoid}(x) = \sigma(wx+b)
$$

那么我们拟合出来的方程实际上就是:

$$
f(x) = \sum_{i=1}^{n} \sigma(w_i x+b_i) + b
$$

然而, 我们在很多情况下可以使用多于一个$x$的参数来拟合更复杂的关系, 这时候我们就需要引入多维输入. 那么我们拟合的方程就变成了:

$$
f(x) = b + \sum_{i} c_i \sigma \left( b_i + \sum_{j} w_{ij} x_j \right)
$$

我们可以用线性代数的方式来表示这个方程:

$$
\begin{pmatrix}
    r_1 \\
    r_2 \\
    \vdots \\
    r_m
\end{pmatrix} = 
\begin{pmatrix}
    b_1 \\
    b_2 \\
    \vdots \\
    b_m
\end{pmatrix} + 
\begin{pmatrix}
    w_{11} & w_{12} & \cdots & w_{1n} \\
    w_{21} & w_{22} & \cdots & w_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    w_{m1} & w_{m2} & \cdots & w_{mn}
\end{pmatrix}
\begin{pmatrix}
    x_1 \\
    x_2 \\
    \vdots \\
    x_n
\end{pmatrix}
$$

更简单的, 我们可以把这个方程写成:
$$
r = b + W x
$$
然后再利用广播机制对$r$进行非线性变换:
$$
a = \sigma(r) = \begin{pmatrix}
    \sigma(r_1) \\
    \sigma(r_2) \\
    \vdots \\
    \sigma(r_m)
\end{pmatrix}
$$

现在模型经过复杂化之后已经具有了相当的表达能力, 那么具体怎么训练呢? 训练的过程和之前一样, 只是现在我们需要优化的参数更多了. 我们需要定义一个损失函数来衡量模型的预测效果, 然后使用梯度下降算法来优化参数. 通过不断迭代这个过程, 我们可以找到一组参数使得模型的预测效果最佳.

为了进一步提高模型的表达能力, 我们可以在模型中引入更多的非线性变换. 这时候我们就需要引入多层神经网络. 多层神经网络的结构如下图所示:

![MLP](figures\Figure_2.svg)

对于此类模型, 夹在输入层和输出层之间的层被称为隐藏层(Hidden Layer). 每个隐藏层都包含多个神经元(Neurons), 每个神经元都进行线性变换和非线性变换.

如此以来, 我们就得到了一个多层感知机(Multi-Layer Perceptron, MLP)模型. 通过增加隐藏层的数量和每层的神经元数量, 我们可以提高模型的表达能力, 从而更好地拟合复杂的关系. 现代神经网络通常包含数十层甚至数百层, 每层包含数千个神经元, 这使得它们能够拟合非常复杂的关系.

## 方法论

然而, 由于机器学习模型的复杂性, 我们在训练过程中可能会遇到一些问题, 这些问题我们不能总是一股脑的归咎为数据不足, 可能是模型结构不合理, 可能是优化算法不合适, 可能是损失函数不合适, 可能是超参数设置不合理等等. 因此, 在训练模型时, 我们需要仔细分析问题的根源, 并针对性地进行调整和优化.

1. 模型偏差

模型偏差(Bias)是指模型在训练数据上的拟合能力不足, 导致在训练数据上表现不佳. 这通常是由于模型过于简单, 无法捕捉到数据中的复杂关系. 解决模型偏差的方法是增加模型的复杂度, 比如增加隐藏层的数量和每层的神经元数量.

2. 过拟合

过拟合(Overfitting)是指模型在训练数据上表现很好, 但是在测试数据上表现不佳. 这通常是由于模型过于复杂, 捕捉到了训练数据中的噪声和偶然关系.

解决过拟合的方法之一: 增加数据量, 通过增加训练数据的数量, 可以让模型更好地学习到数据中的规律, 从而减少过拟合的风险. 但是我们有时候并不需要增加数据量, 我们可以用**数据增强(Data Augmentation)**的方法来增加数据的多样性, 从而减少过拟合的风险. 例如, 在图像分类任务中, 我们可以通过旋转、翻转、缩放等方式来增强图像数据. 在文本分类任务中, 我们可以通过同义词替换、随机删除等方式来增强文本数据.

解决过拟合的方法之二: 降低模型的复杂度. 如果采用的是深度学习模型, 我们可以通过减少隐藏层的数量和每层的神经元数量来降低模型的复杂度. 但是这可能会导致模型偏差增加, 因此我们需要在模型偏差和过拟合之间进行权衡.

其他解决方法还包括正则化(Regularization), 早停(Early Stopping), dropout等. 正则化是通过在损失函数中添加一个惩罚项来限制模型的复杂度, 从而减少过拟合的风险. 早停是在训练过程中监控模型在验证集上的表现, 当模型在验证集上的表现不再提升时, 就停止训练. Dropout是一种随机丢弃神经元的方法, 在训练过程中随机丢弃一部分神经元, 从而减少过拟合的风险.

3. 梯度消失和梯度爆炸

梯度消失(Vanishing Gradient)和梯度爆炸(Exploding Gradient)是指在训练深度神经网络时, 由于链式法则的作用, 导致梯度在反向传播过程中逐渐变小或变大, 从而导致模型无法有效地更新参数. 解决梯度消失和梯度爆炸的方法包括使用ReLU激活函数、使用批归一化(Batch Normalization)、使用残差连接(Residual Connection)等.

- 批归一化是指在每一层的输入上进行归一化处理, 使得输入的分布更加稳定, 从而减少梯度消失和梯度爆炸的风险. 批归一化可以加速训练过程, 提高模型的性能.

- 残差连接是指在每一层的输出上添加一个跳跃连接, 使得输入可以直接传递到后面的层, 从而减少梯度消失和梯度爆炸的风险. 残差连接可以让模型更容易地学习到恒等映射(Identity Mapping), 从而提高模型的性能. 用数学表达式可以写成:

$$
y_k = h(x_k) + x_k
$$

其中, $h(x_k)$ 是第 $k$ 层的输出, $x_k$ 是第 $k$ 层的输入. 通过添加残差连接, 我们可以让模型更容易地学习到恒等映射, 从而提高模型的性能.

4. 交叉验证

交叉验证(Cross Validation)是一种评估模型性能的方法, 通过将数据集划分为多个子集, 轮流使用其中一个子集作为验证集, 其余子集作为训练集, 来评估模型在不同数据上的表现. 交叉验证可以帮助我们更好地评估模型的泛化能力, 从而选择最合适的模型和超参数. 常见的交叉验证方法包括K折交叉验证(K-Fold Cross Validation), 留一交叉验证(Leave-One-Out Cross Validation)等.

## 深度学习基础

