import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def compute_gradient_norms_and_max_diff(filepath):
    grads = np.loadtxt(filepath)
    grad_norms = []
    grads_list = []
    for i in range(grads.shape[0]):
        norm = np.linalg.norm(grads[i])
        grad_norms.append(norm)
        grads_list.append(grads[i])

    # 计算最大差值及对应的 epoch
    max_diff = 0
    max_pair = (0, 0)
    for i in range(len(grads_list)):
        for j in range(i + 1, len(grads_list)):
            diff = np.linalg.norm(grads_list[i] - grads_list[j])
            if diff > max_diff:
                max_diff = diff
                max_pair = (i, j)

    return grad_norms, max_diff, max_pair

# 文件路径与模型名称
filepaths = [
    ('grads_VGG_Dropout.txt', 'VGG_A_Dropout'),
    ('grads_VGG.txt', 'VGG_A'),
    ('grads_VGG_Batch.txt', 'VGG_A_BatchNorm')
]

plt.figure(figsize=(10, 6))
for filepath, label in filepaths:
    grad_norms, max_diff, max_pair = compute_gradient_norms_and_max_diff(filepath)
    epochs = np.arange(len(grad_norms)).reshape(-1, 1)
    grad_array = np.array(grad_norms).reshape(-1, 1)

    # 线性拟合
    reg = LinearRegression().fit(epochs, grad_array)
    slope = reg.coef_[0][0]
    intercept = reg.intercept_[0]

    # 输出信息
    print(f"\n模型: {label}")
    print(f"线性拟合结果: 梯度 = {slope:.6f} * epoch + {intercept:.6f}")
    print(f"最大梯度差值: {max_diff:.6f}")
    print(f"出现在第 {max_pair[0]} 和第 {max_pair[1]} 个 epoch，距离为 {abs(max_pair[1] - max_pair[0])}")

    # 绘图
    plt.plot(epochs, grad_array, marker='o', label=f'{label} (norm)')
    plt.plot(epochs, reg.predict(epochs), linestyle='--', label=f'{label} (fit)')

# 图像设置
plt.title('Gradient Norms with Linear Fit and Max Difference')
plt.xlabel('Epoch')
plt.ylabel('Gradient L2 Norm')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('gradient_norms_linear_fit_maxdiff.png')
plt.show()
