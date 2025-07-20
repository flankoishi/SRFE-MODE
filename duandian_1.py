import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os

# 保存预测结果和标签的路径
save_dir = "./epoch_outputs/"
os.makedirs(save_dir, exist_ok=True)


# 保存每个 epoch 的预测结果和标签
def save_epoch_outputs(epoch, outputs, labels):
    np.save(os.path.join(save_dir, f"epoch_{epoch}_outputs.npy"), outputs)
    np.save(os.path.join(save_dir, f"epoch_{epoch}_labels.npy"), labels)


# 加载所有 epoch 的预测结果和标签
def load_all_epoch_outputs(total_epochs):
    all_outputs = []
    all_labels = []
    for epoch in range(1, total_epochs + 1):
        outputs = np.load(os.path.join(save_dir, f"epoch_{epoch}_outputs.npy"))
        labels = np.load(os.path.join(save_dir, f"epoch_{epoch}_labels.npy"))
        all_outputs.append(outputs)
        all_labels.append(labels)
    return all_outputs, all_labels


# 模型训练代码中保存每个 epoch 的预测结果和标签
for epoch in range(start_epoch, end_epoch + 1):  # 假设有断点再训练
    net.eval()  # 进入评估模式
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for data_test in test_loader:
            images, labels = data_test
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    save_epoch_outputs(epoch, all_outputs, all_labels)  # 保存到文件

# 训练完成后加载所有保存的结果并计算 ROC 曲线
all_outputs, all_labels = load_all_epoch_outputs(total_epochs=60)

# 绘制所有 epoch 的 ROC 曲线
n_classes = all_outputs[0].shape[1]  # 假设是多分类问题
for epoch in range(60):
    outputs = all_outputs[epoch]
    labels = all_labels[epoch]

    # One-hot 编码标签
    labels_onehot = np.eye(n_classes)[labels]

    plt.figure(figsize=(10, 8))
    for class_idx in range(n_classes):
        fpr, tpr, _ = roc_curve(labels_onehot[:, class_idx], outputs[:, class_idx])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {class_idx} (AUC = {roc_auc:.2f})")

    plt.title(f"ROC Curve - Epoch {epoch + 1}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(f"roc_epoch_{epoch + 1}.png")  # 保存每个 epoch 的 ROC 曲线图
    plt.close()
