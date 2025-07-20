import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 保存预测结果和标签的路径
#save_dir = "results/"
#os.makedirs(save_dir, exist_ok=True)


def save_epoch_predictions(epoch, y_true, y_pred, save_dir):
    """保存每个 epoch 的真实标签和预测值"""
    np.save(os.path.join(save_dir, f"epoch_{epoch}_y_true.npy"), y_true)
    np.save(os.path.join(save_dir, f"epoch_{epoch}_y_pred.npy"), y_pred)


def load_all_predictions(num_epochs, save_dir):
    """加载所有 epoch 的真实标签和预测值"""
    all_y_true = []
    all_y_pred = []
    for epoch in range(num_epochs):
        y_true = np.load(os.path.join(save_dir, f"epoch_{epoch}_y_true.npy"))
        y_pred = np.load(os.path.join(save_dir, f"epoch_{epoch}_y_pred.npy"))
        all_y_true.append(y_true)
        all_y_pred.append(y_pred)
    return all_y_true, all_y_pred

def plot_confusion_matrices(num_epochs, class_names, save_dir):
    """绘制每个 epoch 的混淆矩阵"""
    all_y_true, all_y_pred = load_all_predictions(num_epochs)

    for epoch in range(num_epochs):
        y_true = all_y_true[epoch]
        y_pred = all_y_pred[epoch]

        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

        # 绘制混淆矩阵
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap="viridis", xticks_rotation="vertical")
        plt.title(f"Confusion Matrix - Epoch {epoch + 1}")
        plt.savefig(os.path.join(save_dir, f"confusion_matrix_epoch_{epoch + 1}.png"))
        plt.close()






"""
# 在训练循环中修改如下：
for epoch in range(start_epoch, total_epochs):
    y_true_epoch = []
    y_pred_epoch = []

    # 验证阶段，记录预测结果和真实标签
    with torch.no_grad():
        for data_test in test_iter:
            test_image, test_label = data_test
            test_image = test_image.to(device)
            test_label = test_label.to(device)

            output = net(test_image)
            predict_y = torch.argmax(output, dim=1)  # 获取预测值

            # 收集当前 batch 的真实标签和预测值
            y_true_epoch.extend(test_label.cpu().numpy())
            y_pred_epoch.extend(predict_y.cpu().numpy())

    # 保存当前 epoch 的预测值和真实标签
    save_epoch_predictions(epoch, np.array(y_true_epoch), np.array(y_pred_epoch))

    print(f"Epoch {epoch + 1}/{total_epochs} completed.")
"""
