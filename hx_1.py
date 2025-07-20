import torch
import numpy as np
from matplotlib import pyplot as plt
import time
from sklearn.metrics import f1_score
from sklearn.metrics import (precision_score, recall_score,confusion_matrix,
                             classification_report,ConfusionMatrixDisplay)
import logging
import os
from net_checkpoint import save_checkpoint, load_checkpoint
import seaborn as sns
from duandinzaixun_1 import save_epoch_predictions,load_all_predictions


result_save_dir = 'D:/log/dd/resnet_jump_19'
all_y_true, all_y_pred = load_all_predictions(150, result_save_dir)
y_true = []
y_pred = []
class_names = ['RT 1', 'RT 2', 'RT 3', 'RT 4', 'RT 5']

y_ture = np.concatenate(all_y_true).tolist()
y_pred = np.concatenate(all_y_pred).tolist()
y_ture_1 = y_ture[149]
y_pred_1 = y_pred[149]


"""
# 计算混淆矩阵
cm = confusion_matrix(y_ture, y_pred, labels=range(len(
    ["Rock Type 0", "Rock Type 1", "Rock Type 2", "Rock Type 3", "Rock Type 4"])))
# 绘制混淆矩阵
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=
["Rock Type 0", "Rock Type 1", "Rock Type 2", "Rock Type 3", "Rock Type 4"])
disp.plot(cmap="viridis", xticks_rotation="vertical")
plt.figure(figsize=(10, 8))
plt.title(f"Confusion Matrix")
plt.xlabel("Predicted Label")  # x 轴名称
plt.ylabel("True Label")  # y 轴名称
plt.savefig(os.path.join(result_save_dir, f"confusion_matrix_epoch_d.png"))
# plt.close()
plt.show()
time.sleep(1)
"""
cm = confusion_matrix(y_ture, y_pred, labels=range(len(
    ["Rock Type 0", "Rock Type 1", "Rock Type 2", "Rock Type 3", "Rock Type 4"])))
plt.figure(figsize=(10, 8))
#sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=np.arange(cm.shape[1]),
            #yticklabels=np.arange(cm.shape[1]))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=class_names,
            yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig(os.path.join(result_save_dir, f"confusion_matrix_epoch_1.png"))
plt.show()
#print(y_ture_3)