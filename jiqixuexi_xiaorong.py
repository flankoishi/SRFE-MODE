import torchvision
from torchvision import datasets, transforms
import torch.utils.data
from torch import nn
from yanshi_model_xiaorong import yanshi_xiaorong
from yanshi_model_xiaorong_zhuyili import SE_ResNet2262
from yanshi_model_xiaorong_CABM import CBAM_ResNet2262_CABM
from yanshi_train_xiaorong import yanshi_train
import warnings
warnings.filterwarnings("ignore")
import torchvision.models as models
from net_checkpoint import save_checkpoint, load_checkpoint

pretrained_model = CBAM_ResNet2262_CABM()
#pretrained_model = yanshi_xiaorong()

##对图片进行增广,并固定位1000X1000的图片,并经行归一化
color_aug = transforms.ColorJitter(0.2, 0.2, 0.05, 0.05)
shape_aug = transforms.RandomCrop(224)
#train_aug = transforms.Compose([color_aug, shape_aug, transforms.ToTensor(),
                               #transforms.Normalize(mean=[0.61513588, 0.5875835, 0.54538981],
                                                    #std=[0.1633872,  0.16138637,  0.15809886])])
train_aug = transforms.Compose([color_aug,
                                transforms.Resize(512),
                               transforms.RandomCrop(224),
                                transforms.ToTensor()])
##对测试图片进行维度转换，转成pytorch框架的格式,resize从长方形改为正方形，
test_aug = transforms.Compose([transforms.Resize(224),
                               transforms.ToTensor()])

#创建数据集
train_dataset = torchvision.datasets.ImageFolder(
    'D:\yangben\Train_ds', transform=train_aug)
test_dataset = torchvision.datasets.ImageFolder('D:\yangben\Test_ds', transform=test_aug)

#训练量
train_num = len(train_dataset)
test_num = len(test_dataset)

#加载框架数据集函数，batch_size为批量数，shuffle为是否随机
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=8, shuffle=True, num_workers=8)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=8, shuffle=False, num_workers=8)

#使用显卡跑
device_g = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#定义模型

yanshi_net = pretrained_model
#yanshi_net = yanshi_xiaorong()
#yanshi_net.fc = nn.Linear(yanshi_net.fc.in_features,out_features=5)

#nn.init.xavier_uniform_(yanshi_net.fc.weight)#初始化权重
loss_func = torch.nn.CrossEntropyLoss()##损失函数

yanshi_net.to(device_g)
#loss_func = nn.NLLLoss2d()##损失函数
#optimizer_fun = torch.optim.Adam(yanshi_net.parameters(), lr=0.0001)##定义优化算法,lr学习率可调

#tg_save_path = 'D:\yangben\weight\Yanshi_100.pth'
#torch.save(yanshi_net.state_dict(), 'save.pt')##保存训练好的权重

#开始训练
#yanshi_train(yanshi_net, 5, train_loader, test_loader, device_g, loss_func)


if __name__ == '__main__':
    yanshi_train(yanshi_net, 50, train_loader, test_loader, device_g, loss_func)
