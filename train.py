import os
import json

import torch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils import data

from model.MobileNet import MobileNetV2


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device".format(device))
    data_transforms = {
        'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    }
    data_path = os.getcwd()
    image_path = os.path.join(data_path, 'data')
    assert os.path.exists(image_path), "{} path not exist!".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'train'), transform=data_transforms["train"])
    val_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'val'), transform=data_transforms["val"])
    train_num = len(train_dataset)
    val_num = len(val_dataset)
    print("Using {} images for training,{} images for validation".format(train_num, val_num))
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((key, val) for val, key in flower_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open("class_indices.json", 'w') as f:
        f.write(json_str)
    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print("Using {} dataloader worker every process".format(nw))

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=nw)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=nw)
    net = MobileNetV2(num_classes=5)

    model_weight_path = './mobilenet_v2.pth'
    assert os.path.exists(model_weight_path), "file {} not exist.".format(model_weight_path)
    pre_weight = torch.load(model_weight_path, map_location=device)

    # 删除全连接层的权重
    # 读取预训练权重中与现有模型参数设置相同层的权重，
    # 可适用于修改了分类或某些层通道数的情况
    # 结构与命名要和预训练的一致
    pre_dict = {k: v for k, v in pre_weight.items() if net.state_dict()[k].numel() == v.numel()}
    missing_key, unexpected_keys = net.load_state_dict(pre_dict, strict=False)

    # 冻结特征提取层预训练权重
    for param in net.features.parameters():
        param.requires_grad = False

    net.to(device)

    loss_function = nn.CrossEntropyLoss()

    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    best_acc = 0.0
    save_path = "./MobileNetV2.pth"
    train_steps = len(train_loader)
    epochs = 20
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data_ in enumerate(train_bar):
            images, labels = data_
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "training epoch {},loss:{:.3f}".format(epoch + 1, loss)
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                val_bar.desc = "validation epoch {}".format(epoch + 1)
        val_acc = acc / val_num
        print("epoch {} train_loss:{:.3f}  val_accuracy:{:.3f}".format(epoch + 1, running_loss / train_steps, val_acc))
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), save_path)


if __name__ == "__main__":
    train()
