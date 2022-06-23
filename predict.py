import os
import json

import torch
from PIL import Image
from torchvision import transforms

from model.MobileNet import MobileNetV2


def predict():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    # load image

    image_root = "data/images"
    assert os.path.exists(image_root), "{} file not exist!".format(image_root)

    image_path_list = [os.path.join(image_root, i) for i in os.listdir(image_root) if i.endswith(".jpg")]

    json_path = "./class_indices.json"

    with open(json_path, 'r') as f:
        class_dict = json.load(f)

    # 创建网络
    model = MobileNetV2(num_classes=5)
    model.to(device)

    # 加载训练权重 ：
    weight_path = "./MobileNetV2.pth"
    model.load_state_dict(torch.load(weight_path, map_location=device))

    model.eval()
    batch_size = 8

    with torch.no_grad():
        for ids in range(0, len(image_path_list) // batch_size + 1):
            img_list = []
            for img_path in image_path_list[ids * batch_size:min((ids + 1) * batch_size, len(image_path_list))]:
                img = Image.open(img_path)
                img = data_transform(img)
                img_list.append(img)

            batch_img = torch.stack(img_list, dim=0)

            output = model(batch_img.to(device)).cpu()
            predict = torch.softmax(output, dim=1)
            prbs, classes = torch.max(predict, dim=1)
            for idx, (pro, cls) in enumerate(zip(prbs, classes)):
                print("image:{},class:{} prob:{:.3f}".format(image_path_list[ids * batch_size + idx],
                                                             class_dict[str(cls.numpy())],
                                                             pro.numpy()))


if __name__ == "__main__":
    predict()
