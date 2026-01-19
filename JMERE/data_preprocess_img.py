import json
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

print(torch.cuda.is_available())

from datasets import Dataset

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

from tqdm import tqdm
from PIL import Image

# 配置ResNet模型
model_name = 'resnet152'  # 可以选择resnet18, resnet34, resnet50, resnet101, resnet152
pretrained = True  # 使用预训练权重

# 创建ResNet模型并修改
def create_resnet_model(model_name, pretrained=True):
    """创建ResNet模型并修改为特征提取器"""
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=pretrained)
    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=pretrained)
    else:
        raise ValueError(f"不支持的ResNet模型: {model_name}")
    
    # 移除最后的全连接层，用于特征提取
    model = nn.Sequential(*list(model.children())[:-1])  # 移除最后一层(fc)
    
    return model

# 图像转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_img(img_path):
    """加载并预处理图像"""
    img = Image.open(img_path).convert("RGB")
    return img

def data_from_arrow(path):
    print("loading data from folder:", path + '_arrow')
    return Dataset.load_from_disk(path + '_arrow', keep_in_memory=True)

@torch.no_grad()
def get_feature(image, model):
    """使用ResNet提取图像特征"""
    image = transform(image).unsqueeze(0)  # 添加batch维度
    image = image.to(next(model.parameters()).device)  # 将图像移到与模型相同的设备
    features = model(image)
    features = features.squeeze()  # 移除batch维度
    if len(features.shape) == 4:  # 如果特征仍然是4D张量(BxCxHxW)，压缩空间维度
        features = features.mean([2, 3])  # 全局平均池化
    return features.cpu()  # 返回CPU上的特征

def prepare_data(data_path, model, json_path) -> Dataset:
    if os.path.exists(json_path + '_arrow'):
        return data_from_arrow(json_path)
    print("arrow not found, loading data from json:", json_path + '.json')
    if not os.path.exists(json_path + '.json'):
        raise ValueError("json path not exists:", json_path + '.json')

    data = []
    with open(json_path + '.json', "r", encoding='utf-8') as fr:
        pieces = json.load(fr)

        for one in tqdm(pieces):
            data_dict = {'img_id': one['img_id']}
            img = get_feature(load_img(data_path + '/' + one['img_id']), model)
            data_dict['images'] = img
            data.append(data_dict)

        data = Dataset.from_list(data)

    print("saving data to arrow:", json_path + '_arrow')
    data.save_to_disk(json_path + '_arrow')

    return data

def data_folder(model):
    train = prepare_data("/home/nlp/Project/XZY/JMERE-dataset/img_cor/img_myself/train", model, "/home/nlp/Project/XZY/JMERE-dataset/dataset/V2/train_ofa_image_caption_knowledge")
    print(train)
    val = prepare_data("/home/nlp/Project/XZY/JMERE-dataset/img_cor/img_myself/val", model, "/home/nlp/Project/XZY/JMERE-dataset/dataset/V2/val_ofa_image_caption_knowledge")
    print(val)
    test = prepare_data("/home/nlp/Project/XZY/JMERE-dataset/img_cor/img_myself/test", model, "/home/nlp/Project/XZY/JMERE-dataset/dataset/V2/test_ofa_image_caption_knowledge")
    print(test)

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

if __name__ == '__main__':
    # 创建ResNet模型
    resnet_model = create_resnet_model(model_name, pretrained)
    resnet_model.eval()  # 设置为评估模式
    resnet_model.to('cpu')  # 移到GPU
    
    print(f"{model_name} Loaded with pretrained weights: {pretrained}")
    # debug = prepare_data("/home/nlp/Project/XZY/JMERE-dataset/img_cor/img_myself/val", resnet_model, "/home/nlp/Project/XZY/JMERE-dataset/dataset/V2/val_ofa_image_caption_knowledge")
    # print(debug)
    # first_sample = debug[0]
    # print("Image ID:", first_sample['img_id'])
    # print("Image Features Shape:", torch.tensor(first_sample['images']).shape)
    # print("Image Features:", first_sample['images'])
    # debug = prepare_data("../JMERE_text/debug", resnet_model, "../few_shot/debug")
    # print(debug)

    # for i, batch in enumerate(debug):
    #     print(i, batch['img_id'], batch['images'].shape)

    # 处理多个数据集文件夹
    data_folder(resnet_model)

