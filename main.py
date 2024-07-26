import warnings
import yaml
import torch
import argparse
import random
import time
from datetime import timedelta
import os

from scripts import initialize
from models import *
from utils import log
import dng
from trainer import binary_trainer, multi_trainer
from utils import metrics, statistic, log
from visualize import drawG, umapG

from torch_geometric.loader import DataLoader

# 查看代码运行情况
# from heartrate import trace
# trace(browser=True)

warnings.filterwarnings("ignore")

def load_config(file_path):
    """加载配置文件

    Args:
        file_path (string): 配置文件路径

    Returns:
        dict: yaml格式配置文件
    """
    with open(file_path, "rb") as f:
        config = yaml.safe_load(f)
    return config

def data_func(data_config):
    """数据集功能入口"""
    myDNG = dng.DNG(data_config, transform=data_config['transform'], pre_transform=data_config['pre_transform'], pre_filter=data_config['pre_filter'])
    return myDNG

def train_func(dataset, train_config):
    """dataset -> 训练结果"""
    if train_config["is_enabled"]:
        print("***Train Config is enabled, start graph_train_func...***")
        if train_config['task'] == 't1':
            binary_trainer.train(dataset, train_config)
        elif train_config['task'] == 't2':
            multi_trainer.train(dataset, train_config)
    else:
        print("***Train Config is not enabled, skip graph_train_func...***")

def apply_func(dng, model_path):
    # 加载模型，对每张图进行判断，返回每张图对应的结果
    model = torch.load(model_path).to('cuda:0')
    model.eval()
    y_pred = []
    dng = [each.to_homogeneous().to('cuda:0') for each in dng]
    loader = DataLoader(dng, batch_size=len(dng), shuffle=False)
    for data in loader:
        out = model(data.x, data.edge_index, data.batch) # if type == 'homogeneous' else model(data.x_dict, data.edge_index_dict, data.batch_dict)
        pred = out.argmax(dim=1)
        y_pred.extend(pred.tolist())
    
    result = {}
    labels_1 = {
        0: "Background",
        1: "DNS tunneling"
    }
    labels_2 = {
        0: "Background",
        1: "cobaltstrike",
        2: "dns_shell",
        3: "dns2tcp",
        4: "dnscat2",
        5: "dnsexfiltrator",
        6: "iodine",
        7: "ozymandns"
    }
    label_dict = labels_1
    for i in range(len(y_pred)):
        if label_dict[y_pred[i]] not in result:
            result[label_dict[y_pred[i]]] = [dng[i].parentDomain]
        else:
            result[label_dict[y_pred[i]]].append(dng[i].parentDomain)

    # 对result的每个list去重
    for k, v in result.items():
        result[k] = list(set(v))
    # print(f"图{dng[i].parentDomain}的预测结果为{y_pred[i]}")
    logger = log.get_logger()
    logger.info(result)
    
    return y_pred

def draw_func(dataset, draw_config):
    exit()
    """所有绘图功能入口"""
    if draw_config["embedding2pic"]["is_enabled"]:
        print("***Embedding Visualize Config is enabled, start embedding visualization...***")
        # embedding visualization
        if model_list != []:
            for model in model_list:
                embedding_visualize_func(
                    model.embeddings,
                    dataset.father_domain_list,
                    dataset.father_domain_dict,
                    draw_config["embedding2pic"],
                    model.name
                )
        else:
            print("***Model List is empty, skip embedding visualization...***")

    if draw_config["dataset2pic"]["is_enabled"]:
        print("***Draw Config is enabled, start draw_func...***")
        drawG.draw_datasets(dataset.data, draw_config["dataset2pic"])
    else:
        print("***Draw Config is not enabled, skip draw_func...***")


def embedding_visualize_func(embedding, name_list, name2label, config, model_name):
    """embedding -U-MAP-> pic"""
    if config["is_enabled"] == True:
        umapG.embeddings2pic(embedding, name_list, name2label, config, model_name)
    else:
        print("***Embedding Visualize Config is not enabled, skip embedding_visualize_func...***")


def main(args):  # 命令行参数传入主函数
    start_time = time.time()
    # 读取配置
    global_config = load_config(args.global_config_path)  # 加载全局配置文件
    data_config    = load_config(args.data_config_path)  # 加载数据配置文件
    train_config  = load_config(args.train_config_path)  # 加载训练配置文件
    draw_config   = load_config(args.draw_config_path)  # 加载可视化配置文件

    # 所有功能入口
    if global_config["is_enabled"] == True:
        print("***Global function is enabled, start global_func...***")
        random.seed(global_config["seed"])
        
        if global_config["init_dir"]:
            for each_dir in global_config["init_dir"]:
                initialize.clean_folder(each_dir)
        # data
        myDNG = data_func(data_config)
        
        # trainer
        train_func(myDNG, train_config)
        
        # apply 加载模型识别
        # apply_func(myDNG, os.path.join(train_config['save_dir'], 'GAI_30_model.pth'))

        # visualization
        draw_func(myDNG, draw_config)

    else:
        print("***Global function is not enabled, exit...***")


    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time = timedelta(seconds=elapsed_time)
    print(f'程序运行时间{elapsed_time}毫秒')


if __name__ == "__main__":
    print("PyTorch Version: ", torch.__version__)
    # 定义解析器
    parser = argparse.ArgumentParser()
    
    config_dir = 'config/test/t2'
    encoder = ''
    interval = ''
    
    # 加参数
    parser.add_argument(
        "--global_config_path", type=str, default=os.path.join(config_dir, encoder, interval, 'global_config.yaml')
    )
    parser.add_argument(
        "--data_config_path", type=str, default=os.path.join(config_dir, encoder, interval, 'data_config.yaml')
    )
    parser.add_argument(
        "--train_config_path", type=str, default=os.path.join(config_dir, encoder, interval, 'train_config.yaml')
    )
    parser.add_argument(
        "--draw_config_path", type=str, default=os.path.join(config_dir, encoder, interval, 'draw_config.yaml')
    ) 
    args = parser.parse_args()
    main(args)
