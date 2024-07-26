# TODO 找其他数据集做实验
from datetime import timedelta
import time
import torch
import importlib
from torch_geometric.loader import DataLoader
from torch_geometric import seed_everything
from torch_geometric.profile import get_model_size, count_parameters, get_gpu_memory_from_nvidia_smi
# from dng import DNG   # 为什么不同于视频的dataloader.DataLoader
from utils.metrics import get_binary_evaluation, get_multi_confusion_matrix, get_confusion_matrix_plot
from utils import log
import GPUtil
import psutil
import os
import gc
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm

from memory_profiler import profile

def train(dataset, train_config):
    seed_everything(train_config['seed'])
    dataset = dataset.shuffle()
    homo_dataset = [each_graph.to_homogeneous() for each_graph in tqdm(dataset, desc='异质图转同质图')]
    for each_model in train_config['model_list']:
        each_train(dataset, homo_dataset, each_model, train_config)


def each_train(dataset, homo_dataset, each_model, train_config):
    os.makedirs(os.path.join(train_config['logger']['root_dir'], f"{each_model}"), exist_ok=True)
    profile_path = os.path.join(train_config['logger']['root_dir'], f"{each_model}/{each_model}_memory.log")
    @profile(precision=4, stream=open(profile_path, 'w'))
    def inner_function(dataset, homo_dataset, each_model, train_config):
            # 准备
            logger = log.get_logger(os.path.join(train_config['logger']['root_dir'], f"{each_model}/{each_model}{train_config['logger']['postfix']}"), train_config['logger']['verbosity'], each_model)
            logger.info(f'Train config: \n{train_config}')

            ## 系统信息
            # 获取系统内存和CPU信息
            memory_info = psutil.virtual_memory()
            cpu_info = psutil.cpu_percent()

            # 获取GPU信息
            torch.cuda.empty_cache()
            gpus = GPUtil.getGPUs()
            gpu_info = "\n".join([f"GPU {i}: {gpu.name}, Memory Usage: {gpu.memoryUsed} MB / {gpu.memoryTotal} MB" for i, gpu in enumerate(gpus)])

            # 记录信息到日志
            logger.info("Memory Usage: %s", memory_info)
            logger.info("CPU Usage: %s%%", cpu_info)
            logger.info("GPU Info:\n%s", gpu_info)
            
            # CUDA检查 
            device = torch.device(train_config['device'])
            logger.info(f'Using device {device}')
            
            # 在模块中查找模型类并加入表示模型列表
            model_config = load_config(os.path.join(train_config['model_config'], each_model+'.yaml'))
            module = importlib.import_module('.'.join([train_config['model_root'], each_model]))
            model_class = getattr(module, each_model)
            model = model_class(**model_config['parameters']).to(device)
            logger.info(f'模型: \n{model}')
            logger.info(f'模型参数量: {count_parameters(model)}')
            logger.info(f'模型大小: {get_model_size(model)}')
            # TODO pyg记录相关信息
            
            ## 数据集
            logger.info(f'第一张异质图信息：\n{dataset[0]}')
            logger.info(f'第一张同质图信息：\n{homo_dataset[0]}')
            
            # 根据模型选择数据集类型
            dataset = homo_dataset if model_config['type'] == 'homogeneous' else dataset
            
            # 划分数据集
            train_dataset = dataset[:int(len(dataset) * train_config['train_ratio'])]
            val_dataset = dataset[int(len(dataset) * train_config['train_ratio']):int(len(dataset) * (train_config['train_ratio'] + train_config['val_ratio']))]
            test_dataset = dataset[int(len(dataset) * (train_config['train_ratio'] + train_config['val_ratio'])):]
            train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=train_config['batch_size'], shuffle=False)
            # 数据集信息
            logger.info(f'训练集大小: {len(train_dataset)}')
            logger.info(f'训练集loader大小: {len(train_loader)}')
            train_dict = dict()
            for each in train_dataset:
                train_dict[int(each.y[1])] = train_dict.get(int(each.y[1]), 0) + 1
            logger.info(f'训练集label分布: {dict(sorted(train_dict.items()))}')
            
            logger.info(f'验证集大小: {len(val_dataset)}')
            logger.info(f'验证集loader大小: {len(val_loader)}')
            val_dict = dict()
            for each in val_dataset:
                val_dict[int(each.y[1])] = val_dict.get(int(each.y[1]), 0) + 1
            logger.info(f'验证集label分布: {dict(sorted(val_dict.items()))}')
            
            logger.info(f'测试集大小: {len(test_dataset)}')
            logger.info(f'测试集loader大小: {len(test_loader)}')
            test_dict = dict()
            for each in test_dataset:
                test_dict[int(each.y[1])] = test_dict.get(int(each.y[1]), 0) + 1
            logger.info(f'测试集label分布: {dict(sorted(test_dict.items()))}')
            
            
            ## 训练
            # 初始化
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=train_config['lr'])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=train_config['mode'] , factor=train_config['factor'], patience=train_config['patience'], min_lr=train_config['min_lr'], verbose=True)
            
            # 训练历史记录
            columns = ['Epoch', 'Training Time', 'Loss']
            if 'train' in train_config['history']['content']:
                train_columns = ['train_acc', 'train_f1', 'train_precision', 'train_recall', 'train_fpr', 'train_auc']
                columns += train_columns
            
            if 'val' in train_config['history']['content']:
                val_columns = ['val_acc', 'val_f1', 'val_precision', 'val_recall', 'val_fpr', 'val_auc']
                columns += val_columns
            
            if 'test' in train_config['history']['content']:
                test_columns = ['test_acc', 'test_f1', 'test_precision', 'test_recall', 'test_fpr', 'test_auc']
                columns += test_columns
            
            history = pd.DataFrame(columns=columns)
            
            # epoch循环
            epoch_time = 0
            for epoch in range(1, train_config['epochs'] + 1):
                # 垃圾回收
                # logger.info(f'GPU memory usage: {get_gpu_memory_from_nvidia_smi()}')

                gc.collect()
                torch.cuda.empty_cache()
                # logger.info(f'GPU memory usage: {get_gpu_memory_from_nvidia_smi()}')

                
                epoch_start_time = time.time()
                model.train()
                for data in train_loader:  # Iterate in batches over the training dataset.
                    data = data.to(device)
                    out = model(data.x, data.edge_index, data.batch)  if model_config['type'] == 'homogeneous' else model(data.x_dict, data.edge_index_dict, data.batch_dict) # Perform a single forward pass.
                    label_1 = data.y[0::2]
                    loss = criterion(out, label_1)  # Compute the loss.
                    loss.backward()  # Derive gradients.
                    optimizer.step()  # Update parameters based on gradients.
                    optimizer.zero_grad()  # Clear gradients.
                
                scheduler.step(loss, epoch)
                
                epoch_stop_time = time.time()
                epoch_time += epoch_stop_time - epoch_start_time
                
                # 训练信息记录
                logger.info(f'Epoch: {epoch:03d}/{train_config["epochs"]:03d}, Training time: {timedelta(seconds=epoch_time)}, Loss: {loss:.4f}'.center(82, '-'))
                history_row = [epoch, epoch_time, float(loss)]

                if 'train' in train_config['history']['content']:
                    # 将样本统计转为域名数量统计
                    train_y_target, train_y_pred = get_y(model, device, train_loader, model_config['type'])
                    train_y_target = get_y_with_nums(train_y_target, train_dataset, model_config['type'])
                    train_y_pred = get_y_with_nums(train_y_pred, train_dataset, model_config['type'])
                    
                    train_acc, train_f1, train_precision, train_recall, train_fpr, train_auc, train_tb = get_binary_evaluation(train_y_target, train_y_pred)
                    logger.info(f'Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train FPR: {train_fpr}, Train AUC: {train_auc}')
                    history_row += [train_acc, train_f1, train_precision, train_recall, train_fpr, train_auc]

                if 'val' in train_config['history']['content']:
                    val_y_target, val_y_pred = get_y(model, device, val_loader, model_config['type'])
                    val_y_target = get_y_with_nums(val_y_target, val_dataset, model_config['type'])
                    val_y_pred = get_y_with_nums(val_y_pred, val_dataset, model_config['type'])
                    
                    val_acc, val_f1, val_precision, val_recall, val_fpr, val_auc, val_tb = get_binary_evaluation(val_y_target, val_y_pred)
                    logger.info(f'Valid Acc: {val_acc:.4f}, Valid F1: {val_f1:.4f}, Valid Precision: {val_precision:.4f}, Valid Recall: {val_recall:.4f}, Valid FPR: {val_fpr}, Valid AUC: {val_auc}')
                    history_row += [val_acc, val_f1, val_precision, val_recall, val_fpr, val_auc]

                if 'test' in train_config['history']['content']:
                    test_y_target, test_y_pred = get_y(model, device, test_loader, model_config['type'])
                    test_y_target = get_y_with_nums(test_y_target, test_dataset, model_config['type'])
                    test_y_pred = get_y_with_nums(test_y_pred, test_dataset, model_config['type'])
                    
                    test_acc, test_f1, test_precision, test_recall, test_fpr, test_auc, test_tb = get_binary_evaluation(test_y_target, test_y_pred)
                    logger.info(f'Test  Acc: {test_acc:.4f}, Test  F1: {test_f1:.4f}, Test  Precision: {test_precision:.4f}, Test  Recall: {test_recall:.4f}, Test  FPR: {test_fpr}, Test  AUC: {test_auc}')
                    history_row += [test_acc, test_f1, test_precision, test_recall, test_fpr, test_auc]

                # all_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
                # all_y_target, all_y_pred = get_y(model, device, all_loader)
                # all_acc, all_f1, all_precision, all_recall, all_tb = get_multi_evaluation(all_y_target, all_y_pred)
                # logger.info(f'-All- Acc: {all_acc:.4f}, -All- F1: {all_f1:.4f}, -All- Precision: {all_precision:.4f}, -All- Recall: {all_recall:.4f}')

                history.loc[len(history)] = history_row
                
                # 检查点
                if epoch in train_config['check_epoch']:
                    with torch.no_grad():
                        if train_config['save']:
                            os.makedirs(train_config['save_dir'], exist_ok=True)
                            torch.save(model, os.path.join(train_config['save_dir'], f"{each_model}_{epoch}_model.pth"))
                            logger.info(f'模型保存在{os.path.join(train_config["save_dir"], f"{each_model}_{epoch}_model.pth")}')
                        os.makedirs(os.path.join(train_config['logger']['root_dir'], each_model), exist_ok=True)
                        all_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
                        all_y_target, all_y_pred = get_y(model, device, all_loader, model_config['type'])
                        all_y_target = get_y_with_nums(all_y_target, dataset, model_config['type'])
                        all_y_pred = get_y_with_nums(all_y_pred, dataset, model_config['type'])
                        all_cm, all_cm_tb = get_multi_confusion_matrix(all_y_target, all_y_pred, train_config['labels'])
                        logger.info("Confusion Matrix:\n%s", all_cm_tb)
                        cm_plt = get_confusion_matrix_plot(all_cm, train_config['labels'], normalize=False, path=os.path.join(train_config['logger']['root_dir'], f"{each_model}/{epoch}_confusion_matrix.svg"))
                        normalized_cm_plt = get_confusion_matrix_plot(all_cm, train_config['labels'], normalize=True, path=os.path.join(train_config['logger']['root_dir'], f"{each_model}/{epoch}_normalized_confusion_matrix.svg"))

            history_path = os.path.join(train_config['logger']['root_dir'], f"{each_model}/{each_model}_history.csv")
            history.to_csv(history_path, index=False)
            logger.info(f'训练历史记录保存在{history_path}')
    
    inner_function(dataset, homo_dataset, each_model, train_config)
    
def get_y(model, device, loader, type):
    model.eval()
    
    y_pred = []
    y_target = []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch) if type == 'homogeneous' else model(data.x_dict, data.edge_index_dict, data.batch_dict)
        pred = out.argmax(dim=1)
        y_pred.extend(pred.tolist())
        y_target.extend(data.y[0::2].tolist())  # 根据任务进行设定
    
    return y_target, y_pred

def get_y_with_nums(y, dataset, type):
    # 将train_y_target和train_y_pred根据dataloader中相应图中的x数量进行扩充
    if type == 'homogeneous':
        y_nums = [torch.sum(torch.eq(each.node_type, 1)) for each in dataset]
    elif type == 'heterogeneous':
        y_nums = [each['domain'].num_nodes for each in dataset]
    y = np.repeat(y, y_nums)
    return y


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

# def test(model, device, loader):
#     model.eval()

#     correct = 0
#     for data in loader:  # Iterate in batches over the training/test dataset.
#         data.to(device)
#         out = model(data.x, data.edge_index, data.batch)
#         pred = out.argmax(dim=1)  # Use the class with highest probability.
#         correct += int((pred == data.y[1::2]).sum())  # Check against ground-truth labels.
#     return correct / len(loader.dataset)  # Derive ratio of correct predictions.
