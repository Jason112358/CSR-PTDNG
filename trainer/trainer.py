from datetime import timedelta
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import beautifultable as bt
import time
import torch
import importlib
from torch_geometric.loader import DataLoader   # 为什么不同于视频的dataloader.DataLoader
from utils import log
import os

def train(dataset, train_config):
    logger = log.get_logger(**train_config['logger'])
    
    # # CUDA检查
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(train_config['device'])
    logger.info(f'Using device {device}')

    start_time = time.time()
    # 在模块中查找模型类并加入表示模型列表
    module = importlib.import_module('.'.join([train_config['model_root'], train_config['model']]))
    model_class = getattr(module, train_config['model'])
    model_config = train_config[train_config['model']]
    model = model_class(**model_config['parameters']).to(device)
    logger.info(f'Model: {model}')
    
    dataset = [each_graph.to_homogeneous() for each_graph in dataset] if model_config['type'] == 'homogeneous' else dataset
    
    # 初始化
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=train_config['mode'] , factor=train_config['factor'], patience=train_config['patience'], min_lr=train_config['min_lr'], verbose=True)
    
    # 划分数据集
    train_dataset = dataset[:int(len(dataset) * train_config['train_ratio'])]
    val_dataset = dataset[int(len(dataset) * train_config['train_ratio']):]
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    logger.info(f'Train dataset size: {len(train_dataset)}')
    logger.info(f'Val dataset size: {len(val_dataset)}')
    logger.info(f'Train loader size: {len(train_loader)}')
    logger.info(f'Val loader size: {len(val_loader)}')

    # 训练
    for epoch in range(0, train_config['epochs'] + 1):
        model.train()
        for data in train_loader:  # Iterate in batches over the training dataset.
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            loss = criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
            
        train_acc = test(model, device, train_loader)
        test_acc = test(model, device, val_loader)
        logger.info(f'Epoch: {epoch:03d}/{train_config["epochs"]:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {test_acc:.4f}')
        
        # 评价指标
        if epoch % 10 == 0:
            tmp_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
            # 根据模型预测结果得到y_pred
            y_pred = []
            for data in tmp_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.batch)
                pred = out.argmax(dim=1)
                y_pred.extend(pred.tolist())
                
            y_target = data.y.cpu().numpy()
            cm = confusion_matrix(y_target, y_pred)
            auc = roc_auc_score(y_target, y_pred)
            acc = accuracy_score(y_target, y_pred)
            f1 = f1_score(y_target, y_pred)
            precision = precision_score(y_target, y_pred)
            recall = recall_score(y_target, y_pred)
            # 用beautiful table打印混淆矩阵
            cm_tb = bt.BeautifulTable()
            cm_tb.column_headers = ['DNS Tunneling', 'Background']
            cm_tb.append_row([cm[0][0], cm[0][1]])
            cm_tb.append_row([cm[1][0], cm[1][1]])
            cm_tb.rows.header = ['Positive', 'Negative']
            logger.info("Confusion Matrix:\n%s", cm_tb)
            # 用beautiful table打印评价指标
            eval_tb = bt.BeautifulTable()
            eval_tb.column_headers = ['AUC', 'ACC', 'F1', 'Precision', 'Recall']
            eval_tb.append_row([auc, acc, f1, precision, recall])
            logger.info("Evaluation:\n%s", eval_tb)
    
    end_time = time.time()
    logger.info(f'Training time: {timedelta(seconds=end_time - start_time)}')

def test(model, device, loader):
    model.eval()

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.
