import os
import time
import torch
os.environ['TORCH'] = torch.__version__

from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter
from torch.nn.functional import pad

import yaml
import os
import torch
from torch_geometric.data import InMemoryDataset, HeteroData
import torch_geometric.transforms as T
import pandas as pd
from tqdm import tqdm
tqdm.pandas(desc='Pandas bar')

import re
pattern = re.compile(r'[^\"\'\!\@\#\$\%\^\&\*\(\)\_\+\[\\\]\{\}\;\:\,\.\/\<\>\?\|\`\~\-\=0-9a-zA-Z]')

INTERVAL_MAP = {
    'year': '%Y',
    'month': '%Y-%m',
    'day': '%Y-%m-%d',
    'hour': '%Y-%m-%d %H',
    'minute': '%Y-%m-%d %H:%M',
    'second': '%Y-%m-%d %H:%M:%S',
    'millisecond': '%Y-%m-%d %H:%M:%S.%f'
    }


# from pandarallel import pandarallel

# pandarallel.initialize(progress_bar=True)


from utils import log
from feature.encoder import get_encoder

class DNG(InMemoryDataset):
    # 域名图类
    def __init__(self, data_config, transform=None, pre_transform=None, pre_filter=None):
        self.data_config = data_config
        self.y = []
        super().__init__(data_config['raw_path'], transform, pre_transform, pre_filter)
        self.root = data_config['raw_path']
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['all.parquet']

    @property
    def processed_file_names(self):
        return [self.data_config['interval'] + '.pt', 'clientIP.csv', 'domain.csv', 'rdata.csv', 'sld_label.csv', str(self.data_config['sample_rate']) + '_sample.csv', ]

    def download(self):
        pass

    def process(self):
        self.logger = log.get_logger(os.path.join(self.data_config['logger']['root_dir'], 'data' + self.data_config['logger']['postfix']), self.data_config['logger']['verbosity'], self.data_config['logger']['name'])
        self.logger.info(f'Raw config: \n{self.data_config}')
        start_time = time.time()
        # Read data into huge `Data` list.
        data_list = self.load_datas()
        self.y = torch.tensor([data.y.tolist() for data in data_list])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        end_time = time.time()
        self.logger.info(f'处理时间：{end_time - start_time:.4f}秒，处理图数：{len(data_list)}')
        self.logger.info(f'Torch Version:{torch.__version__}')
        self.logger.info(f'第一张图信息：\n{data_list[0]}')
        self.logger.info(f'clientIP长度：\n{data_list[0]["clientIP"].x.shape}')
        self.logger.info(f'clientIP信息：\n{data_list[0]["clientIP"]}')
        self.logger.info(f'domain长度：\n{data_list[0]["domain"].x.shape}')
        self.logger.info(f'domain信息：\n{data_list[0]["domain"]}')
        self.logger.info(f'rdata长度：\n{data_list[0]["rdata"].x.shape}')
        self.logger.info(f'rdata信息：\n{data_list[0]["rdata"]}')
        self.logger.info(f'request长度：\n{data_list[0]["clientIP", "request", "domain"].edge_index.shape}')
        self.logger.info(f'request信息：\n{data_list[0]["clientIP", "request", "domain"]}')
        self.logger.info(f'resolve长度：\n{data_list[0]["domain", "resolve", "rdata"].edge_index.shape}')
        self.logger.info(f'resolve信息：\n{data_list[0]["domain", "resolve", "rdata"]}')
        self.logger.info(f'y信息：\n{data_list[0].y}')
        
        self.logger.info(f'数据保存于{self.processed_paths[0]}')
        self.logger.info(f'数据集信息：\n{data_list}')

    def __repr__(self):
        return f'{self.__class__.__name__}({len(self)})'

    def load_datas(self):
        labels = ['label_1', 'label_2']
            
        # 读取parquet文件
        pdns = pd.read_parquet(os.path.join(self.root, 'raw', self.raw_file_names[0])).sample(frac=self.data_config['sample_rate'], random_state=42).reset_index(drop=True)
        pdns['pdate'] = pdns['pdate'].astype('datetime64[ns]').dt.strftime(INTERVAL_MAP[self.data_config['interval']])
        
        # 只保留pdns['rdata']中的字母、数字、常用符号
        pdns['rdata'] = pdns['rdata'].apply(lambda x: re.sub(pattern, '', x))
        pdns.to_csv(os.path.join(self.root, 'processed', self.processed_file_names[5]), index=True)
        
        # 从pdns中提取sld与label列，保存为sld_label.csv，并以sld排序，去重
        sld_label = pdns[['sld']+labels].drop_duplicates().sort_values(by=labels+['sld'])
        sld_label.to_csv(os.path.join(self.root, 'processed', self.processed_file_names[4]), index=False)
        # 从pdns中删去fqdn列中最后的'sld'部分
        pdns['fqdn'] = pdns['fqdn'].apply(lambda x: '.'.join(x.split('.')[:-2]))
        # 如果fqdn中有空值，用'nan'填充
        pdns['fqdn'] = pdns['fqdn'].fillna('NAN')
        
        # 生成clientIP.csv, domain.csv, rdata.csv，用于实体对应
        clientIP_df, domain_df, rdata_df, clientIP_map, domain_map, rdata_map = get_entity(pdns)
        clientIP_df.to_csv(os.path.join(self.root, 'processed', self.processed_file_names[1]), index=True, index_label='clientIP_node')
        domain_df.to_csv(os.path.join(self.root, 'processed', self.processed_file_names[2]), index=True, index_label='domain_node')
        rdata_df.to_csv(os.path.join(self.root, 'processed', self.processed_file_names[3]), index=True, index_label='rdata_node')
        
        # load_node
        clientIP_encoder = get_encoder(self.data_config['encoder']['clientIP'])
        domain_encoder = get_encoder(self.data_config['encoder']['domain'])
        rdata_encoder = get_encoder(self.data_config['encoder']['rdata'])
        
        encoder_start_time = time.time()
        clientIP_x, clientIP_mapping = load_node(os.path.join(self.root, 'processed', self.processed_file_names[1]), index_col='clientIP_node', encoders={'clientIP': clientIP_encoder}, length=self.data_config['encoder']['length'])
        client_encoder_end_time = time.time()
        self.logger.info(f'{self.data_config["encoder"]["clientIP"]}处理时间：{client_encoder_end_time - encoder_start_time:.4f}秒')
        domain_x, domain_mapping = load_node(os.path.join(self.root, 'processed', self.processed_file_names[2]), index_col='domain_node', encoders={'domain': domain_encoder}, length=self.data_config['encoder']['length'])
        domain_encoder_end_time = time.time()
        self.logger.info(f'{self.data_config["encoder"]["domain"]}处理时间：{domain_encoder_end_time - client_encoder_end_time:.4f}秒')
        rdata_x, rdata_mapping = load_node(os.path.join(self.root, 'processed', self.processed_file_names[3]) , index_col='rdata_node', encoders={'rdata': rdata_encoder}, length=self.data_config['encoder']['length'])
        rdata_encoder_end_time = time.time()
        self.logger.info(f'{self.data_config["encoder"]["rdata"]}处理时间：{rdata_encoder_end_time - domain_encoder_end_time:.4f}秒')
        encoder_end_time = time.time()
        self.logger.info(f'Encoder总处理时间：{encoder_end_time - encoder_start_time:.4f}秒')

        # 将pdns中的client_ip, fqdn, rdata转换为clientIP_df, domain_df, rdata_df中的index
        map_start_time = time.time()
        pdns['client_ip'] = pdns['client_ip'].apply(lambda x: clientIP_map[x])
        self.logger.info('client_ip done')
        pdns['fqdn']      = pdns['fqdn']     .apply(lambda x: domain_map  [x])
        self.logger.info('fqdn done')
        pdns['rdata']     = pdns['rdata']    .apply(lambda x: rdata_map   [x])
        self.logger.info('rdata done')
        map_end_time = time.time()
        self.logger.info(f'map处理时间：{map_end_time - map_start_time:.4f}秒')
        
        # 创建该时间间隔下的data_list
        data_list = []
        # 根据sld和时间分割group
        data_start_time = time.time()
        groups = pdns.groupby(['sld', 'pdate'])
        # 对每个group使用load_data函数得到cur_data
        for parentDomain_time, group in tqdm(groups, desc='PT grouping'):
            # 读取相应时间间隔的pt
            request_index, _ = load_edge(
                group,
                src_index_col='client_ip',
                src_mapping=clientIP_mapping,
                dst_index_col='fqdn',
                dst_mapping=domain_mapping,
            )
            resolve_index, _ = load_edge(
                group,
                src_index_col='fqdn',
                src_mapping=domain_mapping,
                dst_index_col='rdata',
                dst_mapping=rdata_mapping,
            )

            # data = HeteroData(y=torch.tensor([group[label].iloc[0]], dtype=torch.long))
            # data = HeteroData(y=torch.tensor(group[labels].iloc[0].values, dtype=torch.long).squeeze(0))
            data = HeteroData()

            data['clientIP'].x = clientIP_x
            data['domain'].x = domain_x
            data['rdata'].x = rdata_x

            data['clientIP', 'request', 'domain'].edge_index = request_index
            data['domain', 'resolve', 'rdata'].edge_index = resolve_index

            # 给cur_data打标签
            cur_data = data
            cur_data.y = torch.tensor(group[labels].iloc[0].values, dtype=torch.long).squeeze(0)

            # 其他属性
            cur_data.parentDomain = parentDomain_time[0]
            cur_data.time = parentDomain_time[1]
            cur_data['clientIP'].encoder = clientIP_encoder
            cur_data['domain'].encoder = domain_encoder
            cur_data['rdata'].encoder = rdata_encoder

            # 去除孤立点
            # transform = T.Compose([T.remove_isolated_nodes.RemoveIsolatedNodes(), T.ToUndirected()])
            transform = T.Compose([T.remove_isolated_nodes.RemoveIsolatedNodes()])
            
            cur_data = transform(data)

            # 加入data_list
            data_list.append(cur_data)
        
        data_end_time = time.time()
        self.logger.info(f'分组处理时间：{data_end_time - data_start_time:.4f}秒')

        return data_list
    
    # def shuffle(self, seed=None, return_perm: bool = False):
    #     # 使用 torch.manual_seed 设置相同的随机种子
    #     torch.manual_seed(seed)
    #     super(DNG, self).shuffle(self)
        
    #     perm = torch.randperm(len(self))
    #     dataset = self.index_select(perm)
    #     return (dataset, perm) if return_perm is True else dataset

def get_entity(pdns):
    # 提取clientIP
    clientIP_df = pd.DataFrame({'clientIP': pdns['client_ip'].unique()}).reset_index()
    # 提取domain
    domain_df = pd.DataFrame({'domain': pdns['fqdn'].unique()}).reset_index()
    # 提取rdata
    rdata_df = pd.DataFrame({'rdata': pdns['rdata'].unique()}).reset_index()
    
    clientIP_map = clientIP_df.set_index(['clientIP'])['index'].to_dict()
    domain_map = domain_df.set_index(['domain'])['index'].to_dict()
    rdata_map = rdata_df.set_index(['rdata'])['index'].to_dict()
    
    return clientIP_df, domain_df, rdata_df, clientIP_map, domain_map, rdata_map

def load_node(csv, index_col, encoders=None, length=128):
    df = pd.read_csv(csv, index_col=index_col)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)
        if x.shape[1] < length:
            x = pad(x, (0, length - x.shape[1]), mode='constant', value=0)
        elif x.shape[1] > length:
            x = x[:, :length]

    return x, mapping

def load_edge(df, src_index_col, src_mapping, dst_index_col, dst_mapping, encoders=None):
    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr