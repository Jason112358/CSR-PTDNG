import torch
from sentence_transformers import SentenceTransformer
from feature.fanci import extract_all_features, split_all_col

DEFAULT_DIM = 64

def get_encoder(config):
    if config == 'RandomEncoder':
        return RandomEncoder()
    elif config == 'BertEncoder':
        return BertEncoder()
    elif config == 'ZerosEncoder':
        return ZerosEncoder()
    elif config == 'OnesEncoder':
        return OnesEncoder()
    elif config == 'FeatureEncoder':
        return FeatureEncoder()
    else :
        return None

# Encoder部分，用于将Fqdn和Rdata转换为向量
class RandomEncoder:
    def __init__(self, seed=42, device=None, length=DEFAULT_DIM):
        self.device = device
        self.model = None
        self.length = length

    @torch.no_grad()
    def __call__(self, df):
        x = torch.rand([df.values.shape[0], self.length]).long()
        return x.cpu()

class ZerosEncoder:
    def __init__(self, device=None, length=DEFAULT_DIM):
        self.device = device
        self.model = None
        self.length = length
    
    @torch.no_grad()
    def __call__(self, df):
        x = torch.zeros([df.values.shape[0], self.length])
        return x.cpu()

class OnesEncoder:
    def __init__(self, device=None, length=DEFAULT_DIM):
        self.device = device
        self.model = None
        self.length = length
    
    @torch.no_grad()
    def __call__(self, df):
        x = torch.ones([df.values.shape[0], self.length])
        return x.cpu()
    
class BertEncoder:
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None, length=DEFAULT_DIM):
      self.device = device
      self.model = SentenceTransformer(model_name, device=device)
      self.length = length

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(df.values, show_progress_bar=True, convert_to_tensor=True, device=self.device)
        return x.cpu()

class FeatureEncoder:
    def __init__(self, device=None, length=DEFAULT_DIM):
        self.model = extract_all_features
        self.length = length

    @torch.no_grad()
    def __call__(self, df):
        features = self.model(df)
        features = split_all_col(features, column_list=["_parts", "_n_grams"])
        x = torch.tensor(features.iloc[:, 1:].values.tolist())
        return x.cpu()