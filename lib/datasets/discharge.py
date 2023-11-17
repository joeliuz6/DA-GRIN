import os

import numpy as np
import pandas as pd

from lib import datasets_path
from .pd_dataset import PandasDataset
from ..utils import sample_mask



class Discharge(PandasDataset):
    def __init__(self):
        df,  mask,df_raw = self.load()
        #self.dist = dist
        super().__init__(dataframe=df, u=None, mask=mask, name='discharge', freq='1D', aggr='nearest')
        self.df_raw = df_raw

    def load(self, impute_zeros=True):
        #path = os.path.join(datasets_path['discharge'], 'SSC_discharge.csv')
        df = pd.read_csv('./datasets/discharge/SSC_discharge.csv',index_col=0)
        
        df.index = pd.to_datetime(df.index)
        datetime_idx = sorted(df.index)
        date_range = pd.date_range(datetime_idx[0], datetime_idx[-1], freq='1D')
        df = df.reindex(index=date_range) 

        #### log transformation #####
        #df = df.transform(lambda x: np.log(x+1))
        #df.replace(-np.inf,np.nan,inplace=True)
        df = df.loc['2015/4/15':'2021/9/9',:]
        df_raw = df

        #df_target = df_target.loc['2021/1/4':'2021/9/8',:]
        #df = df.loc['2021/1/4':'2021/9/8',:]
        #df = pd.concat([df,df_target],axis=1)
        mask = ~np.isnan(df.values)
        df.fillna(method='ffill', axis=0, inplace=True)
        #print(mask.shape)
        # dist = self.load_distance_matrix(list(df.columns))
        #print(df)
        return df.astype('float32'),  mask.astype('uint8'), df_raw.astype('float32')

    def get_similarity(self, thr=0.1, force_symmetric=False, sparse=False):
        #path = os.path.join(datasets_path['discharge'], 'SSC_sites_flow_direction.csv')
        #adj = np.array(pd.read_csv(r'C:\Users\89457\Desktop\optimizaiton\Spatial-Temporal\spatial-temporal\grin-main\datasets/discharge\SSC_sites_flow_direction.csv',index_col=0).values)
        #adj = np.ones((40,40))
        adj = np.array(pd.read_csv('./datasets/discharge/SSC_sites_flow_direction.csv',index_col=0).values)
        #print(adj.shape)
        if force_symmetric:
            adj = np.maximum.reduce([adj, adj.T])
        if sparse:
            import scipy.sparse as sps
            adj = sps.coo_matrix(adj)
        return adj

    @property
    def mask(self):
        if self._mask is None:
            return self.df.values != 0.
        return self._mask

class MissingValuesDischarge(Discharge):
    SEED = 56789
    def __init__(self, p_fault=0.0015, p_noise=0.05):
        super(MissingValuesDischarge, self).__init__()
        self.rng = np.random.default_rng(self.SEED)
        self.p_fault = p_fault
        self.p_noise = p_noise
        # print(self.df_raw.shape)

        eval_mask = sample_mask(self.mask,p=0.6)
        # self.eval_mask = (eval_mask & self.mask).astype('uint8')
        self.eval_mask = eval_mask
        #self.eval_mask = np.array(pd.read_csv(r'C:\Users\89457\Desktop\optimizaiton\Spatial-Temporal\spatial-temporal\grin-main\mask.csv',index_col=0).values).astype('uint8')

    @property
    def training_mask(self):
        # print(type(self.mask))
        # print(self.mask.size - np.count_nonzero(self.mask))

        return self.mask if self.eval_mask is None else (self.mask & (1 - self.eval_mask))

    def splitter(self, dataset, val_len=0, test_len=0, window=0):
        idx = np.arange(len(dataset))
        # train = idx[:2091]
        # val = idx[2091:2271]
        # test = idx[2271:2324]
        test_len = 180
        val_len = 50

        test_start = len(idx) - test_len
        val_start = test_start - val_len


        return [idx[:val_start - window], idx[val_start:test_start - window], idx[test_start:]]



