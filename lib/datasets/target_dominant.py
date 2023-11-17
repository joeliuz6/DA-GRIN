import os
import numpy as np
import pandas as pd

from lib import datasets_path
from .pd_dataset import PandasDataset
from ..utils import sample_mask,sample_mask_block


class Target_dominant(PandasDataset):
    def __init__(self):
        df,  mask,df_raw = self.load()
        super().__init__(dataframe=df, u=None, mask=mask, name='target', freq='1D', aggr='nearest')
        self.df_raw = df_raw

    def load(self, impute_zeros=True):
        #path = os.path.join(datasets_path['discharge'], 'SSC_discharge.csv')
        df = pd.read_csv('./datasets/discharge/SSC_target_dominant.csv',index_col=0)

        df.index = pd.to_datetime(df.index)
        datetime_idx = sorted(df.index)
        date_range = pd.date_range(datetime_idx[0], datetime_idx[-1], freq='1D')
        df = df.reindex(index=date_range) 
        df = df.loc['2015/4/15':'2021/9/9',:]
        df_raw = df

        mask = ~np.isnan(df.values)
        df.fillna(method='ffill', axis=0, inplace=True)
        # print(df_raw.isna().sum().sum())
        # print(np.count_nonzero(mask==True))
        return df.astype('float32'),  mask.astype('uint8'), df_raw.astype('float32')

    def get_similarity(self, thr=0.1, force_symmetric=False, sparse=False):
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

class MissingValuesTarget_dominant(Target_dominant):
    SEED = 56789
    def __init__(self, p_fault=0.0015, p_noise=0):
        super(MissingValuesTarget_dominant, self).__init__()
        self.rng = np.random.default_rng(self.SEED)
        self.p_fault = p_fault
        self.p_noise = p_noise
        # print(np.count_nonzero(self.mask))
        eval_mask = sample_mask(self.mask[0:2110,:], p = 0.6)
        
        eval_mask_block = sample_mask_block(eval_mask[-230:,:].shape,
                                p=0,
                                p_noise=0.25,
                                min_seq=5,
                                max_seq=15,
                                rng=self.rng)

        # self.eval_mask = (eval_mask & self.mask).astype('uint8')
        
        self.eval_mask = np.concatenate((eval_mask,eval_mask_block),axis=0)
        
        # print(self.df_raw.size - np.count_nonzero(self.eval_mask))
        #self.eval_mask = np.array(pd.read_csv(r'C:\Users\89457\Desktop\optimizaiton\Spatial-Temporal\spatial-temporal\grin-main\mask.csv',index_col=0).values).astype('uint8')

    @property
    def training_mask(self):
        # print(np.count_nonzero(self.mask))
        # # print(np.count_nonzero(1 - self.eval_mask))
        # print(np.count_nonzero(self.mask & (1 - self.eval_mask)))
        # exit()
        return self.mask if self.eval_mask is None else (self.mask & (1 - self.eval_mask))

    def splitter(self, dataset, val_len=0, test_len=0, window=0):
        idx = np.arange(len(dataset))
        # print(idx.shape)
        test_len = 180
        val_len = 50

        test_start = len(idx) - test_len
        val_start = test_start - val_len


        return [idx[:val_start], idx[val_start:test_start], idx[test_start:]]



