import copy
import datetime
import os
import pathlib
import sys
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import pandas as pd 
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import CosineAnnealingLR
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib import fillers, datasets, config
from lib.data.datamodule import SpatioTemporalDataModule
from lib.data.imputation_dataset import ImputationDataset, GraphImputationDataset
from lib.nn import models
from lib.nn.utils.metric_base import MaskedMetric
from lib.nn.utils.metrics import MaskedMAE, MaskedMAPE, MaskedMSE, MaskedMRE
from lib.utils import parser_utils, numpy_metrics, ensure_list, prediction_dataframe
from lib.utils.parser_utils import str_to_bool
from pytorch_lightning.utilities import CombinedLoader

def has_graph_support(model_cls):
    return model_cls in [models.GRINet, models.MPGRUNet, models.BiMPGRUNet]


def get_model_classes(model_str):
    if model_str == 'brits':
        model, filler = models.BRITSNet, fillers.BRITSFiller
    elif model_str == 'grin':
        model, filler = models.GRINet, fillers.GraphFiller
    elif model_str == 'mpgru':
        model, filler = models.MPGRUNet, fillers.GraphFiller
    elif model_str == 'bimpgru':
        model, filler = models.BiMPGRUNet, fillers.GraphFiller
    elif model_str == 'var':
        model, filler = models.VARImputer, fillers.Filler
    elif model_str == 'gain':
        model, filler = models.RGAINNet, fillers.RGAINFiller
    elif model_str == 'birnn':
        model, filler = models.BiRNNImputer, fillers.MultiImputationFiller
    elif model_str == 'rnn':
        model, filler = models.RNNImputer, fillers.Filler
    else:
        raise ValueError(f'Model {model_str} not available.')
    return model, filler


def get_dataset(dataset_name):
    if dataset_name[:3] == 'air':
        dataset = datasets.AirQuality(impute_nans=True, small=dataset_name[3:] == '36')
    elif dataset_name == 'bay_block':
        dataset = datasets.MissingValuesPemsBay()
    elif dataset_name == 'la_block':
        dataset = datasets.MissingValuesMetrLA()
    elif dataset_name == 'discharge_block':
        dataset = datasets.MissingValuesDischarge()
        target_dataset = datasets.MissingValuesTarget()
    elif dataset_name == 'la_point':
        dataset = datasets.MissingValuesMetrLA(p_fault=0., p_noise=0.25)
    elif dataset_name == 'bay_point':
        dataset = datasets.MissingValuesPemsBay(p_fault=0., p_noise=0.25)
    elif dataset_name == 'discharge_point':
        dataset = datasets.MissingValuesDischarge(p_fault=0., p_noise=0.25)
        target_dataset = datasets.MissingValuesTarget(p_fault=0., p_noise=0.25)
    else:
        raise ValueError(f"Dataset {dataset_name} not available in this setting.")
    return dataset,target_dataset


def parse_args():
    # Argument parser
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument("--model-name", type=str, default='grin')
    parser.add_argument("--dataset-name", type=str, default='air36')
    parser.add_argument("--config", type=str, default=None)
    # Splitting/aggregation params
    parser.add_argument('--in-sample', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--val-len', type=float, default=0.1)
    parser.add_argument('--test-len', type=float, default=0.2)
    parser.add_argument('--aggregate-by', type=str, default='mean')
    # Training params
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--l2-reg', type=float, default=0.)
    parser.add_argument('--scaled-target', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--grad-clip-val', type=float, default=5.)
    parser.add_argument('--grad-clip-algorithm', type=str, default='norm')
    parser.add_argument('--loss-fn', type=str, default='l1_loss')
    parser.add_argument('--use-lr-schedule', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--consistency-loss', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--whiten-prob', type=float, default=0.05)
    parser.add_argument('--pred-loss-weight', type=float, default=1.0)
    parser.add_argument('--warm-up', type=int, default=0)
    # graph params
    parser.add_argument("--adj-threshold", type=float, default=0.1)
    # gain hparams
    parser.add_argument('--alpha', type=float, default=10.)
    parser.add_argument('--hint-rate', type=float, default=0.7)
    parser.add_argument('--g-train-freq', type=int, default=1)
    parser.add_argument('--d-train-freq', type=int, default=5)

    #### graph_constructor ### 
    parser.add_argument('--buildA_true', type=bool, default=True, help='whether to construct adaptive adjacency matrix')
    parser.add_argument('--subgraph_size',type=int,default=1,help='k')
    parser.add_argument('--node_dim',type=int,default=40,help='dim of nodes')
    parser.add_argument('--device',type=str,default='cuda:0',help='')

    known_args, _ = parser.parse_known_args()
    model_cls, _ = get_model_classes(known_args.model_name)
    parser = model_cls.add_model_specific_args(parser)
    parser = SpatioTemporalDataModule.add_argparse_args(parser)
    parser = ImputationDataset.add_argparse_args(parser)

    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, 'r') as fp:
            config_args = yaml.load(fp, Loader=yaml.FullLoader)
        for arg in config_args:
            setattr(args, arg, config_args[arg])

    return args


def run_experiment(args):
    # Set configuration and seed
    args = copy.deepcopy(args)
    if args.seed < 0:
        args.seed = np.random.randint(1e9)
    torch.set_num_threads(1)
    pl.seed_everything(args.seed)

    model_cls, filler_cls = get_model_classes(args.model_name)
    dataset,target_dataset = get_dataset(args.dataset_name)

    ########################################
    # create logdir and save configuration #
    ########################################

    exp_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.seed}"
    logdir = os.path.join(config['logs'], args.dataset_name, args.model_name, exp_name)
    # save config for logging
    pathlib.Path(logdir).mkdir(parents=True)
    with open(os.path.join(logdir, 'config.yaml'), 'w') as fp:
        yaml.dump(parser_utils.config_dict_from_args(args), fp, indent=4, sort_keys=True)

    ########################################
    # data module                          #
    ########################################

    # instantiate dataset
    dataset_cls = GraphImputationDataset if has_graph_support(model_cls) else ImputationDataset
    torch_dataset = dataset_cls(*target_dataset.numpy(return_idx=True),
                                mask=target_dataset.training_mask,
                                eval_mask=target_dataset.eval_mask,
                                window=args.window,
                                stride=args.stride)
    
    torch_target_dataset = dataset_cls(*target_dataset.numpy(return_idx=True),
                                mask=target_dataset.training_mask,
                                eval_mask=target_dataset.eval_mask,
                                window=args.window,
                                stride=args.stride)

    # get train/val/test indices
    split_conf = parser_utils.filter_function_args(args, dataset.splitter, return_dict=True)
    train_idxs, val_idxs, test_idxs = dataset.splitter(torch_dataset, **split_conf)


    #### target dataset split ### 

    #split_conf = parser_utils.filter_function_args(args, target_dataset.splitter, return_dict=True)
    #train_idxs, val_idxs, test_idxs = target_dataset.splitter(torch_target_dataset, **split_conf)


    ## no change on  the target dataset yet ### 

    # configure datamodule
    data_conf = parser_utils.filter_args(args, SpatioTemporalDataModule, return_dict=True)
    dm = SpatioTemporalDataModule(torch_dataset,torch_target_dataset, train_idxs=train_idxs, val_idxs=val_idxs, test_idxs=test_idxs,
                                  **data_conf)
    dm.setup()

    # if out of sample in air, add values removed for evaluation in train set
    if not args.in_sample and args.dataset_name[:3] == 'air':
        dm.torch_dataset.mask[dm.train_slice] |= dm.torch_dataset.eval_mask[dm.train_slice]

    # get adjacency matrix
    adj = dataset.get_similarity(thr=args.adj_threshold)
    # force adj with no self loop
    np.fill_diagonal(adj, 0.)


    #iterables = {"source": dm, "target": dm}
    #combined_loader = CombinedLoader(iterables, mode="min_size")

    ########################################
    # predictor                            #
    ########################################

    # model's inputs
    additional_model_hparams = dict(adj=adj, d_in=dm.d_in, n_nodes=dm.n_nodes,device = args.device,
                        num_nodes = 20,
                        subgraph_size = args.subgraph_size,
                        node_dim=args.node_dim,
                        buildA_true = args.buildA_true)
    model_kwargs = parser_utils.filter_args(args={**vars(args), **additional_model_hparams},
                                            target_cls=model_cls,
                                            return_dict=True)

    # loss and metrics
    loss_fn = MaskedMetric(metric_fn=getattr(F, args.loss_fn),
                           compute_on_step=True,
                           metric_kwargs={'reduction': 'none'})

    metrics = {'mae': MaskedMAE(compute_on_step=False),
               'mape': MaskedMAPE(compute_on_step=False),
               'mse': MaskedMSE(compute_on_step=False),
               'mre': MaskedMRE(compute_on_step=False)}

    # filler's inputs
    scheduler_class = CosineAnnealingLR if args.use_lr_schedule else None
    additional_filler_hparams = dict(model_class=model_cls,
                                     model_kwargs=model_kwargs,
                                     optim_class=torch.optim.Adam,
                                     optim_kwargs={'lr': args.lr,
                                                   'weight_decay': args.l2_reg},
                                     loss_fn=loss_fn,
                                     metrics=metrics,
                                     scheduler_class=scheduler_class,
                                     scheduler_kwargs={
                                         'eta_min': 0.0001,
                                         'T_max': args.epochs
                                     },
                                     alpha=args.alpha,
                                     hint_rate=args.hint_rate,
                                     g_train_freq=args.g_train_freq,
                                     d_train_freq=args.d_train_freq)
    filler_kwargs = parser_utils.filter_args(args={**vars(args), **additional_filler_hparams},
                                             target_cls=filler_cls,
                                             return_dict=True)
    filler = filler_cls(**filler_kwargs)

    ########################################
    # training                             #
    ########################################

    # callbacks
    early_stop_callback = EarlyStopping(monitor='val_mae/dataloader_idx_1', patience=args.patience, mode='min')
    checkpoint_callback = ModelCheckpoint(dirpath=logdir, save_top_k=1, monitor='val_mae/dataloader_idx_1', mode='min')

    logger = TensorBoardLogger(logdir, name="model")

    trainer = pl.Trainer(max_epochs=args.epochs,
                         logger=logger,
                         default_root_dir=logdir,
                         accelerator='auto',
                         gradient_clip_val=args.grad_clip_val,
                         gradient_clip_algorithm=args.grad_clip_algorithm,
                         callbacks=[early_stop_callback, checkpoint_callback])

    trainer.fit(filler, datamodule=dm)

    ########################################
    # testing                              #
    ########################################

    filler.load_state_dict(torch.load(checkpoint_callback.best_model_path,
                                      lambda storage, loc: storage)['state_dict'])
    filler.freeze()
    trainer.test(datamodule=dm)
    filler.eval()

    if torch.cuda.is_available():
       filler =  filler.to(args.device)

    with torch.no_grad():
        y_true, y_hat, mask = filler.predict_loader(dm.test_dataloader()['target'], return_mask=True)
    y_hat = y_hat.detach().cpu().numpy().reshape(y_hat.shape[:3])  # reshape to (eventually) squeeze node channels
    #y_hat = np.exp(y_hat) -  1e-5
    # Test imputations in whole series
    eval_mask = target_dataset.eval_mask[dm.test_slice]
    df_true = target_dataset.df.iloc[dm.test_slice]
    df_true_raw = target_dataset.df_raw.iloc[dm.test_slice]

    

    #df_true = df_true.transform(lambda x: np.exp(x) - 1e-5)

    metrics = {
        'mae': numpy_metrics.masked_mae,
        'mse': numpy_metrics.masked_mse,
        'mre': numpy_metrics.masked_mre,
        'mape': numpy_metrics.masked_mape
    }
    # Aggregate predictions in dataframes
    index = dm.torch_target_dataset.data_timestamps(dm.testset.indices, flatten=False)['horizon']
    aggr_methods = ensure_list(args.aggregate_by)
    df_hats = prediction_dataframe(y_hat, index, target_dataset.df.columns, aggregate_by=aggr_methods)
    df_hats = dict(zip(aggr_methods, df_hats))
    for aggr_by, df_hat in df_hats.items():
        # Compute error
        print(f'- AGGREGATE BY {aggr_by.upper()}')
        for metric_name, metric_fn in metrics.items():
            error = metric_fn(df_hat.values, df_true.values, eval_mask).item()
            print(f' {metric_name}: {error:.4f}')

            #### de_transform of df_values ## 
            error_real = metric_fn(df_hat.values, df_true_raw.values, eval_mask).item()
            print(f' {metric_name}: {error_real:.4f}')

    mask = pd.DataFrame(eval_mask)
    #data_pre = (df_hat-1) * 3759.35 + 1237.58
    data_pre = df_hat
    data_pre.to_csv('%s_pre.csv'%(args.model_name),header=None)
    df_true_raw.to_csv('true.csv',header=None)
    mask.to_csv('%s_mask.csv'%(args.model_name),header=None,index = None)
    return y_true, y_hat, mask


if __name__ == '__main__':
    args = parse_args()
    run_experiment(args)
