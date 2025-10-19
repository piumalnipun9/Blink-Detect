import os
import time
import random
import warnings
import argparse
import pathlib

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
from data_trainer import *
from model_trainer import VideoClassificaiton
import data_transform as T
from utils import print_on_rank_zero
import warnings
warnings.filterwarnings("ignore")
from pytorch_lightning import loggers as pl_loggers
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from pytorch_lightning import seed_everything

def parse_args():
    parser = argparse.ArgumentParser(description='lr receiver')
    # Common
    parser.add_argument(
        '-epoch', type=int, default=50,
        help='the max epochs of training')
    parser.add_argument(
        '-batch_size', type=int, required=True,
        help='the batch size of data inputs')
    parser.add_argument(
        '-num_workers', type=int, default=4,
        help='the num workers of loading data')
    parser.add_argument(
        '-resume', default=False, action='store_true')
    parser.add_argument(
        '-resume_from_checkpoint', type=str, default=None,
        help='the pretrain params from specific path')
    parser.add_argument(
        '-log_interval', type=int, default=30,
        help='the intervals of logging')
    parser.add_argument(
        '-objective', type=str, default='mim',
        help='the learning objective from [mim, supervised]')
    parser.add_argument(
        '-eval_metrics', type=str, default='finetune',
        help='the eval metrics choosen from [linear_prob, finetune]')
    # add an arugment for the experiment name
    parser.add_argument(
        '-exp_name', type=str, default='experiment_name',
        help='the name of experiment')

    # Environment
    parser.add_argument(
        '-gpus', nargs='+', type=int, default=-1,
        help='the avaiable gpus in this experiment')
    parser.add_argument(
        '-root_dir', type=str, default="./",
        help='the path to root dir for work space')

    # Data
    parser.add_argument(
        '-dataset', type=str, default=None,
        help='the dataset used in this experiment [synblink50knpy, hust, talkingface]')
    parser.add_argument(
        '-seq_length', type=int, default=13,
        help='number of frames per sample (used by talkingface pipeline)')
    parser.add_argument(
        '-talkingface_root', type=str, default=str(pathlib.Path('BlinkFormer') / 'data_preprocess' / 'talking_face'),
        help='path to Talking Face preprocessed npy files')
        
    # Model
    parser.add_argument(
        '-arch', type=str, default='BlinkFormer',
        help='the choosen model arch from [BlinkFormer, BlinkFormer_with_BSE_head]')
    parser.add_argument(
        '-pretrain_pth', type=str, default=None,
        help='the path to the pretrain weights')
    parser.add_argument(
        '-weights_from', type=str, default='imagenet',
        help='the pretrain params from [imagenet, kinetics]')
    parser.add_argument(
        '-regression', type=bool, default=False)
    parser.add_argument(
        '-dim', type=int, default=128, help='mix data num of syn')

    parser.add_argument(
        '-depth', type=int, default=3, help='mix data num of syn')
    parser.add_argument(
        '-mlp_dim', type=int, default=256, help='mix data num of syn')
    parser.add_argument(
        '-heads', type=int, default=4, help='mix data num of syn')
    parser.add_argument('-mode', type=str, default='train',
        help='the exp mode from [train, test]')

    # Training/Optimization parameters
    parser.add_argument(
        '-seed', type=int, default=0,
        help='the seed of exp')
    parser.add_argument(
        '-optim_type', type=str, default='adamw',
        help='the optimizer using in the training')
    parser.add_argument(
        '-lr_schedule', type=str, default='cosine',
        help='the lr schedule using in the training')
    parser.add_argument(
        '-lr', type=float, default=0.0001,
        help='the initial learning rate')
    parser.add_argument(
        '-layer_decay', type=float, default=0.75,
        help='the value of layer_decay')
    parser.add_argument(
        '--min_lr', type=float, default=1e-9, 
        help="""Target LR at the end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument(
        '-weight_decay', type=float, default=0.05, 
        help="""Initial value of the weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument(
        '-weight_decay_end', type=float, default=0.05, 
        help="""Final value of the weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument(
        '-clip_grad', type=float, default=0, 
        help="""Maximal parameter gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument(
        "-warmup_epochs", default=5, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument(
        '-is_reg', type=bool, default=False,
        help='the dataset used in this experiment [render, hust]')

    # test
    parser.add_argument('-test_ckpt_path', type=str, default="best", help='the path to the checkpoint') # ckpt_path (Optional[str]) – Either "best", "last", "hpc" or path to the checkpoint you wish to test. If None and the model instance was passed, use the current weights. Otherwise, the best model checkpoint from the previous trainer.fit call will be loaded if a checkpoint callback is configured.

    args = parser.parse_args()
    return args

def single_run():
    args = parse_args()
    warnings.filterwarnings('ignore')
    
    # # linear learning rate scale 让学习率随着gpu数量增加而增加
    # if isinstance(args.gpus, int):
    #     num_gpus = torch.cuda.device_count()
    # else:
    #     num_gpus = len(args.gpus)
    # effective_batch_size = args.batch_size * num_gpus
    # args.lr = args.lr * effective_batch_size / 256

    # Experiment Settings
    ROOT_DIR = args.root_dir
    tim = time.strftime("%Y%m%d%H")
    tim = "{}-{}".format(tim,args.exp_name)
    ckpt_dir = os.path.join(ROOT_DIR, f'results/{tim}/ckpt')
    log_dir = os.path.join(ROOT_DIR, f'results/{tim}/log')
    log_dir_default = os.path.join(ROOT_DIR, f'results/{tim}/log/default')
    
    if not args.mode == "test":
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(log_dir_default, exist_ok=True)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir)
    
    if args.dataset == "hust":
        data_module = HUSTDataModule(configs=args)
    elif args.dataset == "synblink50knpy":
        data_module = SynthBlinkNPYDataModule(configs=args)
    elif args.dataset == "talkingface":
        args.talkingface_root = os.path.abspath(os.path.expanduser(args.talkingface_root))
        data_module = TalkingFaceDataModule(configs=args)
    else:
        raise NotImplementedError("No such dataset")
    
    # Resume from the last checkpoint
    if args.resume and not args.resume_from_checkpoint:
        print("Resume from the last checkpoint")
        args.resume_from_checkpoint = os.path.join(ckpt_dir, 'last_checkpoint.pth')

    # Trainer
    find_unused_parameters = True

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        save_top_k=1,
        monitor="val/F1",
        verbose=True,
        mode="max",
    )

    num_of_gpus = torch.cuda.device_count()

    if isinstance(args.gpus, int):
        if args.gpus == -1:
            requested_gpus = list(range(num_of_gpus))
        else:
            requested_gpus = [args.gpus]
    else:
        requested_gpus = [int(gpu) for gpu in args.gpus]

    if num_of_gpus > 0:
        requested_gpus = [gpu for gpu in requested_gpus if 0 <= gpu < num_of_gpus]
        if not requested_gpus:
            requested_gpus = list(range(num_of_gpus))
        accelerator = "gpu"
        devices = requested_gpus if len(requested_gpus) != num_of_gpus else num_of_gpus
        strategy = DDPStrategy(find_unused_parameters=find_unused_parameters) if (isinstance(devices, int) and devices > 1) or (isinstance(devices, list) and len(devices) > 1) else None
    else:
        accelerator = "cpu"
        devices = 1
        strategy = None

    trainer_kwargs = dict(
        logger=tb_logger,
        benchmark=True,
        accelerator=accelerator,
        devices=devices,
    )

    if strategy is not None:
        trainer_kwargs["strategy"] = strategy

    if args.mode == "test":
        trainer = pl.Trainer(
            **trainer_kwargs,
            limit_train_batches=0,
            limit_val_batches=0,
        )
    else:
        trainer = pl.Trainer(
            **trainer_kwargs,
            max_epochs=args.epoch,
            callbacks=[
                checkpoint_callback,
                LearningRateMonitor(logging_interval='step'),
            ],
            check_val_every_n_epoch=1,
            log_every_n_steps=args.log_interval,
        )

    # To be reproducable
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    seed_everything(args.seed, workers=True)
    
    # Model
    model = VideoClassificaiton(configs=args,
                             trainer=trainer,
                              ckpt_dir=ckpt_dir,
                             do_eval=True,
                             do_test=True)
    print_on_rank_zero(args)
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

    print_on_rank_zero(f'{timestamp} - INFO - Start Training')
    fit_kwargs = {}
    if args.resume_from_checkpoint:
        fit_kwargs["ckpt_path"] = args.resume_from_checkpoint
    trainer.fit(model, data_module, **fit_kwargs)

    print_on_rank_zero(f'{timestamp} - INFO - Start Testing')
    trainer.test(ckpt_path=args.test_ckpt_path, datamodule=data_module)

    print("Finish")
    
if __name__ == '__main__':
    single_run()