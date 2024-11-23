import utils
import torch
import numpy as np
import Procedure
import multiprocessing
import dataloader
import model
from os.path import join
import os
from parse import parse_args
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)

def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")

import sys
sys.path.append('sources')



MODELS = {
    'mf': model.PureMF,
    'lgn': model.SAGCN,
    'sagcn': model.SAGCN

}

def buile_dataset(args):
    if args.dataset in ['baby', 'clothing', 'office', 'goodreads']:
        dataset = dataloader.Loader(args, path="./"+args.DATA+"/"+args.dataset)
    else:
        raise NotImplementedError
    return dataset

def build_network(args, dataset):
    Recmodel = MODELS[args.model](args, dataset)
    Recmodel = Recmodel.to(args.device)
    return Recmodel

def build_optimizer(args, Recmodel):
    bpr = utils.BPRLoss(Recmodel, args)
    return bpr

def build_early_stop(args):
    early_stop = utils.EarlyStopManager(args.es_patience)
    return early_stop


def train(args):
    args.topks = eval(args.topks)
    all_dataset = ['baby', 'clothing', 'office', 'goodreads']
    all_models  = ['mf', 'lgn', 'sagcn']
    args.A_split = False
    args.bigdata = False
    args.test_u_batch_size = args.testbatch
    GPU = torch.cuda.is_available()
    args.device = torch.device('cuda' if GPU else "cpu")
    args.CORES = multiprocessing.cpu_count() // 2

    dataset = args.dataset
    if dataset not in all_dataset:
        raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
    if args.model not in all_models:
        raise NotImplementedError(f"Haven't supported {args.model} yet!, try {all_models}")

    weight_file = utils.getFileName(args)
    # def get_path():
    #     ckpt_root = 'checkpoints'
    #     for i in range(10000):
    #         ckpt_dir = os.path.join(ckpt_root, str(i))
    #         if not os.path.exists(ckpt_dir):
    #             os.makedirs(ckpt_dir)
    #             break
    #     return ckpt_dir
    # PATH = get_path()
    # weight_file = join(PATH, weight_file)


    # ==============================
    utils.set_seed(args.seed)
    print(">>SEED:", args.seed)
    # ==============================

    print('===========config================')
    print("cores for test:", args.CORES)
    print("comment:", args.comment)
    print("LOAD:", args.load)
    print("Weight path:", weight_file)
    print("Test Topks:", args.topks)
    print("using bpr loss")
    print('===========end===================')
    print(f"load and save to {weight_file}")

    if args.load:
        try:
            Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
            cprint(f"loaded model weights from {weight_file}")
        except FileNotFoundError:
            print(f"{weight_file} not exists, start from beginning")
    Neg_k = 1


    dataset = buile_dataset(args)
    Recmodel = build_network(args, dataset)
    bpr = build_optimizer(args, Recmodel)
    early_stop = build_early_stop(args)
    max_metric = 0
    w = None
    for epoch in range(args.epochs):
        if epoch % args.validation_epoch == 0:
            cprint("[TEST]")
            results = Procedure.Test(dataset, Recmodel, epoch, w=w, multicore=args.multicore, args=args, es=early_stop)
            if early_stop.stop:
                break
            if results['recall'][0] > max_metric:
                max_metric = results['recall'][0]
                torch.save(Recmodel.state_dict(), weight_file)
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, args=args, neg_k=Neg_k,w=w)
        print(f'EPOCH[{epoch+1}/{args.epochs}] {output_information}')
        torch.save(Recmodel.state_dict(), weight_file)
# %%
args = parse_args()
train(args)