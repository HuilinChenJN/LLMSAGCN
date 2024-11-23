import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Go SAGCN")
    parser.add_argument('--bpr_batch', type=int,default=1024,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int,default=7,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float,default=0.01,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=0.001,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int,default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float,default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int,default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--DATA', type=str, default='datasets', help='directory of all datasets')
    parser.add_argument('--dataset', type=str,default='baby',
                        help="available datasets: [baby, clothing, office]")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[10,20,50]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int,default=1,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str,default="sagcn")
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--epochs', type=int,default=1000)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--model', type=str, default='sagcn', help='rec-model, support [mf, lgn, sagcn]')
    parser.add_argument('--es_patience', type=int, default=100, help='early stop patience')
    parser.add_argument('--validation_epoch', type=int, default=1, help='validation epoch')
    parser.add_argument('--mode', type=str, default='concat', help='aggregation mode', choices=['mean', 'add', 'concat'])
    parser.add_argument('--explicit_factors', nargs='+', default=[], help='explicit factors. baby: "quality", "functionality", "comfort", "ease_of_use", "design", "durability", "size", "price". clothing: "quality", "comfort", "appearance", "style", "fit", "design", "size", "price". office: "quality", "functionality", "ease_of_use", "convenience", "comfort", "durability", "design", "price"')
    parser.add_argument('--has_implicit', type=int, default=0, help='whether we use implicit graph or not')
    parser.add_argument('--explicit_graph', type=str, default='specific', help='explicit graph type', choices=['common', 'specific'])

    return parser.parse_args()