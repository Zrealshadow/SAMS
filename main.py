import calendar
import time
import argparse


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def arch_args(parser):
    # Model Select
    parser.add_argument('--net', default = "sams", type=str, 
                        help="select the model to test")
    # MOE-NET
    parser.add_argument('--K', default=4, type=int,
                        help='# duplication layer of each MOElayer')
    parser.add_argument('--C', default=1, type=int,
                        help= '# chosen experts')
    parser.add_argument('--noise', default=True, type=bool, dest='noise_gating',
                        help= 'whether adding noise when training MoE network')
    parser.add_argument('--moe_num_layers', default=2,
                        type=int, help='# hidden MOElayers of MOENet')
    parser.add_argument('--moe_hid_layer_len', default=10,
                        type=int, help='hidden layer length in MoeLayer')

    # hyperNet
    parser.add_argument('--hyper_num_layers', default=2, type=int,
                        help='# hidden layers of hyperNet')
    parser.add_argument('--hid_layer_len', default=10,
                        type=int, help='hidden layer length in hyerNet')
    parser.add_argument('--output_size', default=1,
                        type = int, help='Binary Classification Task')
    # embedding layer
    parser.add_argument('--data_nemb', type=int,
                        default=10, help='embedding size 10')
    parser.add_argument('--sql_nemb', type=int, default=10,
                        help='embedding size 10')

    # other setting
    parser.add_argument('--dropout', default=0.0,
                        type=float, help='dropout rate')

    # for AFN/ARMNet
    parser.add_argument('--nhead', default= 4,
                        type = int, help = 'number of attention head in armnet ')
    parser.add_argument('--nhid', default = 4,
                        type = int, help = 'number of hidden size of specific module in model \
                            such as attention layer in armnet, afn layer in AFN')
    # for expert choosing
    parser.add_argument('--expert', default="dnn",
                        type = str, help = 'which expert to choose, dnn/afn/cin/armnet')
def trainner_args(parser):

    parser.add_argument('--alpha', default=1.5, type=float,
                        help='entmax alpha to control sparsity')

    parser.add_argument('--beta', default=0.01, type=float,
                        help='coefficient for auxiliary loss')
    
    parser.add_argument('--gamma', default=0.0, type=float,
                        help="sparse for auxiliary loss")
    
    parser.add_argument('--max_filter_col', type=int, default=4,
                        help='the number of columns to choose in select...where...')

    # MLP model config
    parser.add_argument('--nfeat', type=int, default=369,
                        help='the number of features, '
                             'frappe: 5500, '
                             'uci_diabetes: 369,'
                             'criteo: 2100000')
    parser.add_argument('--nfield', type=int, default=43,
                        help='the number of fields, '
                             'frappe: 10, '
                             'uci_diabetes: 43,'
                             'criteo: 39')

    parser.add_argument('--epoch', type=int, default=3,
                        help='number of maximum epochs, '
                             'frappe: 20, uci_diabetes: 40, criteo: 10'
                        )

    parser.add_argument('--batch_size', type=int,
                        default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.002,
                        help="learning reate")

    parser.add_argument('--iter_per_epoch', type=int, default=1000,
                        help="200 for frappe, uci_diabetes, 2000 for criteo")

    # MLP train config
    parser.add_argument('--report_freq', type=int,
                        default=30, help='report frequency')

    parser.add_argument('--save_best_model', type=bool, default=True, 
                        help='whether to save best model in log')

    parser.add_argument('--patience', type=int, default=999, 
                        help='if there is no improving between patience epoch, training will be early stopped')
    
    
def data_set_config(parser):
    parser.add_argument('--data_dir', type=str, default="./third_party/data/",
                        help='path of data and result parent folder')
    # define search space,
    parser.add_argument('--dataset', type=str, default='uci_diabetes',
                        help='credit, diabetes'
                             'frappe, '
                             'criteo, '
                             'uci_diabetes')

    parser.add_argument('--workload', type=str, default='random',
                        help='workload name according to different sample strategy')
    
    parser.add_argument('--num_labels', type=int, default=1, help='[2, 2, 2]')


def tensorboard_config(parser:argparse.ArgumentParser):
    parser.add_argument('--exp', type=str, default = './tensor_log_baseline', help="the directory to store training tensorboard log")
    parser.add_argument('--train_dir', type=str, required=True, help="the name of this train process")
    parser.add_argument('--seed', type = int, default=1998)
def parse_arguments():
    parser = argparse.ArgumentParser(description='system')
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--log_folder', default="sams_logs", type=str)
    parser.add_argument('--log_name', type=str,
                        default="run_log", help="file name to store the log")

    arch_args(parser)
    data_set_config(parser)
    trainner_args(parser)
    tensorboard_config(parser)
    
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_arguments()
    
    # set the log name
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)

    import src.data_loader as data
    from src.tensorlog import setup_tensorboard_writer
    from src.run_time import Wrapper
    
    seed_everything(args.seed)
    # init data loader
    train_loader, val_loader, test_loader = data.sql_attached_dataloader(args=args)


    writer = setup_tensorboard_writer(args)
    # init model
    model = Wrapper(
        args=args,
        writer=writer)

    model.train(train_loader, val_loader, test_loader)
    model.close()
    
    print("Done")
