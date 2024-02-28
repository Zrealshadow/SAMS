import os
import random as ra
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader

import third_party.utils.func_utils as utils
from src.data_loader import sql_attached_dataloader

from src.model.factory import initialize_model
import matplotlib.pyplot as plt



def load_model(tensorboard_path: str):
    """
    Args:
    tensorboard_path: the path of the directory of tensorboard
    """
    arg_file_path = os.path.join(tensorboard_path, "args.txt")
    model_config = reload_argparse(arg_file_path)
    
    net = initialize_model(model_config)
    
    model_pth_path = os.path.join(tensorboard_path, "best_model.pth")
    saved_state_dict = torch.load(model_pth_path)

    net.load_state_dict(saved_state_dict)
    print("successfully load model")
    return net, model_config

def reload_argparse(file_path:str):
    
    d = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            key, value = line.strip('\n').split(',')
            # print(f"{key}, {value}\n")
            try:
                re = eval(value)
            except:
                re = value
            d[key] = re
    
    return argparse.Namespace(**d)   


def fig(x,y,path):
    plt.figure(figsize=(10, 6))  # Set the figure size
    colors = [plt.cm.viridis(ra.random()) for _ in range(len(y))]
    bar_width = 0.7
    plt.bar(x, y, width=bar_width, color=colors, alpha=0.7)
    plt.title('Beautiful Histogram')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(path, dpi=300, bbox_inches='tight')


parser = argparse.ArgumentParser(description='predict FLOPS')
parser.add_argument('path', type=str, 
                    help="directory to model file")
parser.add_argument('--alpha', type=float,
                    default=None,
                    help="if set the value of alpha of sparsemax")
parser.add_argument('--workload', type=str,
                    default="random")
device="cuda:0"
if __name__ == '__main__':
    args = parser.parse_args()
    path = args.path
    net, config = load_model(path)
    config.workload = 'random'
    print(config.workload)
    # config.
    # load data
    
    train_loader, _, workload = sql_attached_dataloader(config)
    
    print(config.K)
    net.eval()
   
    frequency = []
    expert_num_for_each_item = []

    if config.net == "sparsemax_vertical_sams":
        alpha = net.sparsemax.alpha
        print(alpha)
        if args.alpha is not None:
            net.sparsemax.alpha = torch.nn.Parameter(torch.tensor(args.alpha, dtype=torch.float32), requires_grad=True)
      
        with torch.no_grad():
            for i in range(len(workload)):
                _, sql = workload[i]
                sql = sql.unsqueeze(0) # [1, nfield]
                
                score = net.cal_gate_score(sql) # [1, n]
                score = score.squeeze(0)
                
                
                nonzero = score.numpy() != 0            
                # L = expert nnum
                frequency.append(nonzero)
                expert_num_for_each_item.append(np.sum(nonzero))
    
    
        frequency = np.array(frequency)
        frequency = np.sum(frequency, axis=0) / len(workload)
        frequency = frequency.tolist()


        print(frequency)
        fig(range(len(frequency)), frequency, "./Expert.png")
        fig(range(len(expert_num_for_each_item)), expert_num_for_each_item, "./Item.png")        
        
        ave_expert = sum(expert_num_for_each_item)/len(expert_num_for_each_item)    
        print(ave_expert)
    
    # calculate if sql is all padding, means no predicate.
    sql = train_loader.dataset.generate_default_sql()
    
    auc_avg_, pr_avg_ = utils.AvgrageMeter(), utils.AvgrageMeter()
    p_auc_avg_, p_pr_avg_ = utils.AvgrageMeter(), utils.AvgrageMeter()
    

    sql = sql.unsqueeze(0)
    
    if config.net == "sparsemax_vertical_sams":
        score = net.cal_gate_score(sql) # [1, n]
        score = score.squeeze(0)
        print(score)
    
    net = net.to(device)
    psql = sql.to(device)
    cnt = 0
    target_macro, y_macro, py_macro = [], [], []
    res = []
    with torch.no_grad():
        for i in range(len(workload)):
            dataset, sql = workload[i]
            data_laoder = DataLoader(dataset, batch_size = 1024, shuffle = False,pin_memory=True )
            
            target_list = []
            y_list = []
            py_list = []
            
            sql = sql.to(device)
            sql = sql.unsqueeze(0)
            tuple_size = len(dataset)
            for _, data_batch in enumerate(data_laoder):
                target = data_batch['y'].to(device)
                target_list.append(target)
                x_id = data_batch['id'].to(device)
                B = target.shape[0]
                
                sql_ = sql.expand(B, -1)
                psql_ = psql.expand(B, -1)
                
                y = net(x_id, sql_)
                py = net(x_id, psql_)
                
                if isinstance(y, tuple):
                    y = y[0]
                    py = py[0]
                      
                y_list.append(y)
                py_list.append(py)
                
            target = torch.cat(target_list, dim = 0) # [#n]
            y = torch.cat(y_list, dim = 0)
            py = torch.cat(py_list, dim = 0)
            
            target = target.detach().cpu()
            y = y.detach().cpu()
            py = py.detach().cpu()
            

            
            # print(f"target shape{target.shape}, y_shape {y.shape}, pyshape {py.shape}")
            assert target.shape == py.shape
            assert target.shape == y.shape
            
            try:                
                auc = utils.roc_auc_compute_fn(y, target)
                
                pauc = utils.roc_auc_compute_fn(py, target)

                auc_avg_.update(auc, 1)

                target_macro.append(target)
                y_macro.append(y)
                
                if auc > pauc:
                    py_macro.append(py)
                    p_auc_avg_.update(pauc, 1)

                    cnt += 1
                else:
                    py_macro.append(y)
                    p_auc_avg_.update(auc,1)

                # print(f"target shape{target.shape}, y_shape {y.shape}, pyshape {py.shape}")
                print(f"Workload {i}, #tuple {tuple_size}: AUCROC {pauc:8.4f} : {auc:8.4f}") 
                res.append("{:.4f}".format(auc))
            except ValueError:
                print("Skip this workload micro-evaluation")
                pass
        
        
    n = len(workload)
    
    target_macro = torch.cat(target_macro, dim = 0)
    y_macro = torch.cat(y_macro, dim = 0)
    py_macro = torch.cat(py_macro, dim = 0)
    
    # print(f"target shape{target_macro.shape}, y_shape {y_macro.shape}, pyshape {py_macro.shape}")
    
    macro_auc = utils.roc_auc_compute_fn(y_macro, target_macro)
    p_macro_auc = utils.roc_auc_compute_fn(py_macro, target_macro)
    
    print(f"{cnt} / {n}, WithSQL is better than PaddingSQL")
    # print(f'\n--------------------Workload Test, without SQL Predicate ----------------------\n'
    #     f'Without SQL -> AUC-ROC {p_auc_avg_.avg:8.4f} \t AUC-PR {p_pr_avg_.avg:8.4f} \n'
    #     f'With SQL    -> AUC-ROC {auc_avg_.avg:8.4f} \t AUC-PR {pr_avg_.avg:8.4f} \n'
    #     )
    print(f'\n--------------------Workload Test, without SQL Predicate ----------------------\n'
        f'Without SQL -> Micro-AUC-ROC {p_auc_avg_.avg:8.4f} \t Macro-AUC-ROC {p_macro_auc:8.4f} \n'
        f'With SQL    -> Micro-AUC-ROC {auc_avg_.avg:8.4f} \t Macro-AUC-ROC {macro_auc:8.4f} \n'
        )
    
    print(f"auc result list {res[:30]}")