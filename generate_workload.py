import argparse
import os
import glob
import random
import numpy as np
import shutil
from copy import deepcopy
from typing import Callable, List
from src.data_loader import SQLAttacedLibsvmDataset

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

seed_everything(1998)
'''
Generate Workload

frappe: 
python generate_workload.py --output_dir="/hdd1/sams/data/frappe/workload"

bank:   
python generate_workload.py --output_dir="/hdd1/sams/data/bank/workload" --dataset=bank --nfield=16 

adult:
python generate_workload.py --output_dir="/hdd1/sams/data/adult/workload" --dataset=adult --nfield=13 

cardiovascular disease:
python generate_workload.py --output_dir="/hdd1/sams/data/cvd/workload" --dataset=cvd --nfield=11 

diabetes:
python generate_workload.py  --output_dir="/hdd1/sams/data/diabetes/workload" --dataset=diabetes --nfield=48 --n 40
python generate_workload.py  --output_dir="/hdd1/sams/data/diabetes/workload" --dataset=diabetes --nfield=48 --n 100 --output_name random_100


credit:
python generate_workload.py  --output_dir="/hdd1/sams/data/credit/workload" --dataset=credit --nfield=23 --n 40
python generate_workload.py  --output_dir="/hdd1/sams/data/credit/workload" --dataset=credit --nfield=23 --n 100 --output_name random_100

hcdr: Home Credit Default Risk
python generate_workload.py  --output_dir="/hdd1/sams/data/hcdr/workload" --dataset=hcdr --nfield=69 --n 40
python generate_workload.py  --output_dir="/hdd1/sams/data/hcdr/workload" --dataset=hcdr --nfield=69 --n 100 --output_name random_100

census
python generate_workload.py  --output_dir="/hdd1/sams/data/census/workload" --dataset=census --nfield=41 --n 40
python generate_workload.py  --output_dir="/hdd1/sams/data/census/workload" --dataset=census --nfield=41 --n 100 --output_name random_100


Avazu
python generate_workload.py  --output_dir="/hdd1/sams/data/avazu/workload" --dataset=avazu --nfield=22 --n 40
python generate_workload.py  --output_dir="/hdd1/sams/data/avazu/workload" --dataset=avazu --nfield=22 --n 100 --output_name random_100
'''

pwd = os.getcwd()



parser = argparse.ArgumentParser(description='wordload_generation')
parser.add_argument('--output_dir', type=str, default='./workload')
parser.add_argument('--output_name', type=str, default="random")

parser.add_argument('--data_dir', type=str,
                    default='/hdd1/sams/data', help="")
parser.add_argument('--dataset', type=str, default="frappe", help="name of dataset")
parser.add_argument('--nfield', type=int, default=10, help="")
parser.add_argument('--n', type=int, default=30,
                    help="number of sql in workload")
parser.add_argument('--max_select_col', type=int, default=3,
                    help="max selected column number for filter")


def generate_workload(n: int, output_dir: str, data_dir: str, nfield: int, max_select_col: int, sample_func: Callable, min_sub_dataset:int = 20, max_sub_dataset:int = 2000000):

    # read train dataset get dict configuration
    train_file = glob.glob("%s/train.*" % data_dir)[0]
    train_dataset = SQLAttacedLibsvmDataset(train_file, nfield, max_select_col)

    # test dataset
    test_file = glob.glob("%s/test.*" % data_dir)[0]
    test_dataset = SQLAttacedLibsvmDataset(test_file, nfield, max_select_col)

    # value_map = train_dataset.col_cardinalities
    value_map = []
    # construct value map
    for col in range(train_dataset.ncols):
        valueset = train_dataset.feat_id[:,col].unique().numpy().tolist()
        value_map.append(valueset)
    
    col_padding = train_dataset.padding_feature_id

    feat_id = test_dataset.feat_id.numpy()

    sql_file_path = os.path.join(output_dir, 'sql.txt')

    sqlf = open(sql_file_path, 'w', encoding='utf-8')

    data_dir_path = os.path.join(output_dir, 'data_idx')
    
    if os.path.exists(data_dir_path):
        shutil.rmtree(data_dir_path)

    # Create the new directory
    os.mkdir(data_dir_path)
    
    i = 0
    
    freq_map = [0] * len(test_dataset)
    
    while i < n:
        
        sql_query, padding_tuple = sample_func(value_map, max_select_col, col_padding)
        sub_data_idx = filter_data(feat_id, sql_query, col_padding)

        if len(sub_data_idx) < min_sub_dataset or len(sub_data_idx) > max_sub_dataset:
            continue
        
        for idx in sub_data_idx:
            freq_map[idx] += 1
            
        if len(sub_data_idx) == 0:
            continue

        data_idx_file_path = os.path.join(data_dir_path, 'data_idx_%d.txt' % i)
        size = len(sub_data_idx)
        with open(data_idx_file_path, 'w', encoding='utf-8') as dataf:
            dataf.write(str(sub_data_idx))

        sqlf.write(str(size) +":"+str(padding_tuple) + ":" + str(sql_query) + '\n')
        i += 1

        print(f"finish query {i} contain data tuples {len(sub_data_idx)}")
        
    description_path = os.path.join(output_dir, 'description.txt')
    
    with open(description_path, 'w', encoding='utf-8') as f:
        # calculate coverage
        cnt = 0
        for i in freq_map:
            cnt += 1
        f.write(f"test size #{len(test_dataset)}\n")
        f.write(f"coverage {cnt/len(test_dataset):.4f}\n")
        f.write(f"Max Freq {max(freq_map)}\n")
        
    dataf.close()
    sqlf.close()


def filter_data(data: np.array, sql_query: List, col_padding: List):
    
    filter_condition = None
    
    for i, col_value in enumerate(sql_query):
        if col_value == col_padding[i]:
            continue
        if filter_condition is None:
            filter_condition = (data[:, i] == col_value)
        else:
            filter_condition &= (data[:, i] == col_value)

    sub_data_idx = np.where(filter_condition)[0]
    return sub_data_idx.tolist()


def random_sample(value_map: List[List], max_select_col: int, padding_feat:List):
    
    col_num = len(value_map)
    max_select_col = max(col_num, max_select_col)

    select_col_n = random.randint(0, max_select_col)

    # all padding
    sql_tuple = deepcopy(padding_feat)
    padding_tuple = [0] * len(padding_feat)

    # random choice select_col_n colomn
    cols = random.sample(range(0, col_num), select_col_n)

    for col in cols:
        col_unique_list = value_map[col]
        v = random.choice(col_unique_list)  # don't include padding
        sql_tuple[col] = v
        padding_tuple[col] = 1
    return sql_tuple, padding_tuple





if __name__ == "__main__":
    args = parser.parse_args()
    sample_func = random_sample
    


    data_dir = os.path.join(args.data_dir, args.dataset)
    print(data_dir)
    
    output_dir = os.path.join(data_dir,"workload" ,args.output_name)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    generate_workload(args.n, output_dir, data_dir,
                      args.nfield, args.max_select_col, sample_func)
    
    print(f"Finished Generate Workload, saved in {output_dir}")
