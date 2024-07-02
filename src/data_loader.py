import glob
from tqdm import tqdm
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader,Subset
import os
from copy import deepcopy

def decode_libsvm(line):
    columns = line.split(' ')
    def map_func(pair): return (int(pair[0]), float(pair[1]))
    id, value = zip(
        *map(lambda col: map_func(col.split(':')), columns[1:]))
    sample = {'id': torch.LongTensor(id),
                'value': torch.FloatTensor(value),
                'y': float(columns[0])}
    return sample

class SQLAttacedLibsvmDataset(Dataset):
    """ Dataset loader for Libsvm data format """

    def __init__(self, fname, nfields, max_filter_col):
        src_type = fname.split('.')[-1]
        if src_type == 'libsvm':
            self.__read_from_libsvm__(fname, nfields)
        elif src_type == 'npy':
            self.__read_from_np__(fname, nfields)
            
        self.__init_setting__(max_filter_col)
    
    def __read_from_np__(self, fname:str, nfields:int):
        
        data = np.load(fname)
        self.feat_id = torch.LongTensor(data[:,1:])
        self.y = torch.FloatTensor(data[:,0])
        self.feat_value = torch.ones(self.feat_id.shape)
        
        self.nsamples = self.feat_id.shape[0]
        # number of columns
        self.ncols = self.feat_id.shape[1]
        
        print(f'#{self.nsamples} data samples loaded...')
    
    
    def __read_from_libsvm__(self, fname:str, nfields:int):
        with open(fname) as f:
            sample_lines = sum(1 for line in f)

        self.feat_id = torch.LongTensor(sample_lines, nfields)
        self.feat_value = torch.FloatTensor(sample_lines, nfields)
        self.y = torch.FloatTensor(sample_lines)

        self.nsamples = 0
        with tqdm(total=sample_lines) as pbar:
            with open(fname) as fp:
                line = fp.readline()
                while line:
                    try:
                        sample = decode_libsvm(line)
                        self.feat_id[self.nsamples] = sample['id']
                        self.feat_value[self.nsamples] = sample['value']
                        self.y[self.nsamples] = sample['y']
                        self.nsamples += 1
                    except Exception:
                        print(f'incorrect data format line "{line}" !')
                    line = fp.readline()
                    pbar.update(1)
        print(f'# {self.nsamples} data samples loaded...')




    def __init_setting__(self, max_filter_col:int):
        # generate the columsn statics
        self.max_columns = min(self.feat_id.shape[1], max_filter_col)
        
        # number of columns
        self.ncols = self.feat_id.shape[1]

        # max_value
        max_value = self.feat_id.max().item()
        
        # add pedding feature_id to each of the columns unique value.
        self.padding_feature_id = []
        for i in range(self.ncols):
            self.padding_feature_id.append(max_value + 1 + i)

        # print("self.sql_history is initialized !")
        # # this is used in infernece, as infernece must testd on the trained sql.
        # self.sql_history = set()
        
    def generate_sql_by_row(self, row: torch.Tensor):
        # 1. firstly randomly pick number of the columns
        ncol = random.randint(0, self.max_columns)

        # default to not select any value, init all columns feature id to last of each list
        random_sql = deepcopy(self.padding_feature_id)

        # 2. second, randomly pick two col s,
        selected_cols = random.sample(range(len(random_sql)), ncol)

        # 3. assign value from row to selected columns in random_sql
        for col in selected_cols:
            random_sql[col] = row[col].item()   # this is feature id

        # record history
        # self.sql_history.add(tuple(random_sql))
        return torch.tensor(random_sql)
    
    def generate_default_sql(self):
        # default to not select any value, init all columns feature id to last of each list
        default_sql = self.padding_feature_id
        return torch.tensor(default_sql)
    
    
    def __len__(self):
        return self.nsamples

    def __getitem__(self, idx):
        return {'sql': self.generate_sql_by_row(self.feat_id[idx]),
                'default_sql':self.generate_default_sql(),
                'id': self.feat_id[idx],
                'value': self.feat_value[idx],
                'y': self.y[idx]}


    def reset_sql_his(self):
        """
        Called after each epoch. such that infernece can use the same sql set.
        :return:
        """
        self.sql_history.clear()


class SQLAwareDataset(Dataset):
    
    def __init__(self, fname, nfields):
        src_type = fname.split('.')[-1]
        if src_type == 'libsvm':
            self.__read_from_libsvm__(fname, nfields)
        elif src_type == 'npy':
            self.__read_from_np__(fname, nfields)
            
    def __read_from_np__(self, fname, nfields):
        
        data = np.load(fname)
        self.feat_id = torch.LongTensor(data[:,1:])
        self.y = torch.FloatTensor(data[:,0])
        self.feat_value = torch.ones(self.feat_id.shape)
        
        self.nsamples = self.feat_id.shape[0]
        # number of columns
        self.ncols = self.feat_id.shape[1]
        
        print(f'#{self.nsamples} data samples loaded...')
        
    
    
    def __read_from_libsvm__(self, fname:str, nfields:int):
        with open(fname) as f:
            sample_lines = sum(1 for line in f)


        self.feat_id = torch.LongTensor(sample_lines, nfields)
        self.feat_value = torch.FloatTensor(sample_lines, nfields)
        self.y = torch.FloatTensor(sample_lines)

        self.nsamples = 0
        with tqdm(total=sample_lines) as pbar:
            with open(fname) as fp:
                line = fp.readline()
                while line:
                    try:
                        sample = decode_libsvm(line)
                        self.feat_id[self.nsamples] = sample['id']
                        self.feat_value[self.nsamples] = sample['value']
                        self.y[self.nsamples] = sample['y']
                        self.nsamples += 1
                    except Exception:
                        print(f'incorrect data format line "{line}" !')
                    line = fp.readline()
                    pbar.update(1)
        print(f'# {self.nsamples} data samples loaded...')

        # generate the columsn statics

        
    def __len__(self):
        return self.nsamples
    
    def __getitem__(self, idx):
        return {'id': self.feat_id[idx],
                'value': self.feat_value[idx],
                'y': self.y[idx]}



class Workload(object):
    
    def __init__(self, dataset:Dataset, dir_path:str):
        print(dir_path)
        assert os.path.exists(dir_path)
        
        sql_file = os.path.join(dir_path, "sql.txt")
        idx_file_dir = os.path.join(dir_path, "data_idx")
        
        assert os.path.exists(sql_file)
        assert os.path.exists(idx_file_dir)
        
        self.dataset = dataset
        self.init_sql(file_path=sql_file)
        
        # self.sub_idx_path_list = [os.path.join(idx_file_dir, i) for i in os.listdir(idx_file_dir)]
        self.length = len(os.listdir(idx_file_dir))
        self.sub_idx_path_list = [os.path.join(idx_file_dir, "data_idx_{}.txt".format(i)) for i in range(self.length)]
        assert self.length == len(self.sub_idx_path_list)
        
        
    def init_sql(self, file_path:str):
        self.sql_list = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                self.sql_list.append(eval(line.split(":")[-1].strip()))

    def set_dataset(self, dataset:Dataset):
        self.dataset = dataset
    
    def __len__(self):
        return self.length
    
    
    def __getitem__(self, index):
        n = len(self.dataset)
        with open(self.sub_idx_path_list[index], 'r') as f:
            idxs = [i for i in eval(f.read().strip()) if i < n]
            
        sql = torch.LongTensor(self.sql_list[index])
        return Subset(self.dataset, idxs), sql
    
    

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def sql_attached_dataloader(args):
    # g = torch.Generator()
    # g.manual_seed(args.seed)
    data_dir = os.path.join(args.data_dir, args.dataset)
    train_file = glob.glob("%s/train.*" % data_dir)[0]
    val_file = glob.glob("%s/val.*" % data_dir)[0]
    test_file = glob.glob("%s/test.*" % data_dir)[0]

    train_loader = DataLoader(SQLAttacedLibsvmDataset(train_file, args.nfield, args.max_filter_col),
                              batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, worker_init_fn=seed_worker)
    
    val_loader = DataLoader(SQLAttacedLibsvmDataset(val_file, args.nfield, args.max_filter_col),
                              batch_size=args.batch_size, shuffle=False,
                              pin_memory=True, worker_init_fn=seed_worker)
    
    # val_dataset = SQLAwareDataset(val_file, args.nfield)
    # val_workload = Workload(val_dataset, os.path.join(data_dir, args.workload))
    
    test_dataset = SQLAwareDataset(test_file, args.nfield)

    workload_dir = os.path.join(data_dir, "workload", args.workload)

    test_workload = Workload(test_dataset, workload_dir)
    
    return train_loader, val_loader, test_workload

''' ---------------------  Big Dataset Loading ---------------------'''

def load_data(data_dir, namespace):
    print(f'# loading data from '
          f'{data_dir}/{namespace}_feat_id.pt, '
          f'{data_dir}/{namespace}_feat_value.pt'
          f'{data_dir}/{namespace}_y.pt ......')

    feat_id = torch.load(f'{data_dir}/{namespace}_feat_id.pt')
    feat_value = torch.load(f'{data_dir}/{namespace}_feat_value.pt')
    y = torch.load(f'{data_dir}/{namespace}_y.pt')

    print(f'# {int(y.shape[0])} data samples loaded...')

    return feat_id, feat_value, y, int(y.shape[0])


class LibsvmDatasetReadOnce(Dataset):
    """ Dataset loader for Libsvm data format """

    def __init__(self, fname):
        parent_directory = os.path.dirname(fname)
        if "train" in fname:
            namespace = "decoded_train"
        elif "valid" in fname:
            namespace = "decoded_valid"
        else:
            raise
        self.feat_id, self.feat_value, self.y, self.nsamples = load_data(
            parent_directory, namespace)

        print(f'# {self.nsamples} data samples loaded...')

    def __len__(self):
        return self.nsamples

    def __getitem__(self, idx):
        return {'id': self.feat_id[idx],
                'value': self.feat_value[idx],
                'y': self.y[idx]}


def devoded_libsvm_dataloader(args, data_dir, batch_size):
    print("Loading data from ", data_dir)
    workers = args.workers
    train_file_name = f"{data_dir}/train.libsvm"
    valid_file_name = f"{data_dir}/valid.libsvm"
    test_file_name = f"{data_dir}/test.libsvm"
    print(f"using train={train_file_name}, valid={valid_file_name}")
    # read the converted file
    if args.device == "cpu":
        train_loader = DataLoader(LibsvmDatasetReadOnce(train_file_name),
                                  batch_size=batch_size,
                                  shuffle=True)
        val_loader = DataLoader(LibsvmDatasetReadOnce(valid_file_name),
                                batch_size=batch_size * 8,
                                shuffle=False)

    else:
        train_loader = DataLoader(LibsvmDatasetReadOnce(train_file_name),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=workers,
                                  pin_memory=True)

        val_loader = DataLoader(LibsvmDatasetReadOnce(valid_file_name),
                                batch_size=batch_size * 8,
                                shuffle=False,
                                num_workers=workers,
                                pin_memory=True)

    return train_loader, val_loader
