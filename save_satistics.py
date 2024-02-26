

from main import parse_arguments, seed_everything
import os
import glob
import json
from src.data_loader import SQLAttacedLibsvmDataset


def write_json(file_name, data):
    print(f"writting {file_name}...")
    with open(file_name, 'w') as outfile:
        outfile.write(json.dumps(data))


args = parse_arguments()
seed_everything(args.seed)


data_dir = os.path.join(args.data_dir, args.dataset)
print("current data dir is " + data_dir)
train_file = glob.glob("%s/tr*libsvm" % data_dir)[0]


train_loader = SQLAttacedLibsvmDataset(
    train_file,
    args.nfield,
    args.max_filter_col)


write_json(
    f"{args.dataset}_col_cardinalities",
    train_loader.col_cardinalities)

"""
export PYTHONPATH=$PYTHONPATH:./internal/ml/model_slicing
python ./internal/ml/model_slicing/save_satistics.py --dataset frappe --data_dir ../exp_data/ --nfeat 5500 --nfield 10 --max_filter_col 10 --train_dir ./
python ./internal/ml/model_slicing/save_satistics.py --dataset criteo --data_dir ../exp_data/ --nfeat 2100000 --nfield 39
"""

