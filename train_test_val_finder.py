"""
Due to random spilt of the train and validation set, the paper is hard to replicate.

This code takes the data set from the original implementation and find train, test and validation set. 

Find the data set here:
https://worksheets.codalab.org/bundles/0x5b9d528d2101418b87212db92fea6683

"""
data_path= r"Class_attr_data_10/class_attr_data_10"
output_path = r""


import pandas as pd
import os

sets = ['train', 'test', 'val']
set_dict = {}
for set in sets:
    file_path = os.path.join(data_path, f'{set}.pkl')
    type_list = pd.read_pickle(file_path)
    set_list = []
    for d in type_list:
        set_list.append(d["id"])
    print(f"length of {set} set: {len(set_list)}")
    set_dict[set] = set_list

    # Save the new dict with train test val sets
    pd.to_pickle(set_dict, os.path.join(output_path, 'train_test_val.pkl'))


