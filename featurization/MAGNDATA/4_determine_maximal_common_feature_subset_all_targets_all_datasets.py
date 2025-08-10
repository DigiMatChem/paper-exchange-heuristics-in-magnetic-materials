"""
Get maximal common subset of features -> regard all six full target-dataset combis
(three datasets, in two we compute both p and ap, in one we compute p and AFM/FM).
After that, perform 20 train test splits per target-dataset combi and again remove constant and duplicate features.
Perform stratified subsampling of train splits of larger datasets in p and ap models to yield same-size datasets.
Requires prior execution of ../MP/2_add_automatminer_features.py.
"""
from math import ceil
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold, train_test_split


magndata_path = "data"
magndata_files = os.listdir(magndata_path)
mp_path = "../MP/data"
mp_files = os.listdir(mp_path)

dfs = {os.path.join(mp_path, f): pd.read_json(os.path.join(mp_path, f)) for f in mp_files
       if "250525_full_features" in f and not ("train" in f or "test" in f)}
dfs.update({os.path.join(magndata_path, f): pd.read_json(os.path.join(magndata_path, f))
            for f in magndata_files if "250430_full_features" in f and not ("train" in f or "test" in f)})
feat_names = [set([c for c in df.columns if not "target" in c]) for df in dfs.values()]
mcs = feat_names[0].intersection(*feat_names[1:])

# For subsampling of p and ap models
train_set_length = ceil([df for name, df in dfs.items() if "TM-structs" in name][0].shape[0] * 0.95)
splitter = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)

for df_name, df in dfs.items():
    df.drop(columns=[c for c in df.columns if c not in mcs and not "target" in c], inplace=True)
    assert len(df.columns) == len(mcs) + 1
    target_name = [c for c in df.columns if "target" in c][0]
    subsampleids = []
    for i, (train_ids, test_ids) in enumerate(splitter.split(np.zeros(len(df)), df[target_name].tolist())):
        df_test = df.iloc[test_ids, :]
        if "TM-structs" in df_name or "classic" in target_name:
            df_train = df.iloc[train_ids, :]
            subsampleids.append(set(train_ids))
        else:
            # Subsampling (also stratified!)
            train_ys = df.iloc[train_ids, :][target_name].values
            subsampled_train_ids, _ = train_test_split(train_ids,
                                                       train_size=train_set_length,
                                                       stratify=train_ys,
                                                       shuffle=True,
                                                       random_state=42)
            subsampleids.append(set(subsampled_train_ids))
            assert len(set(subsampled_train_ids)) == len(subsampled_train_ids)
            df_train = df.iloc[subsampled_train_ids, :]

        df_train = df_train.T.drop_duplicates().T
        df_train = df_train.loc[:, (df_train != df_train.iloc[0]).any()]
        assert list(df_train.columns) == list(sorted(df_train.columns))

        train_name = df_name.replace(".json", f"train{i}.json")
        test_name = df_name.replace(".json", f"test{i}.json")
        df_train.to_json(train_name)
        df_test.to_json(test_name)
