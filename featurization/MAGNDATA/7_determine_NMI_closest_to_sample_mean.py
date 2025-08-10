"""
For each (train-test split, target, dataset) combination, find the random seed of NMI computation
that minimizes the sum of absolute differences between target-NMI sample mean and the NMI(rs).
This rs is then used for modnet feature selection.
"""
from collections import Counter
import json
import numpy as np
import os
import pandas as pd


targets = ["ap", "p", "classic"]
dbs = ["all-structs_MAGNDATA", "TM-structs_MAGNDATA", "MP"]
results = {t: {d: {i: -1 for i in range(20)} for d in dbs} for t in targets}

for target in targets:
    print(target)
    for db_string in dbs:
        df_path = "data" if "MAGNDATA" in db_string else "../MP/data/"
        date = "250430" if "MAGNDATA" in db_string else "250525"  # corrected missed data leakage
        if "p" in target:
            target_name = f"target_{target}_10.0_excl_binned3" if "MAGNDATA" in db_string \
                else f"target_p_exclude_ligand_multiplicities_binned3"
        else:
            if db_string != dbs[-1]:
                continue
            target_name = "target_classic"
        print(db_string)
        for train_ix in range(20):
            mutual_info = pd.read_json(
                os.path.join(df_path,
                             f"{date}_NMI_MI_H_{date}_full_features_{target_name}_{db_string}train{train_ix}.json"))
            sample_means = mutual_info["nmi_target_sample_mean"]
            mutual_info.drop(columns=[c for c in mutual_info.columns if not c.startswith("nmi_target_rs")],
                             inplace=True)
            summed_distances = {}
            for c in mutual_info.columns:
                summed_distances[int(c.removeprefix("nmi_target_rs"))] = (
                    np.sum(np.abs(np.subtract(mutual_info[c], sample_means))))
            co = Counter([round(v, 1) for v in summed_distances.values()])
            print({k: co[k] for k in sorted(co.keys())})
            print(min(summed_distances, key=summed_distances.get), summed_distances[min(summed_distances, key=summed_distances.get)])
            results[target][db_string][train_ix] = min(summed_distances, key=summed_distances.get)

with open("data/optimal_rs_all_train_sets-targets-datasets.json", "w") as f:
    json.dump(results, f)
