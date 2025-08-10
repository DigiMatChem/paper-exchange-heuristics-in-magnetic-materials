from collections import Counter
import json
from math import sqrt
import numpy as np
import os
import pandas as pd
import random
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import sys
import warnings
warnings.filterwarnings("ignore")

n_sampling_per_size_list = [x * 500 for x in range(1, 21)]
random.seed(42)
rs_list = random.sample(range(0, 1000000000), n_sampling_per_size_list[-1])

dt_list = ["250430_full_features_target_p_10.0_excl_binned3_TM-structs_MAGNDATA",
                  "250430_full_features_target_ap_10.0_excl_binned3_TM-structs_MAGNDATA",
                  "250430_full_features_target_p_10.0_excl_binned3_all-structs_MAGNDATA",
                  "250430_full_features_target_ap_10.0_excl_binned3_all-structs_MAGNDATA",]
dt = sys.argv[1]
idx = sys.argv[2]
assert dt in dt_list

magndata_feat = pd.read_json(os.path.join("data", f"{dt}train{idx}.json"))
target_name = "target_p_10.0_excl_binned3" if "_p_10.0" in dt else "target_ap_10.0_excl_binned3"

# Sanity check -> no duplicate and constant feats
df_copy = magndata_feat.copy()
df_copy = df_copy.T.drop_duplicates().T
df_copy = df_copy.loc[:, (df_copy != df_copy.iloc[0]).any()]
assert df_copy.shape == magndata_feat.shape

feat = magndata_feat.drop(columns=[target_name])
mutual_info = pd.DataFrame([], columns=[], index=feat.columns)
target_entropies = {}
for n_sampling_idx, n_sampling in enumerate(n_sampling_per_size_list):
    lb = n_sampling_per_size_list[n_sampling_idx - 1] if n_sampling_idx > 0 else 0
    for rs in rs_list[lb:n_sampling]:
        mutual_info.loc[:, f"mi_target_rs{rs}"] = mutual_info_classif(feat,
                                                                      magndata_feat[target_name],
                                                                      random_state=rs)
        for fe in mutual_info.index:
            # Not nec. continuous features, but do same as later in modnet feature selection
            mutual_info.at[fe, f"entropy_rs{rs}"] = mutual_info_regression(feat[fe].values.reshape(-1, 1),
                                                                            feat[fe],
                                                                            random_state=rs)[0]
        target_entropy = mutual_info_classif(magndata_feat[target_name].values.reshape(-1, 1),
                                             magndata_feat[target_name],
                                             random_state=rs)[0]
        target_entropies[rs] = target_entropy
        # Symmetric uncertainty as measure for normalized mutual info as in modnet
        mutual_info[f"nmi_target_rs{rs}"] = mutual_info.apply(
            lambda x: (2 * x[f"mi_target_rs{rs}"]) / (target_entropy + x[f"entropy_rs{rs}"]), axis=1)

    for feat_name, row in mutual_info.iterrows():
        mis = row.loc[[c for c in row.keys() if c.startswith("mi_target_rs")]].values
        nmis = row.loc[[c for c in row.keys() if c.startswith("nmi_target_rs")]].values
        assert len(mis) == len(nmis)
        assert len(mis) == n_sampling
        for q, q_string in zip([mis, nmis], ["mi", "nmi"]):
            mutual_info.at[feat_name, f"{q_string}_target_sample_standard_deviation"] = np.std(q) / sqrt(n_sampling)
            mutual_info.at[feat_name, f"{q_string}_target_sample_mean"] = q.mean()

    print(n_sampling, dt, idx)
    print("mean of nmi_target_sample_mean", mutual_info["nmi_target_sample_mean"].values.mean())
    print("mean of nmi_target_sample_standard_deviation ", mutual_info["nmi_target_sample_standard_deviation"].values.mean())

    print("mean of mi_target_sample_mean", mutual_info["mi_target_sample_mean"].values.mean())
    print("mean of mi_target_sample_standard_deviation ", mutual_info["mi_target_sample_standard_deviation"].values.mean())

    print("Feature with highest nmi sample mean: ",
          mutual_info.loc[mutual_info["nmi_target_sample_mean"] == mutual_info["nmi_target_sample_mean"].values.max()].drop(
              columns=[c for c in mutual_info.columns if "_rs" in c or "stop_criterion" in c or c.startswith("mi_")]))

    mutual_info["stop_criterion"] = mutual_info.apply(
        lambda m: m["nmi_target_sample_standard_deviation"] < 0.1 * m["nmi_target_sample_mean"]
        if m["nmi_target_sample_mean"] > 0.0 else True, axis=1)
    co = Counter(mutual_info["stop_criterion"].values)
    print("Stop criterion reached: ", co)

    if co[False] < len(mutual_info) / 100:
        print(n_sampling, dt, idx)
        print(f"Sample stand deviation of more than 99 % of all feature-target NMIs dropped below .1 * feature "
              f"sample mean at {n_sampling} samplings.")
        mutual_info.to_json(os.path.join("data", f"250430_NMI_MI_H_{dt}train{idx}.json"))
        with open(os.path.join("data", f"250430_target_entropies_{dt}train{idx}.json"), "w") as f:
            json.dump(target_entropies, f)
        break

    elif co[False] < 8:
        print(mutual_info.loc[~mutual_info["stop_criterion"]].drop(
            columns=[c for c in mutual_info.columns if "_rs" in c or "stop_criterion" in c or c.startswith("mi_")]))

    if n_sampling == n_sampling_per_size_list[-1]:
        print(n_sampling, dt, idx)
        print(f"NMI computation did not converge ({co}), stopping at {n_sampling} samplings and saving to json.")
        mutual_info.to_json(os.path.join("data", f"250430_NMI_MI_H_{dt}train{idx}.json"))
        with open(os.path.join("data", f"250430_target_entropies_{dt}train{idx}.json"), "w") as f:
            json.dump(target_entropies, f)
            