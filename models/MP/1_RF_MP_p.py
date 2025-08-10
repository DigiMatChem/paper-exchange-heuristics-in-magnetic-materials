"""Trains p score (binned) models on MP dataset. The train set index is given via a command line argument."""

import json
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import sys

from utils_kga.models.RF_classification import train_rf_classifier, select_features


num_classes = 3
eval_metric = "f1_weighted"

n_outer = 10
n_inner = 10
n_features = 20
hyper_param_grid = {"n_estimators": [150, 400, 600, 800],
                    "max_depth": [5, 6, 7],
                    "min_samples_split": [2, 4, 8, 12], }

# Random seeds that give the most average feature-target NMIs
with open("../../featurization/MAGNDATA/data/optimal_rs_all_train_sets-targets-datasets.json") as f:
    train_idx_pick_dict = json.load(f)

train_idx = sys.argv[1]
assert int(train_idx) in list(range(20))

score_type = "p"
target_name = f"target_{score_type}_exclude_ligand_multiplicities_binned3"
data_filter = "MP"

train_feat_df_path = ("../../featurization/MP/data/"
                      f"250525_full_features_{target_name}_{data_filter}train{train_idx}.json")
feat_df = pd.read_json(train_feat_df_path)

optimal_features = select_features(df=feat_df,
                                   target_name=target_name,
                                   n_features=n_features,
                                   random_state=train_idx_pick_dict[score_type][data_filter][train_idx],
                                   n_jobs=2)
outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=42)

model_log = {
    "param_grid": hyper_param_grid,
    "n_data_points": len(feat_df),
    "eval_metric": eval_metric,
}
print(f"Computing {target_name} {data_filter} train idx {train_idx}")
train_rf_classifier(df=feat_df,
                    model_log=model_log,
                    target_name=target_name,
                    target_type="structure",
                    data_filter=data_filter,
                    classes_int_to_name_map={0: f"{score_type}<=0.005",
                                             1: f"0.005<{score_type}<=0.995",
                                             2: f"{score_type}>0.995"},
                    script_path=__file__,
                    feature_dataframe_identifier=f"{train_feat_df_path}, b728ab4",
                    test_feat_df_path=train_feat_df_path.replace("train", "test"),
                    n_features=n_features,
                    optimal_features=optimal_features,
                    random_state_feature_selection=train_idx_pick_dict[score_type][data_filter][train_idx],
                    save_models=True,
                    additional_info_for_model_id=f"trainidx{train_idx}",
                    outer_cv=outer_cv,
                    inner_cv=StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=42),
                    folds_shapley=list(range(n_outer)),
                    eval_metric=eval_metric,
                    n_jobs_rf=1,
                    n_jobs_grid=2,
                    n_jobs_feature_selection=2,
                    )
