"""
Requires automatminer env.
Besides automatminer express featurization, adds an SCM of the sites guessed magnetic.
Clean df before finding minimal common subset
with other target-dataset combinations in next step.
"""
from automatminer.featurization import AutoFeaturizer
import json
from matminer.featurizers.structure import SineCoulombMatrix
from monty.json import MontyDecoder
import pandas as pd


with open("data/250525_structure_cn_features_labels.json") as f:
    feat_df = pd.DataFrame(json.load(f))
with open("data/df_stable_MP_db_additional_nonmag_info.json") as f:
    info_df = json.load(f, cls=MontyDecoder)

original_info_columns = info_df.columns
structure_ids = feat_df.index

pre_amm_df = info_df.loc[structure_ids]
assert len(pre_amm_df) == len(feat_df)
pre_amm_df["structure"] = pre_amm_df["cryst_structure_oxi_states"].apply(
    lambda struct: json.loads(struct, cls=MontyDecoder))
# Choose any available dummy target
pre_amm_df.drop(columns=[c for c in original_info_columns if c not in ["ordering", "structure"]],
                inplace=True)
af = AutoFeaturizer(n_jobs=6, preset="express")
amm_df = af.fit_transform(pre_amm_df, target="ordering")
amm_df.drop(columns=["ordering"], inplace=True)

feat_df = pd.merge(feat_df, amm_df, left_index=True, right_index=True)

for filter_type in ["cryst_structure_dfcat_sites", "cryst_structure_magcat_sites"]:
    scm_df = info_df.loc[structure_ids]
    scm_df["structure"] = scm_df[filter_type].apply(lambda struct: json.loads(struct, cls=MontyDecoder))
    scm_df.drop(columns=[c for c in scm_df.columns if c!= "structure"], inplace=True)

    scm = SineCoulombMatrix()
    scm_df = scm.fit_featurize_dataframe(df=scm_df, col_id="structure")
    scm_df.drop(columns=["structure"], inplace=True)
    feat_df = pd.merge(feat_df, scm_df, left_index=True, right_index=True, suffixes=("", "_"+filter_type.split("_")[2]))

max_na_frac = 0.01
# Issue with NaN cols from automatminer featurization, remove or fill with average val
to_remove = ["crystal_system"]  # Because string-encoded, MI comput., respective int feature also present
for col, content in feat_df.isnull().any().iteritems():
    if content:
        if feat_df[col].isna().sum() / len(feat_df) > max_na_frac:
            to_remove.append(col)
            print(f"Dropping col {col}")
feat_df.drop(columns=to_remove, inplace=True)
feat_df.fillna(value=feat_df.mean(), inplace=True)
for col, content in feat_df.isnull().any().iteritems():
    assert not content
for c in feat_df.columns:
    if set(feat_df[c].values).issubset({True, False}):
        feat_df[c] = feat_df[c].astype(int)

for t in ["target_p_exclude_ligand_multiplicities_binned3",  # no ap as complete label mapping in coll. dataset
          "target_classic"]:
    ta_df = feat_df.copy()
    ta_df.drop(columns=[c for c in ta_df.columns if ("target_" in c and c != t)], inplace=True)
    if "binned3" in t:
        ta_df = ta_df.loc[ta_df[t] < 3]
    # Before dropping duplicates, order feature columns alphabetically for finding meaningful minimal common subset
    ta_df = ta_df.reindex(sorted(ta_df.columns), axis=1)
    ta_df = ta_df.T.drop_duplicates().T
    ta_df = ta_df.loc[:, (ta_df != ta_df.iloc[0]).any()]
    print(t, len(ta_df))
    ta_df.to_json(
        f"data/250525_full_features_{t}_MP.json")
