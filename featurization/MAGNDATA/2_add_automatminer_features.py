"""
Requires automatminer env.
Also add an SCM of the sites guessed magnetic.
"""
from automatminer.featurization import AutoFeaturizer
import json
from matminer.featurizers.structure import SineCoulombMatrix
from monty.json import MontyDecoder, MontyEncoder
import pandas as pd


with open("data/250226_structure_cn_features.json") as f:
    feat_df = pd.DataFrame(json.load(f))
with open("data/df_grouped_and_chosen_commensurate_MAGNDATA_additional_nonmag_info.json") as f:
    info_df = json.load(f, cls=MontyDecoder)

original_info_columns = info_df.columns
structure_ids = feat_df.index

pre_amm_df = info_df.loc[structure_ids]
assert len(pre_amm_df) == len(feat_df)
pre_amm_df["structure"] = pre_amm_df["cryst_structure_oxi_states"].apply(
    lambda struct: json.loads(struct, cls=MontyDecoder))
# Choose any available dummy target
pre_amm_df.drop(columns=[c for c in original_info_columns if c != "total_moment_per_supercell"],
                inplace=True)
af = AutoFeaturizer(n_jobs=6, preset="express")
amm_df = af.fit_transform(pre_amm_df, target="total_moment_per_supercell")
amm_df.drop(columns=["total_moment_per_supercell"], inplace=True)

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
to_remove = ["crystal_system"]  # Because string-encoded, MI comput.
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

# Further cleaning in next script after label addition
with open("data/250226_structure_cn_amm_features_dfcat_magcat.json", "w") as f:
    json.dump(feat_df, f, cls=MontyEncoder)
