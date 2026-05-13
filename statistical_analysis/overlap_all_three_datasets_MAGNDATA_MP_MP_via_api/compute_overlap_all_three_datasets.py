import os
import pandas as pd


overlap_data_dir = "../overlap_MAGNDATA_MP/data"
o_mp_df = pd.read_json(os.path.join(overlap_data_dir, "matches_mp_magndata.json"))

overlap_api_data_dir = "../overlap_MAGNDATA_MP_via_api/data"
o_api_mp_df = pd.read_json(os.path.join(overlap_api_data_dir, "matches_mp_via_api_magndata.json"))


o_df = o_mp_df.loc[(o_mp_df["magndata_is_collinear"]) 
                   & (o_mp_df["magndata_is_chosen_one"]) 
                   & (o_mp_df["magndata_id"].isin(o_api_mp_df["magndata_id"].values))]

api_df = o_api_mp_df.loc[(o_api_mp_df["magndata_is_collinear"]) 
                   & (o_api_mp_df["magndata_is_chosen_one"]) 
                   & (o_api_mp_df["magndata_id"].isin(o_mp_df["magndata_id"].values))]

assert len(api_df) == len(o_df)

results_dict = {
    "all_three_same_p": [],
    "only_mp_magndata_same_p": [],
    "only_mp_via_api_mandata_same_p": [],
    "magndata_unique_p": []
}

for row_id, row in o_df.iterrows():
    md_id = row["magndata_id"]

    md_p = round(row["magndata_p"], 4)
    mp_p = round(row["mp_p"], 4)

    api_match = api_df.loc[api_df["magndata_id"]==md_id]
    assert len(api_match) == 1
    assert round(api_match["magndata_p"].values[0], 4) == md_p

    mp_api_p = round(api_match["mp_p"].values[0], 4)


    if md_p == mp_p:

        if md_p == mp_api_p:
            results_dict["all_three_same_p"].append(md_id)
        else:
            results_dict["only_mp_magndata_same_p"].append(md_id)
    
    elif md_p == mp_api_p:
        results_dict["only_mp_via_api_mandata_same_p"].append(md_id)
    
    else:
        results_dict["magndata_unique_p"].append(md_id)


result_str = f"{len(o_df)} matches between all three datasets. \n"

for k, v in results_dict.items():
    result_str += f"{k}: {len(v)} counts. \n"

with open("string_summary.txt", "w") as f:
    f.write(result_str)
