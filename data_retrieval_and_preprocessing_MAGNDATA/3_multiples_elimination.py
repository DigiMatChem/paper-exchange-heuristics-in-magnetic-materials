import json
from monty.json import MontyDecoder, MontyEncoder
from pymatgen.analysis.structure_matcher import StructureMatcher

from utils_kga.data_retrieval_and_preprocessing.multiples_elimination import *


with open("data/df_commensurate_MAGNDATA.json") as f:
    df = json.load(f, cls=MontyDecoder)

# Get crystallographic primitive structure and groups of multiples,
# sanitize temperature and citation year meta data
df["cryst_structure"] = None
df["group_index"] = None
df["transition_temperature_san"] = None
df["experiment_temperature_san"] = None
df["citation_year_san"] = None

max_group_index = 0

for idx, (row_id, row) in enumerate(df.iterrows()):

    # Get crystallographic primitive
    struct = json.loads(row["mag_structure"], cls=MontyDecoder)
    prim = get_crystallographic_primitive(struct)
    df.at[row_id, "cryst_structure"] = prim

    # Assign multiples group index
    unique = True
    for row_id_comp, row_comp in df[:idx].iterrows():
        if StructureMatcher().fit(prim, row_comp["cryst_structure"],
                                  skip_structure_reduction=True):
            unique = False
            df.at[row_id, "group_index"] = row_comp["group_index"]
            break

    if unique:
        max_group_index += 1
        df.at[row_id, "group_index"] = max_group_index

    # Add sanitized temperatures and citation year
    df.at[row_id, "transition_temperature_san"] = clean_and_convert_temperature_string(row["transition_temperature"])
    df.at[row_id, "experiment_temperature_san"] = clean_and_convert_temperature_string(row["experiment_temperature"])
    df.at[row_id, "citation_year_san"] = convert_citation_year_string(row["citation_year"])


# Assign chosen one in each group by highest transition / experiment temperature > newest publication
df["chosen_one"] = None
for group_idx, group in df.groupby(["group_index"]).groups.items():

    # Case of no multiples
    if len(group) == 1:
        df.at[group[0], "chosen_one"] = True

    else:
        group_df = df.loc[group]

        # Check if transition temperatures present at all
        if True in group_df["transition_temperature_san"].notna().values:

            # Case: transition temperature only present in subset of entries
            if group_df["transition_temperature_san"].isnull().values.any():

                # Is any experiment temperature above highest transition temperature in group?
                transition_temp_present = group_df.loc[group_df["transition_temperature_san"].notna()]
                highest_t_temp = transition_temp_present["transition_temperature_san"].values.max()

                experiment_temp_present = group_df.loc[group_df["experiment_temperature_san"].notna()]
                if len(experiment_temp_present) > 0:
                    highest_e_temp = experiment_temp_present["experiment_temperature_san"].values.max()
                else:
                    highest_e_temp = 0

                if highest_t_temp < highest_e_temp:
                    experiment_temp_highest = experiment_temp_present.loc[
                        experiment_temp_present["experiment_temperature_san"] == highest_e_temp]

                    chosen_one = pick_only_one_or_choose_by_newest_publication_or_pick_lowest_index(
                        df=experiment_temp_highest)

                # Case: no experiment temperatures higher than highest transition temperature
                else:
                    transition_temp_highest = transition_temp_present.loc[
                        transition_temp_present["transition_temperature_san"] == highest_t_temp]

                    chosen_one = pick_only_one_or_choose_by_newest_publication_or_pick_lowest_index(
                        df=transition_temp_highest)

            # All entries in group have transition temperature
            else:
                # How many entries with highest transition temperature?
                transition_temp_highest = group_df.loc[
                    group_df["transition_temperature_san"] == group_df["transition_temperature_san"].values.max()]

                chosen_one = pick_only_one_or_choose_by_newest_publication_or_pick_lowest_index(
                    df=transition_temp_highest)

        # Case: no transition temperatures present at all
        else:
            # Go by highest experiment temperature instead
            if any(group_df["experiment_temperature_san"].values):
                experiment_temp_present = group_df.loc[group_df["experiment_temperature_san"].notna()]
                experiment_temp_highest = experiment_temp_present.loc[
                    experiment_temp_present["experiment_temperature_san"] == experiment_temp_present[
                        "experiment_temperature_san"].values.max()]

                chosen_one = pick_only_one_or_choose_by_newest_publication_or_pick_lowest_index(
                    df=experiment_temp_highest)

            # If no temperature info at all, choose by newest publication, after that by lowest index
            else:
                chosen_one = choose_by_newest_publication_or_pick_lowest_index(df=group_df)

        for df_idx in group:
            if df_idx == chosen_one:
                df.at[df_idx, "chosen_one"] = True
            else:
                df.at[df_idx, "chosen_one"] = False

# Sanity check
assert df["group_index"].values.max() == len(df.loc[df["chosen_one"]])

# Save to json
with open("data/df_grouped_and_chosen_commensurate_MAGNDATA.json", "w") as f:
    json.dump(df, f, cls=MontyEncoder)



