# General
This repository contains code and data of the publication 
*Can simple exchange heuristics guide us in predicting magnetic properties of solids?*  
We review a popular structure-magnetism heuristic on datasets of magnetic structures ([statistical_analysis](statistical_analysis))
and investigate whether we can utilize the heuristic to predict magnetic structures ([featurization](featurization), [models](models)).

TODO insert link

The code has been written by K. Ueltzen with contributions from P. Benner, J. George and A. Naik.

# Installation
For executing all scripts except two (see below), create a new Python environment with Python 3.10 
and pip install the packages from `requirements.txt`. 
After that, install project-specific utils by executing `pip install ./utils_kga` from this location.

Two scripts ([2_add_automatminer_features.py](featurization/MAGNDATA/2_add_automatminer_features.py) 
for MAGNDATA and [2_add_automatminer_features.py](featurization/MP/2_add_automatminer_features.py) for MP) 
contain featurization with the `Automatminer` package, for those, create an environment from `requirements_automatminer.txt`.

Large parts of the data and figures produced by the scripts of this repository are given in the repository in a zipped format.
For unzipping all files run `find . -type f -name '*.gz' -execdir gunzip '{}' \;
` . The total repository size with everything unzipped is about TODO. 
Please note that not all data produced by the scripts is uploaded (e.g., redundant figures in different formats), but can easily 
be replicated with the scripts.

# License
All code (including all *.py and *.ipynb) is licensed under the TODO License (TODO link), 
while the rest of this repository is released under the TODO license (TODO link).

# Project structure
Generally, scripts are numbered to indicate the order of execution.

## data_retrieval_and_preprocessing_MAGNDATA
This folder contains all scripts to 
- retrieve all commensurate magnetic structures from the MAGNDATA database 
([1_commensurate_MAGNDATA_crawler.py](data_retrieval_and_preprocessing_MAGNDATA/1_commensurate_MAGNDATA_crawler.py))
- filter and analyze database entries 
([2_filter_convert_get_provenance_and_coordination_features_commensurate_MAGNDATA.py](data_retrieval_and_preprocessing_MAGNDATA/2_filter_convert_get_provenance_and_coordination_features_commensurate_MAGNDATA.py))
  - represent entries as pymatgen `Structure` and as `CoordinationFeatures` objects containing information on their
  connectivity
  - collect metadata like the magnetic transition temperature
  - exclude erroneous entries (e.g., those with unphysically low distances, for a detailed list see script)
  - convert entries (e.g., convert D as H, for a detailed list see script)
- find a crystallographically unique subset 
([3_multiples_elimination.py](data_retrieval_and_preprocessing_MAGNDATA/3_multiples_elimination.py))
-> this is especially relevant to avoid data leakage in the machine learning part
  - identify entries with the same crystallographic structure (without magnetic information)
  - out of groups of crystallographic multiples, select one entry as per highest magnetic transition / experiment 
  temperature > newest publication > pseudo-random choice  

The subfolder `data` contains the main output of these scripts (`df_grouped_and_chosen_commensurate_MAGNDATA.json`) that is required for the stat. analysis and the ML
as well as the log of MAGNDATA structures
that are excluded in [2_filter_convert_get_provenance_and_coordination_features_commensurate_MAGNDATA.py](data_retrieval_and_preprocessing_MAGNDATA/2_filter_convert_get_provenance_and_coordination_features_commensurate_MAGNDATA.py).

Please note that preprocessing of the MP database is done in [1_get_coordinationnet_features_of_MP_database.py](statistical_analysis/MP/1_get_coordinationnet_features_of_MP_database.py)
in `statistical_analysis/MP`.

## statistical_analysis
This folder contains all scripts to replicate the analysis results for both the MAGNDATA and MP dataset presented 
in the Sections `Statistical Analysis`, `Outlook` and the SI of the paper.
### MAGNDATA
- [1_compute_magnetic_node_and_edge_informations.py](statistical_analysis/MAGNDATA/1_compute_magnetic_node_and_edge_informations.py) 
collects info on all magnetic edges and nodes of the summarized connectivity representation that is contained in the 
`CoordinationFeatures` objects for further analysis for the subset of crystallographically unique,
commensurate MAGNDATA entries
- [2_analyze_TM_octahedra.ipynb](statistical_analysis/MAGNDATA/2_analyze_TM_octahedra.ipynb) analyzes bond angle occurrences 
as a function of the angle between the magnetic vectors (spin angle) of nearest neighbor transition metal (TM) sites 
that are octahedrally coordinated (the subset relevant to the simplification of the Kanamori-Goodenough-Anderson (KGA) rules). Includes plots,
statistics and the Kolmogorov-Smirnov (KS) test to compare AFM and FM bond angle distributions
- The scripts `2_analyze_TM_octahedra_90-deg_*.ipynb` analyze possible reason for KGA rule breaking in AFM 
interactions with 90 deg bond angles that are discussed in the SI
- [2_analyze_TM_sites.ipynb](statistical_analysis/MAGNDATA/2_analyze_TM_sites.ipynb) analyzes bond angle occurrences 
as a function of the spin angle for all magnetic TM--TM interactions (also those that are not octahedrally coordinated).
This is especially relevant for the KS test computation as we aim to compare this to the MaterialsProject (MP, see below)
dataset to understand low importance of bond angle features in a previous ML study.
- [2_plot_sitewise_collinearity_and_ces.ipynb](statistical_analysis/MAGNDATA/2_plot_sitewise_collinearity_and_ces.ipynb) 
plots the distribution of coordination environments of magnetic sites as a function of their sitewise collinearity, 
displayed in the `Outlook` of the paper.
- [2_get_n_structures-dependent_p_value_MAGNDATA_determine_n_sampling_per_size.py](statistical_analysis/MAGNDATA/2_get_n_structures-dependent_p_value_MAGNDATA_determine_n_sampling_per_size.py)
determines the minimum number of samples required for a n_structures-dependent analysis of KS test results that yields 
a p value sample standard deviation below 0.01 for all n_structures. Together with the results from 
[2_get_n_structures-dependent_p_value_MP_determine_n_sampling_per_size.py](statistical_analysis/MP/2_get_n_structures-dependent_p_value_MP_determine_n_sampling_per_size.py),
1,000 repetitions are determined for this analysis
- [3_plot_n_structures-dependent_p_values_test_statistics_and_sample_size_MAGNDATA.py](statistical_analysis/MAGNDATA/3_plot_n_structures-dependent_p_values_test_statistics_and_sample_size_MAGNDATA.py)
plots p values, test statistics d and the true KS test size (FM and AFM bond angle occurrences) as a function of 
n_structures. In the SI, these results are compared to 
[3_plot_n_structures-dependent_p_values_test_statistics_and_sample_size_MP.py](statistical_analysis/MP/3_plot_n_structures-dependent_p_values_test_statistics_and_sample_size_MP.py)
- [3_compute_magnetic_edge_informations_include_crystallographic_multiples.py](statistical_analysis/MAGNDATA/3_compute_magnetic_edge_informations_include_crystallographic_multiples.py)
collects info on all magnetic edges and nodes of the summarized connectivity representation that is contained in the 
`CoordinationFeatures` objects for further analysis for the subset of *all* commensurate MAGNDATA entries. This is done
to investigate in 
[4_analyze_TM_octahedra_include_crystallographic_multiples.ipynb](statistical_analysis/MAGNDATA/4_analyze_TM_octahedra_include_crystallographic_multiples.ipynb)
if the bond angle - magnetism trends found in 
[2_analyze_TM_octahedra.ipynb](statistical_analysis/MAGNDATA/2_analyze_TM_octahedra.ipynb) are a function of the chosen subset
of cryst. unique entries (see SI for discussion)

### MP
- [1_get_coordinationnet_features_of_MP_database.py](statistical_analysis/MP/1_get_coordinationnet_features_of_MP_database.py) 
extracts the structurally unique dataset of the MagneticOrderingsWorkflow dataset from NC Frey et al. Science Advances 2020, 6 (50), eabd1076,
 collects metadata on the entries and represents the magnetic structures as `CoordinationFeatures` objects.
- [2_get_bond_angle_statistics_of_MP_database.py](statistical_analysis/MP/2_get_bond_angle_statistics_of_MP_database.py)
collects info on magnetic edges. Bond angle occurrences are analyzed as a function of the spin angle, 
both for nearest neighbor TM sites and their subset that are octahedrally coordinated. Includes plots, statistics and 
the Kolmogorov-Smirnov (KS) test to compare AFM and FM bond angle distributions.
- [2_get_n_structures-dependent_p_value_MP_determine_n_sampling_per_size.py](statistical_analysis/MP/2_get_n_structures-dependent_p_value_MP_determine_n_sampling_per_size.py)
determines the minimum number of samples required for a n_structures-dependent analysis of KS test results that yields 
a p value sample standard deviation below 0.01 for all n_structures.
- [3_plot_n_structures-dependent_p_values_test_statistics_and_sample_size_MP.py](statistical_analysis/MP/3_plot_n_structures-dependent_p_values_test_statistics_and_sample_size_MP.py)
plots p values, test statistics d and the true KS test size (FM and AFM bond angle occurrences) as a function of n_structures. 
In the SI, these results are compared to [3_plot_n_structures-dependent_p_values_test_statistics_and_sample_size_MAGNDATA.py](statistical_analysis/MAGNDATA/3_plot_n_structures-dependent_p_values_test_statistics_and_sample_size_MAGNDATA.py).

### MP_via_api
Additionally to the MagneticOrderingsWorkflow dataset by Frey et al. (see above), we repeated the statistical analysis
for all structures in the MaterialsProject database that have magnetic structures determined by the MagneticOrderings workflow
(*not* those simply initialized as FM). The statistical test results concerning bond angle trends of FM and AFM interactions
are similar to the dataset by Frey et al. and further work (analysis of mutual information and machine learning) considers
the dataset by Frey et al.  
All MP task ids belonging to the MagneticOrderings workflow ([find_query.json](statistical_analysis/MP_via_api/data/find_query.json)) 
were provided by Jason Munro (thanks!).
- [1_get_MP_magnetism_data_via_api.ipynb](statistical_analysis/MP_via_api/1_get_MP_magnetism_data_via_api.ipynb)
maps the task ids to mp-ids, downloads the structures and filters for those determined unique by the pymatgen StructureMatcher.
- [2_get_coordinationnet_features_of_MP_database_via_api.py](statistical_analysis/MP_via_api/2_get_coordinationnet_features_of_MP_database_via_api.py)
represents the magnetic structures as `CoordinationFeatures` objects and saves additional metadata on the entries.
- [3_get_bond_angle_statistics_of_MP_database_via_api.py](statistical_analysis/MP_via_api/3_get_bond_angle_statistics_of_MP_database_via_api.py)
collects info on magnetic edges. Bond angle occurrences are analyzed as a function of the spin angle, 
both for nearest neighbor TM sites and their subset that are octahedrally coordinated. Includes plots, statistics and 
the Kolmogorov-Smirnov (KS) test to compare AFM and FM bond angle distributions.
- [3_get_n_structures-dependent_p_value_MP_via_api_determine_n_sampling_per_size.py](statistical_analysis/MP_via_api/3_get_n_structures-dependent_p_value_MP_via_api_determine_n_sampling_per_size.py)
determines the minimum number of samples required for a n_structures-dependent analysis of KS test results that yields 
a p value sample standard deviation below 0.01 for all n_structures.
- [4_plot_n_structures-dependent_p_values_test_statistics_and_sample_size_MP_via_api.py](statistical_analysis/MP_via_api/4_plot_n_structures-dependent_p_values_test_statistics_and_sample_size_MP_via_api.py)
plots p values, test statistics d and the true KS test size (FM and AFM bond angle occurrences) as a function of n_structures. 


## featurization
This folder contains all scripts for transforming the non-magnetic parent structures of the cryst. unique MAGNDATA 
dataset, its RE-free subset and MP dataset into structural and compositional descriptors for ML. 
Both general automatminer features as well as custom, magnetism-specific features are computed.
For creation of custom features that include only sites guessed magnetic, e.g., mean distance between nearest-neighbor 
"magnetic" sites, two methods are employed to guess sites to be magnetically ordered or not:
  1. sites are assumed to be magnetic if they are classified as cationic and belong to the *d* or *f* block
  2. sites are assumed to be magnetic if they are classified as cationic and are contained in pymatgen's 
  `default_magmoms.yaml` from the `pymatgen.analysis.magnetism.analyzer` module including originally uncommented entries
  as magnitude of guessed magnetic moment not decisive

Further, this folder contains the computation of feature-target normalized mutual information (NMI) that is 
presented in the Section `Machine-learning model for magnetism` and the SI of the paper.

### MAGNDATA
- [1_get_sitewise_and_structurewise_coordinationnet_features.py](featurization/MAGNDATA/1_get_sitewise_and_structurewise_coordinationnet_features.py) computes the custom, magnetism-specific features.
- [2_add_automatminer_features.py](featurization/MAGNDATA/2_add_automatminer_features.py) computes general
compositional and structural automatminer features.
- [3_add_labels.py](featurization/MAGNDATA/3_add_labels.py) computes the binned structurewise p and ap scores with a 10°
tolerance to count magnetic vectors of neighboring sites as (anti-)parallel.
- [4_determine_maximal_common_feature_subset_all_targets_all_datasets.py](featurization/MAGNDATA/4_determine_maximal_common_feature_subset_all_targets_all_datasets.py)
For all three datasets and all targets (binned p score, binned ap score, AFM / FM classification target), the intersection
of non-duplicate, non-constant features is determined and stored for further NMI computations / models.
Each feature-target dataframe is split into 20 train test splits (with no overlap between the 5% test splits). For p/ap models, in the 
two larger dataframes (MP and all-structs of MAGNDATA), an additional, stratified undersampling of the train sets is
performed to yield train sets of the same size as in the smallest dataset (RE-free structures of MAGNDATA) so ML results 
are not a function of dataset size and can be compared directly.
- [5_compute_normalized_mutual_infomation_as_f_random_seed.py](featurization/MAGNDATA/5_compute_normalized_mutual_infomation_as_f_random_seed.py) computes 
the mutual information (MI) and its normalized variant (NMI) between all features and the target for the p and ap target in the 20 train splits of the 2
MAGNDATA datasets. MI and NMI are computed with different random seeds until convergence is reached (which is, in this case, defined as that the sample 
standard deviation of more than 99 % of all feature-target NMIs dropping below .1 of the respective feature sample mean). 
Convergence is checked for every 500 random seeds.
- [6_plot_normalized_mutual_infomation_features_w_p_and_ap_as_f_random_seed_all_datasets.py](featurization/MAGNDATA/6_plot_normalized_mutual_infomation_features_w_p_and_ap_as_f_random_seed_all_datasets.py) 
plots NMI and MI distributions of the p and ap target in all three datasets.
- [7_determine_NMI_closest_to_sample_mean.py](featurization/MAGNDATA/7_determine_NMI_closest_to_sample_mean.py) determines
the random seed for NMI computation that minimizes the sum of absolute distances of feature-target NMI(random seed) to their respective
feature-target mean NMI (i.e., it determines the random seed that gives the most average feature-target NMI results).
This is done per target per dataset per train split (for both MAGNDATA and MP datasets).
- [8_evaluate_mag_site_guessing_methods.py](featurization/MAGNDATA/8_evaluate_mag_site_guessing_methods.py) evaluates the two
methods applied during featurization to guess magnetically ordered sites for both MAGNDATA and MP datasets.

### MP
- [1_get_labels_sitewise_and_structurewise_coordinationnet_features.py](featurization/MP/1_get_labels_sitewise_and_structurewise_coordinationnet_features.py)
computes the custom, magnetism-specific features, the binned structurewise p and ap labels and the AFM / FM classification labels
as done in Frey et al.
- [2_add_automatminer_features.py](featurization/MP/2_add_automatminer_features.py) computes general
compositional and structural automatminer features.
- [3_compute_normalized_mutual_information_as_f_random_seed.py](featurization/MP/3_compute_normalized_mutual_information_as_f_random_seed.py) computes 
the mutual information (MI) and its normalized variant (NMI) between all features and the target for the p and AFM / FM classification target in the 20 train splits of the
MP dataset. MI and NMI are computed with different random seeds until convergence is reached (which is, in this case, defined as that the sample 
standard deviation of more than 99 % of all feature-target NMIs dropping below .1 of the respective feature sample mean). 
Convergence is checked for every 500 random seeds.
- [4_plot_mutual_infomation_features_classic_target_as_f_random_seed.py](featurization/MP/4_plot_mutual_infomation_features_classic_target_as_f_random_seed.py)
computes and plots the feature-target NMIs as f(random seed) for the AFM / FM classification target of the MP dataset.

**Note regarding the data:** The `data` folders contain the fully featurized dataframes as yielded by [3_add_labels.py](featurization/MAGNDATA/3_add_labels.py)
and [2_add_automatminer_features.py](featurization/MP/2_add_automatminer_features.py). They do *not* contain the redundant data representation
of 20 train test splits created in later scripts. Also, the raw MI and NMI data is not included, but just the analysis results on
the most important features.  
In case you require the raw NMI data or the 20 train test split data (e.g., for running the ML models below), execute the
scripts starting from [4_determine_maximal_common_feature_subset_all_targets_all_datasets.py](featurization/MAGNDATA/4_determine_maximal_common_feature_subset_all_targets_all_datasets.py)
and [3_compute_normalized_mutual_information_as_f_random_seed.py](featurization/MP/3_compute_normalized_mutual_information_as_f_random_seed.py).

## models
This folder contains the training and evaluation of ML models for magnetic structure prediction. The results are presented
in the Section `Machine-learning model for magnetism` and the SI of the paper.

### MAGNDATA
- [1_RF_MAGNDATA_TM-structs_all-structs_p_ap.py](models/MAGNDATA/1_RF_MAGNDATA_TM-structs_all-structs_p_ap.py) trains RF 
models on the binned p and ap scores in a nested ten-fold nested CV approach on the MAGNDATA dataset and its RE-free subset.
This gives 20 models per target per dataset (20 different train-test splits with non-overlapping test sets).
The script requires two command line arguments (the train-test split and the target).  
If you have access to an hpc cluster with slurm, you can modify the following batch script and the `n_jobs_*` parameters in the script
for running all MAGNDATA models.
```
#!/bin/bash
#SBATCH --job-name=RF-md
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2
#SBATCH --time=99:00:00
#SBATCH --output=out.%j.txt
#SBATCH --error=error.%j.txt
#SBATCH --mail-user=your-e-mail
#SBATCH --mail-type=ALL

micromamba activate path-to-your-environment

list1=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19")
list2=("p" "ap")

for param1 in "${list1[@]}"; do
  for param2 in "${list2[@]}"; do
    srun --exclusive -n 1 python 1_RF_MAGNDATA_TM-structs_all-structs_p_ap.py "$param1" "$param2" &
    if [[ $(jobs -r -p | wc -l) -ge 40 ]]; then
      wait -n
    fi
  done
done

# Wait for all background jobs to finish
wait
```
You can also modify the number of jobs for feature selection and RF model training - however,
please note that the Shapley value computation is not parallelized.

- [1_RF_MAGNDATA_TM-structs_all-structs_p_ap_without_heuristic_features.py](models/MAGNDATA/1_RF_MAGNDATA_TM-structs_all-structs_p_ap_without_heuristic_features.py)
trains comparison p and ap models where different groups of features related to the KGA rules of thumb are eliminated before training.

- [2_evaluate_RF_p_ap_all_databases.py](models/MAGNDATA/2_evaluate_RF_p_ap_all_databases.py) evaluates the p and ap models obtained from
[1_RF_MAGNDATA_TM-structs_all-structs_p_ap.py](models/MAGNDATA/1_RF_MAGNDATA_TM-structs_all-structs_p_ap.py) and [1_RF_MP_p.py](models/MP/1_RF_MP_p.py),
including model metrics and plotting of grouped feature importance plots.

### MP
- [1_RF_MP_p.py](models/MP/1_RF_MP_p.py) trains p models on the MP dataset similar to [1_RF_MAGNDATA_TM-structs_all-structs_p_ap.py](models/MAGNDATA/1_RF_MAGNDATA_TM-structs_all-structs_p_ap.py).
The script requires one command line argument for the train-test split.
- [1_RF_MP_afm-fm.py](models/MP/1_RF_MP_afm-fm.py) trains AFM / FM classification models on the MP dataset. 
The script requires one command line argument for the train-test split.
- [1_RF_MP_p_without_heuristic_features.py](models/MP/1_RF_MP_p_without_heuristic_features.py) and [1_RF_MP_afm-fm_without_heuristic_features.py](models/MP/1_RF_MP_afm-fm_without_heuristic_features.py)
train comparison p and AFM / FM models where different groups of features related to the KGA rules of thumb are eliminated before training.
- [2_evaluate_RF_afm-fm_MP.py](models/MP/2_evaluate_RF_afm-fm_MP.py) evaluates the AFM / FM classification models
including model metrics and plotting of grouped feature importance plots.

## utils_kga
This folder hosts the package that contains all project-specific utility functions.

## tests
This folder contains all tests for the `utils_kga` functions.
