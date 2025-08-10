This package contains all utilities to execute the scripts 
of the publication *Can simple exchange heuristics guide us in predicting magnetic properties of solids?*

The functionalities of *coordination_features* are
adapted from the package 
[*pycoordinationnet*](https://github.com/pbenner/pycoordinationnet) 
written by Philipp Benner. Changes have been made to 
*features_datatypes.py* and *features_featurizer.py* to include
a second method of oxidation state guessing and account for 
edge multiplicities in the computation of next-nearest-neighbor
features. *features_coding.py* remains unchanged.