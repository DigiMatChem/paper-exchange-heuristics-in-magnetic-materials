## Comments on tests for data_retrieval_and_preprocessing

### General tests
[test_p_ap_score.py](test_p_ap_score.py) compares the computed p and ap score for different parameters with 
manually computed results.

### Supercell tests
To test the validity of our approach of coordination-based neighbors determination, we asserted 
for the whole set of 938 crystallographically unique MAGNDATA structures that the p and ap score
is the same for each structure and its 2x2x2 superstructure 
(see [test_supercell_p_ap_score.py](test_supercell_p_ap_score.py)). The test output is stored in 
[test_supercell_p_ap_score_SKIPPED_or_FAILED.log](test_supercell_p_ap_score_SKIPPED_or_FAILED.log) - 
please note that in the case of the three failing structures, the determination of cations and anions in the cell and 
its supercell differs, causing the test to fail for these 3 structures because the connectivities are different.

