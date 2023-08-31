# RatStrokeSegmentation

The project web page is here: https://www.creatis.insa-lyon.fr/~grenier/?p=1317
This work is under publication and full material and access will be authorized after paper acceptance.
MRI images and manual annotations can be downloaded from this page.

# PyQT
The first interface is written with PyQT, the input and output directory are chosen and the pipline will automatically do the segmentation

## Input Data
ATTENTION: All the sequences for the same animals should be be in the same folder. 
The DWI sequence should be ended with "DIFF.nii.gz", for PWI is "PERF.nii.gz", for T2H0 and T2H24 are "T2H0.nii" and "T2H24.nii". 
