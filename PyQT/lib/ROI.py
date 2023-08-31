import SimpleITK as sitk
import numpy as np
import os
import matplotlib.pyplot as plot
import pandas as pd
from scipy.spatial.distance import dice
from scipy.spatial.distance import directed_hausdorff


ROI_DIFF = "/home/dliang/Documents/Carmen/23_stage_MISS_WIART/09_OP2Data_ROI/Q1/W3-01/Mask/DIFF/"
Segmentation = "/home/dliang/Documents/Carmen/23_stage_MISS_WIART/07_OP2Data_ADC_segmentation/Q1/W3-01/ Segmentation/MPCSegmentation.nii.gz"
masks = os.listdir(ROI_DIFF)
masks.sort()

ADC = sitk.ReadImage(Segmentation)
ADC_arr = sitk.GetArrayFromImage(ADC)
#print(np.shape(ADC_arr))
#print(type(ADC_arr))
#np.savetxt("/home/dliang/Desktop/file.txt",ADC_arr,delimiter="\t")

ROI = np.zeros((15,128,128))
for file in masks:
    mask_path = os.path.join(ROI_DIFF,file)
    num = file[5:-4]
    index = int(num)
    mask = pd.read_csv(mask_path,delimiter="\t")
    mask_arr = mask.values
    #print(mask_arr.shape[0])
    print(mask)
    nrows = mask_arr.shape[0]

    for i in range(nrows):
        x_coordinate = int(mask_arr[i,0])
        y_coordinate = int(mask_arr[i,1])
        ROI[index-1,y_coordinate,x_coordinate] = 1


iteration = ROI.shape[0]
for i in range(iteration):
    arr1 = ADC_arr[i,:,:].copy()
    arr2 = ROI[i,:,:].copy()
    dice_coeff = 1-dice(arr1.flatten(),arr2.flatten())
    print("Dice coeff",dice_coeff)

    distance1 = directed_hausdorff(arr1,arr2)[0]
    distance2 = directed_hausdorff(arr2,arr1)[0]

    hausdorf_distance = max(distance1,distance2)
    print("Hausdorff distance", hausdorf_distance)


ROI_nifti = sitk.GetImageFromArray(ROI)
sitk.WriteImage(ROI_nifti,"/home/dliang/Desktop/eg.nii.gz")



