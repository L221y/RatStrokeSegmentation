import SimpleITK as sitk
import os
import warnings


from lib.filesfinder import find_files
from lib.visualize import visualize




def deep_learning_segmentation(input,outputPath,diff,perf=False,t2h24=False):
    warnings.filterwarnings("ignore")

    if diff == True:
        input_diff_path = sorted(find_files(input,"diff.nii"))
        for i in range(len(input_diff_path)):
            diff_volume = sitk.ReadImage(input_diff_path[i])
            diff_name = os.path.basename(input_diff_path[i])
            pred = visualize(input_diff_path[i],diff = True)
            output = outputPath+f"{i+1}"
            try:
                os.mkdir(output)
            except FileExistsError:
                j = 0

            sitk.WriteImage(diff_volume,os.path.join(output,diff_name))
            sitk.WriteImage(pred,output+"/pred_diff.nii.gz")

    if perf == True:
        input_perf_path = sorted(find_files(input,"perf.nii"))
        for i in range(len(input_perf_path)):
            perf_volume = sitk.ReadImage(input_perf_path[i])
            perf_name = os.path.basename(input_perf_path[i])
            pred = visualize(input_perf_path[i],perf = True)
            output = outputPath+f"{i+1}"
            try:
                os.mkdir(output)
            except FileExistsError:
                j = 0

            sitk.WriteImage(perf_volume,os.path.join(output,perf_name))
            sitk.WriteImage(pred,output+"/pred_perf.nii.gz")

    if t2h24 == True:
        input_t2h24_path = sorted(find_files(input,"T2H24.nii"))
        for i in range(len(input_t2h24_path)):
            t2h24_volume = sitk.ReadImage(input_t2h24_path[i])
            t2h24_name = os.path.basename(input_t2h24_path[i])
            pred = visualize(input_t2h24_path[i],t2h24 = True)
            output = outputPath+f"{i+1}"
            try:
                os.mkdir(output)
            except FileExistsError:
                j = 0

            sitk.WriteImage(t2h24_volume,os.path.join(output,t2h24_name))
            sitk.WriteImage(pred,output+"/pred_t2h24.nii.gz")
            

    print("Segmentation complete")