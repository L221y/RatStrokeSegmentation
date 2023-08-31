import numpy as np
import SimpleITK as sitk
import os
import argparse
import warnings

from lib.Elastix import Elastix
from lib.normalization_mean import normalization
from lib.filesfinder import find_files
from lib.Transformix import Transformix
from lib.maskVolume import MaskVolume
from lib.adc_segmentation import ADC_segmentation_mediane
from lib.mpc_segmentaion import MPC_segmentation_mediane
from lib.visualize import visualize
from lib.registration import H24_registration



parser = argparse.ArgumentParser()
warnings.filterwarnings("ignore")

#Input folder
parser.add_argument("--input",dest="input",default=-1)
#Output folder
parser.add_argument("--output",dest="output",default=-1)
#DIFF Enable
parser.add_argument("--diff",dest="diff",default=-1)
#PERF Enable
parser.add_argument("--perf",dest="perf",default=-1)
#Threshold value
parser.add_argument("--threshold",dest="threshold",default=0.75)
#T2H24 Enable
parser.add_argument("--t2h24",dest="t2h24",default=-1)

args = parser.parse_args()


reference_T2_volume = sitk.ReadImage("./data/threshold/2dseq_W4-04_T2H0.nii")
reference_brain_mask = sitk.ReadImage("./data/threshold/Segmentation_W4-04.nii.gz")
reference_hemisphere_mask = sitk.ReadImage("./data/threshold/HemisphereMask-W4-04.nii.gz")

input_T2H0_path = sorted(find_files(args.input,"T2H0.nii"))



if int(args.diff) == 1:
    input_DWI_path = sorted(find_files(args.input,"DIFF.nii"))
if int(args.perf) == 1:
    input_PWI_path = sorted(find_files(args.input,"PERF.nii"))
if int(args.t2h24) == 1:
    input_t2h24_path = sorted(find_files(args.input,"T2H24.nii"))


for i in range(len(input_PWI_path)):
    print(i)

    output = args.output +f"{i+1}/"
    os.mkdir(output)
    os.mkdir(output+"./cache")
    
    input_T2_volume = sitk.ReadImage(input_T2H0_path[i])
    t2_name = os.path.basename(input_T2H0_path[i])
    sitk.WriteImage(input_T2_volume,os.path.join(output,t2_name))
    
    if int(args.diff) == 1:
        input_diff_volume = sitk.ReadImage(input_DWI_path[i])
        diff_name = os.path.basename(input_DWI_path[i])
        sitk.WriteImage(input_diff_volume,os.path.join(output,diff_name))
        
    if int(args.perf) == 1:
        input_perf_volume = sitk.ReadImage(input_PWI_path[i])
        perf_name = os.path.basename(input_PWI_path[i])
        sitk.WriteImage(input_perf_volume,os.path.join(output,perf_name))
        
    if int(args.t2h24) == 1:
        input_t2h24_volume = sitk.ReadImage(input_t2h24_path[i])
        t2h24_name = os.path.basename(input_t2h24_path[i])
        sitk.WriteImage(input_t2h24_volume,os.path.join(output,t2h24_name))
    
    #Elastix
    print("Registration start for", t2_name)
    registeredVolume,transformParameterMap = Elastix(reference_T2_volume,input_T2_volume,T2H24=False)
    #Transformix brain
    brain_mask = Transformix(reference_brain_mask,transformParameterMap)
    #Transformix hemisphere
    hemisphere_mask = Transformix(reference_hemisphere_mask,transformParameterMap)

    if int(args.diff) == 1:
        #Mask Volume
        masked_volume_diff = MaskVolume(input_diff_volume,brain_mask)
        #Normalization
        normalized_brain_diff = normalization(masked_volume_diff,hemisphere_mask)
        #ADC segmentation
        result_volume_diff = ADC_segmentation_mediane(normalized_brain_diff)
        sitk.WriteImage(result_volume_diff,output+"./ADC_segmentation.nii.gz")
        sitk.WriteImage(masked_volume_diff,output+"./cache/masked_volume_diff.nii.gz")
        sitk.WriteImage(normalized_brain_diff,output+"./cache/normalized_brain_diff.nii.gz")

    if int(args.perf) == 1:
        #Mask Volume
        masked_volume_perf = MaskVolume(input_perf_volume,brain_mask)
        #Normalization
        normalized_brain_perf = normalization(masked_volume_perf,hemisphere_mask)
        #MPC segmentation
        result_volume_perf = MPC_segmentation_mediane(normalized_brain_perf,hemisphere_mask,float(args.threshold))
        sitk.WriteImage(result_volume_perf,output+"./MPC_segmentation.nii.gz")
        sitk.WriteImage(masked_volume_perf,output+"./cache/masked_volume_perf.nii.gz")
        sitk.WriteImage(normalized_brain_perf,output+"./cache/normalized_brain_perf.nii.gz")

    #t2h24 segmentation
    
    if int(args.t2h24) == 1:
        pred = visualize(input_t2h24_path[i],t2h24 = True)
        sitk.WriteImage(pred,output+"./pred_t2h24.nii.gz")

        #registration for T2H24
        print("Registration start for T2H24")

        transformed_mask, registered_t2h24,trans = H24_registration(input_T2_volume,input_t2h24_volume,pred)
        sitk.WriteImage(transformed_mask,output+"./transformed_pred_t2h24.nii.gz")
        sitk.WriteImage(registered_t2h24,output+"./cache/registered_t2h24.nii")
        sitk.WriteParameterFile(trans,output+"./cache/Transform_t2h24.txt")
    

    sitk.WriteImage(registeredVolume,output+"./cache/registerVolume.nii.gz")
    sitk.WriteParameterFile(transformParameterMap,output+"./cache/Transform.txt")
    sitk.WriteImage(brain_mask,output+"./cache/brain_mask.nii.gz")
    sitk.WriteImage(hemisphere_mask,output+"./cache/hemisphere_mask.nii.gz")
    
    
    

    
    
    
    

