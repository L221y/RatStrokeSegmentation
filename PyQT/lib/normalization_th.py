import SimpleITK as sitk
import numpy as np

def normalization(volume,mask_hemisphere):
    cast_brain = sitk.Cast(volume,sitk.sitkUInt32)
    cast_mask = sitk.Cast(mask_hemisphere, sitk.sitkUInt32)
    stats_filter = sitk.LabelStatisticsImageFilter()
    stats_filter.Execute(volume,cast_mask)

    mediane = stats_filter.GetMedian(1)
    volume_arr = sitk.GetArrayFromImage(volume)
    normalized_arr = np.zeros((15,256,256))
    
    iteration = volume_arr.shape[0]
    for i in range(iteration):
        for row in range(volume_arr.shape[1]):
            for col in range(volume_arr.shape[2]):
                normalized_arr[i,row,col] = volume_arr[i,row,col]/mediane

    normalized_volume = sitk.GetImageFromArray(normalized_arr)

    resamplefilter = sitk.ResampleImageFilter()

    normalized_volume.SetSpacing(volume.GetSpacing())
    normalized_volume.SetOrigin(volume.GetOrigin())
    normalized_volume.SetDirection(volume.GetDirection())
            
    resamplefilter.SetReferenceImage(volume)
    resampled_image = resamplefilter.Execute(normalized_volume)
    
    return resampled_image

# brain = sitk.ReadImage("/home/dliang/Documents/Carmen/23_stage_MISS_WIART/06_OP2Data_resetGrayScale/Q1/W2-15/BrainRescaled.nii.gz")
# hemisphere = sitk.ReadImage("/home/dliang/Documents/Carmen/23_stage_MISS_WIART/04.5_OP2Data_HemisphereMask/Q1/W2-15/hemisphere.nii.gz")
# dodo = normalization(brain,hemisphere)
# sitk.WriteImage(dodo,"dodo.nii.gz")