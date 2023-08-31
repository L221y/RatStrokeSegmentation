import SimpleITK as sitk
import numpy as np

def normalization(volume,mask_hemisphere):
    volume_arr = sitk.GetArrayFromImage(volume)
    mask_hemisphere_arr = sitk.GetArrayFromImage(mask_hemisphere)

    masked_volume = volume_arr*mask_hemisphere_arr

    volume_nonzero = masked_volume[masked_volume != 0]
    mean = np.mean(volume_nonzero)
    volume_arr = sitk.GetArrayFromImage(volume)
    normalized_arr = np.zeros((15,256,256))
    
    iteration = volume_arr.shape[0]
    for i in range(iteration):
        masked_slice = volume_arr[i,:,:]*mask_hemisphere_arr[i,:,:]

        slice_nonzero = masked_slice[masked_slice != 0]
        mean_slice = np.mean(slice_nonzero)

        for row in range(volume_arr.shape[1]):
            for col in range(volume_arr.shape[2]):
                normalized_arr[i,row,col] = volume_arr[i,row,col]/mean_slice

    normalized_volume = sitk.GetImageFromArray(normalized_arr)

    resamplefilter = sitk.ResampleImageFilter()

    normalized_volume.SetSpacing(volume.GetSpacing())
    normalized_volume.SetOrigin(volume.GetOrigin())
    normalized_volume.SetDirection(volume.GetDirection())
            
    resamplefilter.SetReferenceImage(volume)
    resampled_image = resamplefilter.Execute(normalized_volume)
    
    return resampled_image