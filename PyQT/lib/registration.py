import SimpleITK as sitk
import numpy as np
from lib.Elastix import Elastix
from lib.Transformix import Transformix

def H24_registration(h0_volume,h24_volume,mask):
    h0_arr = sitk.GetArrayFromImage(h0_volume)
    
    mean = np.mean(h0_arr)
    std = np.std(h0_arr)

    normalized_image_arr = (h0_arr-mean)/std
    h0_volume_nor = sitk.GetImageFromArray(normalized_image_arr)
    
    h0_volume_nor.SetOrigin(h0_volume.GetOrigin())
    h0_volume_nor.SetDirection(h0_volume.GetDirection())
    h0_volume_nor.SetSpacing(h0_volume.GetSpacing())

    threshold_filter = sitk.ThresholdImageFilter()
    threshold_filter.SetLower(0)
    threshold_filter.SetUpper(20)
    h0_volume_nor_thresh = threshold_filter.Execute(h0_volume_nor)

    h24_arr = sitk.GetArrayFromImage(h24_volume)
    
    mean = np.mean(h24_arr)
    std = np.std(h24_arr)

    normalized_image_arr = (h24_arr-mean)/std
    h24_volume_nor = sitk.GetImageFromArray(normalized_image_arr)
    
    h24_volume_nor.SetOrigin(h24_volume.GetOrigin())
    h24_volume_nor.SetDirection(h24_volume.GetDirection())
    h24_volume_nor.SetSpacing(h24_volume.GetSpacing())

    threshold_filter = sitk.ThresholdImageFilter()
    threshold_filter.SetLower(0)
    threshold_filter.SetUpper(20)
    h24_volume_nor_thresh = threshold_filter.Execute(h24_volume_nor)

    registered_T2H24,trans = Elastix(h24_volume_nor_thresh,h0_volume_nor_thresh,T2H24=True)

    transformed_mask = Transformix(mask,trans)

    return transformed_mask,registered_T2H24,trans