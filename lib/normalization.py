import SimpleITK as sitk
import numpy as np
import os
import glob

def normalization(image_path):
    image = sitk.ReadImage(image_path)
    image_arr = sitk.GetArrayFromImage(image)

    mean = np.mean(image_arr)
    std = np.std(image_arr)

    normalized_image_arr = (image_arr-mean)/std
    normalized_image = sitk.GetImageFromArray(normalized_image_arr)

    normalized_image.SetSpacing(image.GetSpacing())
    normalized_image.SetOrigin(image.GetOrigin())
    normalized_image.SetDirection(image.GetDirection())

    return normalized_image