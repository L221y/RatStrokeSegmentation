import SimpleITK as sitk
import numpy as np

def MaskVolume(image,seg):
    spacing = seg.GetSpacing()
    origin = seg.GetOrigin()
    direction = seg.GetDirection()

    seg_arr = sitk.GetArrayFromImage(seg)
    image_arr = sitk.GetArrayFromImage(image)

    mask_arr = np.zeros_like(seg_arr)
    mask_arr[seg_arr > 0] = 1

    brain_arr = image_arr * mask_arr

    brain = sitk.GetImageFromArray(brain_arr)

    brain.SetSpacing(spacing)
    brain.SetDirection(direction)
    brain.SetOrigin(origin)


    return brain