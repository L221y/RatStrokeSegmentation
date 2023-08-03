import SimpleITK as sitk
import os

def cut_slices(input_volume):

    volume_arr = sitk.GetArrayFromImage(input_volume)
    for j in range(volume_arr.shape[0]):
        image_slice_arr = volume_arr[j,:,:]
        image_slice_volume = sitk.GetImageFromArray(image_slice_arr)

        index = str(j+1)
        output_image_path = str(chr(j+65))+"_"+index+".nii"
        try:
            os.mkdir("./tmp")
        except FileExistsError:
            j = 0

        sitk.WriteImage(image_slice_volume,"./tmp/"+output_image_path)
        
