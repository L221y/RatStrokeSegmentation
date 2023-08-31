import SimpleITK as sitk
import os
import numpy as np

import SimpleITK as sitk
import torch
from skimage.morphology import closing,opening,square
import numpy as np


import monai
from monai.data import decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    ScaleIntensityRanged,
    Orientationd,
)


def ADC_segmentation_mediane(brain):
    lower_threshold = 0.43
    upper_threshold = 0.76

    threshold_filter = sitk.ThresholdImageFilter()
    threshold_filter.SetLower(lower_threshold)
    threshold_filter.SetUpper(upper_threshold)
    thresholded_volume = threshold_filter.Execute(brain)

    rescaler = sitk.RescaleIntensityImageFilter()
    rescaler.SetOutputMinimum(0.0)
    rescaler.SetOutputMaximum(1.0)
    rescaled_volume = rescaler.Execute(thresholded_volume)


    thresh_filter = sitk.BinaryThresholdImageFilter()
    thresh_filter.SetLowerThreshold(0.01)
    thresh_filter.SetUpperThreshold(1)
    thresh_filter.SetInsideValue(1)
    thresh_filter.SetOutsideValue(0)
    binary_volume = thresh_filter.Execute(rescaled_volume)

    median_filter = sitk.BinaryMedianImageFilter()
    median_kernel = [3,3,1]
    median_filter.SetRadius(median_kernel)
    median_volume = median_filter.Execute(binary_volume)

    component_image = sitk.ConnectedComponent(median_volume)
    sorted_component_image = sitk.RelabelComponent(component_image,sortByObjectSize=True)
    largest_component_binary_image = sorted_component_image == 1


    return largest_component_binary_image

def Elastix(movingVolume,fixedVolume,T2H24):
    current = os.path.split(os.path.realpath(__file__))[0]
    print(current)
    
    if T2H24 == False:
        selx = sitk.ElastixImageFilter()
        Rigid = sitk.ReadParameterFile(current+'/doc/Par0020rigid.txt')
        Affine = sitk.ReadParameterFile(current+'/doc/Par0020affine.txt')


        selx.SetParameterMap(Rigid)
        selx.AddParameterMap(Affine)

    if T2H24 == True:
        selx = sitk.ElastixImageFilter()
        Rigid = sitk.ReadParameterFile(current+'/doc/Par0026rigid.txt')
        Bspline = sitk.ReadParameterFile(current+'/doc/TG-ParamAMMIbsplineMRI.txt')


        selx.SetParameterMap(Rigid)
        selx.AddParameterMap(Bspline)
    

    selx.SetMovingImage(movingVolume)
    selx.SetFixedImage(fixedVolume)
    selx.LogToFileOff()
    selx.LogToConsoleOff()
  

    resultVolume = selx.Execute()
    transformParameterMap = selx.GetTransformParameterMap()[0]

    return resultVolume,transformParameterMap

def find_files(directory, extension):
    files_set = set()
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_path = os.path.join(root, file)
                if file_path not in files_set:
                    files_set.add(file_path)
        for dir in dirs:
            files_set.update(find_files(os.path.join(root, dir), extension))
    return files_set

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

def MPC_segmentation_mediane(brain,hemisphere,threshold):

    lower_threshold = 0.0001
    upper_threshold = threshold
    
    threshold_filter = sitk.ThresholdImageFilter()
    threshold_filter.SetLower(lower_threshold)
    threshold_filter.SetUpper(upper_threshold)
    thresholded_volume = threshold_filter.Execute(brain)

    rescaler = sitk.RescaleIntensityImageFilter()
    rescaler.SetOutputMinimum(0.0)
    rescaler.SetOutputMaximum(1.0)
    rescaled_volume = rescaler.Execute(thresholded_volume)

    hemisphere_arr = sitk.GetArrayFromImage(hemisphere)
    rescaled_volume_arr = sitk.GetArrayFromImage(rescaled_volume)

    filtered_arr = rescaled_volume_arr - hemisphere_arr

    filtered_volume = sitk.GetImageFromArray(filtered_arr)
    filtered_volume.SetDirection(brain.GetDirection())
    filtered_volume.SetOrigin(brain.GetOrigin())
    filtered_volume.SetSpacing(brain.GetSpacing())


    thresh_filter = sitk.BinaryThresholdImageFilter()
    thresh_filter.SetLowerThreshold(0.01)
    thresh_filter.SetUpperThreshold(1)
    thresh_filter.SetInsideValue(1)
    thresh_filter.SetOutsideValue(0)
    binary_volume = thresh_filter.Execute(filtered_volume)

    median_filter = sitk.BinaryMedianImageFilter()
    median_kernel = [3,3,1]
    median_filter.SetRadius(median_kernel)
    median_volume = median_filter.Execute(binary_volume)

    morpho_closing = sitk.BinaryMorphologicalClosingImageFilter()
    radius = [9,9,1]
    morpho_closing.SetKernelRadius(radius)
    morpho_volume = morpho_closing.Execute(median_volume)

    morpho_opening = sitk.BinaryMorphologicalOpeningImageFilter()
    radius = [5,5,1]
    morpho_opening.SetKernelRadius(radius)
    morpho_volume = morpho_opening.Execute(median_volume)

    

    component_image = sitk.ConnectedComponent(morpho_volume)
    sorted_component_image = sitk.RelabelComponent(component_image,sortByObjectSize=True)
    largest_component_binary_image = sorted_component_image == 1


    return largest_component_binary_image

def normalization(image):
    image_arr = sitk.GetArrayFromImage(image)

    mean = np.mean(image_arr)
    std = np.std(image_arr)

    normalized_image_arr = (image_arr-mean)/std
    normalized_image = sitk.GetImageFromArray(normalized_image_arr)

    normalized_image.SetSpacing(image.GetSpacing())
    normalized_image.SetOrigin(image.GetOrigin())
    normalized_image.SetDirection(image.GetDirection())

    return normalized_image

def normalization_mean(volume,mask_hemisphere):
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

def normalization_th(volume,mask_hemisphere):
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

def Transformix(fixedMask,Transform):
    Transform ["FixedInternalImagePixelType"] = ["float"]
    Transform ["ResultImagePixelType"] = ["float"]
    Transform ["MovingInternalImagePixelType"] = ["float"]

    #sitk.PrintParameterMap(Transform)
    
    trans = sitk.TransformixImageFilter()
    trans.SetMovingImage(fixedMask)
    trans.SetTransformParameterMap(Transform)
    
    #test = trans.GetTransformParameterMap()
    #sitk.PrintParameterMap(test)
    
    trans.LogToFileOff()
    trans.LogToConsoleOff()
    trans.Execute()

    transformedMask = trans.GetResultImage()



    return transformedMask

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

def cut_slices(input_volume):

    current = os.path.split(os.path.realpath(__file__))[0]

    volume_arr = sitk.GetArrayFromImage(input_volume)
    for j in range(volume_arr.shape[0]):
        image_slice_arr = volume_arr[j,:,:]
        image_slice_volume = sitk.GetImageFromArray(image_slice_arr)

        index = str(j+1)
        output_image_path = str(chr(j+65))+"_"+index+".nii"
        try:
            os.mkdir(current+"/tmp")
        except FileExistsError:
            j = 0
        #print(current+"/tmp/")
        sitk.WriteImage(image_slice_volume,current+"/tmp/"+output_image_path)

def visualize(input_volume,diff = False, perf = False,t2h24 = False):
    normalized = normalization(input_volume)

    cut_slices(normalized)

    test_images_slice = sorted(find_files("./tmp",".nii"))

    test_data_dicts = [
    {"image": image_name}
    for image_name in zip(test_images_slice)
    ]

    test_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityRanged(
                keys=["image"], a_min=0, a_max=5,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            Orientationd(keys=["image"], axcodes="PLF"),
        ]
    )

    test_ds = monai.data.CacheDataset(data=test_data_dicts, transform=test_transforms, cache_rate=1.0, num_workers=4)
    # val_ds = Dataset(data=val_files, transform=val_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=4)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    if diff == True:
        state_dict = torch.load("./data/deep_learning/diff/best_metric_model_segmentation2d_dict.pth", map_location=device)
    else:
        j = 0
    
    if perf == True:
        state_dict = torch.load("./data/deep_learning/perf/best_metric_model_segmentation2d_dict.pth", map_location=device)
    else:
        j = 0

    if t2h24 == True:
        state_dict = torch.load("./data/deep_learning/T2H24/best_metric_model_segmentation2d_dict.pth", map_location=device)
    else:
        j = 0

    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        i = 0
        outputs = np.zeros((15,256,256))
        for test_data in test_loader:
            test_images = test_data["image"].to(device)
            #print(np.shape(test_images))
            # Assuming `test_loader` contains images without labels
            roi_size = (128, 128)
            sw_batch_size = 1
            test_outputs = sliding_window_inference(test_images, roi_size, sw_batch_size, model)
            test_outputs = [post_trans(i) for i in decollate_batch(test_outputs)]
            
            arr = test_outputs[0].cpu().numpy()
            
            arr = closing(arr[0] , square(5))
            arr = opening(arr , square(5))
            arr = np.expand_dims(arr, axis=0)
            
        
            outputs[i] = arr
            output_volume = sitk.GetImageFromArray(outputs)
            reference = input_volume

            output_volume.SetSpacing(reference.GetSpacing())
            output_volume.SetOrigin(reference.GetOrigin())
            output_volume.SetDirection(reference.GetDirection())

            output_volume = sitk.Cast(output_volume, sitk.sitkUInt16)
            
            median_filter = sitk.BinaryMedianImageFilter()
            median_kernel = [3,3,1]
            median_filter.SetRadius(median_kernel)
            median_volume = median_filter.Execute(output_volume)

            morpho_opening = sitk.BinaryMorphologicalOpeningImageFilter()
            radius = [5,5,1]
            morpho_opening.SetKernelRadius(radius)
            morpho_volume = morpho_opening.Execute(median_volume)
            
            component_image = sitk.ConnectedComponent(morpho_volume)
            sorted_component_image = sitk.RelabelComponent(component_image,sortByObjectSize=True)
            largest_component_binary_image = sorted_component_image == 1

            i = i+1
            
    return largest_component_binary_image