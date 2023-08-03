import SimpleITK as sitk

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
