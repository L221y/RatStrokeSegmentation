import SimpleITK as sitk

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
