import SimpleITK as sitk
import torch
from skimage.measure import label
from skimage.morphology import closing,opening,square
import numpy as np

from lib.normalization import normalization
from lib.slice import cut_slices
from lib.filesfinder import find_files

import monai
from monai.utils import first, set_determinism
from monai.data import create_test_image_2d, list_data_collate, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    CropForegroundd,
    Orientationd,
)

def visualize(input_volume_path,diff = False, perf = False,t2h24 = False):

    normalized = normalization(input_volume_path)

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
            reference = sitk.ReadImage(input_volume_path)

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