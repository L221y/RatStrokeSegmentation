import logging
import os
import warnings

import vtk,qt,ctk

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

import SimpleITK as sitk
import sitkUtils

from lib.util import normalization
from lib.util import cut_slices
from lib.util import find_files
from lib.util import normalization_mean
from lib.util import MaskVolume
from lib.util import MPC_segmentation_mediane
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


#
# ratbrainseg
#

class ratbrainseg(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "ratbrainseg"  # TODO: make this more human readable by adding spaces
        self.parent.categories = ["Segmentation"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Duoyao LIANG (CarMeN)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#ratbrainseg">module documentation</a>.
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

        # Additional initialization step after application startup is complete
        #slicer.app.connect("startupCompleted()", registerSampleData)


#
# ratbrainsegWidget
#

class ratbrainsegWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

        

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/ratbrainseg.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = ratbrainsegLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        self.ui.inputT2H0Selector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.inputDWISelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.inputPWISelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.inputT2H24Selector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.outputDWISelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.outputPWISelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.outputT2H24Selector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.outputT2H24BefRegSegSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.outputT2H24RegVolSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

        self.ui.ThresholdSliderWidget.connect("valueChanged(double)", self.updateParameterNodeFromGUI)
        
        self.ui.PWIOutputCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        self.ui.DWIOutputCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        self.ui.T2H24OutputCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        self.ui.RegisterOutputCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)

        #
        # self.ui.invertedOutputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

        # Buttons
        self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.GetNodeReference("InputT2H0Volume"):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID("InputT2H0Volume", firstVolumeNode.GetID())

    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None and self.hasObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode):
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # Update node selectors and sliders
        self.ui.inputT2H0Selector.setCurrentNode(self._parameterNode.GetNodeReference("InputT2H0Volume"))
        self.ui.inputDWISelector.setCurrentNode(self._parameterNode.GetNodeReference("InputDWIVolume"))
        self.ui.inputPWISelector.setCurrentNode(self._parameterNode.GetNodeReference("InputPWIVolume"))
        self.ui.inputT2H24Selector.setCurrentNode(self._parameterNode.GetNodeReference("InputT2H24Volume"))

        self.ui.outputDWISelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputDWISeg"))
        self.ui.outputPWISelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputPWISeg"))
        self.ui.outputT2H24Selector.setCurrentNode(self._parameterNode.GetNodeReference("OutputT2H24Seg"))
        self.ui.outputT2H24BefRegSegSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputT2H24BefSeg"))  
        self.ui.outputT2H24RegVolSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputT2H24RegVol"))

        self.ui.ThresholdSliderWidget.value = float(self._parameterNode.GetParameter("Threshold"))
        
        self.ui.DWIOutputCheckBox.checked = (self._parameterNode.GetParameter("DWI") == "true")
        self.ui.PWIOutputCheckBox.checked = (self._parameterNode.GetParameter("PWI") == "true")
        self.ui.T2H24OutputCheckBox.checked = (self._parameterNode.GetParameter("T2H24") == "true")
        self.ui.RegisterOutputCheckBox.checked = (self._parameterNode.GetParameter("Register") == "true")

        self.ui.applyButton.toolTip = "Compute output volume"
        self.ui.applyButton.enabled = True


        # Update buttons states and tooltips
        # if (self._parameterNode.GetNodeReference("DWI") == "true" and self._parameterNode.GetNodeReference("InputDWIVolume") and self._parameterNode.GetNodeReference("OutputDWISeg"))\
        #     or (self._parameterNode.GetNodeReference("PWI") == "true" and self._parameterNode.GetNodeReference("InputPWIVolume") and self._parameterNode.GetNodeReference("OutputPWISeg"))\
        #         or (self._parameterNode.GetNodeReference("T2H24") == "true" and self._parameterNode.GetNodeReference("InputT2H24Volume") and self._parameterNode.GetNodeReference("OutputT2H24Seg"))\
        #             or (self._parameterNode.GetNodeReference("DWI") == "true" and self._parameterNode.GetNodeReference("PWI") == "true" and self._parameterNode.GetNodeReference("InputDWIVolume")\
        #              and self._parameterNode.GetNodeReference("OutputDWISeg") and self._parameterNode.GetNodeReference("InputPWIVolume") and self._parameterNode.GetNodeReference("OutputPWISeg"))\
        #                 or (self._parameterNode.GetNodeReference("DWI") == "true" and self._parameterNode.GetNodeReference("T2H24") == "true" and self._parameterNode.GetNodeReference("InputDWIVolume")\
        #                      and self._parameterNode.GetNodeReference("OutputDWISeg") and self._parameterNode.GetNodeReference("InputT2H24Volume") and self._parameterNode.GetNodeReference("OutputT2H24Seg"))\
        #                         or (self._parameterNode.GetNodeReference("PWI") == "true" and self._parameterNode.GetNodeReference("T2H24") == "true" and self._parameterNode.GetNodeReference("InputPWIVolume")\
        #                              and self._parameterNode.GetNodeReference("OutputPWISeg") and self._parameterNode.GetNodeReference("InputT2H24Volume") and self._parameterNode.GetNodeReference("OutputT2H24Seg"))\
        #                                 or (self._parameterNode.GetNodeReference("DWI") == "true" and self._parameterNode.GetNodeReference("PWI") == "true" and self._parameterNode.GetNodeReference("T2H24") == "true" and\
        #                                     self._parameterNode.GetNodeReference("InputDWIVolume") and self._parameterNode.GetNodeReference("OutputDWISeg") and self._parameterNode.GetNodeReference("InputPWIVolume") and\
        #                                           self._parameterNode.GetNodeReference("OutputPWISeg") and self._parameterNode.GetNodeReference("InputT2H24Volume") and self._parameterNode.GetNodeReference("OutputT2H24Seg")):
        #     #if self._parameterNode.GetNodeReference("InputVolume") and self._parameterNode.GetNodeReference("OutputVolume"):
        #     self.ui.applyButton.toolTip = "Compute output volume"
        #     self.ui.applyButton.enabled = True
        # else:
        #     self.ui.applyButton.toolTip = "Select proper input and output volume nodes"
        #     self.ui.applyButton.enabled = False

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        self._parameterNode.SetNodeReferenceID("InputT2H0Volume", self.ui.inputT2H0Selector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("InputDWIVolume", self.ui.inputDWISelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("InputPWIVolume", self.ui.inputPWISelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("InputT2H24Volume", self.ui.inputT2H24Selector.currentNodeID)

        self._parameterNode.SetNodeReferenceID("OutputT2H24Seg", self.ui.outputT2H24Selector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("OutputDWISeg", self.ui.outputDWISelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("OutputPWISeg", self.ui.outputPWISelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("OutputT2H24BefSeg", self.ui.outputT2H24BefRegSegSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("OutputT2H24RegVol", self.ui.outputT2H24RegVolSelector.currentNodeID)
        
        self._parameterNode.SetParameter("Threshold", str(self.ui.ThresholdSliderWidget.value))
        self._parameterNode.SetParameter("DWI", "true" if self.ui.DWIOutputCheckBox.checked else "false")
        self._parameterNode.SetParameter("PWI", "true" if self.ui.PWIOutputCheckBox.checked else "false")
        self._parameterNode.SetParameter("T2H24", "true" if self.ui.T2H24OutputCheckBox.checked else "false")
        self._parameterNode.SetParameter("Register", "true" if self.ui.RegisterOutputCheckBox.checked else "false")

        self._parameterNode.EndModify(wasModified)

    def onApplyButton(self):
        """
        Run processing when user clicks "Apply" button.
        """
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):
            if self.ui.DWIOutputCheckBox.checked == True:
                self.logic.process_diff(self.ui.inputDWISelector.currentNode(),self.ui.outputDWISelector.currentNode(),showResult=False)
            if self.ui.T2H24OutputCheckBox.checked == True:
                self.logic.process_t2h24(self.ui.inputT2H24Selector.currentNode(),self.ui.outputT2H24BefRegSegSelector.currentNode(),showResult=False)
            if self.ui.PWIOutputCheckBox.checked == True:
                self.logic.process_perf(self.ui.inputT2H0Selector.currentNode(),self.ui.inputPWISelector.currentNode(),self.ui.outputPWISelector.currentNode(),self.ui.ThresholdSliderWidget.value,showResult=False)
            if self.ui.RegisterOutputCheckBox.checked == True:
                self.logic.process_register(self.ui.inputT2H0Selector.currentNode(),self.ui.inputT2H24Selector.currentNode(),self.ui.outputT2H24BefRegSegSelector.currentNode(),self.ui.outputT2H24Selector.currentNode(),self.ui.outputT2H24RegVolSelector.currentNode(),showResult = False)
            # Compute output
            #self.logic.process(self.ui.inputSelector.currentNode(), self.ui.outputSelector.currentNode(),
            #                   self.ui.ThresholdSliderWidget.value, self.ui.DWIOutputCheckBox.checked)

            # Compute inverted output (if needed)
            #if self.ui.invertedOutputSelector.currentNode():
                # If additional output volume is selected then result with inverted threshold is written there
            #    self.logic.process(self.ui.inputSelector.currentNode(), self.ui.invertedOutputSelector.currentNode(),
            #                       self.ui.ThresholdSliderWidget.value, not self.ui.invertOutputCheckBox.checked, showResult=False)


    def refreshVersion(self):
        print("Refreshing version...")
        monai = self.logic.importMONAI()
        nibabel = monai = self.logic.importNibabel()
        version = monai.__version__
        #print(version)

#
# ratbrainsegLogic
#

class ratbrainsegLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)
        self.elastixBinDir = None
        self.scriptPath = os.path.split(os.path.realpath(__file__))[0]
        self.registrationParameterFilesDirs = os.path.abspath(os.path.join(self.scriptPath,'data','threshold'))
        self.customElastixBinDirSettingsKey = 'Elastix/CustomElastixPath'
        import platform
        executableExt = '.exe' if platform.system() == 'Windows' else ''
        self.elastixFilename = 'elastix'+executableExt
        self.transformixFilename = 'transformix'+executableExt
        import Elastix
        self.elastixLogic = Elastix.ElastixLogic()
        self.elastixLogic.deleteTemporaryFiles = False

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        if not parameterNode.GetParameter("Threshold"):
            parameterNode.SetParameter("Threshold", "0.75")
        if not parameterNode.GetParameter("DWI"):
            parameterNode.SetParameter("DWI", "true")
        if not parameterNode.GetParameter("PWI"):
            parameterNode.SetParameter("PWI", "true")
        if not parameterNode.GetParameter("T2H24"):
            parameterNode.SetParameter("T2H24", "true")
        if not parameterNode.GetParameter("Register"):
            parameterNode.SetParameter("Register", "true")

    def importMONAI(self):
        if not self.torchLogic.torchInstalled():
            logging.info("PyTorch module not found")
            torch = self.torchLogic.installTorch(askConfirmation=True)
            if torch is None:
                slicer.util.errorDisplay(
                    "PyTorch needs to be installed to use the MONAI extension."
                    " Please reload this module to install PyTorch."
                )
                return None
        try:
            import monai
        except ModuleNotFoundError:
            with self.showWaitCursor(), self.peakPythonConsole():
                monai = self.installMONAI()
        logging.info(f"MONAI {monai.__version__} imported correctly")
        return monai
    
    def importNibabel(self):    
        try:
            import nibabel
        except ModuleNotFoundError:
            with self.showWaitCursor(), self.peakPythonConsole():
                monai = self.installNIBABEL()
        logging.info(f"Nibabel {nibabel.__version__} imported correctly")
        return nibabel
    
    @staticmethod
    def installMONAI(confirm=True):
        if confirm and not slicer.app.commandOptions().testingEnabled:
            install = slicer.util.confirmOkCancelDisplay(
                "MONAI will be downloaded and installed now. The process might take some minutes."
            )
            if not install:
                logging.info("Installation of MONAI aborted by user")
                return None
        slicer.util.pip_install("monai[itk,nibabel,tqdm]")
        import monai

        logging.info(f"MONAI {monai.__version__} installed correctly")
        return monai
    
    @staticmethod
    def installNIBABEL(confirm=True):
        if confirm and not slicer.app.commandOptions().testingEnabled:
            install = slicer.util.confirmOkCancelDisplay(
                "Nibabel will be downloaded and installed now. The process might take some minutes."
            )
            if not install:
                logging.info("Installation of Nibabel aborted by user")
                return None
        slicer.util.pip_install("nibabel")
        import nibabel

        logging.info(f"nibabel {nibabel.__version__} installed correctly")
        return nibabel

    def process_diff(self, inputDWIVolume, outputDWISeg, showResult=False):
    
        warnings.filterwarnings("ignore")
        #print(self.getElastixBinDir())
        import time
        startTime = time.time()
        logging.info('DWI Processing started')

        inputDWIImage = sitkUtils.PullVolumeFromSlicer(inputDWIVolume)
        
        normalized = normalization(inputDWIImage)

        cut_slices(normalized)

        current = os.path.split(os.path.realpath(__file__))[0]

        test_images_slice = sorted(find_files(current+"/lib/tmp",".nii"))

        test_data_dicts = [{"image": image_name} for image_name in test_images_slice]
        

        test_transforms = Compose(
            [
                LoadImaged(keys=["image"],reader=["NibabelReader"]),
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
        test_loader = DataLoader(test_ds, batch_size=1, num_workers=0)
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
        state_dict = torch.load(current+"/data/deep_learning/diff/best_metric_model_segmentation2d_dict.pth", map_location=device)

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
                reference = inputDWIImage

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
            

        sitkUtils.PushVolumeToSlicer(largest_component_binary_image,outputDWISeg)
        outputDWISeg.SetName("DWISegmentation")

        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')

    def process_t2h24(self, inputT2H24Volume, outputT2H24Seg, showResult=False):
        warnings.filterwarnings("ignore")
        
        import time
        startTime = time.time()
        logging.info('T2H24 Processing started')

        inputT2H24Image = sitkUtils.PullVolumeFromSlicer(inputT2H24Volume)
        
        normalized = normalization(inputT2H24Image)

        cut_slices(normalized)

        current = os.path.split(os.path.realpath(__file__))[0]

        test_images_slice = sorted(find_files(current+"/lib/tmp",".nii"))

        test_data_dicts = [{"image": image_name} for image_name in test_images_slice]
        

        test_transforms = Compose(
            [
                LoadImaged(keys=["image"],reader=["NibabelReader"]),
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
        test_loader = DataLoader(test_ds, batch_size=1, num_workers=0)
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
        state_dict = torch.load(current+"/data/deep_learning/T2H24/best_metric_model_segmentation2d_dict.pth", map_location=device)

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
                reference = inputT2H24Image

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
            

        sitkUtils.PushVolumeToSlicer(largest_component_binary_image,outputT2H24Seg)
        outputT2H24Seg.SetName("T2H24SegmentationBeforeRegistration")

        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')

    def process_perf(self, inputT2H0VolumeNode, inputPWIVolumeNode, outputPWISegNode, threshold, showResult=False):
        import time
        startTime = time.time()
        #print(type(self.elastixLogic))
        #print(dir(self.elastixLogic))
        logging.info('PWI Processing started')
        current = os.path.split(os.path.realpath(__file__))[0]
        
        # Paths to your reference images and parameter files
        reference_T2_volume_path = os.path.join(current, "data/threshold/2dseq_W4-04_T2H0.nii")
        #reference_brain_mask_path = os.path.join(current, "data/threshold/Segmentation_W4-04.nii.gz")
        #reference_hemisphere_mask_path = os.path.join(current, "data/threshold/HemisphereMask-W4-04.nii.gz")
        parameter_file1 = os.path.join(current, "lib/doc/Par0020rigid.txt")
        parameter_file2 = os.path.join(current, "lib/doc/Par0020affine.txt")

        # Load reference volumes into Slicer
        reference_T2_node = slicer.util.loadVolume(reference_T2_volume_path)
        #reference_brain_mask_node = slicer.util.loadVolume(reference_brain_mask_path)
        #reference_hemisphere_mask_node = slicer.util.loadVolume(reference_hemisphere_mask_path)

        
        # Create and setup Elastix parameter node
        registeredT2 = slicer.vtkMRMLScalarVolumeNode()
        transformT2 = slicer.vtkMRMLTransformNode()
        slicer.mrmlScene.AddNode(transformT2)
        transformT2.SetName("TransformT2")
        parameterFilesname = [parameter_file1,parameter_file2]
        self.elastixLogic.registerVolumes(fixedVolumeNode=inputT2H0VolumeNode,movingVolumeNode=reference_T2_node,
                                          parameterFilenames=parameterFilesname,outputVolumeNode=registeredT2,outputTransformNode=transformT2,
                                            fixedVolumeMaskNode=None, movingVolumeMaskNode=None,forceDisplacementFieldOutputTransform=False, 
                                                initialTransformNode=None)
        
        tempDirBase = self.elastixLogic.getTempDirectoryBase()
        subfolders = [f.name for f in os.scandir(tempDirBase) if f.is_dir()]
        sorted_subfolders = sorted(subfolders)
        if sorted_subfolders:
            last_subfolder = sorted_subfolders[-1]
            tempDir = os.path.join(tempDirBase, last_subfolder)

        #brain_mask_node = slicer.vtkMRMLScalarVolumeNode()

        #hemisphere_mask_node = slicer.vtkMRMLScalarVolumeNode()
        
        # Transformix using cmd line
        brain_mask_node = slicer.vtkMRMLScalarVolumeNode()
        transformFileNameBase = os.path.join(tempDir,"result-transform/TransformParameters.1")
        resultResampleDir = os.path.join(current,"data/tmp")
        brainParamsTransformix = [
        '-tp', f'{transformFileNameBase}.txt',
        '-out', os.path.join(resultResampleDir,"brain"), 
        '-in', os.path.join(current, "data/threshold/Segmentation_W4-04.mha"),
        '-def','all'
        ]
        self.elastixLogic.startTransformix(brainParamsTransformix)
        time.sleep(5)
        self.elastixLogic._loadTransformedOutputVolume(brain_mask_node, os.path.join(resultResampleDir,"brain"))
        del_list_brain = os.listdir(os.path.join(resultResampleDir,"brain"))
        for f in del_list_brain:
            file_path = os.path.join(os.path.join(resultResampleDir,"brain"), f)
            if os.path.isfile(file_path):
                os.remove(file_path)
        slicer.mrmlScene.AddNode(brain_mask_node)
        brain_mask_node.SetName("Regi-mask")

        hemisphere_mask_node = slicer.vtkMRMLScalarVolumeNode()
        hemisphereParamsTransformix = [
        '-tp', f'{transformFileNameBase}.txt',
        '-out', os.path.join(resultResampleDir,"hemisphere"), 
        '-in', os.path.join(current, "data/threshold/HemisphereMask-W4-04.mha"),
        '-def','all'
        ]
        self.elastixLogic.startTransformix(hemisphereParamsTransformix)
        time.sleep(5)
        self.elastixLogic._loadTransformedOutputVolume(hemisphere_mask_node, os.path.join(resultResampleDir,"hemisphere"))
        del_list_hemi = os.listdir(os.path.join(resultResampleDir,"brain"))
        for f in del_list_hemi:
            file_path = os.path.join(os.path.join(resultResampleDir,"brain"), f)
            if os.path.isfile(file_path):
                os.remove(file_path)
        slicer.mrmlScene.AddNode(hemisphere_mask_node)
        hemisphere_mask_node.SetName("Regi-hemi")


        inputPWIImage = sitkUtils.PullVolumeFromSlicer(inputPWIVolumeNode)
        brain_mask = sitkUtils.PullVolumeFromSlicer(brain_mask_node)
        hemisphere_mask = sitkUtils.PullVolumeFromSlicer(hemisphere_mask_node)
        # # Your existing processing steps
        masked_volume_perf = MaskVolume(inputPWIImage, brain_mask)
        normalized_brain_perf = normalization_mean(masked_volume_perf, hemisphere_mask)
        
        normalized_brain_perf_node = slicer.vtkMRMLScalarVolumeNode()
        slicer.mrmlScene.AddNode(normalized_brain_perf_node)
        normalized_brain_perf_node.SetName("normalizedPWI")
        
        result_volume_perf = MPC_segmentation_mediane(normalized_brain_perf, hemisphere_mask, float(threshold))

        # # Push the result back to Slicer
        sitkUtils.PushVolumeToSlicer(result_volume_perf, targetNode=outputPWISegNode)
        outputPWISegNode.SetName("PWISegmentation")

        slicer.mrmlScene.RemoveNode(reference_T2_node)
        slicer.mrmlScene.RemoveNode(brain_mask_node)
        slicer.mrmlScene.RemoveNode(hemisphere_mask_node)
        #slicer.mrmlScene.RemoveNode(reference_hemisphere_mask_node)

        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime - startTime:.2f} seconds')

    def process_register(self,inputT2H0VolumeNode, inputT2H24VolumeNode,outputBefRegSegNode,outputT2H24SegNode,outputT2H24RegVolumeNode,showResult=False):
        import time
        startTime = time.time()

        logging.info('T2H24 Processing started')
        current = os.path.split(os.path.realpath(__file__))[0]
        
        # Paths to parameter files
        parameter_file1 = os.path.join(current, "lib/doc/Par0026rigid.txt")
        parameter_file2 = os.path.join(current, "lib/doc/TG-ParamAMMIbsplineMRI.txt")

        # Create and setup Elastix parameter node
        transformT2H24 = slicer.vtkMRMLTransformNode()
        slicer.mrmlScene.AddNode(transformT2H24)
        transformT2H24.SetName("transformT2H24")
        parameterFilesname = [parameter_file1,parameter_file2]
        self.elastixLogic.registerVolumes(fixedVolumeNode=inputT2H0VolumeNode,movingVolumeNode=inputT2H24VolumeNode,
                                          parameterFilenames=parameterFilesname,outputVolumeNode=outputT2H24RegVolumeNode,outputTransformNode=transformT2H24,
                                            fixedVolumeMaskNode=None, movingVolumeMaskNode=None,forceDisplacementFieldOutputTransform=False, 
                                                initialTransformNode=None)

        outputT2H24RegVolumeNode.SetName("T2H24RegisteredVolume")
        #outputBefRegSegNode_cloned = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
        id = outputT2H24SegNode.GetID()
        outputT2H24SegNode.Copy(outputBefRegSegNode)
        #outputT2H24SegNode.SetID(id)


        outputT2H24SegNode.SetAndObserveTransformNodeID(transformT2H24.GetID())
        outputT2H24SegNode.HardenTransform()
        outputT2H24SegNode.SetName("T2H24RegisteredSegmentation")


        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime - startTime:.2f} seconds')
# #
# ratbrainsegTest
#

# class ratbrainsegTest(ScriptedLoadableModuleTest):
#     """
#     This is the test case for your scripted module.
#     Uses ScriptedLoadableModuleTest base class, available at:
#     https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
#     """

#     def setUp(self):
#         """ Do whatever is needed to reset the state - typically a scene clear will be enough.
#         """
#         slicer.mrmlScene.Clear()

#     def runTest(self):
#         """Run as few or as many tests as needed here.
#         """
#         self.setUp()
#         self.test_ratbrainseg1()

#     def test_ratbrainseg1(self):
#         """ Ideally you should have several levels of tests.  At the lowest level
#         tests should exercise the functionality of the logic with different inputs
#         (both valid and invalid).  At higher levels your tests should emulate the
#         way the user would interact with your code and confirm that it still works
#         the way you intended.
#         One of the most important features of the tests is that it should alert other
#         developers when their changes will have an impact on the behavior of your
#         module.  For example, if a developer removes a feature that you depend on,
#         your test should break so they know that the feature is needed.
#         """

#         self.delayDisplay("Starting the test")

#         # Get/create input data

#         import SampleData
#         registerSampleData()
#         inputVolume = SampleData.downloadSample('ratbrainseg1')
#         self.delayDisplay('Loaded test data set')

#         inputScalarRange = inputVolume.GetImageData().GetScalarRange()
#         self.assertEqual(inputScalarRange[0], 0)
#         self.assertEqual(inputScalarRange[1], 695)

#         outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
#         threshold = 100

#         # Test the module logic

#         logic = ratbrainsegLogic()

#         # Test algorithm with non-inverted threshold
#         logic.process(inputVolume, outputVolume, threshold, True)
#         outputScalarRange = outputVolume.GetImageData().GetScalarRange()
#         self.assertEqual(outputScalarRange[0], inputScalarRange[0])
#         self.assertEqual(outputScalarRange[1], threshold)

#         # Test algorithm with inverted threshold
#         logic.process(inputVolume, outputVolume, threshold, False)
#         outputScalarRange = outputVolume.GetImageData().GetScalarRange()
#         self.assertEqual(outputScalarRange[0], inputScalarRange[0])
#         self.assertEqual(outputScalarRange[1], inputScalarRange[1])

#         self.delayDisplay('Test passed')
