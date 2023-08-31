import SimpleITK as sitk

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
