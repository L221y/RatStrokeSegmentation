import SimpleITK as sitk

def Elastix(movingVolume,fixedVolume,T2H24):
    if T2H24 == False:
        selx = sitk.ElastixImageFilter()
        Rigid = sitk.ReadParameterFile('./doc/Par0020rigid.txt')
        Affine = sitk.ReadParameterFile('./doc/Par0020affine.txt')


        selx.SetParameterMap(Rigid)
        selx.AddParameterMap(Affine)

    if T2H24 == True:
        selx = sitk.ElastixImageFilter()
        Rigid = sitk.ReadParameterFile('./doc/Par0026rigid.txt')
        Bspline = sitk.ReadParameterFile('./doc/TG-ParamAMMIbsplineMRI.txt')


        selx.SetParameterMap(Rigid)
        selx.AddParameterMap(Bspline)
    

    selx.SetMovingImage(movingVolume)
    selx.SetFixedImage(fixedVolume)
    selx.LogToFileOff()
    selx.LogToConsoleOff()
  

    resultVolume = selx.Execute()
    transformParameterMap = selx.GetTransformParameterMap()[0]

    return resultVolume,transformParameterMap