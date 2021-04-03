import SimpleITK as sitk
import numpy as np
import os

def save_image(i,pet,ct,truth,pred,writePath):
    concat1 = np.concatenate((pet,ct),axis=1)
    concat2 = np.concatenate((truth,pred),axis=1)
    concat3 = np.concatenate((concat1,concat2),axis=0)
    outputImage = sitk.GetImageFromArray(concat3.transpose(2,0,1).astype(np.float32))
    sitk.WriteImage(outputImage,os.path.join(writePath,str(i)+'.nii'))

def rescale_img(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

