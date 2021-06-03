'''
Calculate the tumor-wise balance factor for each tumor.
'''
import os
import SimpleITK as sitk
import skimage
import copy
import numpy as np

readPath = ""
writePath = ""
#We set a=1 here, which can be further adjusted in "make_input.py" for better flexiblility, but b
#can not be changed after the file generated, because it's calculated tumor-wisely.
a=1; b = 3/4
patientList = os.listdir(readPath)
patientList.sort(key=lambda x:int(x))
for patient in patientList:
    tumorOri = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(readPath,patient,'Tumor.nii')))
    tumors,tumorNum = skimage.measure.label(tumorOri, connectivity=3, return_num=True)
    tumorNew = np.zeros_like(tumorOri,dtype=np.float32)
    print(patient)
    for i in range(tumorNum):
        tumorsCopy = copy.deepcopy(tumors)
        tumorsCopy[tumors!=i+1] = 0
        tumorPixel = np.count_nonzero(tumorsCopy)
        #Line 27 is corresponding to f(Ti)=a/(T_i^b).
        newValue = a/tumorPixel**(b)
        tumorNew[tumors==i+1] = newValue
        print(tumorPixel,newValue)
    print('---')
    sitk.WriteImage(sitk.GetImageFromArray(tumorNew),os.path.join(writePath,patient+'.nii'))
    
