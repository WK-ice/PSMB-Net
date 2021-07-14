from denseinference import CRFProcessor
import SimpleITK as  sitk
import os
import save_image
import numpy as np
import metrics
from tensorflow import contrib
autograph = contrib.autograph
import traceback
import sys
import multiprocessing as mp
os.environ['CUDA_VISIBLE_DEVICES']='0'

def crf(readPath,refine_output_path,patientList,use_crf=True):
    try:
        while len(patientList)>0:
            patient = patientList.pop()
            i = len(patientList)  
            totalNii = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(readPath,patient))).transpose(1,2,0)
            pet = totalNii[0:256,0:256,:]
            softmax = totalNii[0:256,256:,:]
            tumor = totalNii[256:,0:256,:]
            if use_crf:
                petRescale = save_image.rescale_img(pet)
                softmaxRescale = save_image.rescale_img(softmax)
                softmaxRescale = np.stack([softmaxRescale,1-softmaxRescale],axis=-1)
                pro = CRFProcessor.CRF3DProcessor(pos_x_std=1,pos_y_std=1,pos_z_std=1,pos_w=1, \
                    bilateral_x_std=3,bilateral_y_std=3,bilateral_z_std=3,bilateral_intensity_std=10,\
                    bilateral_w=10,ignore_memory=True)
                predVolume = 1-pro.set_data_and_run(petRescale, softmaxRescale)
            else:
                predVolume = np.zeros_like(softmax)
                predVolume[softmax>=0.9] = 1
            save_image.save_image(patient,pet,softmax,tumor, predVolume,refine_output_path)
            sys.stdout.write('%d %s\n'%(96-i,patient[:-4]))
    except:
        traceback.print_exc()
        print('Thread finished.')

if __name__ == '__main__':
    readPath = "  "
    refine_output_path = "  "
    os.mkdir(refine_output_path)
    patientList = os.listdir(readPath)
    sharedList = mp.Manager().list()
    sharedList.extend(patientList)
    processList = []
    #Set the count of parallel process. This will consume CPU memory linearly with increased filled value.
    for i in range(8):
        processList.append(mp.Process(target=crf,args=(readPath,refine_output_path,sharedList,True)))
    for p in processList:
        p.start()
    for p in processList:
        p.join()
    #Write a .xlsx file in disk, including Dice, Sensitivity and Precision of each sample and their averaged value.
    metrics.get_all_metrics(refine_output_path,os.path.dirname(refine_output_path)+'/metrics.xlsx')
