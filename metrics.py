import tensorflow as tf
import os
import SimpleITK as sitk
import pandas
import numpy
from tensorflow import contrib
autograph = contrib.autograph

def Dice(truth,prediction):
    intersection_pixel_num = tf.count_nonzero(truth*prediction)
    prediction_pixel_num = tf.count_nonzero(prediction)
    truth_pixel_num = tf.count_nonzero(truth)
    dice = (intersection_pixel_num*2)/(prediction_pixel_num+truth_pixel_num)
    cond = tf.not_equal(tf.constant(0,dtype=tf.int64),prediction_pixel_num+truth_pixel_num)
    res = tf.where(cond,dice,1)
    return res

@autograph.convert()
def Sensitivity(truth,prediction):
    intersection_pixel_num = tf.count_nonzero(truth*prediction)
    prediction_pixel_num = tf.count_nonzero(prediction)
    truth_pixel_num = tf.count_nonzero(truth)
    if tf.equal(truth_pixel_num,tf.constant(0,tf.int64)):
        if tf.equal(prediction_pixel_num,tf.constant(0,tf.int64)):
            res = tf.constant(1,tf.float64)
        else:
            res = tf.constant(0,tf.float64)
    else:
        res = intersection_pixel_num/truth_pixel_num
    return res

@autograph.convert()
def Precision(truth,prediction):
    intersection_pixel_num = tf.count_nonzero(truth*prediction)
    prediction_pixel_num = tf.count_nonzero(prediction)
    truth_pixel_num = tf.count_nonzero(truth)
    if tf.equal(prediction_pixel_num,tf.constant(0,tf.int64)):
        if truth_pixel_num == tf.constant(0,dtype=tf.int64):
            res = tf.constant(1,tf.float64)
        else:
            res = tf.constant(0,tf.float64)
    else:
        res = intersection_pixel_num/prediction_pixel_num
    return res
    
def scalar_summary(model):
    dice = Dice(model.label[0][:,:,:,:,0],model.get_binary_prediction()[:,:,:,:,0])
    sensitivity = Sensitivity(model.label[0][:,:,:,:,0],model.get_binary_prediction()[:,:,:,:,0])
    precision = Precision(model.label[0][:,:,:,:,0],model.get_binary_prediction()[:,:,:,:,0])
    tf.compat.v1.summary.scalar('Dice',dice)
    tf.compat.v1.summary.scalar('Sensitivity',sensitivity)
    tf.compat.v1.summary.scalar('Precision',precision)


def get_all_metrics(readPath,writePath):
    '''
    Calculate Dice,Sensitivity and Precision for each patient, and store results in a .xlsx file.
    '''
    title = ["ID","Dice","Sensitivity","Precision"]
    metricsList = []
    patientList = os.listdir(readPath)
    for i in range(len(patientList)):
        patientList[i] = int(patientList[i].replace(".nii",""))
    patientList.sort()
    sesson = tf.Session()
    truthHolder = tf.placeholder(tf.float32,shape=[256,256,None])
    predHolder = tf.placeholder(tf.float32,shape=[256,256,None])
    dice = Dice(truthHolder,predHolder)
    sensitivity = Sensitivity(truthHolder,predHolder)
    precision = Precision(truthHolder,predHolder)
    for patient in patientList:
        totalNii = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(readPath,str(patient)+".nii"))).transpose(1,2,0)
        truth = totalNii[256:,0:256,:]
        pred = totalNii[256:,256:,:]
        dice_ = sesson.run(dice,feed_dict={truthHolder:truth,predHolder:pred})
        sensitivity_ = sesson.run(sensitivity,feed_dict={truthHolder:truth,predHolder:pred})
        precision_ = sesson.run(precision,feed_dict={truthHolder:truth,predHolder:pred})
        metricsList.append([patient,dice_,sensitivity_,precision_])
        print([patient,dice_,sensitivity_,precision_])
    arr = numpy.array(metricsList)
    #Calculate mean value of three metrics and write to .xlsx file.
    metricsList.append(['average',numpy.average(arr[:,1]),numpy.average(arr[:,2]),numpy.average(arr[:,3])])
    xlsx = pandas.DataFrame(metricsList,columns=title)
    writer = pandas.ExcelWriter(writePath)
    xlsx.to_excel(writer,'sheet1',index=False)
    writer.save()
    return

if __name__ == "__main__":
    get_all_metrics("/media/wangkun/Data2/实验结果/src21/全身/5/test_output_refine", \
                    "/media/wangkun/Data2/实验结果/src21/全身/5/metrics_noCRF.xlsx")
    