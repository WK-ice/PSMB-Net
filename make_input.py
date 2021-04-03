import tensorflow as tf
#import glob
import os
import pickle

def parse_total_patient(patientPath,startIndex,endIndex,isTraining):
    '''
    Parse input from *.tfrecord file. Each *.tfrecord file contains all the modalities and labels of a patient.
    
    The explanation to keys in *.tfrecord file:
        
    ct: CT volumetric image, slice concatenated in last dimension. Shape: (row,column,sliceNumber)
    pet: PET volumetric image, slice concatenated in last dimension. Shape: (row,column,sliceNumber)
    tumor: Tumor volumetric image, slice concatenated in last dimension. Shape: (row,column,sliceNumber)
    heightManual: A vector contains relative position of each slice. Shape: (sliceNumber)
    shape: Shape of CT, PET, and Tumor volumetric image. Shape: (3)
    name: Name of a patient.
    '''
    
    def parse(record):
        keys_to_features = {
            'ct': tf.io.FixedLenFeature([], tf.string),
            'pet': tf.io.FixedLenFeature([], tf.string),
            'tumor': tf.io.FixedLenFeature([], tf.string),
            'heightManual': tf.io.FixedLenFeature([], tf.string),
            'shape': tf.io.FixedLenFeature([3],tf.int64),
            'name': tf.io.FixedLenFeature([], tf.string),
        }
        features = tf.io.parse_single_example(record, keys_to_features)
        
        pt = tf.decode_raw(features['pet'], tf.float16)
        ct = tf.decode_raw(features['ct'], tf.float16)
        tu = tf.decode_raw(features['tumor'], tf.float16)
        height = tf.decode_raw(features['heightManual'], tf.float16)
        shape = features['shape']
        name = features['name']
        
        pt = tf.reshape(pt,shape)
        ct = tf.reshape(ct,shape)
        tu = tf.reshape(tu,shape)
        
        pt = tf.cast(pt,tf.float32)
        ct = tf.cast(ct,tf.float32)
        tu = tf.cast(tu,tf.float32)
        height = tf.cast(height,tf.float32)
        
        ct = tf.image.per_image_standardization(ct)
        
        #Random flipping in a probability of 0.5.
        if isTraining:
            imgConcat = tf.concat([pt,ct,tu],2)
            flipped = tf.image.random_flip_left_right(imgConcat)
            pt = tf.slice(flipped,tf.cast([0,0,0],tf.int64),shape)
            ct = tf.slice(flipped,tf.cast([0,0,shape[2]],tf.int64),shape)
            tu = tf.slice(flipped,tf.cast([0,0,shape[2]*2],tf.int64),shape)
        
        pt = tf.expand_dims(pt,axis=-1)
        #The parameter 'a' of tumor-wise balance factor.
        tu = tf.expand_dims(tu,axis=-1)*tf.constant(10000,tf.float32)
        
        tuCond = tf.equal(tf.constant(0,dtype=tf.float32),tu)
        zeros = tf.zeros_like(tu,dtype=tf.float32)
        ones = zeros+1
        bg = tf.where(tuCond,ones,zeros)
        
        return (pt,ct),(tu,bg,height),name
    
    #Read a random shuffled list of patient names, which is stored in disk.
    with open('/data/patient_order.txt','rb') as f:
        nameList = pickle.load(f)
#    print(nameList)
    fileList = list(map(lambda x:os.path.join(patientPath,x+'.tfrecord'),nameList))
    if isTraining:
        fileList = fileList[:startIndex-1]+fileList[endIndex:]
    else:
        fileList = fileList[startIndex-1:endIndex]
    dataSet = tf.data.TFRecordDataset(fileList).map(parse)
    if isTraining:
        dataSet = dataSet.repeat(-1).shuffle(2)
    dataSet = dataSet.batch(1).prefetch(2)
    return dataSet
