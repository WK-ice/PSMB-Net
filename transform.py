import numpy as np
import tensorflow as tf
import random

#Split a patient volume to patches according to FLAGS.slice_number_per_patch.
def split_volume_to_subvolumes(FLAGS,pet,ct,tumor,bg,height,skip=True):
    shape = pet.shape
    sliceNumPerGroup = FLAGS.slice_number_per_patch
    subvolumeNum = np.ceil(shape[-2]/sliceNumPerGroup)
    #Different modalities of a patch is stored using dictionary object。
    def split(i,pet,ct,tumor,bg,height):
        volumeMap = {}
        volumeMap['pet'] = pet[:,:,:,i*sliceNumPerGroup:(i+1)*sliceNumPerGroup,:]
        volumeMap['ct'] = ct[:,:,:,i*sliceNumPerGroup:(i+1)*sliceNumPerGroup]
        volumeMap['tumor'] = tumor[:,:,:,i*sliceNumPerGroup:(i+1)*sliceNumPerGroup,:]
        volumeMap['bg'] = bg[:,:,:,i*sliceNumPerGroup:(i+1)*sliceNumPerGroup,:]
        volumeMap['height'] = height[:,i*sliceNumPerGroup:(i+1)*sliceNumPerGroup]
        sliceNum = volumeMap['pet'].shape[-2]
        #If slices cannot fulfill a patch，slices for previous patch are used for padding。
        if sliceNum<sliceNumPerGroup:
            volumeMap['pet'] = np.concatenate([pet[:,:,:,i*sliceNumPerGroup-(sliceNumPerGroup-sliceNum):i*sliceNumPerGroup,:],volumeMap['pet']],axis=-2)
            volumeMap['ct'] = np.concatenate([ct[:,:,:,i*sliceNumPerGroup-(sliceNumPerGroup-sliceNum):i*sliceNumPerGroup],volumeMap['ct']],axis=-1)
            volumeMap['tumor'] = np.concatenate([tumor[:,:,:,i*sliceNumPerGroup-(sliceNumPerGroup-sliceNum):i*sliceNumPerGroup,:],volumeMap['tumor']],axis=-2)
            volumeMap['bg'] = np.concatenate([bg[:,:,:,i*sliceNumPerGroup-(sliceNumPerGroup-sliceNum):i*sliceNumPerGroup,:],volumeMap['bg']],axis=-2)
            volumeMap['height'] = np.concatenate([height[:,i*sliceNumPerGroup-(sliceNumPerGroup-sliceNum):i*sliceNumPerGroup],volumeMap['height']],axis=-1)
        return volumeMap
        
    i = -1
    while i+1<subvolumeNum:
        i = i+1
        res = split(i,pet,ct,tumor,bg,height)
        #Discarding patches do not contain tumor voxel in a fixed probability.
        if skip:
            if np.count_nonzero(res['tumor']) == 0 and random.randint(1,10)<=8 and FLAGS.mode=='train':
                continue
        yield res

def concat_subvolumes(batch):
    petList = []
    tumorList = []
    bgList = []
    heightList = []
    for single in batch:
        petList.append(single['pet'])
        tumorList.append(single['tumor'])
        bgList.append(single['bg'])
        heightList.append(single['height'])
    pet = np.concatenate(petList,axis=0)
    tumor = np.concatenate(tumorList,axis=0)
    bg = np.concatenate(bgList,axis=0)
    height = np.concatenate(heightList,axis=0)
    return pet,tumor,bg,height

#CT slices are concatenated with it's upper and lower neighboor in channel dimension.
def split_volume_to_3slice_groups(ct):
    ct = np.concatenate([ct[:,:,:,0:1],ct],-1)
    ct = np.concatenate([ct,ct[:,:,:,-1:]],-1)
    for i in range(ct.shape[3]-2):
        yield ct[:,:,:,i:i+3]

#CT 3-slice groups are concatenated in batch dimension.
def concat_3slice_groups(batch):
    ct = None
    for patch in batch:
        transformedList = []
        iterator = split_volume_to_3slice_groups(patch['ct'])
        try:
            while True:
                transformedList.append(next(iterator))
        except Exception:
            if not isinstance(ct,np.ndarray):
                ct = np.concatenate(transformedList,axis=0)
            else:
                ct = np.concatenate([ct,np.concatenate(transformedList,axis=0)],axis=0)
            transformedList = []
    return ct

#Transfer batch dimension to vertical axis dimension.
def batch_dim_to_zaxis(vol,sliceNum,batchSize):
    vol = tf.transpose(tf.expand_dims(vol,axis=3),(3,1,2,0,4))
    subVolList = []
    for i in range(batchSize):
        subVolList.append(vol[:,:,:,i*sliceNum:(i+1)*sliceNum,:])
    res = tf.concat(subVolList,axis=0)
    return res

#Transfer vertical axis dimension to batch dimension.
def zaxis_dim_to_batch(vol,batchSize):
    vol_ = tf.transpose(vol,(3,1,2,0,4))
    sliceList = []
    for i in range(batchSize):
        sliceList.append(vol_[:,:,:,i,:])
    res = tf.concat(sliceList,axis=0)
    return res
    