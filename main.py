import os
import tensorflow as tf
from make_input import parse_total_patient
from network import Network
import shutil
import save_image
import transform
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#Model setting
FLAGS = tf.app.flags.FLAGS
#Running mode: train or test.
tf.app.flags.DEFINE_string('mode','train','')
#Restore parameters from *.ckpt 
tf.app.flags.DEFINE_boolean('restore',False,'')
#Dir contains all *.tfrecord file.
tf.app.flags.DEFINE_string('data_path','/data/PET-CT-3-4/','')
'''
Start and end index of patients for test, two value are included. Start index starts from 1, not 0.
Patients for training and test are in same directory, they are choosed for training/test by assigning 
index of patients for test, and residule patients are used for training.
Example:
A dir "/media/xxx/PET-CT/" contains 9 patients: 1.tfrecord, 2.tfrecord, ... ,9.tfrecord. Their shuffled name list
(see make_input.py) is ['2','3','6',7','5','8','1','4','9']. In first training-test loop, use first three patients in
the list (name=2 and name=3). Thus only need to set: test_data_start_index=1 and test_data_end_index=3.
In next loop, set test_data_start_index=4 and test_data_end_index=6. 
In final loop, set test_data_start_index=7 and test_data_end_index=9. 
'''
tf.app.flags.DEFINE_integer('test_data_start_index',1,'')
tf.app.flags.DEFINE_integer('test_data_end_index',96,'')
#Dir to store log file, which can be read by tensorboard.
tf.app.flags.DEFINE_string('log_dir1','/media/wangkun/Data2/Python_workspace/my-model/log/','')
#Dir to store trained parameters and model sturcture.
tf.app.flags.DEFINE_string('ckpt_dir','/media/wangkun/Data2/Python_workspace/my-model/checkpoint','')
#Dir to store outputted volumetric image in test stage.
tf.app.flags.DEFINE_string('test_output_path','/media/wangkun/Data2/Python_workspace/my-model/test_output/','')
#In each iteration, how many patch is used for training.
tf.app.flags.DEFINE_integer('batch_size',1,'')
#How many consequent slices of a patch contains.
tf.app.flags.DEFINE_integer('slice_number_per_patch',64,'')
#How many iteration to save log to disk.
tf.app.flags.DEFINE_integer('log_iteration',50,'')
#How many iteration to save trainable parameters to disk.
tf.app.flags.DEFINE_integer('ckpt_iteration',1000,'')
#Generate hard label during training stage, for supervision of metrics in tensorboard.
tf.app.flags.DEFINE_float('binary_threshold',0.9,'')
#How many iteration in training stage.
tf.app.flags.DEFINE_integer('end_iteration',100000,'')

def train(FLAGS):
    #Parse data
    trainData = parse_total_patient(FLAGS.data_path,FLAGS.test_data_start_index,FLAGS.test_data_end_index,isTraining=True)
    trainIterator = tf.compat.v1.data.make_initializable_iterator(trainData)
    modality,label,name = trainIterator.get_next()
    #Definite Placeholder
    petHolder = tf.compat.v1.placeholder(tf.float32,shape=[None,256,256,FLAGS.slice_number_per_patch,1])
    ctHolder = tf.compat.v1.placeholder(tf.float32,shape=[None,256,256,3])
    tumorHolder = tf.compat.v1.placeholder(tf.float32,shape=[None,256,256,FLAGS.slice_number_per_patch,1])
    bgHolder = tf.compat.v1.placeholder(tf.float32,shape=[None,256,256,FLAGS.slice_number_per_patch,1])
    heightHolder = tf.compat.v1.placeholder(tf.float32,shape=[None,FLAGS.slice_number_per_patch])
    #Build network
    model = Network(FLAGS,[petHolder,ctHolder],[tumorHolder,bgHolder,heightHolder])
    model.build_architecture()
    #Iniiialize variables
    globleInitial = tf.compat.v1.global_variables_initializer()
    localInitial = tf.compat.v1.local_variables_initializer()
    #Set GPU parameters
    gpuOptions = tf.compat.v1.GPUOptions(allow_growth=True)
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,device_count={'GPU':1},gpu_options=gpuOptions)) as sesson:
        sesson.run([trainIterator.initializer,globleInitial,localInitial])
        #A saver for storing trained parameters
        saver = tf.compat.v1.train.Saver(max_to_keep=1)
        #A writer for visualization using Tensorboard
        trainWriter = tf.compat.v1.summary.FileWriter(FLAGS.log_dir1,sesson.graph)
        #Wether restore trained parameters, preventing from incident during training.
        if FLAGS.restore == True:
            ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_dir)
            saver.restore(sesson,ckpt.model_checkpoint_path)
        #Count trainable parameter.
        model.count_trainable_parameter()
        sesson.graph.finalize()
        #Obtain first sample.
        (pet,ct),(tu,bg,height),name_ = sesson.run([modality,label,name])
        #Record the number of obtaineed patches.
        i = 0
        #To store patches in a batch.
        batch = []
        #A flag to control random discarding of patches does not contain tumor voxel.
        skipFlag = True
        #Iterator to generate patches along vertical axis.
        subVolumeIterator = transform.split_volume_to_subvolumes(FLAGS,pet,ct,tu,bg,height,skip=skipFlag)
        #Train iteratively until meet the ending condition.
        while True:
            try:
                if i<FLAGS.batch_size:
                    batch.append(next(subVolumeIterator))
                    i = i+1
                elif i==FLAGS.batch_size:
                    pet_,tumor_,bg_,height_ = transform.concat_subvolumes(batch)
                    ct_ = transform.concat_3slice_groups(batch)
                    globalStep, trainOp = sesson.run([model.globalStep,model.trainOperation], \
                        feed_dict={petHolder:pet_,ctHolder:ct_,tumorHolder:tumor_,bgHolder:bg_, \
                        heightHolder:height_,model.isTraining:True})
                    i = 0
                    batch = [] 
                    #Save log to tensorboard.
                    if globalStep % FLAGS.log_iteration == 0 or globalStep == 0:
                        summary, posLoss, segLoss, reguLoss = sesson.run([model.summary, \
                            model.posLoss, model.segLoss, model.reguLoss],
                            feed_dict={petHolder:pet_,ctHolder:ct_,tumorHolder:tumor_,bgHolder:bg_, \
                            heightHolder:height_,model.isTraining:True})
                        trainWriter.add_summary(summary,globalStep)
                        print(str(globalStep)+'\n'+'Position'.ljust(14)+' = '+str(posLoss)+ \
                            '\n'+'Segmentation'.ljust(14)+' = '+str(segLoss)+ \
                            '\n'+'Regularization'.ljust(14)+' = '+str(reguLoss)+'\n'+'-'*20)
                    #Wether to end training process.
                    if globalStep % FLAGS.end_iteration ==0 and globalStep>0:
                        fileName = FLAGS.ckpt_dir+'/end.ckpt'
                        saver.save(sesson,fileName)
                        return
                    #Save model to disk.
                    if globalStep % FLAGS.ckpt_iteration == 0 and globalStep>0:
                        fileName = FLAGS.ckpt_dir+'/'+str(globalStep)+'.ckpt'
                        saver.save(sesson,fileName)
            #Iterator point to next sample.
            except StopIteration:
                (pet,ct),(tu,bg,height),name_ = sesson.run([modality,label,name])
                subVolumeIterator = transform.split_volume_to_subvolumes(FLAGS,pet,ct,tu,bg,height)
 
def test(FLAGS):
    testData = parse_total_patient(FLAGS.data_path,FLAGS.test_data_start_index,FLAGS.test_data_end_index,isTraining=False)
    testIterator = tf.compat.v1.data.make_initializable_iterator(testData)
    modality,label,name = testIterator.get_next()
    petHolder = tf.compat.v1.placeholder(tf.float32,shape=[None,256,256,FLAGS.slice_number_per_patch,1])
    ctHolder = tf.compat.v1.placeholder(tf.float32,shape=[None,256,256,3])
    tumorHolder = tf.compat.v1.placeholder(tf.float32,shape=[None,256,256,FLAGS.slice_number_per_patch,1])
    bgHolder = tf.compat.v1.placeholder(tf.float32,shape=[None,256,256,FLAGS.slice_number_per_patch,1])
    heightHolder = tf.compat.v1.placeholder(tf.float32,shape=[None,FLAGS.slice_number_per_patch])
    model = Network(FLAGS,[petHolder,ctHolder],[tumorHolder,bgHolder,heightHolder])
    model.build_architecture()
    globleInitial = tf.compat.v1.global_variables_initializer()
    localInitial = tf.compat.v1.local_variables_initializer()
    if not os.path.exists(FLAGS.test_output_path):
        os.mkdir(FLAGS.test_output_path)
    gpuOptions = tf.compat.v1.GPUOptions(allow_growth=True)
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,device_count={'GPU':1},gpu_options=gpuOptions)) as sesson:
        sesson.run([globleInitial,localInitial,testIterator.initializer])
        ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_dir)
        saver = tf.compat.v1.train.Saver()
        saver.restore(sesson,ckpt.model_checkpoint_path)
        i = 1
        try:
            while True:   
                (pet,ct),(tu,bg,height),name_ = sesson.run([modality,label,name])
                subVolumeNum = np.ceil(pet.shape[-2]/FLAGS.slice_number_per_patch)
                subVolumeIterator = transform.split_volume_to_subvolumes(FLAGS,pet,ct,tu,bg,height)
                subSoftmaxList = []
                for j in range(int(subVolumeNum)):
                    batch = []
                    batch.append(next(subVolumeIterator))
                    pet_,tumor_,bg_,height_ = transform.concat_subvolumes(batch)
                    ct_ = transform.concat_3slice_groups(batch)
                    softmax_ = sesson.run(model.softmax, \
                        feed_dict={petHolder:pet_,ctHolder:ct_,tumorHolder:tumor_,bgHolder:bg_, \
                            heightHolder:height_,model.isTraining:False})
                    if j!=subVolumeNum-1 or (j+1)*subVolumeNum==pet.shape[-2]:
                        subSoftmaxList.append(softmax_)
                    else:
                        subSoftmaxList.append(softmax_[:,:,:,-int(pet.shape[-2]%FLAGS.slice_number_per_patch):,:])
                wholeSoftmax = np.concatenate(subSoftmaxList,axis=-2)
                predBlack = np.zeros_like(pet)
                print(str(i)+' '+str(name_[0],'utf-8'))
                ct = np.expand_dims(ct,axis=-1)
                tu[tu>0] = 1
                save_image.save_image(str(name_[0],'utf-8'),pet[0,:,:,:,0],wholeSoftmax[0,:,:,:,0],tu[0,:,:,:,0], \
                    predBlack[0,:,:,:,0],FLAGS.test_output_path)
                i = i+1
        except tf.errors.OutOfRangeError:
            print("Test finished. Using the CRF to generate hard label and then calculate metrics.")
            
def main(_):
    if FLAGS.mode == 'train':
        train(FLAGS)
    elif FLAGS.mode == 'test':
        test(FLAGS)

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    if FLAGS.mode == 'train' and FLAGS.restore==False and os.path.exists(FLAGS.log_dir1) :
        shutil.rmtree(FLAGS.log_dir1)
    if FLAGS.mode == 'test' and os.path.exists(FLAGS.test_output_path):
        shutil.rmtree(FLAGS.test_output_path)
    tf.compat.v1.app.run()
