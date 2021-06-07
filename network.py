import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops
import math
import summary as sm
import metrics
import transform
from tensorflow import contrib
autograph = contrib.autograph

class Network(object):

    def __init__(self,FLAGS,modality,label):
        self.flags = FLAGS
        self.data = modality
        self.label = label
        self.isTraining = tf.compat.v1.placeholder(tf.bool,name="is_training")
                
    def build_architecture(self):
        self.globalStep = tf.compat.v1.train.get_or_create_global_step()
        #CT positional learning. Part A.
        ct_conv1 = self.conv_2D(self.data[1],8,times=2,residule=False)
        ct_down1 = self.down_2D(ct_conv1,16)
        ct_conv2 = self.conv_2D(ct_down1,16,times=2)
        ct_down2 = self.down_2D(ct_conv2,32)
        ct_conv3 = self.conv_2D(ct_down2,32,times=2)
        ct_down3 = self.down_2D(ct_conv3,64)
        ct_conv4 = self.conv_2D(ct_down3,64,times=2)
        ct_down4 = self.down_2D(ct_conv4,128)
        ct_conv5 = self.conv_2D(ct_down4,128,times=2)
        ct_avg = tf.layers.average_pooling2d(ct_conv5,(16,16),1)
        ct_avg_squeeze = tf.squeeze(ct_avg,axis=(1,2))
        ct_fc1 = self.fully_connect(ct_avg_squeeze,128,actv=True)
        ct_fc2 = self.fully_connect(ct_fc1,32,actv=True)
        self.ct_pos= tf.squeeze(self.fully_connect(ct_fc2,1,actv=False),axis=-1)
        #Convert relative position vector to gating vector. Part C.
        self.get_position_based_weight()
        #Multi-scale CT feature fusion. Part B.
        ct_conv4_reduc = self.conv_2D(ct_conv4,32,residule=False,lastActv=False,filterSize=(1,1))
        ct_conv5_reduc = self.conv_2D(ct_conv5,32,residule=False,lastActv=False,filterSize=(1,1))
        ct_conv4_reduc_up = self.up_2D(ct_conv4_reduc,scale=2)
        ct_conv5_reduc_up = self.up_2D(ct_conv5_reduc,scale=4)
        ct_concat = tf.concat([ct_conv3,ct_conv4_reduc_up,ct_conv5_reduc_up],axis=-1)
        ct_concat_reduc = self.conv_2D(ct_concat,32,residule=False,lastActv=False,filterSize=(1,1))
        ct_concat_reverse = transform.batch_dim_to_zaxis(ct_concat_reduc,self.flags.slice_number_per_patch,self.flags.batch_size)
        #PET shared encoder. Part D.
        pet_conv1 = self.conv_3D(self.data[0],8,times=2,residule=False)
        pet_down1 = self.down_3D(pet_conv1,16)
        pet_conv2 = self.conv_3D(pet_down1,16,times=2)
        pet_down2 = self.down_3D(pet_conv2,32)
        pet_conv3 = self.conv_3D(pet_down2,32,times=2)
        #PET/CT feature fusion. Part B.
        ct_concat_reverse_zdown = self.up_3D(ct_concat_reverse,aimSize=pet_conv3,channel=32)
        fuse_concat = tf.concat([ct_concat_reverse_zdown,pet_conv3],axis=-1)
        fuse_conv1 = self.conv_3D(fuse_concat,64,times=2,residule=False,filterSize=(1,1,1))
        #BPS-decoder for upper semi-body. Part D.
        fuse_conv2_upper = self.conv_3D(fuse_conv1,16,residule=False,lastActv=False,filterSize=(1,1,1))
        fuse_up1_upper = self.up_3D(fuse_conv2_upper,aimSize=pet_conv2,channel=16)
        fuse_pet_concat1_upper = tf.concat([fuse_up1_upper,pet_conv2],axis=-1)
        fuse_conv3_upper = self.conv_3D(fuse_pet_concat1_upper,16,times=2,residule=False)
        fuse_conv4_upper = self.conv_3D(fuse_conv3_upper,8,residule=False,lastActv=False,filterSize=(1,1,1))
        fuse_up2_upper = self.up_3D(fuse_conv4_upper,aimSize=pet_conv1,channel=8)
        fuse_pet_concat2_upper = tf.concat([fuse_up2_upper,pet_conv1],axis=-1)
        fuse_conv5_upper = self.conv_3D(fuse_pet_concat2_upper,8,times=2,residule=False)
        softmax_upper = self.get_softmax(fuse_conv5_upper)
        #BPS-decoder for lower semi-body. Part D.
        fuse_conv2_lower = self.conv_3D(fuse_conv1,16,residule=False,lastActv=False,filterSize=(1,1,1))
        fuse_up1_lower = self.up_3D(fuse_conv2_lower,aimSize=pet_conv2,channel=16)
        fuse_pet_concat1_lower = tf.concat([fuse_up1_lower,pet_conv2],axis=-1)
        fuse_conv3_lower = self.conv_3D(fuse_pet_concat1_lower,16,times=2,residule=False)
        fuse_conv4_lower = self.conv_3D(fuse_conv3_lower,8,residule=False,lastActv=False,filterSize=(1,1,1))
        fuse_up2_lower = self.up_3D(fuse_conv4_lower,aimSize=pet_conv1,channel=8)
        fuse_pet_concat2_lower = tf.concat([fuse_up2_lower,pet_conv1],axis=-1)
        fuse_conv5_lower = self.conv_3D(fuse_pet_concat2_lower,8,times=2,residule=False)
        softmax_lower = self.get_softmax(fuse_conv5_lower)
        #Gating. Part D.
        self.weighted_softmax_upper = softmax_upper*self.upperWeight
        self.weighted_softmax_lower = softmax_lower*self.lowerWeight
        self.softmax = self.weighted_softmax_upper+self.weighted_softmax_lower
        #Calculate loss.
        self.loss = self.get_total_loss()
        self.trainOperation = self.train_operation()
        #Output log to tensorboard.
        sm.image_summary(self)
        with tf.compat.v1.variable_scope('Metrics'):
            metrics.scalar_summary(self)
        self.summary = tf.compat.v1.summary.merge_all()
    
    def fully_connect(self,inputVector,outputChannel,actv):
        res = tf.keras.layers.Dense(outputChannel)(inputVector)
        if actv:
            res = self.prelu(res)
        return res
            
    
    def conv_3D(self,inputMap,filterNum,times=1,residule=True,lastActv=True,filterSize=(3,3,3),stride=(1,1,1),BN=False):
        inputMapCopy = inputMap
        for i in range(times):
            conv = tf.keras.layers.Conv3D(
                    filters=filterNum,
                    kernel_size=filterSize,
                    kernel_initializer=self.MSRA_initializer(),
                    bias_initializer=tf.zeros_initializer(),
                    padding='same',
                    )(inputMap)
            if lastActv==True or i<times-1:
                actv = self.prelu(conv)
                inputMap = actv
            else:
                inputMap = conv
        if residule:
            res = tf.add(inputMapCopy,inputMap)
        else:
            res = inputMap
        if BN:
            res = tf.layers.batch_normalization(
                    inputs=res,
                    momentum=0.9,
                    epsilon=0.001,
                    axis=-1,
                    center=True,
                    scale=True,
                    training=self.isTraining
                    )
        return res
    
    def conv_2D(self,inputMap,filterNum,times=1,residule=True,lastActv=True,filterSize=(3,3),stride=(1,1),BN=False):
        inputMapCopy = inputMap
        for i in range(times):
            conv = tf.keras.layers.Conv2D(
                    filters=filterNum,
                    kernel_size=filterSize,
                    kernel_initializer=self.MSRA_initializer(),
                    bias_initializer=tf.zeros_initializer(),
                    padding='same',
                    )(inputMap)
            if lastActv==True or i<times-1:
                actv = self.prelu(conv)
                inputMap = actv
            else:
                inputMap = conv
        if residule:
            res = tf.add(inputMapCopy,inputMap)
        else:
            res = inputMap
        if BN:
            res = tf.layers.batch_normalization(
                    inputs=res,
                    momentum=0.9,
                    epsilon=0.001,
                    axis=-1,
                    center=True,
                    scale=True,
                    training=self.isTraining
                    )
        return res
    
    def down_3D(self,inputMap,filterNum,downSize=(3,3,3),downStride=(2,2,2)):
        conv = tf.keras.layers.Conv3D(
                    filters=filterNum,
                    kernel_size=downSize,
                    strides=downStride,
                    kernel_initializer=self.MSRA_initializer(),
                    bias_initializer=tf.zeros_initializer(),
                    padding='same',
                    )(inputMap)
        actv = self.prelu(conv)
        return actv
    
    def down_2D(self,inputMap,filterNum,downSize=(3,3),downStride=(2,2)):
        conv = tf.keras.layers.Conv2D(
                    filters=filterNum,
                    kernel_size=downSize,
                    strides=downStride,
                    kernel_initializer=self.MSRA_initializer(),
                    bias_initializer=tf.zeros_initializer(),
                    padding='same',
                    )(inputMap)
        actv = self.prelu(conv)
        return actv
    
    #Interpolation for volumetric image.
    def up_3D(self,inputMap,aimSize,channel):
        inputShape = tf.shape(inputMap)
        aimShape = tf.shape(aimSize)
        reshapedForXY = tf.reshape(inputMap,[inputShape[0],inputShape[1],inputShape[2],-1])
        interpolatedOnXY = tf.image.resize(reshapedForXY,[aimShape[1],aimShape[2]],tf.image.ResizeMethod.BILINEAR,align_corners=True)
        reshapedOnXY = tf.reshape(interpolatedOnXY,[inputShape[0],aimShape[1],aimShape[2],inputShape[3],channel])
        inputShape = tf.shape(reshapedOnXY)
        transposedForYZ = tf.transpose(reshapedOnXY,[0,2,3,1,4])
        reshapedForYZ = tf.reshape(transposedForYZ,[inputShape[0],inputShape[2],inputShape[3],-1])
        interpolatedOnYZ = tf.image.resize(reshapedForYZ,[aimShape[2],aimShape[3]],tf.image.ResizeMethod.BILINEAR,align_corners=True)
        reshapedOnXY = tf.reshape(interpolatedOnYZ,[inputShape[0],aimShape[2],aimShape[3],aimShape[1],channel])
        transposedBack = tf.transpose(reshapedOnXY,[0,3,1,2,4])
        return transposedBack
    
    #Interpolation for slice image.
    def up_2D(self,inputMap,scale=2):
        inputShape = tf.shape(inputMap)
        x,y = inputShape[1],inputShape[2]
        res = tf.image.resize(inputMap,[x*scale,y*scale],tf.image.ResizeMethod.BILINEAR,align_corners=True)
        return res
    
    #Random parameter generator.
    def MSRA_initializer(self,p=0.25):
        
        def initializer(shape, dtype=dtypes.float32, partition_info=None):
            n = float(shape[-2])
            for dim in shape[:-2]:
                n *= float(dim)
            stddev = math.sqrt(2/((1+p**2)*n))
            return random_ops.random_normal(shape,stddev=stddev)
        
        return initializer

    def prelu(self,x,p=0.25):
        w_shape = x.get_shape().as_list()[-1]
        alpha = tf.compat.v1.get_variable("alpha_"+x.name.replace("/","_").replace(":",""), shape=w_shape, initializer=tf.constant_initializer(p))
        x = tf.nn.relu(x) + tf.multiply(alpha, (x - tf.abs(x))) * 0.5
        x.alpha = alpha
        return x
    
    def get_softmax(self,inputMap,filterSize=(1,1,1),filterNum=2,stride=(1,1,1)):
        beforeSoftmax = tf.keras.layers.Conv3D(
                filters=filterNum,
                kernel_size=filterSize,
                strides=stride,
                kernel_initializer=self.MSRA_initializer(),
                bias_initializer=tf.zeros_initializer(),
                padding='same',
                )(inputMap)
        softmax = tf.nn.softmax(beforeSoftmax)
        return softmax
    
    #Convert relative position to gate signal of each slice.
    def get_position_based_weight(self):
        upperBound = 0.05
        lowerBound = -0.05
        #In training stage, gate signal is calculated by position label.
        if self.flags.mode == 'train':
            pos_3D = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.label[2],axis=1),axis=1),axis=-1)
        #In test stage, gate signal is calculated by position output from Part A.
        else:
            pos_3D = self.ct_pos
            for i in range(3):
                pos_3D = tf.expand_dims(pos_3D,axis=-1)
            pos_3D = transform.batch_dim_to_zaxis(pos_3D,self.flags.slice_number_per_patch,self.flags.batch_size)
        #Convert
        zeros = tf.zeros_like(pos_3D)
        changedWeight = (pos_3D-lowerBound)/(upperBound-lowerBound)
        cond1 = tf.greater(pos_3D,upperBound)
        upperWeight = tf.where(cond1,zeros+1,changedWeight)
        cond2 = tf.greater(zeros,upperWeight)
        upperWeight = tf.where(cond2,zeros,upperWeight)
        lowerWeight = 1-upperWeight
        #Output to check.
        self.upperWeight1D = tf.squeeze(upperWeight)
        #Boardcast vector to volume.
        self.upperWeight = tf.tile(upperWeight,[1,256,256,1,2])
        self.lowerWeight = tf.tile(lowerWeight,[1,256,256,1,2])
    
    def get_total_loss(self):
        self.posLoss = self.get_regression_loss(self.ct_pos)*0.1
        self.segLoss = self.get_crossEntropy_loss(self.softmax)
        self.reguLoss = self.get_regular_loss()*5e-6
        totalLoss = self.posLoss+self.segLoss+self.reguLoss
        with tf.compat.v1.variable_scope('Loss'):
            tf.compat.v1.summary.scalar('Total_loss',totalLoss)
            tf.compat.v1.summary.scalar('Position_loss',self.posLoss)
            tf.compat.v1.summary.scalar('Segmentation_loss',self.segLoss)
            tf.compat.v1.summary.scalar('Regularization_loss',self.reguLoss)
        return totalLoss
    
    def get_regular_loss(self):
        varList = []
        for var in tf.compat.v1.trainable_variables():
            if not "alpha" in var.name:
                varList.append(tf.nn.l2_loss(var))
        regularLoss = tf.add_n(varList)
        return regularLoss
    
    def get_regression_loss(self,prediction):
        label = tf.squeeze(tf.reshape(self.label[2],(-1,1)),axis=-1)
        regression_loss = tf.reduce_mean((prediction-label)**2)
        return regression_loss
    
    def get_crossEntropy_loss(self,softmax):
        softmax_log = tf.math.log(softmax+1e-8)
        concatLabel = tf.concat(self.label[0:2],axis=-1)
        loss = tf.reduce_mean(-tf.reduce_sum(concatLabel*softmax_log,axis=-1))
        return loss
    
    def get_binary_prediction(self):
        zero = tf.zeros_like(self.softmax[:,:,:,:,0:1])
        one = tf.ones_like(self.softmax[:,:,:,:,0:1])
        tumor = tf.where(self.softmax[:,:,:,:,0:1]>=self.flags.binary_threshold,x=one,y=zero)
        backgroud = tf.where(self.softmax[:,:,:,:,0:1]<self.flags.binary_threshold,x=zero,y=one)
        pred = tf.concat([tumor,backgroud],axis=-1)
        return pred
    
    def expo_decay_LR(self):
        maxLR = 0.0001
        decayCycle = 500
        decayRate = 0.99
        learningRate = maxLR*decayRate**tf.floor(self.globalStep/decayCycle)
        tf.compat.v1.summary.scalar('Learning_rate',learningRate)
        return learningRate
    
    def train_operation(self):
        updateOps = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(updateOps):
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.expo_decay_LR())
            trainOpration = optimizer.minimize(loss=self.loss,global_step=self.globalStep)
        return trainOpration
    
    def count_trainable_parameter(self):
        totalNum = 0
        for variable in tf.compat.v1.trainable_variables():
            singleNum = 1
            for dim in variable.shape:
                singleNum *= dim
            totalNum += singleNum
        print('Trainable parameter count: %s'%totalNum)
        
