import tensorflow as tf

def image_summary(model):
    maxIndex = tf.argmax(tf.count_nonzero(model.label[0][:,:,:,:,0:1],axis=[1,2,-1]),axis=1)
    for i in range(model.flags.batch_size):
        with tf.compat.v1.variable_scope('Image_'+str(i+1)):
            tf.compat.v1.summary.image('1-PET',rescale_pet(model.data[0][i:i+1,:,:,maxIndex[i],0:1]))
            tf.compat.v1.summary.image('2-GroundTruth',model.label[0][i:i+1,:,:,maxIndex[i],0:1])
            tf.compat.v1.summary.image('3-Softmax',model.softmax[i:i+1,:,:,maxIndex[i],0:1])

def statistic_summary(model):
    for i in range(model.flags.batch_size):
        with tf.compat.v1.variable_scope('Image_'+str(i+1)):
            tf.compat.v1.summary.histogram('1PET',model.data[0][i:i+1,:,:,:,0,1])
            tf.compat.v1.summary.histogram('2CT',model.data[1][i:i+1,:,:,0,1])
            tf.compat.v1.summary.histogram('3Softmax',model.softmax[i:i+1,:,:,0:1])
            
            
def rescale_0_1(x):
    return (x-tf.reduce_min(x))/(tf.reduce_max(x)-tf.reduce_min(x))

def rescale_pet(x):
    upperBound = tf.constant(3.5,shape=[1,256,256,1],dtype=tf.float32)
    truncated = tf.where(x>upperBound,upperBound,x)
    return truncated/3.5


