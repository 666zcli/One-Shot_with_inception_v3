#coding:utf-8
import time
import sys
import numpy as np
from scipy.misc import imread,imresize 
import os
import tensorflow as tf
from sklearn.svm  import SVC
import lightgbm as lgb
slim=tf.contrib.slim


### ckpt 
checkpoint_exclude_scopes="InceptionV1/Logits,InceptionV1/AuxLogits"
#checkpoint_exclude_scopes=None    #表示直接从ckpt中回复
checkpoint_path="/home/gui/work2017/ckpt_data/inception_v1_ckpt/inception_v1.ckpt"
is_train=True       # 确定是否要训练 
batch_length=10     # 一个epoch里面的batch次数
batch_number=32     # 一个batch里面的样本个数         
epoch_length=35     # epoch 次数
test_times  = 5     # 测试集合的batch次数，取决与测试集合的样本数//batc_number
#workpath=sys.path[0]
workpath    ='/home/gui/work2017/Research_of_transfer_Learning/dataSets/office-dataset/'
start_jpg=workpath+'/start.jpg'
cate_Li=["back_pack", "bike_helmet", 'bottle',  'desk_lamp', 'desktop_computer', 'file_cabinet', 'keyboard','laptop_computer', 
        'mobile_phone', 'mouse',     'printer', 'projector', 'ring_binder',       'ruler',        'speaker','trash_can']
##### create 4 dirs
paths_source=[];
paths_target=[];
paths_target_train=[];
paths_target_test =[];
for cate in cate_Li:
    dir0=workpath+'amazon_source/'+cate+'/'
    dir1=workpath+'webcam_target/'+cate+'/'
    dir2=workpath+'webcam_target_train/'+cate+'/'
    dir3=workpath+'webcam_target_test/'+cate+'/'
    paths_source.append(dir0)
    paths_target.append(dir1)
    paths_target_train.append(dir2)
    paths_target_test.append(dir3)
##### define path_train, path_test
path_train=paths_source
path_test =paths_target_test

#trainable_scopes="InceptionV3/Logits,InceptionV3/AuxLogits"  # 先训练这几个层，其他的层保持ckpt的参数不变
trainable_scopes="all"  #表示训练所有层
trunc_normal    = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)




###################################################################################
#############################  inception_v1 #######################################



def inception_v1_base(inputs,
                      final_endpoint='Mixed_5c',
                      scope='InceptionV1'):
  """Defines the Inception V1 base architecture.

  This architecture is defined in:
    Going deeper with convolutions
    Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
    Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
    http://arxiv.org/pdf/1409.4842v1.pdf.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    final_endpoint: specifies the endpoint to construct the network up to. It
      can be one of ['Conv2d_1a_7x7', 'MaxPool_2a_3x3', 'Conv2d_2b_1x1',
      'Conv2d_2c_3x3', 'MaxPool_3a_3x3', 'Mixed_3b', 'Mixed_3c',
      'MaxPool_4a_3x3', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e',
      'Mixed_4f', 'MaxPool_5a_2x2', 'Mixed_5b', 'Mixed_5c']
    scope: Optional variable_scope.

  Returns:
    A dictionary from components of the network to the corresponding activation.

  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values.
  """
  end_points = {}
  with tf.variable_scope(scope, 'InceptionV1', [inputs]):
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_initializer=trunc_normal(0.01)):
      with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                          stride=1, padding='SAME'):
        end_point = 'Conv2d_1a_7x7'
        net = slim.conv2d(inputs, 64, [7, 7], stride=2, scope=end_point,normalizer_fn=slim.batch_norm)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points
        end_point = 'MaxPool_2a_3x3'
        net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points
        end_point = 'Conv2d_2b_1x1'
        net = slim.conv2d(net, 64, [1, 1], scope=end_point,normalizer_fn=slim.batch_norm)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points
        end_point = 'Conv2d_2c_3x3'
        net = slim.conv2d(net, 192, [3, 3], scope=end_point,normalizer_fn=slim.batch_norm)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points
        end_point = 'MaxPool_3a_3x3'
        net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_3b'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
            branch_1 = slim.conv2d(branch_1, 128, [3, 3], scope='Conv2d_0b_3x3',normalizer_fn=slim.batch_norm)
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
            branch_2 = slim.conv2d(branch_2, 32, [3, 3], scope='Conv2d_0b_3x3',normalizer_fn=slim.batch_norm)
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope='Conv2d_0b_1x1',normalizer_fn=slim.batch_norm)
          net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_3c'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
            branch_1 = slim.conv2d(branch_1, 192, [3, 3], scope='Conv2d_0b_3x3',normalizer_fn=slim.batch_norm)
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
            branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3',normalizer_fn=slim.batch_norm)
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1',normalizer_fn=slim.batch_norm)
          net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'MaxPool_4a_3x3'
        net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4b'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
            branch_1 = slim.conv2d(branch_1, 208, [3, 3], scope='Conv2d_0b_3x3',normalizer_fn=slim.batch_norm)
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
            branch_2 = slim.conv2d(branch_2, 48, [3, 3], scope='Conv2d_0b_3x3',normalizer_fn=slim.batch_norm)
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1',normalizer_fn=slim.batch_norm)
          net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4c'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 112, [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
            branch_1 = slim.conv2d(branch_1, 224, [3, 3], scope='Conv2d_0b_3x3',normalizer_fn=slim.batch_norm)
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 24, [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
            branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3',normalizer_fn=slim.batch_norm)
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1',normalizer_fn=slim.batch_norm)
          net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4d'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
            branch_1 = slim.conv2d(branch_1, 256, [3, 3], scope='Conv2d_0b_3x3',normalizer_fn=slim.batch_norm)
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 24, [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
            branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3',normalizer_fn=slim.batch_norm)
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1',normalizer_fn=slim.batch_norm)
          net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4e'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 112, [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 144, [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
            branch_1 = slim.conv2d(branch_1, 288, [3, 3], scope='Conv2d_0b_3x3',normalizer_fn=slim.batch_norm)
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
            branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3',normalizer_fn=slim.batch_norm)
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1',normalizer_fn=slim.batch_norm)
          net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4f'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 256, [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
            branch_1 = slim.conv2d(branch_1, 320, [3, 3], scope='Conv2d_0b_3x3',normalizer_fn=slim.batch_norm)
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
            branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0b_3x3',normalizer_fn=slim.batch_norm)
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1',normalizer_fn=slim.batch_norm)
          net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'MaxPool_5a_2x2'
        net = slim.max_pool2d(net, [2, 2], stride=2, scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_5b'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 256, [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
            branch_1 = slim.conv2d(branch_1, 320, [3, 3], scope='Conv2d_0b_3x3',normalizer_fn=slim.batch_norm)
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
            branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0a_3x3',normalizer_fn=slim.batch_norm)
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1',normalizer_fn=slim.batch_norm)
          net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_5c'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
            branch_1 = slim.conv2d(branch_1, 384, [3, 3], scope='Conv2d_0b_3x3',normalizer_fn=slim.batch_norm)
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
            branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0b_3x3',normalizer_fn=slim.batch_norm)
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1',normalizer_fn=slim.batch_norm)
          net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points
    raise ValueError('Unknown final endpoint %s' % final_endpoint)


def inception_v1(inputs,
                 num_classes=1000,
                 is_training=True,
                 dropout_keep_prob=0.8,
                 prediction_fn=slim.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='InceptionV1'):
  """Defines the Inception V1 architecture.

  This architecture is defined in:

    Going deeper with convolutions
    Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
    Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
    http://arxiv.org/pdf/1409.4842v1.pdf.

  The default image size used to train this network is 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether is training or not.
    dropout_keep_prob: the percentage of activation values that are retained.
    prediction_fn: a function to get predictions out of logits.
    spatial_squeeze: if True, logits is of shape is [B, C], if false logits is
        of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.

  Returns:
    logits: the pre-softmax activations, a tensor of size
      [batch_size, num_classes]
    end_points: a dictionary from components of the network to the corresponding
      activation.
  """
  # Final pooling and prediction
  with tf.variable_scope(scope, 'InceptionV1', [inputs, num_classes],
                         reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      net, end_points = inception_v1_base(inputs, scope=scope)
      with tf.variable_scope('Logits'):
        net = slim.avg_pool2d(net, [7, 7], stride=1, scope='MaxPool_0a_7x7')
        net = slim.dropout(net,
                           dropout_keep_prob, scope='Dropout_0b')
        logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                              scope='Conv2d_0c_1x1',normalizer_fn=slim.batch_norm)
        if spatial_squeeze:
          logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')

        end_points['Logits'] = logits
        end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
  return logits, end_points
inception_v1.default_image_size = 224



#############################  inception_v1 #######################################
###################################################################################



###########################################################
####################### class my_net#######################

class my_nets:
    def __init__(self,inputs,weights=None,sess=None,num_classes=3):
        self.inputs = inputs
        self.num_classes=num_classes
        self.build_net_and_get_logits()   #搭建网络，之后才能加载权重，和初始化权重 
        self.probs = self.end_points["Predictions"]
        if weights is not None and sess is not None:
            #sess.run(tf.global_variables_initializer()) 
            if checkpoint_exclude_scopes:
                self.load_weights(sess,weights)#variables_to_restore
            else:
                self.load_weights_just_ckpt(sess,weights)
   
   
    def check_var_not_inited(self):
        check_tensor=tf.report_uninitialized_variables(var_list=tf.trainable_variables())  #check 该网络中是否有未初始化的变量
        return check_tensor



    def build_net_and_get_logits(self):
        #with slim.arg_scope(inception_v3_arg_scope()):   #表示 在这个参数体系下面使用inception_v3,后来发现这个参数体系有问题，与ckpt不对应
        self.logits, self.end_points = inception_v1(inputs=self.inputs,num_classes=self.num_classes)
        print("[v.op.name for v in tf.trainable_variables()]: ",len([v.op.name for v in tf.trainable_variables()]))


    def is_inited(self,sess,var_object):
        '''
        return: True or False
        '''
        return sess.run(tf.is_variable_initialized(var_object))



    def load_weights_just_ckpt(self,sess,weigths):
        save=tf.train.Saver()
        save.restore(sess,weigths)



    def load_weights(self,sess,weigths):
        self.var_li_restore,self.var_li_init=get_var_restored_and_init()  # 这里面的var_li,var_li_init 并不是字符串，而是一个OP(操作)
        print ("var_li_restore:  ",len(self.var_li_restore))  #输出看看是什么
        print ("var_li_init:  ",len(self.var_li_init))
        #save=tf.train.Saver(var_list=self.var_li_restore)
        #save.restore(sess,weigths)
        self.restore_op=slim.assign_from_checkpoint_fn(model_path=weigths, var_list=self.var_li_restore,ignore_missing_vars=True)
        self.restore_op(sess)  #在sess里面执行一下这个restore_op
        #print ("tf.GraphKeys.GLOBAL_VARIABLES: ",tf.GraphKeys.GLOBAL_VARIABLES)                #输出 "variables"
        #self.var_biases=tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope='biases')   #这个scope是根据re.match进行匹配的 
        self.var_biases=[var for var in self.var_li_restore if not self.is_inited(sess,var)]
        print ("var_biases: ",len(self.var_biases))
        self.var_li_init=self.var_li_init+self.var_biases
        sess.run(tf.variables_initializer(var_list=self.var_li_init,name='init'))         #variables_to_init
        print("check_var_not_inited:\n\n",len(sess.run(self.check_var_not_inited())))#check 这些是需要restore,但是没有restore成功的

    

def save_weights(weigths,sess):
    save=tf.train.Saver()
    save.save(sess,weigths)


####################### class my_net#######################
###########################################################




def get_batch_data(has_batched=0,batch_num=32,paths=paths_source):
    x_data=[];y_label=[]
    ### 每类batch同样的数目   batch_num//len(cate_Li)=32//16=2 
    batch_li=[batch_num//len(cate_Li) for i in range(len(cate_Li))]
    print (batch_li)
    ### 遍历所有类别
    for i in range(len(paths)):
        files_list=os.listdir(paths[i])
        batch_len=batch_li[i]
        ### 读取该类别下的图片
        for j in range(has_batched*batch_len,(has_batched+1)*batch_len):
            j=j%len(files_list)   
            ## avoid the out index 
            if not os.path.isdir(files_list[j]):
                img = imread(paths[i]+'/'+files_list[j])    
                # load the  image 
		# print img
                x0=img.tolist()
                # print ('np.shape(x0): ',np.np.shape(x0))
                x_data.append(x0)   
                y_label.append(i)  
                # the label of back_pack is 0;................
    #### shuffle the data-label 
    index_li=range(len(x_data))
    np.random.shuffle(index_li)
    data=[];label=[]
    for j in index_li:
        data.append(x_data[j])
        label.append(y_label[j])
    ########################
    label=label_binary(np.mat(label),range(len(cate_Li)))
    return np.array(data,np.float32),np.array(label)



def label_binary(label,classes=[0,1,2]):
    if np.shape(label)[0]==1 and np.shape(label)[1]>=1:
        label=np.array(label).reshape(-1,).tolist()
        #print(label)
        for i in range(len(label)):
            init_list=np.zeros_like(classes).tolist()
            #print(init_list)
            for j in classes:
                if label[i]==j:
                    init_list[j]=1
                    label[i]=init_list
    else:
        print ('please check the shape==??==(1,m)')
    return np.mat(label)



def get_all_test_data(paths=paths_target_test):
    test_data=[]
    test_label=[]
    for i in range(len(paths)):
        files_list=os.listdir(paths[i])
        for files in files_list:
            if not os.path.isdir(files):
                img = imread(paths[i]+'/'+files)     
                #load the  image 
                x0=img.tolist()
                test_data.append(x0)
                test_label.append(i)
    ### shuffle the test_data-label
    index_li=range(len(test_data))
    np.random.shuffle(index_li)
    data=[];label=[]
    for j in index_li:
        data.append(test_data[j])
        label.append(test_label[j])
    ###############################
    label=label_binary(np.mat(label),range(len(cate_Li)))
    return np.array(data,np.float32),np.array(label)



def recall_rate(predictmat,labelmat,classlabel=1):  #compute the recall rate of classlabel which may be 0,1,2
    '''argument:two  matrix;  np.shape=(1,m)    one is the predict label ,the other is the true label  
       return : recall_rate   :  tp/(tp+fn)
    '''
    if np.shape(predictmat)[0]==np.shape(labelmat)[0] and np.shape(predictmat)[0]==1 and np.shape(predictmat)[1]>=1 :
        tp_fn=np.shape(labelmat[labelmat==classlabel])[1]
        m=np.shape(labelmat)[1]
        tp=0
        for i in range(m):
            if predictmat[0,i]==labelmat[0,i] and predictmat[0,i]==classlabel:
                tp=tp+1
        try:
            recall_rate=tp*1.0/tp_fn
            return recall_rate
        except:
            return 0.0 #tp=tp_fn=0
    else:
        print "please check the np.shape=??==(1,m) "



def correct_rate(predictmat,labelmat,classlabel=1):  #compute the correct rate of classlabel which may be 0,1,2
    '''argument:two  matrix;  np.shape=(1,m)    one is the predict label ,the other is the true label  
       return : correct_rate   :  tp/(tp+fp)
    '''
    if np.shape(predictmat)[0]==np.shape(labelmat)[0] and np.shape(predictmat)[0]==1 and np.shape(predictmat)[1]>=1 :
        tp_fp=np.shape(predictmat[predictmat==classlabel])[1]
        m=np.shape(labelmat)[1]
        tp=0
        for i in range(m):
            if predictmat[0,i]==labelmat[0,i] and predictmat[0,i]==classlabel:
                tp=tp+1
        try: 
            correct_rate=tp*1.0/tp_fp
            return correct_rate
        except:
            return 0.0 #tp=tp_fp=0
    else:
        print "please check the np.shape=??==(1,m) "



def AveP(predictmat,labelmat,classlabel=1):
    if np.shape(predictmat)[0]==np.shape(labelmat)[0] and np.shape(predictmat)[0]==1 and np.shape(predictmat)[1]>=1 :
        m1=np.shape(predictmat)[1]
        cor=[];rec=[]
        for i in range(m1):
            pre=predictmat[:,:i+1]
            label=labelmat[:,:i+1]
            #print np.shape(pre),np.shape(label)
            cor.append(correct_rate(pre,label,classlabel))
            rec.append(recall_rate(pre,label,classlabel))
        # print "cor,rec: ",cor,rec
        cor=[ele for ele in cor if ele is not None]
        rec=[ele for ele in rec if ele is not None]
        #print "cor,rec: ",cor,rec
        index_li=np.argsort(rec)
        new_rec=[rec[index] for index in index_li]
        new_cor=[cor[index] for index in index_li]
        #print 'new_cor,new_rec: ',new_cor,new_rec
        AP=0
        m2=len(new_cor) 
        for j  in range(m2):
            if j==0:
                ap=new_cor[j]*new_rec[j]
            else:
                ap=new_cor[j]*(new_rec[j]-new_rec[j-1])
           # print ap
            AP=AP+ap
        return AP
    else:
        print "please check the np.shape=??=(1,m)"




def Mean_AP(predictmat,labelmat,classlabels_li):
    num=len(classlabels_li)
    sum_AP=0
    for i in range(num):
        sum_AP+=AveP(predictmat,labelmat,classlabels_li[i])
    MAP=round(sum_AP/num,5)
    return MAP



def accuracy(predictmat,labelmat):
    merge=predictmat[predictmat==labelmat]
    num=np.shape(merge)[1]
    return num*1.0/np.shape(labelmat)[1]





def train_test(istrain=is_train):
    start=time.time()
    if istrain==False:
        epoch_len=1
        batch_len=1
    else:
        epoch_len=epoch_length
        batch_len=batch_length
    sess = tf.Session()
    xs=tf.placeholder(tf.float32,[None,224,224,3])
    my_net = my_nets(xs,checkpoint_path,sess,num_classes=16)
    ys=tf.placeholder(tf.float32,[None,16])
    prediction=my_net.probs
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), axis=[0]))
    if checkpoint_exclude_scopes:
        train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01,
                                                    use_locking=True,                                             #使用锁定       
                                                    name='Grad').minimize(loss,var_list=get_variables_to_train()) #只训练指定变量
    else:
        train_step=tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    #sess.run(tf.global_variables_initializer())
    for epoch_i in range(epoch_len):
        # training
        if istrain:
            print ' go to the  epoch_i = '+str(epoch_i)
            print " one epoch will batch until batch_i = "+str(batch_len)
        for batch_i in range(batch_len):
            if istrain:
                t1=time.time()
                print 'this is batch_i= '+str(batch_i)
                x_data,y_data=get_batch_data(has_batched=batch_i,batch_num=batch_number,paths=path_train)
                sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
                print('loss: ',sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
                #print("prediction:  ",sess.run(prediction,feed_dict={xs:x_data}))
                #print("ys:          ",sess.run(ys,        feed_dict={ys:y_data}))
                prob = sess.run(prediction, feed_dict={xs: x_data})
                #####compute train correct_rate ####
                pre=prob.tolist()
                pre=[pre_i.index(np.array(pre_i).max()) for pre_i in pre]
                y_true=y_data.tolist()
                y_true=[y_i.index(np.array(y_i).max()) for y_i in y_true]
                cor_train=[];rec_train=[]
                for i in range(len(cate_Li)):
                    cor_train.append(correct_rate(np.mat(pre),np.mat(y_true),i))
                    rec_train.append(recall_rate(np.mat(pre),np.mat(y_true),i))
		print "accuracy_train=  ",accuracy(np.mat(pre),np.mat(y_true))
                print "cor_train=       ",cor_train
                print "rec_train=       ",rec_train
                t2=time.time()
                print (" run_time:  "+str(t2-t1)+" s")
        ############ compute test correct_rate #############
        print ' start test '
        y_test  =[]
        pre_test=[]
        for k in range(test_times): 
            x_test_0,y_test_0 =get_batch_data(has_batched=k,batch_num=batch_number,paths=path_test)
            prob_test=sess.run(prediction,feed_dict={xs: x_test_0})
            pre_test.extend(prob_test.tolist())                                        
            y_test.extend(y_test_0.tolist())
        #print ("pre_test: ",np.array(pre_test))
        #print ("y_test:   ",np.array(y_test))
        pre_test=[pre_i.index(np.array(pre_i).max()) for pre_i in pre_test]
        y_test  =[y_i.index(np.array(y_i).max()) for y_i in y_test]
        MAP     =Mean_AP(np.mat(pre_test),np.mat(y_test),range(len(cate_Li)))
        print 'MAP_of_test_set = ',MAP
        cor_test=[];rec_test=[]
        for i in range(len(cate_Li)):
            cor_test.append(correct_rate(np.mat(pre_test),np.mat(y_test),i))
            rec_test.append(recall_rate(np.mat(pre_test),np.mat(y_test),i))
        cor_test=[round(ele,5) for ele in cor_test]
        rec_test=[round(ele,5) for ele in rec_test]
	print "accuracy_test=   ",accuracy(np.mat(pre_test),np.mat(y_test))
        print "cor_test=        ",cor_test
        print "rec_test=        ",rec_test
        ####################################################
        print " has finished epoch_i = "+str(epoch_i)
        save_weights(checkpoint_path,sess)
        print " has save the checkpoint_path "
    end=time.time()
    run_time=end-start
    print 'run_time:    ',run_time
    return cor_test,rec_test,round(MAP,4),round(run_time,4)




def get_var_restored_and_init(checkpoint_exclude_scopes=checkpoint_exclude_scopes):
    """Returns a function run by the chief worker to warm-start the training.
    Note that the init_fn is only run when initializing the model during the very
    first global step.
    Returns:
    An init function run by the supervisor.
    """
    exclusions = []
    if checkpoint_exclude_scopes:
        exclusions = [scope.strip() for scope in checkpoint_exclude_scopes.split(',')]
    variables_to_restore = []
    variable_to_init=[]
    for var in slim.get_model_variables():    #获取当前的slim网络的变量
        excluded = False  
        for exclusion in exclusions:   #exclusions=[InceptionV3/Logits,InceptionV3/AuxLogits]
            if var.op.name.startswith(exclusion):
                excluded = True       
                break            #只要var.op.name 是以上面列表中的任何一个开头，那么就会有exclude=True ,即表示：此var不能从ckpt中restore
        if not excluded:
            variables_to_restore.append(var)
        else: 
            variable_to_init.append(var)
    return variables_to_restore,variable_to_init
    



def get_variables_to_train(trainable_scopes=trainable_scopes):
    """Returns a list of variables to train.
    Returns:
    A list of variables to train by the optimizer.
    """
    if trainable_scopes =="all":
        print 'variables_to_train:   ',len(tf.trainable_variables())
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in trainable_scopes.split(',')]
    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    print 'variables_to_train:   ',len(variables_to_train)
    return variables_to_train


if __name__=='__main__':
   cor_test,rec_test,MAP,run_time=train_test(istrain=is_train) 
