
# 1. first  you should get the office-datasets  from  https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view
-----------

# 2. then build the one_hot and zero_hot dataset from original office-datasets
## ./build_data_set.sh your_path_of_office-datasets
-----------

# 3. then you should resize your picture to [299,299]
## python ./resize_all_image.py  --dataset_path  your_path  --size [299,299]
-----------



# 4.依赖库(Dependent library)
## tensorflow
## scipy
## numpy
-----------


# 5.change some area of my_inception_v3.py
## 运行程序之前先修改my_inception_v3.py中的 checkpoint_path 和 dataset_path
###  (before run ,please change the checkpoint_path and dataset_path to be yours)
##  另外保证tf_inception_v3.py 和 my_inception_v3.py 在同一文件夹下面
###  (and ensure tf_inception_v3.py and my_inception_v3.py are in same path)
-----------



# 6.zero-shot
## 必须先做zero-shot实验,在my_inception_v3.py中确认设置如下
###  (must do zero-shot first ,please ensure these in my_inception_v3.py)
#### path_train=paths_source
#### path_test =paths_target_test
#### checkpoint_exclude_scopes="InceptionV3/Logits,InceptionV3/AuxLogits"
## 确认设置跟上面一样后，运行  my_inception_v3.py
## (after ensure , python ./my_inception_v3.py)




# 7.one-shot 
#### path_train=paths_target_train
#### path_test =paths_target_test
#### checkpoint_exclude_scopes=None 
## 使用上面的设置，运行   my_inception_v3.py
## use the above settiings,   
## python ./my_inception_v3.py





