<<<<<<< HEAD
### 1.first  download the office-datasets  from  https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view                        
-----------
-----------

### 2.then build the one_hot and zero_hot dataset from original office-datasets
#### ./build_data_set.sh     dataSets/
-----------
-----------
### 3.download  inception_v3.ckpt  from  http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz


### 4.Dependent library
#### tensorflow
#### scipy
#### numpy
#### fire
=======
## 1.first  download the office-datasets  from  https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view                        
-----------
-----------

## 2.then build the one_hot and zero_hot dataset from original office-datasets
### ./build_data_set.sh     dataSets/
-----------
-----------
## 3.download  inception_v3.ckpt  from  http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz


## 4.Dependent library
### tensorflow
### scipy
### numpy
### fire
>>>>>>> 22b64def45f9776969700b9cb23bb657a4a2106e
-----------
-----------


<<<<<<< HEAD
### 5.change some area of my_inception_v3.py
#### before run ,please ensure the checkpoint_path and dataset_path is right 
=======
## 5.change some area of my_inception_v3.py
### before run ,please ensure the checkpoint_path and dataset_path is right 
>>>>>>> 22b64def45f9776969700b9cb23bb657a4a2106e
-----------
-----------


<<<<<<< HEAD
### 6.zero-shot
#### must do zero-shot first ,please ensure these in my_inception_v3.py
> path_train=paths_source
> path_test =paths_target_test
> checkpoint_exclude_scopes="InceptionV3/Logits,InceptionV3/AuxLogits"
#### after ensure , python ./my_inception_v3.py
=======
## 6.zero-shot
### must do zero-shot first ,please ensure these in my_inception_v3.py
#### path_train=paths_source
#### path_test =paths_target_test
#### checkpoint_exclude_scopes="InceptionV3/Logits,InceptionV3/AuxLogits"
### after ensure , python ./my_inception_v3.py
>>>>>>> 22b64def45f9776969700b9cb23bb657a4a2106e
-----------
-----------


<<<<<<< HEAD
### 7.one-shot 
#### please ensure these in my_inception_v3.py
> path_train=paths_target_train
> path_test =paths_target_test
> checkpoint_exclude_scopes=None 
#### after ensure   python ./my_inception_v3.py
=======
## 7.one-shot 
### please ensure these in my_inception_v3.py
#### path_train=paths_target_train
#### path_test =paths_target_test
#### checkpoint_exclude_scopes=None 
### after ensure   python ./my_inception_v3.py
>>>>>>> 22b64def45f9776969700b9cb23bb657a4a2106e
-----------
-----------

