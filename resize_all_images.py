import fire
from scipy.misc import imread,imresize,imsave
import os



def create_paths(dataset_path):
    #dataset_path    ='/home/gui/work2017/Research_of_transfer_Learning/dataSets/office-dataset/'
    cate_Li=["back_pack", "bike_helmet", 'bottle', 'desk_lamp', 'desktop_computer', 'file_cabinet', 'keyboard',
            'laptop_computer', 'mobile_phone', 'mouse', 'printer', 'projector', 'ring_binder', 'ruler', 'speaker','trash_can']
    # create 4 dirs
    paths_source=[];
    paths_source_train=[];
    paths_source_test =[];

    paths_target=[];
    paths_target_train=[];
    paths_target_test =[];
    for cate in cate_Li:
        dir0=dataset_path+'amazon_source/'+cate+'/'
        dir1=dataset_path+'amazon_source_train/'+cate+'/'
        dir2=dataset_path+'amazon_source_test/'+cate+'/'
        dir3=dataset_path+'webcam_target/'+cate+'/'
        dir4=dataset_path+'webcam_target_train/'+cate+'/'
        dir5=dataset_path+'webcam_target_test/'+cate+'/'
        paths_source.append(dir0)
        paths_source_train.append(dir1)
        paths_source_test.append(dir2)
        paths_target.append(dir3)
        paths_target_train.append(dir4)
        paths_target_test.append(dir5)
    return paths_source,paths_source_train,paths_source_test,paths_target,paths_target_train,paths_target_test



def get_data(paths,size):
    for i in range(len(paths)):
        files_list=os.listdir(paths[i])
        for files in files_list:
            if not os.path.isdir(files):
                img = imread(paths[i]+'/'+files)
                img=imresize(img,(size,size))
                imsave(paths[i]+'/'+files,img)


def main(dataset_path,size):
    paths_source,paths_source_train,paths_source_test,paths_target,paths_target_train,paths_target_test=create_paths(dataset_path)
    get_data(paths_source,size)
    get_data(paths_source_train,size)
    get_data(paths_source_test,size)
    get_data(paths_target,size)
    get_data(paths_target_train,size)
    get_data(paths_target_test,size)



if __name__=='__main__':
    fire.Fire(main)
