import numpy as np
import cv2
import sklearn.model_selection as sk
from analysis import parse_protobufs
from ipdb import set_trace as debug
import os
from sklearn.preprocessing import scale

class struct():
    pass

def parse_data(save=True):

    for c in range(1,5):
        path = struct()
        path.data_folder = 'TeleOpVRSession_2018-03-07_14-38-06_Camera'+str(c)
        #path.data_folder ='TeleOpVRSession_2018-02-05_15-44-11'
        path.data_name = '_SessionStateData.proto'
        save_path_train = 'train/camera'
        save_path_test = 'dev/camera'
        data = parse_protobufs(path)



        # example data extraction of x value of object/item 0 in training example 0: data.states[0].items[0].x
        num_examples = len(data.states) # number or screenshots
        num_items = []  # number of items in each example


        # format labels into n x 6 array
        for i in range(2000):
            print("loop ", i)
            num_items.append(len(data.states[i].items))
            img_name = str(data.states[i].snapshot.name)
            depth_name = img_name[:-4] + '-Depth.jpg'

            # read in rgb and depth images 
            rgb_img_in = cv2.imread(img_name, 1)
            depth_img_in = cv2.imread(depth_name, 0)

            # render color to depth image using heatmap
            depth_img_raw = cv2.applyColorMap(depth_img_in, cv2.COLORMAP_JET)

            # mean and std scale on both rgb and d    
            rgb_mean = np.mean(rgb_img_in)
            rgb_std = np.std(rgb_img_in) ** 2
            rgb_img_raw = [(rgb_img_in[i]-rgb_mean)/rgb_std for i in range(len(rgb_img_in))]

            depth_mean = np.mean(depth_img_raw)
            depth_std = np.std(depth_img_raw) ** 2
            depth_img_raw = [(depth_img_raw[i]-depth_mean)/depth_std for i in range(len(depth_img_raw))]                                                              

            # add a new axis to them to indicate which snapshot index for each image
            rgb_img = np.expand_dims(rgb_img_raw, axis=0)
            depth_img = np.expand_dims(depth_img_raw, axis = 0)


            for j in range(num_items[i]):
                # input data (X)
                #X_rgb = np.vstack([X_rgb, rgb_img])
                #X_d = np.vstack([X_d, depth_img])

                item_id = str(data.states[i].items[j].id)
                if item_id != '35':
                    continue
                '''
                RGB label, classified by name
                input label (X)
                D label, classified by name
                input label (X)
                '''
                
                if i<1600:
                    try:
                        train_x
                    except NameError:
                        train_x = np.empty([0,299,299,3])
                        train_d = np.empty([0,299,299,3])
                        train_y = np.empty([0,3])
                        
                    
                    train_x = np.vstack([train_x, rgb_img])
                    train_d = np.vstack([train_d, depth_img])

                    '''
                    RGB-D label, classified by name
                    Batch split
                    '''

                    # Output label (Y)
                    rlabel = data.states[i].items[j]
                    current_label = [data.states[i].snapshot.name, rlabel.x, rlabel.y, rlabel.z, rlabel.roll, rlabel.pitch, rlabel.yaw]
                    current_label_rgb = [rlabel.x, rlabel.y, rlabel.z]
                    train_y = np.vstack([train_y, current_label_rgb])
                    print("train_x:",train_x.shape)
                    print("train_d:",train_d.shape)
                    print("train_y:",train_y.shape)
                    
                    
                                    
                    
                else: 
                    try:
                        dev_x
                    except NameError:
                        dev_x = np.empty([0,299,299,3])
                        dev_d = np.empty([0,299,299,3])
                        dev_y = np.empty([0,3])

                    dev_x = np.vstack([dev_x, rgb_img])
                    dev_d = np.vstack([dev_d, depth_img])

                    '''
                    RGB-D label, classified by name
                    Batch split
                    '''

                    # Output label (Y)
                    rlabel = data.states[i].items[j]
                    current_label = [data.states[i].snapshot.name, rlabel.x, rlabel.y, rlabel.z, rlabel.roll, rlabel.pitch, rlabel.yaw]
                    current_label_rgb = [rlabel.x, rlabel.y, rlabel.z]
                    dev_y = np.vstack([dev_y, current_label_rgb])
                    print("dev_x:",dev_x.shape)
                    print("dev_d:",dev_d.shape)
                    print("dev_y:",dev_y.shape)

                    
    if not os.path.exists(save_path_train):
                    os.makedirs(save_path_train)
    if not os.path.exists(save_path_test):
                    os.makedirs(save_path_test)
                           
    np.save(save_path_train +"X.npy", train_x)
    np.save(save_path_train +"D.npy", train_d)
    np.save(save_path_train +"Y.npy", train_y)

    np.save(save_path_test +"X.npy", dev_x)
    np.save(save_path_test +"D.npy", dev_d)
    np.save(save_path_test +"Y.npy", dev_y)


if __name__ == '__main__':
    parse_data(save=True)



