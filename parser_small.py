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
    path = struct()
    path.data_folder = 'TeleOpVRSession_2018-02-05_15-44-11'
    path.data_name = '_SessionStateData.proto'
    save_path_train = 'train/'
    save_path_test = 'dev/'
    
    data = parse_protobufs(path)
    data_dict_x = {}#rgb classified by object
    data_dict_d = {}#rgb classified by object
    data_dict_y = {}#rgb classified by object
    batch_dict = {}
    
    # example data extraction of x value of object/item 0 in training example 0: data.states[0].items[0].x
    num_examples = len(data.states) # number or screenshots
    num_items = []  # number of items in each example
    labels = []
    labels_rgb = []
    X_rgb = np.empty([0,299,299,3])
    
    X_d = np.empty([0,299,299])
    batch_size = 32
    
    # format labels into n x 6 array
    for i in range(10000):
        num_items.append(len(data.states[i].items))
        img_name = str(data.states[i].snapshot.name)
        depth_name = img_name[:-4] + '-Depth.jpg'

        # read in rgb and depth images 
        rgb_img_in = cv2.imread(img_name, 1)
        depth_img_in = cv2.imread(depth_name, 0)
        
        # render color to depth image using heatmap
        depth_img_raw = cv2.applyColorMap(depth_img_in, cv2.COLORMAP_JET)
                
        # mean and std scale on both rgb and d    
        #rgb_img_raw = [scale(rgb_img_in[i], axis=0, with_mean=True, with_std=True, copy=True ) for i in range(len(rgb_img_raw)) ]        
        #depth_img_raw = [scale(depth_img_raw[i], axis=0, with_mean=True, with_std=True, copy=True ) for i in range(len(depth_img_raw)) ]
        rgb_mean = np.mean(rgb_img_in)
        rgb_std = np.std(rgb_img_in) ** 2
        rgb_img_raw = [(rgb_img_in[i]-rgb_mean)/rgb_std for i in range(len(rgb_img_in))]
                                                                        
        depth_mean = np.mean(depth_img_raw)
        depth_std = np.std(depth_img_raw) ** 2
        depth_img_raw = [(depth_img_raw[i]-depth_mean)/depth_std for i in range(len(depth_img_raw))]                                                              
        
        # add a new axis to them to indicate which snapshot index for each image
        rgb_img = np.expand_dims(rgb_img_raw, axis=0)
        depth_img = np.expand_dims(depth_img_raw, axis = 0)
        
        #print("RGB image shape = ",np.shape(rgb_img))
        #print("depth image shape = ",np.shape(depth_img))
        
        #print(depth_img.shape)
        
        for j in range(num_items[i]):
            # input data (X)
            #X_rgb = np.vstack([X_rgb, rgb_img])
            #X_d = np.vstack([X_d, depth_img])
            
            item_id = str(data.states[i].items[j].id)
            if item_id != '48':
                continue
            '''
            RGB label, classified by name
            input label (X)
            D label, classified by name
            input label (X)
            '''
            if item_id not in data_dict_x:
                data_dict_x[item_id] = np.empty([0,299,299,3])
                data_dict_d[item_id] = np.empty([0,299,299,3])
            #print("#####################")
            #print(item_id)
            #print(np.size(data_dict_x[item_id]))
            #print(np.size(rgb_img))
            data_dict_x[item_id] = np.vstack([data_dict_x[item_id], rgb_img])
            print(data_dict_d[item_id].shape,depth_img.shape)
            data_dict_d[item_id] = np.vstack([data_dict_d[item_id], depth_img])
            
            '''
            RGB-D label, classified by name
            Batch split
            '''
            if item_id not in batch_dict:
                batch_dict[item_id] = 0
                              
            # Output label (Y)
            rlabel = data.states[i].items[j]
            current_label = [data.states[i].snapshot.name, rlabel.x, rlabel.y, rlabel.z, rlabel.roll, rlabel.pitch, rlabel.yaw]
            #print data.states[i].items[j].id
            labels.append(current_label)
            '''
            RGB label
            '''
            current_label_rgb = [rlabel.x, rlabel.y, rlabel.z]
            labels_rgb.append(current_label_rgb)
            
            '''
            RGB label, classified by name
            Output label (Y)
            '''
            
            if item_id not in data_dict_y:
                data_dict_y[item_id] = []
            data_dict_y[item_id].append(current_label_rgb)
            
            if len(data_dict_x[item_id]) == batch_size:
                batch = batch_dict[item_id]
                if i < 8000:
                    tmp_path = save_path_train + item_id
                else:
                    tmp_path = save_path_test + item_id
                if not os.path.exists(tmp_path):
                    os.makedirs(tmp_path)
                           
                np.save(tmp_path +"/"+ str(batch) +"_x.npy", data_dict_x[item_id])
                np.save(tmp_path +"/"+ str(batch) +"_d.npy", data_dict_d[item_id])
                np.save(tmp_path +"/"+ str(batch) +"_y.npy", np.array(data_dict_y[item_id]))

                data_dict_x[item_id] = np.empty([0,299,299,3])
                data_dict_d[item_id] = np.empty([0,299,299,3])
                data_dict_y[item_id] = []
                batch_dict[item_id] = 1 + batch
                
    # convert to numpy array
    #y = np.array(labels)
    #y_rgb = np.array(labels_rgb)
    #print X_rgb.shape
    #print y_rgb.shape

    if save:
        save_path = 'train/'
        save_path_train = 'train/'
        save_path_test = 'dev/'
        
        #np.save(save_path + "X_rgb.npy", X_rgb)
        #np.save(save_path + "X_d.npy", X_d)
        #np.save(save_path + "y.npy", y)
        #np.save(save_path + "y_rgb.npy", y_rgb)
        
        #np.save(save_path + "x_train", X_rgb)
        #np.save(save_path + "y_train",y_rgb)
        
        '''
        RGB label, classified by name
        Output label (Y)
        
        for cur_snap in data_dict_x:
            x_snap = data_dict_x[cur_snap]
            y_snap = np.array(data_dict_y[cur_snap])
            x_train, x_test, y_train, y_test = sk.train_test_split(x_snap,y_snap,test_size=.3, random_state=42)
            np.save(save_path_train + cur_snap +"_x.npy", x_train)
            np.save(save_path_train + cur_snap +"_y.npy", y_train)
            np.save(save_path_test + cur_snap +"_x.npy", x_test)
            np.save(save_path_test + cur_snap +"_y.npy", y_test)
        '''
    #return X_rgb, X_d, y


if __name__ == '__main__':
    parse_data(save=True)
    #X_rgb, X_d, y = parse_data(save=True)
    #X = (np.concatenate((X_rgb,np.expand_dims(X_d, axis=3)), axis=3))
    #X_train, X_test, y_train, y_test = sk.train_test_split(X,y,test_size=.3, random_state=42)    # random_state=42 ensure indices are same for train/test set for X_rgb and X_d since they must match



