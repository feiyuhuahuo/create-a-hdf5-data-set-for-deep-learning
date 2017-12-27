import h5py
import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# The .hdf5 file may be very large depending on the image size. A batch of images 
# are read at one time and are used for train or test. The corresponding indexes 
# are shuffled using "np.random.seed(seed)" every epoch.
#------------------------------------------------------------------------------

def minibatch_train(train_batch_size,seed):
    
    hdf5_path = 'E:/All about AI/cats_dogs_128.hdf5'
    dataset = h5py.File(hdf5_path, "r")
    
    np.random.seed(seed)
    
    # shuffle indexes,int numbers range from 0 to 20000
    permutation = list(np.random.permutation(20000))
    
    # get the "train_batch_size" indexes    
    train_batch_index=permutation[0:train_batch_size]
    
    # the shape of "train_labels" now is (20000,1)
    train_labels=np.array(dataset["train_labels"]).reshape(20000,-1)
    
    # get the corresponding labels according "train_batch_index"
    train_batch_labels=train_labels[train_batch_index]

    train_batch_labels= np.eye(2)[train_batch_labels.reshape(-1)] #convert to one_hot code
    
    train_batch_imgs=[]
    for i in range(train_batch_size):
        img=(dataset['train_img'])[train_batch_index[i]]
        img=img/255.
        train_batch_imgs.append(img)    
    train_batch_imgs=np.array(train_batch_imgs)
    
    dataset.close()
    
    return(train_batch_imgs,train_batch_labels)

def minibatch_test(test_batch_size,seed): 
     
    hdf5_path = 'E:/All about AI/cats_dogs_128.hdf5'
    dataset = h5py.File(hdf5_path, "r")
    
    np.random.seed(seed)
    
    permutation = list(np.random.permutation(5000))    
    test_batch_index= permutation[0:test_batch_size]  
    test_labels= np.array(dataset["test_labels"]).reshape(5000,-1)
    test_batch_labels= test_labels[test_batch_index]
    test_batch_labels= np.eye(2)[test_batch_labels.reshape(-1)]
    
    test_batch_imgs=[]
    for i in range(test_batch_size):
        img=(dataset['test_img'])[test_batch_index[i]]
        img=img/255.
        test_batch_imgs.append(img)    
    test_batch_imgs=np.array(test_batch_imgs)
    
    dataset.close()  
    
    return(test_batch_imgs,test_batch_labels)
