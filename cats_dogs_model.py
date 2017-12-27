import tensorflow as tf
from cats_dogs_batch import minibatch_train,minibatch_test
import matplotlib.pyplot as plt
import numpy as np

tf.reset_default_graph()        #this code is for retraining 

# create placeholder
X=tf.placeholder(tf.float32,[None,128,128,3])
Y=tf.placeholder(tf.float32,[None,2])

keep_prob1 = tf.placeholder(tf.float32)  #use for dropout
keep_prob2= tf.placeholder(tf.float32)

# conv1
with tf.variable_scope('conv1') as scope:
    weights = tf.get_variable('weights',shape = [3,3,3,16],dtype = tf.float32,                  
                              initializer=tf.contrib.layers.xavier_initializer(seed = 0))
    
    biases = tf.get_variable('biases',shape = [16],dtype=tf.float32,
                             initializer=tf.constant_initializer(0.1))
    
    conv1 = tf.nn.relu(tf.nn.conv2d(X,weights,strides=[1,1,1,1],padding='SAME')+biases)
        
# pool1
with tf.variable_scope('pool1') as scope:
    pool1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],
                           padding = 'SAME',name='pool1')

# conv2
with tf.variable_scope('conv2') as scope:
    weights = tf.get_variable('weights',shape = [3,3,16,32],dtype = tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer(seed = 0))
    
    biases = tf.get_variable('biases',shape = [32],dtype=tf.float32,
                             initializer=tf.constant_initializer(0.1))
    
    conv2 = tf.nn.relu(tf.nn.conv2d(pool1,weights,strides=[1,1,1,1],padding='SAME')+biases)
   
# pool2
with tf.variable_scope('pool2') as scope:
     pool2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],
                            padding = 'SAME',name='pool2')
# conv3
with tf.variable_scope('conv3') as scope:
    weights = tf.get_variable('weights',shape = [3,3,32,32],dtype = tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer(seed = 0))
    
    biases = tf.get_variable('biases',shape = [32],dtype=tf.float32,
                             initializer=tf.constant_initializer(0.1))
    
    conv3 = tf.nn.relu(tf.nn.conv2d(pool2,weights,strides=[1,1,1,1],padding='SAME')+biases)
   
# pool3
with tf.variable_scope('pool3') as scope:
     pool3 = tf.nn.max_pool(conv3,ksize=[1,2,2,1],strides=[1,2,2,1],
                            padding = 'SAME',name='pool3')
    
pool3_flatten=tf.contrib.layers.flatten(pool3)  #flatten pool3 to shape of (128,16*16*32)

# fully connected layer 1 and dropout
fc1=tf.contrib.layers.fully_connected(pool3_flatten,1024,activation_fn=tf.nn.relu)

fc1_drop = tf.nn.dropout(fc1, keep_prob1)

# fully connected layer 2 and dropout
fc2=tf.contrib.layers.fully_connected(fc1_drop,64,activation_fn=tf.nn.relu)

fc2_drop = tf.nn.dropout(fc2, keep_prob2)

# fully connected layer 3                here activation_fn is None
fc3=tf.contrib.layers.fully_connected(fc2_drop,2,activation_fn=None)

# cost, train  
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = fc3, labels = Y))

train=tf.train.AdamOptimizer(0.001).minimize(cost)

# accuracy
correct_prediction = tf.equal(tf.argmax(fc3, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

train_batch_size=128   # minibatch for training
test_batch_size=500    # minibatch for test
seed=1                 # seed for shuffle training images,see cats_dogs_batch.py
costs=[]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(10000):
        seed=seed+1    # seed+1 every epoch to obtain different training batches
       
        # get the training images and labels 
        (train_imgs,train_labels)=minibatch_train(train_batch_size,seed)  
        
        _ , temp_cost = sess.run((train,cost),feed_dict={X:train_imgs,Y:train_labels,
                                                         keep_prob1:0.5,keep_prob2:0.8})
        if epoch % 50 == 0:
            # keep_prob1 and keep_prob2 should be 1.0 when calculating accuracy
            train_acc = accuracy.eval({X: train_imgs, Y: train_labels, 
                                       keep_prob1:1.0,keep_prob2:1.0})
    
            print("epoch: %d, cost: %f, training accuracy: %f"%(epoch,temp_cost,train_acc))
            
        if epoch % 10 == 0:
            # collect temp_cost for ploting the cost convergence figure
            costs.append(temp_cost)   
                
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.show()

    #test accuracy
    (test_imgs,test_labels)=minibatch_test(test_batch_size,seed)  
    test_acc = accuracy.eval({X:test_imgs, Y:test_labels, keep_prob1:1.0, keep_prob2:1.0})
    print("Test Accuracy:", test_acc)
