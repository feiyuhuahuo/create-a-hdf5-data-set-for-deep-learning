# create-a-hdf5-data-set-for-deep-learning
Create your own data set with python library h5py and a simple example for image recognition

1.The famous data set "cats vs dogs" data set is used to create .hdf5 file with the python library: h5py.

2.The data set contains 12500 dog pictures and 12500 cat pictures. All the images are shuffled randomly and 20000 images are used to train,   5000 images are used to test. 

3.The images can be resized to different sizes but the size of the .hdf5 file differs very far depending on the size of the images. The       file is 1.14G when the size of the images is (128,128) and 4.57G for (256,256), 18.3G for (512,512). If you are going to modify the code,   please pay attention to the size of the training batch. A simple 6 layers model is applied to train these images.

4.The training accuracy is about 97% after 2000 epochs.

5.Forget my poor English.

● create h5 file.py： use your own images to create a hdf5 data set.

● cats_dogs_batch.py: read your hdf5 file and prepare the train batch, test batch.

● cats_dogs_model.py: a simple 6 layers model using the created hdf5 file.

References: More detailed tutorial for creating the hdf5 file can be found here: http://machinelearninguru.com/deep_learning/data_preparation/hdf5/hdf5.html
