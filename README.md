# keras_learning
In this repository, I will make notes about keras application

Recently, I tried to train a deep learning model for tumor or non-tumor classifcation from pathology images. I used 'keras.preprocessing.image.ImageDataGenerator(...)' class to perform image augumentation. As we could find from keras website (https://keras.io/preprocessing/image/), there are many arugments that can be set for the function ImageDataGenerator(). 

Here there is a question: we could perfrom many different image augumentations together, but how about the orders of different augumentations??
To get answer for this question, I trid to read keras website and search online, but I am still not sure about it. Therefore, I did some experiments and read the original code image.py to verify it here.
