# VGG16_keras_tf1
This is a VGG16 model build from keras, using tensorflow r1.14, and using the distributed strategy "tf.distribute.experimental.MultiWorkerMirroredStrategy()"



to run this program, you need to download the dataset in the same floder first.  
  using this command:
  
    wget --no-check-certificate https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip -O cats_and_dogs_filtered.zip
  
  then unzip it in the same floder:
  
     unzip cats_and_dogs_filtered.zip
     
  after that, run w0.py and w1.py parallelly.
