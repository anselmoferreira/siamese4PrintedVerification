# import the necessary packages

#build_siamese_model: Constructs the sister network components of the siamese network architecture
from pyimagesearch.siamese_network import build_siamese_model
#config: Stores our training configurations
from pyimagesearch import config
#utils: Holds our helper function utilities used to create image pairs, plot training history, and compute the Euclidean distance using Keras/TensorFlow functions
from pyimagesearch import utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
#Lambda: Takes our implementation of the Euclidean distances and embeds it inside the siamese network architecture itself
from tensorflow.keras.layers import Lambda
from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras import backend as K
#from keras_radam import RAdam

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config2 = ConfigProto()
config2.gpu_options.allow_growth = True
session = InteractiveSession(config=config2)

import pydotplus

from tensorflow.keras.utils import plot_model

#from tensorflow.keras.utils import plot_model
import pywt

    
def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def scheduler(epoch, lr):
  if epoch < 10:
    return lr
  else:
    return lr * tf.math.exp(-0.1)


# load MNIST dataset and scale the pixel values to the range of [0, 1]
print("[INFO] loading dataset...")

trainX=np.load("features_and_labels/energy/train1-images.npy") 
trainY=np.load("features_and_labels/energy/train1-labels.npy") 
#testX=np.load("test1-images.npy")
#testY=np.load("test1-labels.npy") 

trainX, trainY = shuffle(trainX, trainY)
#testX, testY = shuffle(testX, testY) 

trainX, testX, trainY, testY = train_test_split(trainX, trainY, test_size=0.30, stratify=trainY, random_state=42)
#x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.30, stratify=y_train, random_state=42)

trainX = trainX / 255.0
testX = testX / 255.0


# add a channel dimension to the images (?????)
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)


# prepare the positive and negative pairs
print("[INFO] preparing positive and negative pairs...")
(pairTrain, labelTrain) = utils.make_pairs(trainX, trainY)
(pairTest, labelTest) = utils.make_pairs(testX, testY)


# configure the siamese network
print("[INFO] building siamese network...")
imgA = Input(shape=config.IMG_SHAPE)
imgB = Input(shape=config.IMG_SHAPE)

#builds the sister network architecture, which serves as featureExtractor.
#even though there are two sister networks, we actually implement them as a single instance. 
#Essentially, this single network is treated as a feature extractor (hence why we named it featureExtractor). 
#The weights of the network are then updated via backpropagation as we train the network.
featureExtractor = build_siamese_model(config.IMG_SHAPE)

#Each image in the pair will be passed through the featureExtractor, resulting in a 48-d feature vector 
#Since there are two images in a pair, we thus have two 48-d feature vectors.
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)

# finally, construct the siamese network
distance = Lambda(utils.euclidean_distance)([featsA, featsB])

#The sigmoid activation function is used here because the output range of the function is [0, 1]. 
#An output closer to 0 implies that the image pairs are less similar (and therefore from different classes), 
#while a value closer to 1 implies they are more similar (and more likely to be from the same class).
outputs = Dense(1, activation="sigmoid")(distance)
model = Model(inputs=[imgA, imgB], outputs=outputs)

lr_reducer= ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
#An approach to stop training before the whole epochs are processed
#early_stopper=EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=20)
#Policy to save weights
model_checkpoint= ModelCheckpoint("weights.h5", monitor="val_accuracy", save_best_only=True, save_weights_only=True,mode='auto')
#callbacks
scheduler=tf.keras.callbacks.LearningRateScheduler(scheduler)
callbacks=[lr_reducer, model_checkpoint, scheduler]

# compile the model
# We use binary cross-entropy here because this is essentially a two-class classification problem — given a pair of input images, 
# we seek to determine how similar these two images are and, more specifically, if they are from the same or different classes.

print("[INFO] compiling model...")
#sgd = SGD()
#adam=opt = tf.keras.optimizers.Nadam(learning_rate=0.001)

#adamax=bad
#adam=good
#sgd=good
#rmsprop=good
#nadam=good
#adadelta=bad
#adagrad
#opt = RAdam()

model.compile(loss="binary_crossentropy", optimizer="adagrad", metrics="accuracy")
plot_model(model, to_file='output/model.png', show_shapes=True, show_layer_names=True,  rankdir="TB", expand_nested=True,  dpi=96)

# train the model
print("[INFO] training model...")
history = model.fit(
	[pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:],
	validation_data=([pairTest[:, 0], pairTest[:, 1]], labelTest[:]),
	batch_size=config.BATCH_SIZE, 
	epochs=config.EPOCHS)
	
# serialize the model to disk
print("[INFO] saving siamese model...")
model.save(config.MODEL_PATH)

# plot the training history
print("[INFO] plotting training history...")
utils.plot_training(history, config.PLOT_PATH)
