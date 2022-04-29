import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pywt
import cv2
from tensorflow.keras.preprocessing.image import array_to_img
import cv2

#images: The images in our dataset
#labels: The class labels associated with the images
#In the case of the MNIST dataset, our images are the 
#digits themselves, while the labels are the class label (0-9) 
#for each image in the images array.
np.random.seed(42)
def dwt_image(img):
    size_image=img.shape[2]
    
    coeffs2 = pywt.dwt2(img[:,:,0],'bior1.3')
    LL, (LH, HL, HH) = coeffs2
    print(HH.shape)
    input()
    resulting_image = np.zeros((HH.shape[0],HH.shape[1],img.shape[2]))
    for channel in range(size_image):
	
	     coeffs2 = pywt.dwt2(img[:,:,channel],'bior1.3')
	     LL, (LH, HL, HH) = coeffs2
	   
	     resulting_image[:,:,channel]=HH

    return resulting_image


def make_pairs(images, labels):
	# initialize two empty lists to hold the (image, image) pairs and
	# labels to indicate if a pair is positive or negative
	pairImages = []
	pairLabels = []
	
	# calculate the total number of classes present in the dataset
	# and then build a list of indexes for each class label that
	# provides the indexes for all examples with a given label
	#np.unique function finds all unique class labels in our labels list. 
	#Taking the len of the np.unique output yields the total number of unique class labels in the dataset. 
	#In the case of the MNIST dataset, there are 10 unique class labels, corresponding to the digits 0-9.
	
	numClasses = len(np.unique(labels))
	
		
	#idxs have a list of indexes that belong to each class
	idx = [np.where(labels == i)[0] for i in range(0, numClasses)]
	
	#let’s now start generating our positive and negative pairs
	for idxA in range(len(images)):
		
		# grab the current image and label belonging to the current
		# iteration
		currentImage = images[idxA]
		
		label = labels[idxA]
		
		# randomly pick an image that belongs to the *same* class
		# label
		idxB = np.random.choice(idx[label])
		posImage = images[idxB]
		#posImage=dwt_image(posImage)
		
		# prepare a positive pair and update the images and labels
		# lists, respectively
		pairImages.append([currentImage, posImage])
		pairLabels.append([1])
		
               #grab the indices for each of the class labels *not* equal to
               #the current label and randomly pick an image corresponding
               #to a label *not* equal to the current label
		negIdx = np.where(labels != label)[0]
		negImage = images[np.random.choice(negIdx)]
		# prepare a negative pair of images and update our lists
		pairImages.append([currentImage, negImage])
		pairLabels.append([0])
        #return a 2-tuple of our image pairs and labels
	pairImages=np.array(pairImages)
	
	return (pairImages, np.array(pairLabels))
      
def euclidean_distance(vectors):
	# unpack the vectors into separate lists
	(featsA, featsB) = vectors
	# compute the sum of squared distances between the vectors
	sumSquared = K.sum(K.square(featsA - featsB), axis=1,
		keepdims=True)
	# return the euclidean distance between the vectors
	return K.sqrt(K.maximum(sumSquared, K.epsilon()))
	
def plot_training(H, plotPath):
	# construct a plot that plots and saves the training history
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H.history["loss"], label="train_loss")
	plt.plot(H.history["val_loss"], label="val_loss")
	plt.plot(H.history["accuracy"], label="train_acc")
	plt.plot(H.history["val_accuracy"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(plotPath)

