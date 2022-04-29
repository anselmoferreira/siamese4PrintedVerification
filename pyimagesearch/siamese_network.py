# import the necessary packages
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Add

#inputShape: The spatial dimensions (width, height, and number channels) of input images. For the MNIST dataset, our input images will have the shape 28x28x1.
#embeddingDim: Output dimensionality of the final fully-connected layer in the network.
def build_siamese_model(inputShape, embeddingDim=48):
	# specify the inputs for the feature extractor network
	inputs = Input(inputShape)
	
	# define the first set of CONV => RELU => POOL => DROPOUT layers
	#x = Conv2D(64, (2, 2), padding="same", activation="relu")(inputs)
	#x = MaxPooling2D(pool_size=(2, 2))(x)
	#x = Dropout(0.3)(x)
	## second set of CONV => RELU => POOL => DROPOUT layers
	#x = Conv2D(64, (2, 2), padding="same", activation="relu")(x)
	#x = MaxPooling2D(pool_size=2)(x)
	#x = Dropout(0.3)(x)
	
	## first set of CONV => RELU => RESID=> POOL => DROPOUT layers
	first_conv1 = Conv2D(32, (3, 3), padding="same")(inputs)
	first_batch_norm1=BatchNormalization()(first_conv1)
	first_act1= LeakyReLU()(first_batch_norm1)
	
	second_conv1 = Conv2D(32, (5, 5), padding="same")(inputs)
	second_batch_norm1=BatchNormalization()(second_conv1)
	second_act1= LeakyReLU()(second_batch_norm1)
	
	third_conv1 = Conv2D(32, (7, 7), padding="same")(inputs)
	third_batch_norm1=BatchNormalization()(third_conv1)
	third_act1= LeakyReLU()(third_batch_norm1)
	
	residual_block1= Add()([first_act1, second_act1, third_act1])
	pool1 = MaxPooling2D(pool_size=(2, 2))(residual_block1)
	dropout1 = Dropout(0.3)(pool1)
	
	
	#receiver Convolutional layer
	receiver1_conv = Conv2D(32, (3, 3), padding="same")(dropout1)
	receiver1_batch_norm=BatchNormalization()(receiver1_conv)
	act_receiver1=LeakyReLU()(receiver1_batch_norm)
	
	## second set of CONV => BN=> RELU => RESID=> POOL => DROPOUT layers
	first_conv2 = Conv2D(32, (3, 3), padding="same")(act_receiver1)
	first_batch_norm2=BatchNormalization()(first_conv2)
	first_act2= LeakyReLU()(first_batch_norm2)
	
	second_conv2 = Conv2D(32, (5, 5), padding="same")(act_receiver1)
	second_batch_norm2=BatchNormalization()(second_conv2)
	second_act2= LeakyReLU()(second_batch_norm2)
	
	third_conv2 = Conv2D(32, (7, 7), padding="same")(act_receiver1)
	third_batch_norm2=BatchNormalization()(third_conv2)
	third_act2= LeakyReLU()(third_batch_norm2)
	
		
	residual_block2= Add()([first_act2, second_act2, third_act2])
	pool2 = MaxPooling2D(pool_size=(2, 2))(residual_block2)
	dropout2 = Dropout(0.3)(pool2)
		
	#receiver Convolutional layer
	receiver2_conv = Conv2D(32, (3, 3), padding="same")(dropout2)
	receiver2_batch_norm=BatchNormalization()(receiver2_conv)
	act_receiver2=LeakyReLU()(receiver2_batch_norm)
	
	## last set of CONV => BN=> RELU => RESID=> POOL => DROPOUT layers
	first_conv3 = Conv2D(32, (3, 3), padding="same")(act_receiver2)
	first_batch_norm3=BatchNormalization()(first_conv3)
	first_act3= LeakyReLU()(first_batch_norm3)
	
	second_conv3 = Conv2D(32, (5, 5), padding="same")(act_receiver2)
	second_batch_norm3=BatchNormalization()(second_conv3)
	second_act3= LeakyReLU()(second_batch_norm3)
	
	third_conv3 = Conv2D(32, (7, 7), padding="same")(act_receiver2)
	third_batch_norm3=BatchNormalization()(third_conv3)
	third_act3= LeakyReLU()(third_batch_norm3)
			
	residual_block3= Add()([first_act3, second_act3, third_act3])
	pool3 = MaxPooling2D(pool_size=(2, 2))(residual_block3)
	dropout3 = Dropout(0.3)(pool3)
		
	#last receiver Convolutional layer (botar tudo 3x3 novamente)
	receiver3_conv = Conv2D(32, (3, 3), padding="same")(dropout3)
	receiver3_batch_norm=BatchNormalization()(receiver3_conv)
	act_receiver3=LeakyReLU()(receiver3_batch_norm)
	
	
	
	
	
	# prepare the final outputs
	#pooledOutput = GlobalAveragePooling2D()(bn_3)
	pooledOutput = GlobalAveragePooling2D()(act_receiver3)
	outputs = Dense(embeddingDim)(pooledOutput)
	# build the model
	model = Model(inputs, outputs)

	# return the model to the calling function
	return model
