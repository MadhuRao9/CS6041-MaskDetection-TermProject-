# 1 Installing the dependencies
#  tensorflow>=1.15.2
# keras==2.3.1
# imutils==0.5.3
#  numpy==1.18.2
#  opencv-python==4.2.0.*
#  matplotlib==3.2.1
#  scipy==1.4.1

# 2  Datasets
# Will create 2 datasets 
# one with images wearing fase mask and 
# other with images not wearing mask 


# 3)  Data preprocessing


# import the necessary packages
import os
import numpy as np
from imutils import paths
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.layers import Flatten
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelBinarizer




# Initial Learning Rage - better to have very low value 

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-6    # When LR is less, then Loss will be calculated accurately
EPOCHS = 28       # Number of times we train & retrain the model 
BS = 36

# Directory is where our image files are present. 
# Categories is the names of two folders containing images


DIRECTORY = r"C:\Mask Detection\CODE\Face-Mask-Detection-master\dataset"
CATEGORIES = ["with_mask", "without_mask"]

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")   #  To see whats happending during training.

# will create two empty lists

data = []      #  will append all image arrays into this array
labels = []    # Will append all those corresponding labels (with mask & without mask) 

#  Looping through Dataset Categories, first whrough With Mask, & next
# with os.path.join --> Joining the Directory & category.  

# First we will lopp with_mask and then loop through without_mask
# os.listdir --> list down all images in that particular dir. (first with mask) 
# with os.path.join --> join path of withmask to a particular image

#The function Load_img is coming from tensorflow.keras.preprocessing.image
# load_img --> loads image path & we convert all images uniformly to size 224 x 224 for a better model_selection
# and save this to an variable called image

# img_to_array function will convert the image to array

# Then use preprocess_input --> we are using mobilenets for this model
# (if we use MobileNetV2 in our model, then we need to use preprocess_input function)

#  data.append(image) --> Now the array data has all the data converted into numerical values and appended into it.
#  to the predefined list data 

# labels.append(category) then we append lables / categories( ie: either withmask or without Mask)to each of the images  
# Now after apppending data[] is in numbers and lables[] in alphabets/words, 
# so convert lable into 1 & 0 array. For this we use LabelBinarizer function.
# 


for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        image = load_img(img_path, target_size=(224, 224))
    	img_path = os.path.join(path, img)
    	image = preprocess_input(image)
    	image = img_to_array(image)
    	
    	data.append(image)
    	labels.append(category)

# Here we convert "with_mask","without_mask" into categoricial variables.  ie: to 
#  Here using LabelBinarizer we convert even the values in array labels 
# from alphabets/words into numerical values 1s & 0s  ie: categoricial variables
# this is function from sklearn.preprocessing library 

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

#  Now we need to convert both the arrays (in 0's & 1's ) into an Numpy Array

data = np.array(data, dtype="float32")  # 
labels = np.array(labels)

# Here test size is 20% of total data size and remaining 80% for training purpose
# 

(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# 4 Here use the preprocessed data and start the TRAINING

# Heere rather than passing the data through convolution, we pass it to mobilenet (as our data is limited) 

# ImageDataGenerator --> creates data documentation, will create many images from one image, by flipping,
# rotating the image, So will create more dataset with this.
# Weights = "Imagenet" --> ie: using a pretrained images, and those weights will be initialized for us.
# include_top = false --> wether to connect fully connected layer, will do it later.
# So the base model is construced 


# construct the training image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# First construct a Headmodel and passing basemodel's output to it.
# Average Pooling is 7x7,.  And then 
# relu is a go-to-activation function for non-linear use cases (like images) 
# layer --> Relu is 
# finla output is 2 (with and without mask), activattion function is softmax or sigmoid


# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the head of the network
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# serialize the model to disk
print("[INFO] saving mask detector model...")
model.save("mask_detector.model", save_format="h5")

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")