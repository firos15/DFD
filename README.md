# DFD
Drug Abusers Face Detection (Python)

Deep Learning Framework for Detection of an Illicit Drug Abuser Using Facial Image


Objective

To build a deep learning mechanism to detect drug abusers from their facial images.

GUI

GUI will be created using Python Tkinter module

	GUI will contain a page for giving input image
	The output (drug abuser/not) will be predicted on the same page 


Notes
We can  collect non abuser’s facial images from LFW dataset and we have to create drug abuser’s facial images for our project implementation. 

Steps

1) Image Preprocessing
	- Loading Image dataset
	- Perform image resizing
	- Splitting image dataset into training and testing set
2) Feature Extraction
	- Create DCNN Architecture
	- Input  preprocessed image dataset to DCNN model
	- Extract facial features using DCNN
	- Collect extracted facial features
3) Machine Learning Algorithms
a) RandomForest Model
		- Load RandomForest classifier
		- Training the model with  extracted facial features
		- Calculate accuracy
		- Save trained model
b) KNN Model
		- Load KNN classifier
		- Training the model with  extracted facial features
		- Calculate accuracy
		- Save trained model
c) SVM  Model
		- Load SVM classifier
		- Training the model with   extracted facial features
		- Calculate accuracy
		- Save trained model
4) Comparison of different model accuracy
5) Prediction
	- Input image
	- Preprocess image
	- Extract features using DCNN
	- Loading the saved trained model which have high accuracy
	- Prediction using the model
	- View result (Drug abuser/not)
  

Risks and Note
Accuracy of machine learning algorithms depends on quantity and quality of dataset used during training.The output will be predicted on the basis of dataset. Accuracy issues might occur. 


Hardware Specification
•	Processor: i5 or i7
•	RAM: 8GB (Minimum)
•	Hard Disk: 500GB or above
•	Mouse
•	Keyboard

Software Specification
•	Tool: Python IDLE, Visual Studio Code, Anaconda
•	Python: version3
•	Operating System: Windows 10
•	Front End: Python Tkinter



