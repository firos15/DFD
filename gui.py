import tkinter as tk
from tkinter import filedialog
import tkinter.messagebox
import customtkinter
from PIL import Image, ImageTk
from tkinter import Label,Canvas,Entry,PhotoImage,Button,SUNKEN,FLAT,GROOVE,Toplevel,messagebox
import numpy as np
from os import listdir
from os.path import isfile, join
import cv2
import pandas as pd
import os
from tensorflow.keras.backend import concatenate
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import tensorflow as tf
import pandas as pd
import pickle
import joblib
from tensorflow.keras.models import load_model
customtkinter.set_appearance_mode("Light")
global filename,label1
filename=0
label1=0
t=customtkinter.CTk()
t.title("Drug Abuser Face Detection")
t.minsize(900,550)
t.maxsize(900,550)

frame1 = customtkinter.CTkFrame(t,
							   width=465,
							   height=550,
							   corner_radius=0,fg_color='#ffc14d')
frame1.place(x=0,y=0)
frame1i = customtkinter.CTkFrame(frame1,width=415,
							   height=470,corner_radius=20,fg_color='#ffdb99')
frame1i.place(x=25,y=40)
im=ImageTk.PhotoImage(Image.open("Extra/r.png").resize((380,380)))
icon= Label(frame1i, image=im, bg='#ffdb99')
icon.image = im
icon.place(x=8, y=45)
frame2= customtkinter.CTkFrame(t,width=300,
							   height=350,corner_radius=20)
frame2.place(x=531,y=100)
frame2i=customtkinter.CTkFrame(frame2,width=250,
							   height=310,corner_radius=20)
frame2i.place(x=25,y=20)
frame2i1=customtkinter.CTkFrame(frame2i,width=230,
							   height=170,corner_radius=20)
frame2i1.place(x=10,y=70)
imw=ImageTk.PhotoImage(Image.open("Extra/fg.png").resize((45, 45)))
iconww= Label(frame2i1, image=imw, bg='#d1d5d8')
iconww.image = imw
iconww.place(x=91, y=60)
global model3,models1,models2
model3=pickle.load(open('Model/svm.pkl', 'rb'))
models1=load_model('Model/s1.h5')
models2=load_model('Model/s2.h5')
def pred():
	global filename,model3,models1,models2
	if(filename==0):
		messagebox.showinfo("","Please select an image first.")
	else:
		f=str(filename)
		#################################################(Bounding Box)
		img = cv2.imread(f)
		saliency = cv2.saliency.StaticSaliencyFineGrained_create()
		(success, saliencyMap) = saliency.computeSaliency(img)
		saliencyMap = (saliencyMap * 255).astype("uint8")
		cv2.imwrite('Prediction Image/'+'b'+'.jpg',saliencyMap)
		################################################(Face)  
		height=227
		width=227
		img1=cv2.imread(f)
		img1=cv2.resize(img1,(height,width))
		img1=np.array(img1)
		#img1=np.expand_dims(img1, axis=0)
		img1.reshape(-1, 227, 227, 3)
		path2='Prediction Image/b.jpg'
		img2=cv2.imread(path2)
		img2=cv2.resize(img2,(height,width))
		img2=np.array(img2)
		#img2=np.expand_dims(img2, axis=0)
		img2.reshape(-1, 227, 227, 3)
		ip=[]
		a=img1
		a=np.expand_dims(a, axis=0)
		model11=models1.predict(a)
		b=img2
		b=np.expand_dims(b, axis=0)
		model22=models2.predict(b)
		combined=np.concatenate([model11,model22])
		print(combined)
		ip.append(combined)
		print(ip)
		ip=np.array(ip)
		print(ip.shape)
		ip = ip.reshape(ip.shape[0],-1)
		print(ip.shape)
		output=model3.predict(ip)
		print(output)
		output=output[0]
		if output==1:
			result='Not a drug user.'
		elif output==0:
			result='Drugs using person detected'
	messagebox.showinfo("",result)
def refresh():
	global label1
	if os.path.exists('Prediction Image/b.jpg'):
		os.remove('Prediction Image/b.jpg')
	if(label1==0):
		messagebox.showinfo("","Please select an image first.") 
	else:
		label1.destroy()
		label1=0
def browseFiles(fr):
	global filename,label1
	filename = filedialog.askopenfilename(initialdir ="F:/(project)_Drug face detection/For Testing/",title = "Select Image",
		filetypes=[('Image Files', ['.jpeg', '.jpg', '.png', '.gif','.tiff', '.tif', '.bmp'])])
	filename=str(filename)
	image = Image.open(filename)
	resize_image = image.resize((215, 135))
 
	img = ImageTk.PhotoImage(resize_image)

	label1 = Label(fr,image=img)
	label1.image = img
	label1.place(x=5,y=15)
button = customtkinter.CTkButton(frame2i,text="SELECT IMAGE",border_width=3,fg_color="#94b8b8",hover_color="#cce6ff",cursor='hand2',command=lambda:browseFiles(frame2i1))
button.place(x=50, y=20)
button1 = customtkinter.CTkButton(frame2i, text="Predict",border_width=2,text_color="white",corner_radius=20,cursor='hand2',width=20,height=35,command=lambda:pred())
button1.place(x=83,y=260)
rimg=ImageTk.PhotoImage(Image.open("Extra/refresh.png").resize((20,20)))
refreshb= customtkinter.CTkButton(frame2i, image=rimg, text="", width=30, height=30,
												corner_radius=10, fg_color="gray40", hover_color="gray25",
												cursor='hand2',command=lambda:refresh())
refreshb.place(x=196,y=19)
t.mainloop()