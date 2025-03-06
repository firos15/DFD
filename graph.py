import numpy as np 
import matplotlib.pyplot as plt  
import pickle

def addlabels(models,values):
    for i in range(len(models)):
        plt.text(i,values[i],values[i])


def plot1():

    # creating the dataset 
    data = {'Random Forest':100,'KNN':100,'SVM':100} 
    models = list(data.keys()) 
    values = list(data.values()) 
    
    fig = plt.figure(figsize = (10, 5)) 
    
    # creating the bar plot 
    plt.bar(models, values, color ='green', width = 0.4)

    addlabels(models,values)
    
    plt.xlabel("Models") 
    plt.ylabel("Accuracy") 
    plt.title("Performance Graph(400images(200-Normal Images,200-Drugs Abuser Images))")
    plt.savefig('Models Performance Graph.png')
    plt.show() 

plot1()