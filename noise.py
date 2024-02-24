#This program is for preprocessing the dataset by adding noise to raw data


import pandas as pd
import numpy as np
from random import randrange


dataset = pd.read_csv('wikipedia_color_names.csv', header=None, names=['label', 'hex', 'r', 'g', 'b','hue1','hue2','hue3'])
#print(dataset)

np.random.seed(5)
X = np.random.rand(len(dataset)*10, 3)

for i in range(len(dataset)*10):
    X[i][0] = int((X[i][0]-0.5)*2*5)
    X[i][1] = int((X[i][1]-0.5)*2*5)
    X[i][2] = int((X[i][2]-0.5)*2*5)

iFull = len(dataset)*10
for i in range(len(dataset)*10):
    
    value = randrange(0,len(dataset))
    r = int(dataset.loc[value]['r']+X[i][0])
    g = int(dataset.loc[value]['g']+X[i][1])
    b = int(dataset.loc[value]['b']+X[i][2])

    if r<0:
        r=0
    elif r>255:
        r=255
    if g<0:
        g=0
    elif g>255:
        g=255
    if b<0:
        b=0
    elif b>255:
        b=255
    print(i/iFull*100)
    dataset.loc[len(dataset)] = {'label':dataset.loc[value]['label'], 'hex':dataset.loc[value]['hex'],'r':r, 'g':g, 'b':b,'hue1':dataset.loc[value]['hue1'],'hue2':dataset.loc[value]['hue2'],'hue3':dataset.loc[value]['hue3']}                            

#print(dataset.loc[0]['color_name'])


##example 
##dataset.loc[len(dataset)] = {'color_name':dataset.loc[value]['color_name'], 'label':dataset.loc[value]['label'], 'hex':dataset.loc[value]['hex'],'r':dataset.loc[value]['r']+X[i][0], 'g':dataset.loc[value]['g']+X[i][1], 'b':dataset.loc[value]['b']+X[i][2]}                                    


#X = dataset[['r', 'g', 'b']]
#y = dataset['label']


dataset.to_csv("noisycolors.csv", index=False, header=None)