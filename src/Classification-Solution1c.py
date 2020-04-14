"""
Evaluate classifiers created from the data generated based on the Panel A and B
Problem 1 - Part c
@author: raziei.z@husky.neu.edu

"""


import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import keras
from sklearn.utils import resample
from keras.callbacks import EarlyStopping,ModelCheckpoint
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import roc_auc_score
from keras.models import model_from_json
from tqdm import tqdm_notebook as tqdm
import json

def generate_data(n, box, pos_boxes, pos_box_len, plot = False):
    """
        n = number of data points
        box = A list with format = [ [Top left corner of the box], [Bottom right corner of the box]]
        pos_boxes = A list with format = [ [Top left corner of the box1], [Top left corner of the box2], ...]
        pos_box_len = length of positive boxes
        plot = whether to plot the graph or not
    """
    x_points = (np.random.rand(n) * (box[1][0] - box[0][0]))+box[0][0]  # Generating n number of point in x range
    y_points = (np.random.rand(n) * (box[1][1] - box[0][1]))+box[0][1]  # Generating n number of point in y range
    points = np.column_stack((x_points,y_points)) # Column stacking these two dimention in 1 array
    all_pos = np.zeros(n) == 1
    for pos_box in pos_boxes:   # applying the condition for positive candidates as defined in the question
        pos_x = np.logical_and(x_points > pos_box[0],x_points < (pos_box[0]+pos_box_len)) # specifying x axis range of positive boxes
        pos_y = np.logical_and(y_points > (pos_box[1]-pos_box_len), y_points < pos_box[1]) # specifying y axis range of positive boxes
        pos = np.logical_and(pos_x,pos_y) # Taking AND of both the x and y axis
        all_pos = np.logical_or(all_pos,pos) # Now taking OR to join the currect rectangle with the previous ones
    if plot:   # plot if require
        plt.axis('equal')
        plt.scatter(x_points,y_points) # plotting all point default in blue color
        plt.scatter(x_points[all_pos],y_points[all_pos]) # plotting all the positive rectangles default in orange color
        plt.show()
    return points, (all_pos.reshape(-1,1)*2)-1 # reshaping y to single axis and normalizing between -1 and 1

def return_model(h1, h2, lear_rate=0.001):
    """
        h1 = Number of hidden units in layer 1
        h2 = Number of hidder units in layer 2
        lear_rate = learning rate
    """
    model = Sequential() # Specifying a sequential keras model
    model.add(Dense(units=h1, activation='tanh', input_dim=2)) # Adding 1st hidden layer
    if h2!=0: 
        model.add(Dense(units=h2, activation='tanh')) # Adding 2nd hidden layer
    model.add(Dense(units=1, activation='tanh')) # output unit with tanh activation
    model.compile(loss=keras.losses.hinge, # since hinge loss caters loss for output value -1 to 1
              optimizer = keras.optimizers.Adam(lr = lear_rate,decay=1e-6,amsgrad =True), # chosing Adam Optimizer
              metrics=['accuracy']) # set metrics as accuracy for monitoring
    return model

# Generating data
X1,y1 = generate_data(10000,[[-6,4],[6,-4]],[[-4,3],[-2,-1],[2,1]],3,plot=True)
X2,y2 = generate_data(10000,[[-6,4],[6,-4]],[[-4,3],[-1,-2],[2,0]],1,plot=True)
XT = [X1,X2]
yT = [y1,y2]

#Specifying hidden layers
h1 = [12,24]
h2 = [3,9]

acc={}
roc={}
for data_No,(X,y) in enumerate(zip(XT,yT)): # selecting data set (a) or (b)
    for fh,sh in zip(h1,h2):                # Selecting each neurons in hidden layer combination
        keras.backend.clear_session()
        model = return_model(fh,sh)
        es = EarlyStopping(monitor='loss', min_delta=0.001,patience=20, verbose=1)
        training_No = './1c/'+str(data_No)+"_"+str(fh)
   
        history = model.fit(X, y, epochs=50, callbacks=[es])
        # serialize model to JSON
        model_json = model.to_json()
        with open(training_No+".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(training_No+".h5")

        scores = model.evaluate(X, y, verbose=0)
        acc[training_No+'_acc']=scores[1] * 100
        try:
            roc[training_No+'_acc'] = roc_auc_score(y, model.predict(X))
        except:
            roc[training_No+'_acc'] = 0

        # generate 2 2d grids for the x & y bounds
        y1, x = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-6, 6, 100))
        z = np.zeros((np.shape(x)[0],np.shape(x)[1]))
        for i in range(len(x)):
            for j in range(len(y1)):
                z[i][j]=model.predict(np.array([[x[i][i],y1[j][j]]]))

        z = z[:-1, :-1]
        z_min, z_max = -np.abs(z).max(), np.abs(z).max()

        fig, ax = plt.subplots(figsize=(12,8))

        c = ax.pcolormesh(x, y1, z, cmap='RdBu', vmin=z_min, vmax=z_max)
        ax.set_title('pcolormesh')
        # set the limits of the plot to the limits of the data
        ax.axis([x.min(), x.max(), y1.min(), y1.max()])
        fig.colorbar(c, ax=ax)

        fig.savefig(training_No+'.png', dpi=fig.dpi)
            
with open("./1c/acc.json",'w') as f:
    json.dump(acc,f)
with open("./1c/roc.json",'w') as g:
    json.dump(roc,g)