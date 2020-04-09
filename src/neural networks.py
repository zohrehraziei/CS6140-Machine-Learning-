"""
Principal Component Analysis (PCA)

@author: raziei.z@husky.neu.edu

"""


import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import keras
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.metrics import roc_auc_score
from tqdm import tqdm_notebook as tqdm
import json


# In[3]:


def generate_data(n, box, pos_boxes, pos_box_len, plot = False):
    """
        n = number of data points
        box = A list with format = [ [Top left corner of the box], [Bottom right corner of the box]]
        pos_boxes = A list with format = [ [Top left corner of the box1], [Top left corner of the box2], ...]
        pos_box_len = length of positive boxes
        plot = whether to plot the graph or not
    """
    x_points = (np.random.rand(n) * (box[1][0] - box[0][0]))+box[0][0]
    y_points = (np.random.rand(n) * (box[1][1] - box[0][1]))+box[0][1]
    points = np.column_stack((x_points,y_points))
    all_pos = np.zeros(n) == 1
    for pos_box in pos_boxes:
        pos_x = np.logical_and(x_points > pos_box[0],x_points < (pos_box[0]+pos_box_len))
        pos_y = np.logical_and(y_points > (pos_box[1]-pos_box_len), y_points < pos_box[1])
        pos = np.logical_and(pos_x,pos_y)
        all_pos = np.logical_or(all_pos,pos)
    if plot:   
        plt.axis('equal')
        plt.scatter(x_points,y_points)
        plt.scatter(x_points[all_pos],y_points[all_pos])
    return points, (all_pos.reshape(-1,1)*2)-1


# In[4]:


# Specifying Number of folds
num_folds = 10

# Generating data
X1,y1 = generate_data(1000,[[-6,4],[6,-4]],[[-4,3],[-2,-1],[2,1]],3,plot=True)
X2,y2 = generate_data(1000,[[-6,4],[6,-4]],[[-4,3],[-1,-2],[2,0]],1,plot=False)
XT = [X1,X2]
yT = [y1,y2]

#Specifying hidden layers
h1 = [1,4,8]
h2 = [0,3]


# In[26]:


def return_model(h1,h2):
    """
        h1 = Number of hidden units in layer 1
        h2 = Number of hidder units in layer 2
    """
    model = Sequential()
    model.add(Dense(units=h1, activation='tanh', input_dim=2))
    if h2!=0:
        model.add(Dense(units=h2, activation='tanh'))
    model.add(Dense(units=1, activation='tanh'))
    model.compile(loss=keras.losses.hinge, 
              optimizer = keras.optimizers.Adam(lr = 0.001,decay=1e-6,amsgrad =True), 
              metrics=['accuracy'])
    return model


# <h2> Solution Q1 part (a)</h2>

# In[ ]:


# https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
# https://www.machinecurve.com/index.php/2020/02/18/how-to-use-k-fold-cross-validation-with-keras/
# https://keras.io/callbacks/
acc_score={}
roc_score={}
total = len(XT)*len(h1)*len(h2)*num_folds
pbar = tqdm(total=total, desc="Max Iterations: ")
for data_No,(X,y) in enumerate(zip(XT,yT)): # selecting data set (a) or (b)
    for fh in h1:                           # selecting number of neurons in 1st hidden layer
        for sh in h2:                       # selecting number of neurons in 2nd hidden layer
            acc_per_fold = []
            roc_per_fold = []
            kfold = KFold(n_splits=num_folds, shuffle=True)
            fold_no = 1
            for idx, (train, test) in enumerate(kfold.split(X, y)):      # loop on each fold
                keras.backend.clear_session()
                model = return_model(fh,sh)

                es = EarlyStopping(monitor='val_loss', min_delta=0.001,patience=50, verbose=0)
                training_No = str(data_No)+'_'+str(fh)+'_'+str(sh)+'_'+str(idx)
                mc = ModelCheckpoint('./1a/'+training_No+'.h5', monitor='val_accuracy', mode='max', verbose=0)

                history = model.fit(X[train], y[train],validation_data=(X[test], y[test]), epochs=1000, callbacks=[es,mc],verbose=0)

                # Scores on test set
                scores = model.evaluate(X[test], y[test], verbose=0)
                acc_per_fold.append(scores[1] * 100)
                try:
                    roc_per_fold.append(roc_auc_score(y[test], model.predict(X[test])))
                except:
                    roc_per_fold.append(0)

                # Increase fold number
                fold_no = fold_no + 1
                pbar.update(1)
                
            # accuracy status after 10 folds CV
            acc_per_fold = np.asarray(acc_per_fold)
            roc_per_fold = np.asarray(roc_per_fold)
            acc_per_fold_mean = np.mean(acc_per_fold)
            acc_per_fold_std = np.std(acc_per_fold)
            roc_per_fold_mean = np.mean(roc_per_fold)
            roc_per_fold_std = np.std(roc_per_fold)
            acc_score[str(data_No)+'_'+str(fh)+"_"+str(sh)+"_acc_mean"] = acc_per_fold_mean
            acc_score[str(data_No)+'_'+str(fh)+"_"+str(sh)+"_acc_std"] = acc_per_fold_std
            roc_score[str(data_No)+'_'+str(fh)+"_"+str(sh)+"_roc_mean"] = roc_per_fold_mean
            roc_score[str(data_No)+'_'+str(fh)+"_"+str(sh)+"_roc_std"] = roc_per_fold_std
import json
with open("./1a/acc.json",'w') as f:
    json.dump(acc_score,f)
with open("./1a/roc.json",'w') as g:
    json.dump(roc_score,g)


# <h2> Solution Q1 part (b) </h2>

# In[12]:


total = len(XT)*len(h1)*len(h2)
pbar.close()
pbar = tqdm(total=total, desc="Max Iterations: ")
acc={}
roc={}
for data_No,(X,y) in enumerate(zip(XT,yT)): # selecting data set (a) or (b)
    for fh in h1:                           # selecting number of neurons in 1st hidden layer
        for sh in h2:                       # selecting number of neurons in 2nd hidden layer
            keras.backend.clear_session()
            model = return_model(fh,sh)
            
            es = EarlyStopping(monitor='loss', min_delta=0.0001,patience=200, verbose=0)
            
            training_No = './1b/'+str(data_No)+'_'+str(fh)+'_'+str(sh)
            history = model.fit(X, y, epochs=10000, callbacks=[es],verbose=0)
            
            pbar.update(1)
            
            # serialize model to JSON
            model_json = model.to_json()
            with open(training_No+".json", "w") as json_file:
                json_file.write(model_json)
            
            # serialize weights to HDF5
            model.save_weights(training_No+".h5")
            
            # Overall score
            scores = model.evaluate(X, y, verbose=0)
            acc[training_No+'_acc']=scores[1] * 100
            try:
                roc[training_No+'_acc'] = roc_auc_score(y, model.predict(X))
            except:
                roc[training_No+'_acc'] = 0
                
            # Generating heatmap
            y1, x = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-6, 6, 100))
            z = np.zeros((np.shape(x)[0],np.shape(x)[1]))
            for i in range(len(x)):
                for j in range(len(y1)):
                    z[i][j]=model.predict(np.array([[x[i][i],y1[j][j]]]))

            z = z[:-1, :-1]
            z_min, z_max = -np.abs(z).max(), np.abs(z).max()

            fig, ax = plt.subplots(figsize=(18,8))

            c = ax.pcolormesh(x, y1, z, cmap='RdBu', vmin=z_min, vmax=z_max)
            ax.set_title('pcolormesh')
            # set the limits of the plot to the limits of the data
            ax.axis([x.min(), x.max(), y1.min(), y1.max()])
            fig.colorbar(c, ax=ax)

            fig.savefig(training_No+'.png', dpi=fig.dpi)
            
with open("./1b/acc.json",'w') as f:
    json.dump(acc,f)
with open("./1b/roc.json",'w') as g:
    json.dump(roc,g)


# <h2> Solution Q1 part (c) </h2>

# In[28]:


# Generating data
X1,y1 = generate_data(10000,[[-6,4],[6,-4]],[[-4,3],[-2,-1],[2,1]],3,plot=False)
X2,y2 = generate_data(10000,[[-6,4],[6,-4]],[[-4,3],[-1,-2],[2,0]],1,plot=False)
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
        es = EarlyStopping(monitor='loss', min_delta=0.0001,patience=200, verbose=0)
        training_No = './1c/'+str(data_No)+"_"+str(fh)
   
        history = model.fit(X, y, epochs=5000, callbacks=[es])
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


# In[ ]:




