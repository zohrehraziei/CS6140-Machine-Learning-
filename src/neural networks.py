"""
Evaluate classifiers created from the data generated based on the Panel A and B
@author: raziei.z@husky.neu.edu

"""


# In[2]:


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


# In[4]:


# Specifying Number of folds
num_folds = 10

# Generating data
X1,y1 = generate_data(1000,[[-6,4],[6,-4]],[[-4,3],[-2,-1],[2,1]],3,plot=False) # For the 1st dataset of Figure 1
X2,y2 = generate_data(1000,[[-6,4],[6,-4]],[[-4,3],[-1,-2],[2,0]],1,plot=False) # For the 2nd dataset of Figure 1
XT = [X1,X2] # Concatenating X of both the datasets
yT = [y1,y2] # Concatenating y of both the datasets

#Specifying hidden layers
h1 = [1,4,8] # 1st hidden layer
h2 = [0,3]   # 2nd hidder layer


# In[40]:


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
            kfold = KFold(n_splits=num_folds, shuffle=True) # Splitting the data into 10 folds
            fold_no = 1
            for idx, (train, test) in enumerate(kfold.split(X, y)):      # loop on each fold
                keras.backend.clear_session()  # Clearing any previous model if saved
                model = return_model(fh,sh)    # Geneting the required model

                es = EarlyStopping(monitor='val_loss', min_delta=0.001,patience=50, verbose=0) # Specifying the earlystopping criteria
                training_No = str(data_No)+'_'+str(fh)+'_'+str(sh)+'_'+str(idx)  # specifying a unique key for each fold in every combination
                mc = ModelCheckpoint('./1a/'+training_No+'.h5', monitor='val_accuracy', mode='max', verbose=0)

                history = model.fit(X[train], y[train],validation_data=(X[test], y[test]), epochs=1000, callbacks=[es,mc],verbose=0) # Training the model for max of 1000 epochs

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
            
# Saving the scores in json format
import json
with open("./1a/acc.json",'w') as f:
    json.dump(acc_score,f)
with open("./1a/roc.json",'w') as g:
    json.dump(roc_score,g)


# <h2> Solution Q1 part (b) </h2>

# In[ ]:


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
            ax.set_title('Heat Map')
            # set the limits of the plot to the limits of the data
            ax.axis([x.min(), x.max(), y1.min(), y1.max()])
            fig.colorbar(c, ax=ax)

            fig.savefig(training_No+'.png', dpi=fig.dpi)
            
with open("./1b/acc.json",'w') as f:
    json.dump(acc,f)
with open("./1b/roc.json",'w') as g:
    json.dump(roc,g)


# <h2> Solution Q1 part (c) </h2>

# In[ ]:


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


# <h2> Solution Q2 part (a) </h2>

# In[ ]:


#Specifying hidden layers
h1 = [1,4,8]
h2 = [0,3]

learning_rates = np.linspace(0.0001,0.01,30)

acc={}
roc={}

total = len(learning_rates)*len(h1)*len(h2)
try:
    pbar.close()
except:
    pass
pbar = tqdm(total=total, desc="Max Iterations: ")

for learn_rate in learning_rates:         # Training 30 models
    for fh in h1:
        for sh in h2:
            X,y = generate_data(1000,[[-6,4],[6,-4]],[[-4,3],[-1,-2],[2,0]],1,plot=False)
            keras.backend.clear_session()
            model = return_model(fh,sh, learn_rate)
            
            es = EarlyStopping(monitor='loss', min_delta=0.001,patience=100, verbose=0)
            
            training_No = './2a/2_'+str(learn_rate)+'_'+str(fh)+'_'+str(sh)
            history = model.fit(X, y, epochs=2000, callbacks=[es],verbose=0)
            
            pbar.update(1)
            scores = model.evaluate(X, y, verbose=0)
            acc[training_No+'_acc']=scores[1] * 100
            try:
                roc[training_No+'_acc'] = roc_auc_score(y, model.predict(X))
            except:
                roc[training_No+'_acc'] = 0
                
            # serialize model to JSON
            model_json = model.to_json()
            with open(training_No+".json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            
            model.save_weights(training_No+".h5")
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
with open("./2a/2acc.json",'w') as f:
    json.dump(acc,f)
with open("./2a/2roc.json",'w') as g:
    json.dump(roc,g)


# In[ ]:


# Now load all these models together to ensamble
from keras.models import model_from_json
dataset = '2'

def load_model(path):
    # load json and create model
    json_file = open(path+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(path+".h5")
    return loaded_model

#Ensambling on the basis of roc score
with open('./2a/'+dataset+'roc.json','r') as f:
    roc = json.load(f)

# Now I am doing weighted ensambling so that the model with better AUC gets more weight
for fh in h1:
    for sh in h2:
        roc_arr = []
        for lr in learning_rates:
            k = str('./2a/'+dataset+'_'+str(lr)+'_'+str(fh)+'_'+str(sh)+'_acc')
            roc_arr.append(roc[k])
        roc_arr_min = min(roc_arr)
        roc_arr_diff = max(roc_arr) - min(roc_arr)
        roc_total = sum(np.array(roc_arr) - roc_arr_min) / roc_arr_diff
        y1, x = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-6, 6, 100))
        z = np.zeros((np.shape(x)[0],np.shape(x)[1]))
        for lr in learning_rates:
            print(fh,sh,lr)
            k = str('./2a/'+dataset+'_'+str(lr)+'_'+str(fh)+'_'+str(sh))
            weig = roc[k+'_acc']
            weigh = ((weig - roc_arr_min) / roc_arr_diff) / roc_total
            model = load_model(k)
            for i in range(len(x)):
                for j in range(len(y1)):
                    z[i][j] = z[i][j] + (model.predict(np.array([[x[i][i],y1[j][j]]])) * weigh)
            
        z = z[:-1, :-1]
        z_min, z_max = -np.abs(z).max(), np.abs(z).max()

        fig, ax = plt.subplots(figsize=(12,8))

        c = ax.pcolormesh(x, y1, z, cmap='RdBu', vmin=z_min, vmax=z_max)
        ax.set_title('pcolormesh')
        # set the limits of the plot to the limits of the data
        ax.axis([x.min(), x.max(), y1.min(), y1.max()])
        fig.colorbar(c, ax=ax)
        print(k)
        fig.savefig(k+'_ensambled.png', dpi=fig.dpi)


# <h2> Solution Q2 part (b) </h2>

# In[43]:


from imblearn.over_sampling import RandomOverSampler
#Specifying hidden layers
h1 = [1,4,8]
h2 = [0,3]

learning_rates = np.linspace(0.0001,0.01,10)

acc={}
roc={}

total = len(learning_rates)*len(h1)*len(h2)
try:
    pbar.close()
except:
    pass
pbar = tqdm(total=total, desc="Max Iterations: ")

for learn_rate in learning_rates:
    for fh in h1:
        for sh in h2:
            X1,y1 = generate_data(1000,[[-6,4],[6,-4]],[[-4,3],[-2,-1],[2,1]],3,plot=False)
            X, y = oversample.fit_resample(X1, y1)
            keras.backend.clear_session()
            model = return_model(fh,sh, learn_rate)
            
            es = EarlyStopping(monitor='loss', min_delta=0.001,patience=50, verbose=0)
            
            training_No = './2b/0_'+str(learn_rate)+'_'+str(fh)+'_'+str(sh)
            history = model.fit(X, y, epochs=500, callbacks=[es],verbose=0)
            
            pbar.update(1)
            scores = model.evaluate(X, y, verbose=0)
            acc[training_No+'_acc']=scores[1] * 100
            try:
                roc[training_No+'_acc'] = roc_auc_score(y, model.predict(X))
            except:
                roc[training_No+'_acc'] = 0
                
            # serialize model to JSON
            model_json = model.to_json()
            with open(training_No+".json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            
            model.save_weights(training_No+".h5")
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
with open("./2b/0acc.json",'w') as f:
    json.dump(acc,f)
with open("./2b/0roc.json",'w') as g:
    json.dump(roc,g)


# In[62]:


from keras.models import model_from_json
dataset = '1'

learning_rates = np.linspace(0.0001,0.01,10)

def load_model(path):
    # load json and create model
    json_file = open(path+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(path+".h5")
    return loaded_model

#Ensambling on the basis of roc score
with open('./2b/'+dataset+'roc.json','r') as f:
    roc = json.load(f)


for fh in h1:
    for sh in h2:
        roc_arr = []
        for lr in learning_rates:
            k = str('./2b/'+dataset+'_'+str(lr)+'_'+str(fh)+'_'+str(sh)+'_acc')
            roc_arr.append(roc[k])
        roc_arr_min = min(roc_arr)
        roc_arr_diff = max(roc_arr) - min(roc_arr)
        roc_total = sum(np.array(roc_arr) - roc_arr_min) / roc_arr_diff
        y1, x = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-6, 6, 100))
        z = np.zeros((np.shape(x)[0],np.shape(x)[1]))
        for lr in learning_rates:
            print(fh,sh,lr)
            k = str('./2b/'+dataset+'_'+str(lr)+'_'+str(fh)+'_'+str(sh))
            weig = roc[k+'_acc']
            weigh = ((weig - roc_arr_min) / roc_arr_diff) / roc_total
            model = load_model(k)
            for i in range(len(x)):
                for j in range(len(y1)):
                    z[i][j] = z[i][j] + (model.predict(np.array([[x[i][i],y1[j][j]]])) * weigh)
            
        z = z[:-1, :-1]
        z_min, z_max = -np.abs(z).max(), np.abs(z).max()

        fig, ax = plt.subplots(figsize=(12,8))

        c = ax.pcolormesh(x, y1, z, cmap='RdBu', vmin=z_min, vmax=z_max)
        ax.set_title('pcolormesh')
        # set the limits of the plot to the limits of the data
        ax.axis([x.min(), x.max(), y1.min(), y1.max()])
        fig.colorbar(c, ax=ax)
        print(k)
        fig.savefig(k+'_ensambled.png', dpi=fig.dpi)


# In[66]:


dataset = '1'
#Ensambling on the basis of roc score
with open('./2b/'+dataset+'roc.json','r') as f:
    roc = json.load(f)
with open('./2b/'+dataset+'acc.json','r') as f:
    acc = json.load(f)
learning_rates = np.linspace(0.0001,0.01,10)
rc={}
ac={}
for fh in h1:
    for sh in h2:
        roc_arr = []
        for lr in learning_rates:
            k = str('./2b/'+dataset+'_'+str(lr)+'_'+str(fh)+'_'+str(sh)+'_acc')
            roc_arr.append(roc[k])
        roc_arr_min = min(roc_arr)
        roc_arr_diff = max(roc_arr) - min(roc_arr)
        roc_total = sum(np.array(roc_arr) - roc_arr_min) / roc_arr_diff
        y1, x = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-6, 6, 100))
        z = np.zeros((np.shape(x)[0],np.shape(x)[1]))
        l = str(fh)+str(sh)
        rc[l] = 0
        ac[l] = 0
        for lr in learning_rates:
            k = str('./2b/'+dataset+'_'+str(lr)+'_'+str(fh)+'_'+str(sh))
            weig = roc[k+'_acc']
            weigh = ((weig - roc_arr_min) / roc_arr_diff) / roc_total
            rc[l] += (weig * weigh)*100
            ac[l] += (acc[k+'_acc'] * weigh)


# In[ ]:




