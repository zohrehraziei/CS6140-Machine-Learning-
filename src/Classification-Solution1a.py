"""
Evaluate classifiers created from the data generated based on the Panel A and B
Problem 1 - Part a
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
from tqdm import tqdm
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

# Specifying Number of folds
num_folds = 10

# Generating data
X1,y1 = generate_data(1000,[[-6,4],[6,-4]],[[-4,3],[-2,-1],[2,1]],3,plot=True) # For the 1st dataset of Figure 1
X2,y2 = generate_data(1000,[[-6,4],[6,-4]],[[-4,3],[-1,-2],[2,0]],1,plot=True) # For the 2nd dataset of Figure 1
XT = [X1,X2] # Concatenating X of both the datasets
yT = [y1,y2] # Concatenating y of both the datasets

#Specifying hidden layers
h1 = [1,4,8] # 1st hidden layer
h2 = [0,3]   # 2nd hidder layer


bootstrap_iterations=100
acc_score={}
roc_score={}
total = len(XT)*len(h1)*len(h2)*num_folds*bootstrap_iterations
pbar = tqdm(total=total, desc="Max Iterations: ")
for data_No,(X,y) in enumerate(zip(XT,yT)): # selecting data set (a) or (b)
    for fh in h1:                           # selecting number of neurons in 1st hidden layer
        for sh in h2:                       # selecting number of neurons in 2nd hidden layer
            acc_per_fold = {}
            roc_per_fold = {}
            kfold = KFold(n_splits=num_folds, shuffle=True) # Splitting the data into 10 folds
            fold_no = 1
            for idx, (train, test) in enumerate(kfold.split(X, y)):      # loop on each fold
                acc_per_fold[idx] = []
                roc_per_fold[idx] = []
                for boot_step in range(bootstrap_iterations):
                    bt_train = resample(np.arange(len(train)), n_samples=len(train))
                    bt_test = np.array([x for x in np.arange(len(train)) if x not in bt_train])
                    keras.backend.clear_session()  # Clearing any previous model if saved
                    model = return_model(fh,sh)    # Geneting the required model
                    es = EarlyStopping(monitor='val_loss', min_delta=0.001,patience=20, verbose=0) # Specifying the earlystopping criteria
                    training_No = str(data_No)+'_'+str(fh)+'_'+str(sh)+'_'+str(idx)+'_'+str(boot_step)  # specifying a unique key for each fold in every combination

                    history = model.fit(X[bt_train], y[bt_train],validation_data=(X[bt_test], y[bt_test]), epochs=50, callbacks=[es],verbose=1) # Training the model for max of 1000 epochs

                    # serialize model to JSON
                    model_json = model.to_json()
                    with open('./1a/'+training_No+".json", "w") as json_file:
                        json_file.write(model_json)

                    # serialize weights to HDF5
                    model.save_weights('./1a/'+training_No+".h5")
                    # Scores on test set
                    scores = model.evaluate(X[bt_test], y[bt_test], verbose=0)
                    acc_per_fold[idx].append(scores[1] * 100)
                    try:
                        roc_per_fold[idx].append(roc_auc_score(y[bt_test], model.predict(X[bt_test])))
                    except:
                        roc_per_fold[idx].append(0)
                    pbar.update(1)

                # Increase fold number
                fold_no = fold_no + 1
            
            tr_No = training_No = str(data_No)+'_'+str(fh)+'_'+str(sh)
            acc_score[tr_No] = acc_per_fold
            roc_score[tr_No] = roc_per_fold
            
# Saving the scores in json format
with open("./1a/acc.json",'w') as f:
    json.dump(acc_score,f)
with open("./1a/roc.json",'w') as g:
    json.dump(roc_score,g)

# Bootstrapping calculations
acc_score_per_bootstrap_iter_data_1 = {}
roc_score_per_bootstrap_iter_data_1 = {}
acc_score_per_bootstrap_iter_data_2 = {}
roc_score_per_bootstrap_iter_data_2 = {}
keys = list(acc_score.keys())
for i in keys[:6]:
    acc_stats = []
    roc_stats = []
    for j in range(bootstrap_iterations):
        sm = 0
        sm2 = 0
        for k in acc_score[i].keys():
            sm = sm + acc_score[i][k][j]
            sm2 = sm2 + roc_score[i][k][j]
        acc_stats.append( sm / num_folds)
        roc_stats.append( sm2 / num_folds)
    acc_score_per_bootstrap_iter_data_1[str(i)+'_mean'] = np.mean(np.array(acc_stats))
    acc_score_per_bootstrap_iter_data_1[str(i)+'_std'] = np.std(np.array(acc_stats))
    roc_score_per_bootstrap_iter_data_1[str(i)+'_mean'] = np.mean(np.array(roc_stats))
    roc_score_per_bootstrap_iter_data_1[str(i)+'_std'] = np.std(np.array(roc_stats))

for i in keys[6:]:
    acc_stats = []
    roc_stats = []
    for j in range(bootstrap_iterations):
        sm = 0
        sm2 = 0
        for k in acc_score[i].keys():
            sm = sm + acc_score[i][k][j]
            sm2 = sm2 + roc_score[i][k][j]
        acc_stats.append( sm / num_folds)
        roc_stats.append( sm2 / num_folds)
    acc_score_per_bootstrap_iter_data_2[str(i)+'_mean'] = np.mean(np.array(acc_stats))
    acc_score_per_bootstrap_iter_data_2[str(i)+'_std'] = np.std(np.array(acc_stats))
    roc_score_per_bootstrap_iter_data_2[str(i)+'_mean'] = np.mean(np.array(roc_stats))
    roc_score_per_bootstrap_iter_data_2[str(i)+'_std'] = np.std(np.array(roc_stats))