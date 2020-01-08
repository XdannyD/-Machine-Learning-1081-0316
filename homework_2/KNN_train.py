#%% this is my import Library

import numpy as np 
import pickle 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

#%% here we need to input pickle file to data list

path_list = [
        # "FPS_55_1.pickle",
        # "FPS_55_2.pickle",
        # "FPS_55_3.pickle",
        # "FPS_50_1.pickle",
        # "FPS_50_2.pickle",
        # "FPS_50_3.pickle",
        # "FPS_20_1.pickle",
        # "FPS_20_2.pickle",
        # "FPS_20_3.pickle",
        # "level_4_1.pickle",
        # "level_4_2.pickle",
        # "2019-08-05_16-24-55.pickle",
        # "FPS_30_1_1004_1.pickle",
        # "2019-10-04_21-53-47.pickle",
        # "2019-10-04_22-00-52.pickle",
        # "2019-10-04_22-19-45.pickle",
        "2019-10-04_22-18-41.pickle",
        "2019-10-04_22-17-32.pickle",
        "2019-10-04_22-36-13.pickle",
        "2019-10-04_22-36-56.pickle",
        "2019-10-04_22-38-17.pickle",
        "2019-10-05_00-08-45.pickle",
        "2019-10-07_02-22-15.pickle",
        "2019-10-07_02-27-08.pickle"
        # "2019-10-07_02-30-55.pickle"

        ]
#------------------------------------------------- 3Pass knn
        # "2019-10-04_22-18-41.pickle",
        # "2019-10-04_22-17-32.pickle",
        # "2019-10-04_22-36-13.pickle",
        # "2019-10-04_22-36-56.pickle",
        # "2019-10-04_22-38-17.pickle",
        # "2019-10-05_00-08-45.pickle",
        # "2019-10-07_02-22-15.pickle",
        # "2019-10-07_02-27-08.pickle"
#-------------------------------------------------

# with open("FPS_55_1.pickle", "rb") as f: 
# 	data_list = pickle.load(f) 

#%% let we take some data from the data list


big_data_flag =0

for file_path in path_list :
    with open(file_path, "rb") as f: 
    	data_list = pickle.load(f)
    
    
    Frame=[] 
    Status=[] 
    Ballposition=[] 
    PlatformPosition=[] 
    Bricks=[] 
    for i in range(0,len(data_list)): 
        Frame.append(data_list[i].frame) 
        Status.append(data_list[i].status) 
        Ballposition.append(data_list[i].ball) 
        PlatformPosition.append(data_list[i].platform) 
        Bricks.append(len(data_list[i].bricks))
    
    ######################################################################################################
    #%% i need some different data
    # so... let we calculate it 

    PlatX = np.array(PlatformPosition)[:,0][:, np.newaxis] 
    PlatX_next=PlatX[1:,:] 
    instruct = (PlatX_next-PlatX[0:len(PlatX_next),0][:,np.newaxis])/5 
        
    # seLect some features to make x 
    # x is the features
    # so we need careful choice
    Ball = np.array(Ballposition)[1:][:, np.newaxis]
    Plat = np.array(PlatformPosition)[1:][:, np.newaxis]
    Ball_Plat_diff = Plat[...,1] - Ball[...,1]
    Ball_Plat_distance_1 = np.linalg.norm( Plat - Ball, axis = 1 )
    Ball_Plat_distance_2 = np.linalg.norm( Plat - Ball, axis = 2 )

    Ballarray = np.array(Ballposition[:-1])
    Ballarray_next = np.array(Ballposition[1:])
    Bricks_size = np.array(Bricks)[1:][:, np.newaxis] 

    Ball_move = Ballarray_next - Ballarray

    test_PlatX_normalize = (PlatX[0:-1,0][:,np.newaxis])

    test_Ballarray = np.array(Ballarray,np.float32)

    x = np.hstack((test_Ballarray, test_PlatX_normalize, Ball_move, Ball_Plat_distance_1, Bricks_size))
    # x = np.hstack((test_Ballarray, test_PlatX_normalize, Ball_move, Ball_Plat_distance_1))
    
    # seLect intructions as y 
    # According to x we give the corresponding y 
    # y is ( the label of the machine according to features ) 
    y = instruct
    
    
    if big_data_flag == 0:
        big_data_flag = 1
        big_data_x = x
        big_data_y = y
    else :
        big_data_x = np.vstack((big_data_x, x))
        big_data_y = np.vstack((big_data_y, y))
    
# remember! let y's data structure is (1,N) array  not the (N) array
big_data_y = big_data_y.ravel()

#%% data is ok
# next is use K-Nearest Neighbors to train mobel

# set KNN's parameter
neigh = KNeighborsClassifier(n_neighbors=3)

# star the KNN train
neigh.fit(big_data_x, big_data_y)

predict_y = neigh.predict(big_data_x)

# this accuracy_score_answer is retrun inside test accuracy score
# maybe it not the 100% , but don't wory. it's OK.
# 100% is not the prefect. model can pass the game is the first.
accuracy_score_answer = accuracy_score(predict_y, big_data_y)
print(accuracy_score_answer)

#%% after testing, the results are not ideal
# we may have some problems with data of x
# we haven't normalize x
# so we add normalization in our code

scaler = StandardScaler()
scaler.fit(big_data_x)
normalize_big_data_x = scaler.transform(big_data_x)

neigh.fit(normalize_big_data_x, big_data_y)

normalize_predict_y = neigh.predict(normalize_big_data_x)

accuracy_score_answer = accuracy_score(normalize_predict_y, big_data_y)
print(accuracy_score_answer)

#%% save model

model_name = "KNN_normalize_model.sav"
pickle.dump(scaler, open(model_name, 'wb'))


model_name = "KNN_model.sav"
pickle.dump(neigh, open(model_name, 'wb'))
