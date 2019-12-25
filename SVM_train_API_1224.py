import numpy as np
from numpy import *
# import random
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score 

import os
import pickle

#--------------------------------------------------------------------------------------
C = 0.6
tol = 0.001


in_path = "D:\\MLGame-master\\games\\pingpong\\log\\"

command_cod = "python MLGame.py -f 50 -1 -r pingpong -i number_11_P1_API.py number_11_P2_API.py"

numer = 0
end_flag = 0
who_data_fix = 0
first_run = 0
#-----------------------------------------initial data
Initial_data_x_P1_L = np.array( [0,   490, 7, -7, 7,  80, 80, 80,   -70, 80,   -440] )
Initial_data_x_P1_R = np.array( [190, 490, 7,  7, 7,  80, 80, -110, -70, -110, -440] )

Initial_data_x_P2_L = np.array( [0,   0,   7, -7, -7, 80, 80, 80,   420, 80,    50] )
Initial_data_x_P2_R = np.array( [190, 0,   7,  7, -7, 80, 80, -110, 420, -110,  50] )

final_fix_data_x_P1 =  np.vstack(( Initial_data_x_P1_L, Initial_data_x_P1_R, Initial_data_x_P2_L, Initial_data_x_P2_R  ))
final_fix_data_x_P2 =  np.vstack(( Initial_data_x_P1_L, Initial_data_x_P1_R, Initial_data_x_P2_L, Initial_data_x_P2_R  ))

final_Label_Y_P1 = np.array([-1, 1, 0, 0])
final_Label_Y_P2 = np.array([0, 0, -1, 1])

svm = SVC(gamma = 0.001, C=18.0, decision_function_shape= 'ovo')

svm.fit(final_fix_data_x_P1, final_Label_Y_P1) 

model_name = "P1_SVM_parameter_API.sav"
pickle.dump(svm, open(model_name, 'wb'))

#----------------------
svm.fit(final_fix_data_x_P2, final_Label_Y_P2)

model_name = "P2_SVM_parameter_API.sav"
pickle.dump(svm, open(model_name, 'wb'))

#-----------------------------------------ping pong  game
# Ball_position    = [] 
# Ball_speed_list  = []
# Plat_Position_P1 = []
# Plat_Position_P2 = []

#--------------------------------------------------------------------------------------
while(end_flag == 0 ) :

#-----------------------------------------ping pong  game
    os.system(command_cod)

#-----------------------------------------win ?
    name_list = os.listdir(in_path)
    last_path = in_path + name_list[-1]

    with open(last_path, "rb") as f: 
    	data_list = pickle.load(f)

    who_win = data_list[-1].status
    last_Ball_speed = data_list[-1].ball_speed

    #-------------------
    if last_Ball_speed >= 40 :
        end_flag =1

        P1_data = {
            "final_fix_data_x_P1" : final_fix_data_x_P1,
            "final_Label_Y_P1" : final_Label_Y_P1
        }
        output = open('P1_data.pkl', 'wb')
        pickle.dump(P1_data, output)
        output.close()

        P2_data = {
            "final_fix_data_x_P1" : final_fix_data_x_P2,
            "final_Label_Y_P1" : final_Label_Y_P2
        }
        output = open('P2_data.pkl', 'wb')
        pickle.dump(P2_data, output)
        output.close()

        break

    elif who_win == "GAME_1P_WIN" :
        who_lose_need_fix = 2
    elif who_win == "GAME_2P_WIN" :
        who_lose_need_fix = 1

#-----------------------------------------data fix and prepare
    Ball_position    = [] 
    Ball_speed_list  = []
    Plat_Position_P1 = []
    Plat_Position_P2 = []

    for i in range(1,len(data_list)) :
        Ball_position.append    ( data_list[i].ball ) 
        Ball_speed_list.append  ( data_list[i].ball_speed )
        Plat_Position_P1.append ( data_list[i].platform_1P ) 
        Plat_Position_P2.append ( data_list[i].platform_2P ) 

    #------------------ calculate ball
    Ball = np.array(Ball_position)       [1:]

    #------------------ calculate ball speed
    Ball_speed = np.array(Ball_speed_list)       [1:] [:, np.newaxis]

    #------------------ calculate ball move
    Ball_array = np.array(Ball_position[:-1])
    Ball_array_next = np.array(Ball_position[1:])
    Ball_move = Ball_array_next - Ball_array

    #------------------ calculate plat
    Plat_X_P1 = np.array(Plat_Position_P1) [1:,0] [:, np.newaxis]
    Plat_X_P2 = np.array(Plat_Position_P2) [1:,0] [:, np.newaxis]

    #------------------ calculate Plat Ball distance
    Plat_P1 = np.array(Plat_Position_P1) [1:] 
    Plat_P2 = np.array(Plat_Position_P2) [1:] 

    Plat_Ball_distance_P1 = (Plat_P1 - Ball)
    Plat_Ball_distance_P2 = (Plat_P2 - Ball)

    # print("Plat_Ball_distance_P1 : ", Plat_Ball_distance_P1)

    #------------------ mix
    original_data_x = np.hstack((Ball, Ball_speed, Ball_move, Plat_X_P1, Plat_X_P2, Plat_Ball_distance_P1, Plat_Ball_distance_P2))

    #------------------ make feature
    if who_lose_need_fix == 1 : 
        last_ball_y = Ball[-1][1]
        catch_frame = int( (last_ball_y - 420) / Ball_speed[-1]  + 2.5 )

        final_ball_x =  Ball[-catch_frame][0]
        last_plat_x = Plat_P1[-catch_frame][0]
        # last_plat_x += 10

        last_distance = abs(last_plat_x - final_ball_x)
        fix_quantity  = int( last_distance / 5 +0.5 )

        train_starting_point = fix_quantity + catch_frame

        if last_plat_x >= final_ball_x :
            P1_y = [-1] * train_starting_point
        else : 
            P1_y = [1] * train_starting_point
        P2_y = [0] * train_starting_point

    elif who_lose_need_fix == 2 :
        last_ball_y = Ball[-1][1]
        catch_frame = int( (50-last_ball_y) / Ball_speed[-1]  + 2.5  )

        final_ball_x =  Ball[-catch_frame][0]
        last_plat_x = Plat_P2[-catch_frame][0]
        # last_plat_x += 10

        last_distance = abs(last_plat_x - final_ball_x)
        fix_quantity  = int( last_distance / 5 +0.5 )

        train_starting_point = fix_quantity + catch_frame

        if last_plat_x >= final_ball_x :
            P2_y = [-1] * train_starting_point
        else : 
            P2_y = [1] * train_starting_point
        P1_y = [0] * train_starting_point

    fix_data_x = original_data_x[ -train_starting_point : ]
    Label_Y_P1 = np.array(P1_y)
    Label_Y_P2 = np.array(P2_y)


    final_fix_data_x_P1 = np.vstack(( final_fix_data_x_P1, fix_data_x ))
    final_Label_Y_P1 = np.hstack(( final_Label_Y_P1, Label_Y_P1 ))

    final_fix_data_x_P2 = np.vstack(( final_fix_data_x_P2, fix_data_x ))
    final_Label_Y_P2 = np.hstack(( final_Label_Y_P2, Label_Y_P2 ))

    print("fix_data_x : ", fix_data_x)
    # print("final_Label_Y_P1 : ", final_Label_Y_P1)
#-----------------------------------------SVM tain
    svm = SVC(gamma = 0.001, C=18.0, decision_function_shape= 'ovo')
    svm.fit(final_fix_data_x_P1, final_Label_Y_P1) 

    # predict_y=svm.predict(final_fix_data_x_P1) 
    # accuracy_score_answer = accuracy_score(predict_y, final_Label_Y_P1)
    # print(" accuracy_score_answer : ", accuracy_score_answer)

    model_name = "P1_SVM_parameter_API.sav"
    pickle.dump(svm, open(model_name, 'wb'))

    #--------------
    svm = SVC(gamma = 0.001, C=18.0, decision_function_shape= 'ovo')
    svm.fit(final_fix_data_x_P2, final_Label_Y_P2)

    # predict_y=svm.predict(final_fix_data_x_P2) 
    # accuracy_score_answer = accuracy_score(predict_y, final_Label_Y_P2)
    # print(accuracy_score_answer)

    model_name = "P2_SVM_parameter_API.sav"
    pickle.dump(svm, open(model_name, 'wb'))

    #-----------------------------------------model choose
    # for test_x in final_fix_data_x_P2:
        
    #     predict_lenght = np.dot(test_x ,np_P2_w) + np_P2_b
    #     # predict_lenght = test_x ,testwewe + P2_b
    #     if predict_lenght < 0 :
    #         predict_y = -1
    #     elif predict_lenght >=0 :
    #         predict_y = 1
    #     print("predict_lenght :", predict_lenght)
    #     print("predict_y :", predict_y)
    
    
    # pkl_file = open('P2_SVM_parameter.pkl', 'rb')
    # mydict2 = pickle.load(pkl_file)
    # pkl_file.close()

    # print (mydict2['w'])
    
    # end_flag = 1
    print("number : ", numer)
    numer += 1
    
#-----------------------------------------end

    

