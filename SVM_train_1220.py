import numpy as np
from numpy import *
import random
import os
import pickle


def selectJrandom(i,m):
    j=i
    while (j==i):
        j = int(random.uniform(0,m))
    # print("i : ", i)
    # print("j : ", j)
    return j

def clipAlphasJ(aj,H,L):
      if aj > H:
            aj = H
      if L > aj:
            aj = L
      return aj

def simplifiedSMO(dataX, classY, C, tol, max_passes):
    X = np.mat(dataX)
    Y = np.mat(classY).T
    m,n = X.shape
    # Initialize b: threshold for solution
    b = 0
    # Initialize alphas: lagrange multipliers for solution
    alphas = np.mat(zeros((m,1)))
    passes = 0

    # print("X : ", X)
    # print("Y : ", Y)
    # print("M : ", m)
    # print("n : ", n)
    # print("alphas : ", alphas)

    while (passes < max_passes):
        num_changed_alphas = 0
        
        for i in range(m):
            
            
            # print("sum_X : ", sum_X)

            # Calculate Ei = f(xi) - yi
            alphas_Y_multiply = multiply(alphas,Y).T
            sum_X = (X*X[i,:].T)
            fXi = ( alphas_Y_multiply * sum_X ) + b
            Ei = fXi - (Y[i])

            if ((Y[i]*Ei < -tol) and (alphas[i] < C)) or ((Y[i]*Ei > tol) and (alphas[i] > 0)):
                # select j # i randomly
                j = selectJrandom(i,m)
                # Calculate Ej = f(xj) - yj
                fXj = float(multiply(alphas,Y).T*(X*X[j,:].T)) + b
                Ej = fXj - float(Y[j])
                # save old alphas's
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # compute L and H
                if (Y[i] != Y[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                # if L = H the continue to next i
                if L==H:
                    continue
                # compute eta
                eta = 2.0 * X[i,:]*X[j,:].T - X[i,:]*X[i,:].T - X[j,:]*X[j,:].T
                # if eta >= 0 then continue to next i
                if eta >= 0:
                    continue
                # compute new value for alphas j
                alphas[j] -= Y[j]*(Ei - Ej)/eta
                # clip new value for alphas j
                alphas[j] = clipAlphasJ(alphas[j],H,L)
                # if |alphasj - alphasold| < 0.00001 then continue to next i
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    continue
                # determine value for alphas i
                alphas[i] += Y[j]*Y[i]*(alphaJold - alphas[j])
                # compute b1 and b2
                b1 = b - Ei- Y[i]*(alphas[i]-alphaIold)*X[i,:]*X[i,:].T - Y[j]*(alphas[j]-alphaJold)*X[i,:]*X[j,:].T
                b2 = b - Ej- Y[i]*(alphas[i]-alphaIold)*X[i,:]*X[j,:].T - Y[j]*(alphas[j]-alphaJold)*X[j,:]*X[j,:].T
                # compute b
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2)/2.0                      
                num_changed_alphas += 1

            if (num_changed_alphas == 0): 
                passes += 1
            # else: 
            #     passes = 0

    # print(passes)
    return b,alphas

def calculate_w(a_array, Y_array, x_array ):
    flag_w = 0

    for def_a, def_y, def_x in zip(a_array, Y_array, x_array ) :
        def_w = (def_a * def_y * def_x)
        if flag_w == 0 : 
            sum_w = def_w.copy()
            flag_w = 1
        else : 
            sum_w += def_w

    return sum_w
#--------------------------------------------------------------------------------------
C = 0.6
tol = 0.001


in_path = "D:\\MLGame-master\\games\\pingpong\\log\\"

command_cod = "python MLGame.py -f 50 -1 -r pingpong -i number_11_P1.py number_11_P2.py"

numer = 0
end_flag = 0
who_data_fix = 0
first_run = 0
#-----------------------------------------initial data
Initial_data_x_P2_L = np.array(  [0, 0, 7, -7, -7, 80, 80, 80, 420, 80, 50] )
Initial_data_x_P2_R = np.array( [190, 0, 7,  7, -7, 80, 80, -110, 420, -110, 50] )
Initial_data_x_P1_L = np.array(  [0, 490, 7, -7, 7, 80, 80, 80, -70, 80, -440] )
Initial_data_x_P1_R = np.array( [190, 490, 7,  7, 7, 80, 80, -110, -70, -110, -440] )

final_fix_data_x_P1 =  np.vstack(( Initial_data_x_P1_L, Initial_data_x_P1_R ))
final_fix_data_x_P2 =  np.vstack(( Initial_data_x_P2_L, Initial_data_x_P2_R ))

final_Label_Y_P1 = np.array([-1, 1])
final_Label_Y_P2 = np.array([-1, 1])

#-----------------------------------------ping pong  game
Ball_position    = [] 
Ball_speed_list  = []
Plat_Position_P1 = []
Plat_Position_P2 = []

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
    Plat_P1 = np.array(Plat_Position_P1) [1:] [:, np.newaxis]
    Plat_P2 = np.array(Plat_Position_P2) [1:] [:, np.newaxis]

    Plat_Ball_distance_P1 = np.linalg.norm( Plat_P1 - Ball [:, np.newaxis], axis = 1 )
    Plat_Ball_distance_P2 = np.linalg.norm( Plat_P2 - Ball [:, np.newaxis], axis = 1 )

    # print("Plat_Ball_distance_P1 : ", Plat_Ball_distance_P1)

    #------------------ mix
    original_data_x = np.hstack((Ball, Ball_speed, Ball_move, Plat_X_P1, Plat_X_P2, Plat_Ball_distance_P1, Plat_Ball_distance_P2))

    if who_lose_need_fix == 1 : 
        last_ball_y = Ball[-1][1]
        catch_frame = int( (last_ball_y - 420) / Ball_speed[-1]  + 0.5 )

        final_ball_x =  Ball[-catch_frame][0]
        last_plat_x = Plat_P1[-catch_frame][0][0]
        last_plat_x += 10

        last_distance = abs(last_plat_x - final_ball_x)
        fix_quantity  = int( last_distance / 5 +0.5 )

        if last_plat_x > final_ball_x :
            y = [-1] * fix_quantity
            
        else : 
            y = [1] * fix_quantity
        print("y :", y)

        fix_data_x_P1 = original_data_x[-(fix_quantity + catch_frame): -catch_frame]
        Label_Y_P1 = np.array(y)

        final_fix_data_x_P1 = np.vstack(( final_fix_data_x_P1, fix_data_x_P1 ))
        final_Label_Y_P1 = np.hstack(( final_Label_Y_P1, Label_Y_P1 ))

        max_passes = int( (np.shape(final_fix_data_x_P2)[0]) /2+0.5)

    elif who_lose_need_fix == 2 :
        last_ball_y = Ball[-1][1]
        catch_frame = int( (50-last_ball_y) / Ball_speed[-1]  + 0.5 )

        final_ball_x =  Ball[-catch_frame][0]
        last_plat_x = Plat_P2[-catch_frame][0][0]
        last_plat_x += 10

        last_distance = abs(last_plat_x - final_ball_x)
        fix_quantity  = int( last_distance / 5 +0.5 )

        if last_plat_x > final_ball_x :
            y = [-1] * fix_quantity
        else : 
            y = [1] * fix_quantity

        fix_data_x_P2 = original_data_x[-(fix_quantity + catch_frame): -catch_frame]
        Label_Y_P2 = np.array(y)

        final_fix_data_x_P2 = np.vstack(( final_fix_data_x_P2, fix_data_x_P2 ))
        final_Label_Y_P2 = np.hstack(( final_Label_Y_P2, Label_Y_P2 ))

        max_passes = int( (np.shape(final_fix_data_x_P2)[0]) /2 +0.5)

    
    #-----------------------------------------SVM tain
    if who_lose_need_fix == 1 :
        P1_b, P1_a = simplifiedSMO(final_fix_data_x_P1, final_Label_Y_P1, C, tol, max_passes)
        P1_w = calculate_w(P1_a, final_Label_Y_P1, final_fix_data_x_P1)
        np_P1_w = np.squeeze(np.asarray(P1_w))
        np_P1_b = np.squeeze(np.asarray(P1_b))
        
        P1_dictionary = {
            "w" : np_P1_w,
            "b" : np_P1_b
        }
        output = open('P1_SVM_parameter.pkl', 'wb')
        pickle.dump(P1_dictionary, output)
        output.close()

        if first_run == 0 :
            first_run = 1
            P2_b, P2_a = simplifiedSMO(final_fix_data_x_P2, final_Label_Y_P2, C, tol, max_passes)
            P2_w = calculate_w(P2_a, final_Label_Y_P2, final_fix_data_x_P2)
            
            np_P2_w = np.squeeze(np.asarray(P2_w))
            np_P2_b = np.squeeze(np.asarray(P2_b))
            P2_dictionary = {
                "w" : np_P2_w,
                "b" : np_P2_b
            }

            output = open('P2_SVM_parameter.pkl', 'wb')
            pickle.dump(P2_dictionary, output)
            output.close()

    elif who_lose_need_fix == 2 :
        P2_b, P2_a = simplifiedSMO(final_fix_data_x_P2, final_Label_Y_P2, C, tol, max_passes)
        P2_w = calculate_w(P2_a, final_Label_Y_P2, final_fix_data_x_P2)
        
        np_P2_w = np.squeeze(np.asarray(P2_w))
        np_P2_b = np.squeeze(np.asarray(P2_b))
        P2_dictionary = {
            "w" : np_P2_w,
            "b" : np_P2_b
        }

        output = open('P2_SVM_parameter.pkl', 'wb')
        pickle.dump(P2_dictionary, output)
        output.close()
    
        if first_run == 0 :
            first_run = 1
            P1_b, P1_a = simplifiedSMO(final_fix_data_x_P1, final_Label_Y_P1, C, tol, max_passes)
            P1_w = calculate_w(P1_a, final_Label_Y_P1, final_fix_data_x_P1)
            np_P1_w = np.squeeze(np.asarray(P1_w))
            np_P1_b = np.squeeze(np.asarray(P1_b))
            
            P1_dictionary = {
                "w" : np_P1_w,
                "b" : np_P1_b
            }
            output = open('P1_SVM_parameter.pkl', 'wb')
            pickle.dump(P1_dictionary, output)
            output.close()

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

    

