"""
The template of the script for the machine learning process in game pingpong
"""

# Import the necessary modules and classes
import games.pingpong.communication as comm
from games.pingpong.communication import (
    SceneInfo, GameStatus, PlatformAction
)
import numpy as np
import pickle

def ml_loop(side: str):
    """
    The main loop for the machine learning process

    The `side` parameter can be used for switch the code for either of both sides,
    so you can write the code for both sides in the same script. Such as:
    ```python
    if side == "1P":
        ml_loop_for_1P()
    else:
        ml_loop_for_2P()
    ```

    @param side The side which this script is executed for. Either "1P" or "2P".
    """

    # === Here is the execution order of the loop === #
    # 1. Put the initialization code here
    filename = 'P2_SVM_parameter_API.sav'
    model = pickle.load( open(filename, 'rb'))

    ball_position_history = []
    wait_frame = 0

    # 2. Inform the game process that ml process is ready
    comm.ml_ready()

    # 3. Start an endless loop
    while True:
        # 3.1. Receive the scene information sent from the game process
        scene_info = comm.get_scene_info()
        ball_position_history.append( scene_info.ball )

        # 3.2. If either of two sides wins the game, do the updating or
        #      resetting stuff and inform the game process when the ml process
        #      is ready.
        if scene_info.status == GameStatus.GAME_1P_WIN or \
           scene_info.status == GameStatus.GAME_2P_WIN:
            # Do some updating or resetting stuff

            # 3.2.1 Inform the game process that
            #       the ml process is ready for the next round
            comm.ml_ready()
            continue

        # 3.3 Put the code here to handle the scene information
        if wait_frame ==1 :
            #------------------ calculate ball
            Ball = np.asarray( scene_info.ball)
            # print("Ball : ", Ball)

            #------------------ calculate ball speed
            Ball_speed = np.asarray(scene_info.ball_speed) 
            # print("Ball_speed : ", Ball_speed)

            #------------------ calculate ball move
            Ball_move = np.asarray(ball_position_history[-1]) - np.asarray( ball_position_history[-2])
            # print("Ball_move : ", Ball_move)

            #------------------ calculate plat
            Plat_X_P1 = scene_info.platform_1P [0]
            Plat_X_P2 = scene_info.platform_2P [0]
            # print("Plat_X_P1 : ", Plat_X_P1)
            # print("Plat_X_P2 : ", Plat_X_P2)

            #------------------ calculate Plat Ball distance
            Plat_P1 = np.asarray( scene_info.platform_1P )
            Plat_P2 = np.asarray( scene_info.platform_2P)
            # print("Plat_P1 : ", Plat_P1)
            # print("Plat_P2 : ", Plat_P2)

            # Plat_Ball_distance_P1 = np.linalg.norm( Plat_P1 - Ball, axis = 1 )
            # Plat_Ball_distance_P2 = np.linalg.norm( Plat_P2 - Ball, axis = 1 )
            Plat_Ball_distance_P1 = Plat_P1 - Ball
            Plat_Ball_distance_P2 = Plat_P2 - Ball
            # print("Plat_Ball_distance_P1 : ", Plat_Ball_distance_P1)
            # print("Plat_Ball_distance_P2 : ", Plat_Ball_distance_P2)

            #------------------ mix
            data_x = np.hstack((Ball, Ball_speed, Ball_move, Plat_X_P1, Plat_X_P2, Plat_Ball_distance_P1, Plat_Ball_distance_P2))
            
            input_data_x = data_x[np.newaxis, :]

            move = model.predict(input_data_x)

            # 3.4 Send the instruction for this frame to the game process
            if move == -1 :
                comm.send_instruction(scene_info.frame, PlatformAction.MOVE_LEFT)
            elif move ==  1  :
                comm.send_instruction(scene_info.frame, PlatformAction.MOVE_RIGHT)
            # elif Plat_X_P2 > 80:
            #     comm.send_instruction(scene_info.frame, PlatformAction.MOVE_LEFT)
            # elif Plat_X_P2 < 80:
            #     comm.send_instruction(scene_info.frame, PlatformAction.MOVE_RIGHT)
            else:
                comm.send_instruction(scene_info.frame, PlatformAction.NONE)

            # comm.send_instruction(scene_info.frame, PlatformAction.MOVE_LEFT)
        else : 
            wait_frame = 1
