"""The template of the main script of the machine learning process
"""
import pickle 
import arkanoid.communication as comm
from arkanoid.communication import SceneInfo, GameInstruction
import numpy as np
from sklearn.preprocessing import StandardScaler

filename = 'KNN_model.sav'
KNN_model = pickle.load( open(filename, 'rb'))

filename = 'KNN_normalize_model.sav'
KNN_normalize_model = pickle.load( open(filename, 'rb'))

def ml_loop():


	# === Here is the execution order of the loop === #
	# 1. Put the initialization code here.
	ball_position_history = []
	
	
	ball_flag = 0

	# 2. Inform the game process that ml process is ready before start the loop.
	comm.ml_ready()

	# 3. Start an endless loop.
	while True:
		# 3.1. Receive the scene information sent from the game process.
		scene_info = comm.get_scene_info()
		
		if ball_flag == 0 :
			ball_flag = 1
			last_ball = np.array(scene_info.ball)
			ball_move = [0,0]
			# Ball_Play_distance = 0
		else :		
			ball_move = np.array(scene_info.ball) - last_ball
			last_ball = np.array(scene_info.ball)
			# Ball_Play_distance = 400 - scene_info.ball[1]
		
		len_bricks = len(scene_info.bricks)
		Ball_Plat_diff = np.array(scene_info.platform) - np.array(scene_info.ball)
		Ball_Plat_diff_abs = np.absolute(Ball_Plat_diff)
		Ball_Plat_distance = np.linalg.norm( Ball_Plat_diff )
		# print(Ball_Plat_diff)
		
		# inp_temp = np.array([scene_info.ball[0], scene_info.ball[1], scene_info.platform[0], ball_move[0], ball_move[1], Ball_Plat_distance ])
		inp_temp = np.array([scene_info.ball[0], scene_info.ball[1], scene_info.platform[0], ball_move[0], ball_move[1], Ball_Plat_diff_abs[0], Ball_Plat_diff_abs[1], len_bricks ])
		# inp_temp = np.array([scene_info.ball[0], scene_info.ball[1], scene_info.platform[0], ball_move[0], ball_move[1], Ball_Plat_diff_abs[0], Ball_Plat_diff_abs[1] ])
		
		input = inp_temp[np.newaxis, :]
		
####################################################################################################
		# 3.2. If the game is over or passed, the game process will reset
		#      the scene immediately and send the scene information again.
		#      Therefore, receive the reset scene information.
		#      You can do proper actions, when the game is over or passed.
		if scene_info.status == SceneInfo.STATUS_GAME_OVER or \
			scene_info.status == SceneInfo.STATUS_GAME_PASS:
			scene_info = comm.get_scene_info()

####################################################################################################
		# 3.3. Put the code here to handle the scene information
		#print(input)
		test_input = np.array(input, np.float32)
		# test_input[:,0] = test_input[:,0] /200
		# test_input[:,1] = test_input[:,1] /500
		# test_input[:,2] = test_input[:,2] /195
		#print(test_input)
		
		normalize_test_input = KNN_normalize_model.transform(test_input)
		move = KNN_model.predict(normalize_test_input)

		# move = KNN_model.predict(test_input)
		# print(move)

####################################################################################################
		# 3.4. Send the instruction for this frame to the game process
		#print(ball_destination)
		#print(platform_center_x)
		if move <0 :
			comm.send_instruction(scene_info.frame, GameInstruction.CMD_LEFT)

		elif move >0 :
			comm.send_instruction(scene_info.frame, GameInstruction.CMD_RIGHT)

		else :
			comm.send_instruction(scene_info.frame, GameInstruction.CMD_NONE)


		# comm.send_instruction(scene_info.frame, GameInstruction.CMD_LEFT)
