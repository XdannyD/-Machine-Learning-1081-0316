"""The template of the main script of the machine learning process
"""

import arkanoid.communication as comm
from arkanoid.communication import SceneInfo, GameInstruction

def ml_loop():
	"""The main loop of the machine learning process

	This loop is run in a separate process, and communicates with the game process.

	Note that the game process won't wait for the ml process to generate the
	GameInstruction. It is possible that the frame of the GameInstruction
	is behind of the current frame in the game process. Try to decrease the fps
	to avoid this situation.
	"""

	# === Here is the execution order of the loop === #
	# 1. Put the initialization code here.
	ball_position_history = []

	# 2. Inform the game process that ml process is ready before start the loop.
	comm.ml_ready()

	# 3. Start an endless loop.
	while True:
		# 3.1. Receive the scene information sent from the game process.
		scene_info = comm.get_scene_info()
		ball_position_history.append( scene_info.ball
                                              )
		platform_center_x = scene_info.platform[0]+20
		
		if len(ball_position_history) == 1:
			ball_going_down = 0
		elif ball_position_history [-1][1] - ball_position_history[-2][1] >0 :
			ball_going_down =1
			vy = ball_position_history[-1][1] - ball_position_history[-2][1]
			vx = ball_position_history[-1][0] - ball_position_history[-2][0]
			
		# else: ball_going_down =0
		
		
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
		# if ball_going_down ==1 and ball_position_history[-1][1] >=180 :
		if ball_going_down ==1 :
			ball_destination = ball_position_history[-1] [0] +((395- ball_position_history[-1][1])/vy)*vx
			if ball_destination >=195 :
				ball_destination = 195-(ball_destination-195)
			
			elif ball_destination <=0 :
				ball_destination = - ball_destination
		else:
			ball_destination = platform_center_x
			
		# if ball_going_down ==1 :
			# ball_destination = ball_position_history[-1] [0] +((395- ball_position_history[-1][1])/vy)*vx
			# if ball_destination >=195 :
				# ball_destination = 195-(ball_destination-195)
			
			# elif ball_destination <=0 :
				# ball_destination = - ball_destination

####################################################################################################
		# 3.4. Send the instruction for this frame to the game process
		#print(ball_destination)
		#print(platform_center_x)
		if (platform_center_x - 10) > ball_destination :
			comm.send_instruction(scene_info.frame, GameInstruction.CMD_LEFT)

		elif (platform_center_x +10) < ball_destination :
			comm.send_instruction(scene_info.frame, GameInstruction.CMD_RIGHT)

		else :
			comm.send_instruction(scene_info.frame, GameInstruction.CMD_NONE)


		# comm.send_instruction(scene_info.frame, GameInstruction.CMD_LEFT)
