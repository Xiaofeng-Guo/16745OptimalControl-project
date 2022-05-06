import gym
import d4rl
import mujoco_py
import numpy as np

import imageio



def set_state(env, qpos):

	state = env.sim.get_state()
	for i in range(env.n_jnt):
		state[i] = qpos[i]
	for i in range(env.model.nq,env.model.nq+env.n_jnt):
		state[i] = 0.0
	env.sim.set_state(state)
	env.sim.forward()
	env.sim.step()

def linearize(env, Xref, Uref):

	N = Xref.shape[0]
	n,m = 18, 9
	env.reset()
	A = [np.zeros((n,n)) for k in range(0,N-1)]
	B = [np.zeros((n,m)) for k in range(0,N-1)]

	delta = 0.00001

	for step in range(N-1):

		x = Xref[step]
		u = Uref[step]

		set_state(env, x[:9])
		observation, reward, done, info = env.step(u)

		fxu = observation[:7]

		for i in range(7):
			dx = np.zeros(9)
			dx[i] += delta
			# env.sim.data.qpos[:9] = x[:9] + dx[:9]
			# env.sim.data.qvel[:9] = x[9:18] + dx[9:]
			# env.sim.forward()
			set_state(env, x[:9] + dx)
			observation, reward, done, info = env.step(u)


			fdxu = observation[:7]
			A[step][0:7, i] = (fdxu - fxu) / delta

		for i in range(7):
			du = np.zeros(9)
			du[i] += delta
			# env.sim.data.qpos[:9] = x[:9]
			# # env.sim.data.qvel[:9] = np.zeros(9)
			# env.sim.data.qvel[:9] = x[9:18]
			# env.sim.forward()
			set_state(env,x[:9])
			observation, reward, done, info = env.step(u + du)

			fxdu = observation[:7]
			B[step][:7, i] = (fxdu - fxu) / delta
		print(step)
	np.save('A.npy', A)
	np.save('B.npy', B)

	return A, B


def tvlqr(A,B,Q,R,Qf):

	n,m = B[1].shape
	N = len(A)+1
	K = [np.zeros((m,n)) for k in range(0,N-1)]
	P = [np.zeros((n,n)) for k in range(0,N)]
	P[-1] = Qf
	for k in reversed(range(0,N-1)):
		K[k] = np.linalg.pinv(R + B[k].T@P[k+1]@B[k]) @ (B[k].T@P[k+1]@A[k])
		# K[k] .= (R + B[k]'P[k+1]*B[k])\(B[k]'P[k+1]*A[k])
		P[k] = Q + A[k].T@P[k+1]@A[k] - A[k].T@P[k+1]@B[k]@K[k]
	return K,P

# def cost()

def forward_sim(env,K,P,Xref,Uref):
	# return cost
	import matplotlib.pyplot as plt
	cost = 0
	N = len(K)+1
	N = 200

	X = [np.zeros(18) for k in range(0,N)]
	U = [np.zeros(9) for k in range(0,N-1)]
	observation = env.reset()
	X[0] = observation[:9]

	pid_k = np.concatenate((np.identity(9)*1,np.identity(9)*0.0),axis=1)

	for k in range(0,N-1):
		U[k] = Uref[k] - K[k][0:9,0:9]@(X[k]-Xref[k])
		# U[k] = clamp.(U[k], -u_bnd, u_bnd)
		observation, reward, done, info = env.step(U[k])
		env.render()
		X[k+1] = observation[:9]
		# X[k+1]  = true_dynamics_rk4(model, X[k], U[k], dt)
	X = np.asarray(X)
	Xref = np.asarray(Xref)

	plt.plot(X[:,5])
	plt.plot(Xref[:,5])
	plt.show()

	return cost


def forward_ref_sim(env,Xref,Uref):
	# return cost
	import matplotlib.pyplot as plt
	cost = 0
	# N = len(Xref)+1
	N = 200
	X = [np.zeros(18) for k in range(0,N)]
	U = [np.zeros(9) for k in range(0,N-1)]
	observation = env.reset()
	X[0] = observation[:18]

	pid_k = np.concatenate((np.identity(9)*1,np.identity(9)*0.01),axis=1)

	for k in range(0,N-1):
		# U[k] = Uref[k] - K[k]@(X[k]-Xref[k])
		U[k] = Uref[k]
		# if Xref[k][7] < 0.01:
		# 	U[k][7] = -1
		# if Xref[k,8] < 0.01:
		# 	U[k][8] = -1
		# U[k] = clamp.(U[k], -u_bnd, u_bnd)
		observation, reward, done, info = env.step(U[k])
		env.render()
		X[k+1] = observation[:18]
		# X[k+1]  = true_dynamics_rk4(model, X[k], U[k], dt)
	X = np.asarray(X)
	Xref = np.asarray(Xref)

	plt.plot(X[:,1])
	plt.plot(Xref[:,1])
	plt.show()

	return cost



def fake_forward_ref_sim(env,Xref,Uref):
	# return cost
	import matplotlib.pyplot as plt
	cost = 0
	N = len(K)+1

	X = [np.zeros(18) for k in range(0,N)]
	U = [np.zeros(9) for k in range(0,N-1)]
	observation = env.reset()
	X[0] = observation[:18]

	pid_k = np.concatenate((np.identity(9)*1,np.identity(9)*0.01),axis=1)

	for k in range(0,N-1):
		set_state(env,Xref[k])
		env.render()
	return




if __name__ == '__main__':
	env = gym.make('kitchen-complete-v0')
	env.reset()
	# A = np.load('A.npy')
	# B = np.load('B.npy')
	U_ref = np.load('data/trial3/Uref.npy')
	X_ref = np.load('data/trial3/obs.npy')

	q = [0.001,0.01,1,1,0.000000001,1,0.1]+[0]*2+[0]*9
	Q = np.diag(q)
	R = np.identity(9)*0.1

	X_ref[20:29, 7:9] = 0.04
	X_ref[29:45, 7:9] = 0.002
	X_ref[45:60, 7:9] = 0.04
	X_ref[65:75, 7:9] = 0.002

	# U_ref[20:29, 7:9] = 0.04
	U_ref[29:45, 7:9] = -1
	# U_ref[45:60, 7:9] = 0.04
	U_ref[65:75, 7:9] = -1



	# A,B = linearize(env,X_ref,U_ref)
	# K,P = tvlqr(A,B,Q,R,Q)
	# print(K)
	# import ipdb;ipdb.set_trace()

	# forward_sim(env,K,P,X_ref,U_ref)
	forward_ref_sim(env,X_ref,U_ref)
	# fake_forward_ref_sim(env,X_ref,U_ref)

	# video_name = 'video.mp4'
	# frame = grabFrame(env)
	# height, width, layers = frame.shape
	# video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width, height))
	#
	# # First pass - Step through an episode and capture each frame
	# action_spec = env.action_spec()
	# time_step = env.reset()
	# while not time_step.last():
	# 	action = np.random.uniform(action_spec.minimum,
	# 							   action_spec.maximum,
	# 							   size=action_spec.shape)
	# 	time_step = env.step(action)
	# 	frame = grabFrame(env)
	# Render env output to video

	while True:
		env.render()
		rgb = env.render(mode='rgb_array')
		print(rgb)
	# 	env.sim.data.qpos[:self.n_jnt] = reset_pose[:self.n_jnt].copy()
	# 	env.sim.data.qvel[:self.n_jnt] = reset_vel[:self.n_jnt].copy()
	# 	env.step(action)
