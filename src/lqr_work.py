import gym
import d4rl
import mujoco_py
import numpy as np


def set_state(env, qpos, qvel):

	state = env.sim.get_state()
	for i in range(env.n_jnt):
		state[i] = qpos[i]
	for i in range(env.model.nq,env.model.nq+env.n_jnt):
		state[i] = qvel[i-env.model.nq]
	env.sim.set_state(state)
	env.sim.forward()


def linearize(env, Xref, Uref):

	# import ipdb;ipdb.set_trace()
	N = Xref.shape[0]
	n,m = 18, 9
	env.reset()
	A = [np.zeros((n,n)) for k in range(0,N-1)]
	B = [np.zeros((n,m)) for k in range(0,N-1)]

	delta = 0.0001

	for step in range(N-1):

		x = Xref[step]
		u = Uref[step]

		set_state(env,x[:9],x[9:18])

		# observation, reward, done, info = env.step(u)
		# forward sim
		for i in range(env.model.nu):
			env.sim.data.ctrl[i] = u[i]
		env.sim.step()

		fxu = np.concatenate((env.sim.data.qpos[:9],env.sim.data.qpos[:9]))

		for i in range(18):
			dx = np.zeros(18)
			dx[i] += delta

			set_state(env,x[:9]+ dx[:9],x[9:18]+ dx[9:])
			for i in range(env.model.nu):
				env.sim.data.ctrl[i] = u[i]
			env.sim.step()

			fdxu = np.concatenate((env.sim.data.qpos[:9],env.sim.data.qpos[:9]))
			A[step][:, i] = (fdxu - fxu) / delta

		for i in range(9):
			du = np.zeros(9)
			du[i] += delta

			set_state(env,x[:9],x[9:18])
			for i in range(env.model.nu):
				env.sim.data.ctrl[i] = (u+du)[i]
			env.sim.step()

			fxdu = np.concatenate((env.sim.data.qpos[:9],env.sim.data.qpos[:9]))
			B[step][:, i] = (fxdu - fxu) / delta
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

	X = [np.zeros(18) for k in range(0,N)]
	U = [np.zeros(9) for k in range(0,N-1)]
	observation = env.reset()
	X[0] = observation[:18]

	pid_k = np.concatenate((np.identity(9)*1,np.identity(9)*0.01),axis=1)

	for k in range(0,N-1):
		U[k] = Uref[k] - K[k]@(X[k]-Xref[k])
		# U[k] = clamp.(U[k], -u_bnd, u_bnd)
		observation, reward, done, info = env.step(U[k])
		env.render()
		X[k+1] = observation[:18]
		# X[k+1]  = true_dynamics_rk4(model, X[k], U[k], dt)
		cost += 0.5*(X[k]-Xref[k])@Q@((X[k]-Xref[k])) + 0.5*(U[k])@R@(U[k])
	cost += 0.5*(X[N-1]-Xref[N-1])@Q@((X[N-1]-Xref[N-1]))

	
	X = np.asarray(X)
	Xref = np.asarray(Xref)

	plt.plot(X[:,0],label='est')
	plt.plot(Xref[:,0],label='GT')
	plt.legend()
	plt.show()

	print(cost)
	return cost

if __name__ == '__main__':
	env = gym.make('kitchen-complete-v0')
	env.reset()
	A = np.load('A.npy')
	B = np.load('B.npy')
	U_ref = np.load('uref.npy')
	X_ref = np.load('xref.npy')

	q = [10]*9+[1]*9
	Q = np.diag(q)
	R = np.identity(9)*0.01

	A,B = linearize(env,X_ref,U_ref)
	K,P = tvlqr(A,B,Q,R,Q)
	import ipdb;ipdb.set_trace()

	forward_sim(env,K,P,X_ref,U_ref)

	# while True:
	# 	env.render()
	# 	env.sim.data.qpos[:self.n_jnt] = reset_pose[:self.n_jnt].copy()
	# 	env.sim.data.qvel[:self.n_jnt] = reset_vel[:self.n_jnt].copy()
	# 	env.step(action)
