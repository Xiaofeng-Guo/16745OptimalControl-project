import gym
import d4rl
import numpy as np


def linearize(env, Xref, Uref):

	N = Xref.shape[0]
	n,m = 18, 9
	env.reset()
	A = [np.zeros((n,n)) for k in range(0,N-1)]
	B = [np.zeros((n,m)) for k in range(0,N-1)]

	delta = 0.0001

	for step in range(N-1):

		x = Xref[step]
		u = Uref[step]

		env.sim.data.qpos[:9] = x[:9]
		# env.sim.data.qvel[:9] = np.zeros(9)
		env.sim.data.qvel[:9] = x[9:18]
		env.sim.forward()
		observation, reward, done, info = env.step(u)

		fxu = observation[:18]

		for i in range(18):
			dx = np.zeros(18)
			dx[i] += delta
			env.sim.data.qpos[:9] = x[:9] + dx[:9]
			env.sim.data.qvel[:9] = x[9:18] + dx[9:]
			env.sim.forward()
			observation, reward, done, info = env.step(u)

			fdxu = observation[:18]
			A[step][:, i] = (fdxu - fxu) / delta

		for i in range(9):
			du = np.zeros(9)
			du[i] += delta
			env.sim.data.qpos[:9] = x[:9]
			# env.sim.data.qvel[:9] = np.zeros(9)
			env.sim.data.qvel[:9] = x[9:18]
			env.sim.forward()
			observation, reward, done, info = env.step(u + du)

			fxdu = observation[:18]
			B[step][:, i] = (fxdu - fxu) / delta
		print(step)
	np.save('data/trial2/A.npy', A)
	np.save('data/trial2/B.npy', B)

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
	observation = env.reset()
	X = observation[:18]
	cost = 0
	N = len(K)+1
	for k in range(0,N-1):
		U = Uref[k] #- K[k]@(X-Xref[k])
		# U[k] = clamp.(U[k], -u_bnd, u_bnd)
		print(U)
		observation, reward, done, info = env.step(U)
		env.render()
		X = observation[:18]
		# X[k+1]  = true_dynamics_rk4(model, X[k], U[k], dt)
	

	return cost


def forward_ref_sim(env, Xref, Uref):
	# return cost
	observation = env.reset()
	X = observation[:18]
	cost = 0
	N = len(K) + 1
	for k in range(0, N - 1):
		U = Uref[k]
		observation, reward, done, info = env.step(U)
		env.render()
		X = observation[:18]
	# X[k+1]  = true_dynamics_rk4(model, X[k], U[k], dt)

	return cost


if __name__ == '__main__':
	env = gym.make('kitchen-complete-v0')
	env.reset()
	A = np.load('data/trial2/A.npy')
	B = np.load('data/trial2/B.npy')
	U_ref = np.load('data/trial2/uref.npy')
	X_ref = np.load('data/trial2/xref.npy')

	q = [0.1]*9+[0]*9
	Q = np.diag(q)
	R = np.identity(9)*0.1

	# A,B = linearize(env,X_ref,U_ref)
	K,P = tvlqr(A,B,Q,R,Q)
	# import ipdb;ipdb.set_trace()

	forward_sim(env,K,P,X_ref,U_ref)
	# forward_ref_sim(env,X_ref, U_ref)

	input('\n')
# while True:
	# 	env.render()
	# 	env.sim.data.qpos[:self.n_jnt] = reset_pose[:self.n_jnt].copy()
	# 	env.sim.data.qvel[:self.n_jnt] = reset_vel[:self.n_jnt].copy()
	# 	env.step(action)
