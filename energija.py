from scipy.integrate import solve_ivp
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from pendulum import *


fig, ax = plt.subplots(2, 1)
for ik in range(2):
	if ik == 0:
		dt =  0.001
		t = np.arange(0.0, np.pi*2, dt)
		axs = ax[0]
	else:
		dt =  0.1
		t = np.arange(0.0, np.pi*40, dt)
		axs = ax[1]

	print(len(t))
	fi0=1.
	v0=0.

	E_0 = 1-np.cos(fi0)+v0**2/2
	E_0_arr = [E_0 for i in range(len(t))]

	iconds=np.array([fi0,v0])

	functions = [euler, heun, rku4]

	x_ivp = solve_ivp(pendulum_ivp, [0, t[-1]], iconds, atol=1e-13, rtol=1e-13, method='DOP853',)
	energ = energy(x_ivp.y[0], x_ivp.y[1])
	print(len(energ))
	E_0_arr_ivp = [E_0 for i in range(len(energ))]
	axs.plot(x_ivp.t, abs(energ-E_0_arr_ivp), label='solve_ivp')

	for i in range(len(functions)):
		x = functions[i](pendulum,iconds,t)
		energ = energy(x[:,0], x[:,1])
		axs.plot(t, abs(energ-E_0_arr), label=str(functions[i])[10:-23])

	functions_2 = [verlet, pefrl]

	for i in range(len(functions_2)):
		x = functions_2[i](pendulum_frog,iconds[0], iconds[1],t)
		energ = energy(x[0], x[1])
		axs.plot(t, abs(energ-E_0_arr), label=str(functions_2[i])[10:-23])


	axs.legend(loc='lower right')
	if ik == 1:
		axs.set_xlabel('t')
	axs.set_ylabel(r'$\Delta E$')
	axs.set_yscale('log')
plt.show()