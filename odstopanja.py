from scipy.integrate import solve_ivp
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from pendulum import *

plt.figure(figsize=(8, 4))
dt =  0.1
t = np.arange(0.0, np.pi*40, dt)

fi0=1.
v0=0.

iconds=np.array([fi0,v0])

x_ivp = solve_ivp(pendulum_ivp, [0, t[-1]], iconds, atol=1e-13, rtol=1e-13, method='DOP853', t_eval=t)
x_acc = x_ivp.y[0]


functions = [euler, heun, rku4]


for i in range(len(functions)):
	x = functions[i](pendulum,iconds,t)
	x = x[:,0]
	plt.plot(t, abs(x-x_acc), label=str(functions[i])[10:-23])

functions_2 = [verlet, pefrl]

for i in range(len(functions_2)):
	x = functions_2[i](pendulum_frog,iconds[0], iconds[1],t)
	plt.plot(t, abs(x[0]-x_acc), label=str(functions_2[i])[10:-23])

plt.legend(loc='lower right')
plt.xlabel('t')
plt.ylabel(r'$\Delta \varphi$')
plt.yscale('log')
plt.show()