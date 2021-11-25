from scipy.integrate import solve_ivp
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from pendulum import *

dt =  0.3
t = np.arange(0.0, np.pi*40, dt)

fi0=1
v0=0.

iconds=np.array([fi0,v0])

functions = [euler, heun]

x_ivp = solve_ivp(pendulum_ivp, [0, t[-1]], iconds, t_eval=t)
energ = energy(x_ivp.y[0], x_ivp.y[1])
print(energ)
plt.plot(t, energ, label='solve_ivp')

for i in range(len(functions)):
	x = functions[i](pendulum,iconds,t)
	print(x[:,1])
	energ = energy(x[:,0], x[:,1])
	plt.plot(t, energ, label=str(functions[i])[10:-23])

functions_2 = [verlet, pefrl]

for i in range(len(functions_2)):
	x = functions_2[i](pendulum_frog,iconds[0], iconds[1],t)
	energ = energy(x[0], x[1])
	plt.plot(t, energ, label=str(functions_2[i])[10:-23])

plt.ylim(0.25, 0.75)
plt.legend(loc='lower left')
plt.xlabel('t')
plt.ylabel(r'E')
plt.show()

