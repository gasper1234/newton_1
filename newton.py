from scipy.integrate import solve_ivp
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from pendulum import *

dt =  0.1
t = np.arange(0.0, np.pi*6, dt)

fi0=np.pi/2
v0=0.

iconds=np.array([fi0,v0])


colormap = plt.cm.viridis
num_plots = 10
plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.viridis(np.linspace(0, 1, num_plots))))

normalize = mcolors.Normalize(0, fi0)

scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(np.arange(0, fi0, num_plots))
cbar = plt.colorbar(scalarmappaple, location='right')

#cbar.set_label('k', rotation=0)


for i in range(num_plots):
	iconds = np.array([fi0*i/num_plots,v0])
	x_scipy = integrate.odeint(pendulum,iconds,t)
	x_ivp = solve_ivp(pendulum_ivp, [0, t[-1]], iconds, t_eval=t)


	plt.plot(t, x_ivp.y[0])
plt.xticks([np.pi*i*2 for i in range(4)], ['0']+[str(i)+r'$t_0$' for i in range(1, 4)])
plt.yticks([np.pi*i/4 for i in range(-2, 3)], [r'$-\pi/2$', r'$-\pi/4$', r'$0$', r'$\pi/4$', r'$\pi/2$'])
for i in range(1, 4):
	plt.vlines(2*np.pi*i, -1.5, 1.5, 'k')
plt.xlabel(' '*53+'t'+' '*53+'k')
plt.ylabel(r'$\varphi_0$')
plt.show()
