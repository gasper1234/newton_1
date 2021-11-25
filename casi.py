from scipy.integrate import solve_ivp
from pendulum import *

plt.figure(figsize=(8,4))

def step_finder(err, f):
	fi0=1.
	v0=0.

	t_real = t_0(fi0)
	iconds=np.array([fi0,v0])

	err_0 = 10

	dt_map = []
	dt_step = []
	N = 0

	dt_min = 0.0001
	dt_max = 0.5
	while abs(err_0 - err) / err > 0.1:
		dt_mid = (dt_max+dt_min)/2
		dt_map.append(dt_mid)
		N += 1
		dt_step.append(N)
		t = np.arange(0.0, np.pi*3, dt_mid)
		x = f(pendulum,iconds,t)
		err_0 = abs(find(t, x[:,0])-t_real)
		#print('napaka', err_0)
		#print('pogoj', abs(err_0 - err) / err)
		if err_0 < err:
			dt_min = dt_mid
		else:
			dt_max = dt_mid
		if abs(dt_max-dt_min) / dt_min < 0.0001:
			return dt_min
	return dt_mid

def step_finder_frog(err, f):
	fi0=1.
	v0=0.

	t_real = t_0(fi0)
	iconds=np.array([fi0,v0])

	err_0 = 10

	dt_map = []
	dt_step = []
	N = 0

	dt_min = 0.0001
	dt_max = 0.5
	while abs(err_0 - err) / err > 0.1:
		dt_mid = (dt_max+dt_min)/2
		dt_map.append(dt_mid)
		N += 1
		dt_step.append(N)
		t = np.arange(0.0, np.pi*3, dt_mid)
		x = f(pendulum_frog,iconds[0], iconds[1],t)
		err_0 = abs(find(t, x[0])-t_real)
		#print('napaka', err_0)
		#print('pogoj', abs(err_0 - err) / err)
		if err_0 < err:
			dt_min = dt_mid
		else:
			dt_max = dt_mid
		if abs(dt_max-dt_min) / dt_min < 0.0001:
			return dt_min
	return dt_mid

functions = [euler, heun, rk2a, rk2b, rku4, pc4, integrate.odeint]
#functions = [rku4]
fi0=1
v0=0.

t_real = t_0(fi0)
iconds=np.array([fi0,v0])


for k in range(2, 5):
	fun = []
	val = []
	err = 10**(-k)
	for i in range(len(functions)):
		dt = step_finder(err, functions[i])
		fun.append(str(functions[i])[10:-23])
		val.append(dt)

functions_2 = [verlet, pefrl]

	for i in range(len(functions_2)):
		dt = step_finder_frog(err, functions_2[i])
		fun.append(str(functions_2[i])[10:-23])
		val.append(dt)


	plt.plot(fun, val, 'o', label=str(err))
plt.legend(title='err')
plt.yscale('log')
plt.xlabel('metoda')
plt.ylabel(r'$dt$')
plt.ylim(10**(-4)/2, 1)
plt.show()
