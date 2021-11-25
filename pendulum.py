from numpy import sin, cos, pi, array
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy.special import ellipk
from diffeq_2 import *

#-----------------------------------------------------------------------------
# global variable as frequency

omega2=0.1

#m = 1kg
#g = 1m/s^2
#l = 1m

#force_real
def F(fi):
    return np.sin(fi)

#dx/dt for different methods
def pendulum(state, t):

    dydt=np.zeros_like(state)
    dydt[0]=state[1]
    dydt[1]=-F(state[0])  # v' = F(x)

    return dydt

def pendulum_frog(state):
    return -F(state)

def pendulum_ivp(t, state):

    dydt=np.zeros_like(state)
    dydt[0]=state[1]
    dydt[1]=-F(state[0])  # v' = F(x)

    return dydt

#real pendulum solution
def mathematical(t):

    return np.cos(t)

#pendulum period soluiton
def t_0(fi0):
    return 4*ellipk(np.sin(fi0/2)**2)

#period calculator
def find(t, x):
    for i in range(1, len(t)-1):
        if x[i-1] < x[i] and x[i+1] < x[i]:
            t_osc = [t[j] for j in range(i-3, i+4)]
            x_osc = [x[j] for j in range(i-3, i+4)]
            break


    fit_7 = np.polynomial.polynomial.Polynomial.fit(t_osc, x_osc, 2).convert().coef

    t_fit = np.linspace(t_osc[0], t_osc[-1], 50)
    t_fit_val = np.array([x**2 * fit_7[2] + x * fit_7[1] + fit_7[0] for x in t_fit])
    '''
    plt.figure(figsize=(8,4))
    plt.plot(t_osc, x_osc, 'x', label='data')
    plt.plot(t_fit, t_fit_val, 'r-', label='polinom')
    plt.xlabel('t')
    plt.ylabel(r'$\varphi$')
    plt.vlines(-fit_7[1]/(2*fit_7[2]), min(x_osc), 1.005, color='k', label=r'$t_0$')
    plt.legend()
    plt.show()
    '''
    return -fit_7[1]/(2*fit_7[2])

#-----------------------------------------------------------------------------
# linear force (e.g. spring pendulum)
def forcel(y):
    return -omega2*y # sin(y)

#-----------------------------------------------------------------------------
# linear force (e.g. spring pendulum)
def energy_l(y,v):
    return (omega2*y*y+v*v)/2.

#energy pendulum
def energy(phi, v):
    E = (1-np.cos(phi))+v**2/2
    return E
#-----------------------------------------------------------------------------
# a simple pendulum y''= F(y) , state = (y,v)
def pendulum_1(state,t): 

    dydt=np.zeros_like(state)
    dydt[0]=state[1] # x' = v 
    dydt[1]=forcel(state[0])  # v' = F(x)

    return dydt

#-----------------------------------------------------------------------------
# a simple pendulum
'''
if __name__ == "__main__":

    #import diffeq
    from pylab import *

    ion()
    # create a time array from 0..100 sampled at 0.1 second steps
    dt =  0.2
    t = np.arange(0.0, 100, dt)

    #initial conditions
    x0=1.
    v0=0.

    iconds=np.array([x0,v0])
    x_scipy=integrate.odeint(pendulum,iconds,t)

    x_euler=euler(pendulum,iconds,t)
    x_er_eu=x_euler[:,0]-x_scipy[:,0]

    x_rk4=rku4(pendulum,iconds,t)
    x_er_rk=x_rk4[:,0]-x_scipy[:,0]

    x_verlet=verlet(forcel,x0,v0,t)
    x_er_ver=x_verlet[0,:]-x_scipy[:,0]

    x_pefrl=pefrl(forcel,x0,v0,t)
    x_er_pef=x_pefrl[0,:]-x_scipy[:,0]

    plot(t,x_scipy[:,0],'r-o',t,x_euler[:,0],'b-o',t,x_verlet[0,:],'g-o')
    title('Solutions of $d^2x/dt^2=-\omega^2*x$,$x(0)=1$')
    legend(('Scipy','Euler','Verlet'),loc='lower left')
    draw()

    input( "Press Enter to continue... " )
    cla()
    plot(t,x_er_eu,'g-o',t,x_er_ver,'b-o')
    title('Solutions-differences of $d^2x/dt^2=-\omega^2*x$,$x(0)=1$')
    legend(('Euler-Scipy','Verlet-Scipy'),loc='lower left')
    draw()

    input( "Press Enter to continue... " )
    cla()
    plot(t,x_scipy[:,0],'r-o',t,x_rk4[:,0],'g-o',t,x_verlet[0,:],'b-o')
    title('Solutions of $d^2x/dt^2=-\omega^2*x$,$x(0)=1$')
    legend(('Scipy','RK4','Verlet'),loc='lower left')
    draw()

    input( "Press Enter to continue... " )
    cla()
    plot(t,x_er_rk,'g-o',t,x_er_ver,'b-o')
    title('Solutions-differences of $d^2x/dt^2=-\omega^2*x$,$x(0)=1$')
    legend(('RK4-Scipy','Verlet-Scipy'),loc='lower left')
    draw()

    input( "Press Enter to continue... " )
    cla()
    plot(t,x_scipy[:,0],'r-o',t,x_verlet[0,:],'g-o',t,x_pefrl[0,:],'b-o')
    title('Solutions of $d^2x/dt^2=-\omega^2*x$,$x(0)=1$')
    legend(('Scipy','Verlet','PEFRL'),loc='lower left')
    draw()

    input( "Press Enter to continue... " )
    cla()
    plot(t,x_er_ver,'g-o',t,x_er_pef,'b-o')
    title('Solutions-differences of $d^2x/dt^2=-\omega^2*x$,$x(0)=1$')
    legend(('Verlet-Scipy','PEFRL-Scipy'),loc='lower left')
    draw()

    input( "Press Enter to continue... " )
    cla()
    plot(t,x_scipy[:,0],'r-o',t,x_rk4[:,0],'g-o',t,x_pefrl[0,:],'b-o')
    title('Solutions of $d^2x/dt^2=-\omega^2*x$,$x(0)=1$')
    legend(('Scipy','RK4','PEFRL'),loc='lower left')
    draw()

    input( "Press Enter to continue... " )
    cla()
    plot(t,x_er_rk,'g-o',t,x_er_pef,'b-o')
    title('Solutions-differences of $d^2x/dt^2=-\omega^2*x$,$x(0)=1$')
    legend(('RK4-Scipy','PEFRL-Scipy'),loc='lower left')
    draw()


    #energy plots
    en_scipy=energy(x_scipy[:,0],x_scipy[:,1])
    en_euler=energy(x_euler[:,0],x_euler[:,1])
    en_rk4=energy(x_rk4[:,0],x_rk4[:,1])
    en_pefrl=energy(x_pefrl[0,:],x_pefrl[1,:])
    en_verlet=energy(x_verlet[0,:],x_verlet[1,:])
    en_true= numpy.array( [ omega2*0.5 ] * len(t) )

    input( "Press Enter to continue... " )
    cla()
    plot(t,en_true,'k-',t,en_euler,'g-o',t,en_verlet,'b-o')
    title('Energy of the system')
    legend(('True','Euler','Verlet'),loc='upper left')
    draw()

    input( "Press Enter to continue... " )
    cla()
    plot(t,en_true,'k-',t,en_rk4,'r-o',t,en_verlet,'b-o')
    title('Energy of the system')
    legend(('True','RK4','Verlet'),loc='upper left')
    draw()

    input( "Press Enter to continue... " )
    cla()
    plot(t,en_true,'k-',t,en_scipy,'r-o',t,en_rk4,'g-o',t,en_pefrl,'b-o')
    title('Energy of the system')
    legend(('True','Scipy','RK4','PEFRL'),loc='upper left')
    draw()


    input( "Press Enter to continue... " )
'''