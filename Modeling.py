import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import jax.numpy as jnp


class DoublePendulum:
    
    def __init__(self, m1 :float = 1.0,m2 :float = 1.0,l1 :float = 0.5,l2 :float = 0.5):
        # Parameters
        self.name :str = "DoublePendulum"
        self.base_coord :ndarray = np.array([[0.,0.]]).T 
        # Link 1 parameters
        self.m1 = m1 # kg
        self.l1 = l1 # m-meters
        # Link 2 parameters
        self.m2 = m2 # kg
        self.l2 = l2 # m-meters
        
        
    def _mass_matrix(self,x :ndarray):
        
        # The joint position of the 2nd pendulum
        q2 = x[1,0]
        
        l1 = self.l1
        m1 = self.m1
        
        l2 = self.l2
        m2 = self.m2
        
        A0 = (l1**2)*m1 + (l2**2)*m2 + (l1**2)*m2 + 2*l1*m2*l2*jnp.cos(q2)
        A1 = (l2**2)*m2 + l1*m2*l2*jnp.cos(q2)
        A2 = (l2**2)*m2 + l1*m2*l2*jnp.cos(q2)
        A3 = (l2**2)*m2
        
        M = jnp.array([[A0,A1],[A2,A3]])
        
        return M # 2x2 Mass Matrix of the double Pendulum
    
    def _coriolis_matrix(self,x :ndarray):
        
        # Unpacking the state of the system
        q2 = x[1,0] # The joint position of the 2nd pendulum
        q1_dot = x[2,0] # The joint velocity of the 1st pendulum
        q2_dot = x[3,0] # The joint velocity of the 2nd pendulum
        
        l1 = self.l1
        
        l2 = self.l2
        m2 = self.m2
        
        B0 = -2*q2_dot*l1*m2*l2*jnp.sin(q2)
        B1 = -q2_dot*l1*m2*l2*jnp.sin(q2)
        B2 = q1_dot*l1*m2*l2*jnp.sin(q2)
        B3 = 0.0
        
        C = jnp.array([[B0,B1],[B2,B3]])
        
        return C # 2x2 Coriolis of the double pendulum
    
    def _gravity_matrix(self,x :ndarray, g :float = 9.81):
        
        q1 = x[0,0] # The joint position of the 1st pendulum
        q2 = x[1,0] # The joint position of the 2nd pendulum
        
        l1 = self.l1
        m1 = self.m1
        
        l2 = self.l2
        m2 = self.m2
        
        D0 = -g*m1*l1*jnp.sin(q1) - g*m2*(l1*jnp.sin(q1) + l2*jnp.sin(q1 + q2))
        D1 = -g*m2*l2*jnp.sin(q1 + q2)
        
        G = jnp.array([[D0,D1]]).T
        
        return G
        
    # The dynamics of the Double Pendulum
    def dynamics(self,x :ndarray,u :ndarray):
        
        q_dot = jnp.array([[x[2,0],x[3,0]]]).T
        
        M_inv = jnp.linalg.inv(self._mass_matrix(x))
        
        q_ddot = M_inv @ (u - self._coriolis_matrix(x) @ q_dot + self._gravity_matrix(x))
        
        x_dot = jnp.vstack((q_dot,q_ddot))
        return x_dot
    
    # The discretization of the system using Runge and Kutta 4 order
    def runge_kutta4(self, x : ndarray, u : ndarray, dt :float = 0.01) -> ndarray:
        f1 = self.dynamics(x, u)
        f2 = self.dynamics(x + f1 *  (dt / 2), u)
        f3 = self.dynamics(x + f2 * (dt / 2), u)
        f4 = self.dynamics(x + f3 * dt, u)
        return x + (dt / 6) * (f1 + 2 * f2 + 2 * f3 + f4)
    
class Visualization:
    
    def __init__(self,base_coord :ndarray, ball_size :int,l1,l2, step = 0.01):
        
        self.base_coord = base_coord # The position of 1st servo 
        self.l1 = l1 # length of link1
        self.l2 = l2 # Length of Link2
        self.step = step # The discrete step, dt
        
        self.fig,self.ax = plt.subplots()
    
        # The size of the window
        xlim = 1.5*(self.l1 + self.l2)
        ylim = 1.5*(self.l1 + self.l2)
        self.ax.set_xlim(-xlim,xlim) 
        self.ax.set_ylim(-ylim,ylim)
        self.ax.set_aspect(1.0)
        self.ax.set_title("Double_Pendulum_Simulation")

        # Creating the Double Pendulum graphics using plot
        self.lineDP, = self.ax.plot([],[],"o-",markersize = ball_size,lw = 2)

        
        self.lineDP.set_data([],[]) # Initialize the first ploting line with no data
        
        plt.ion() # Turn on interactive mode
        plt.grid()
        plt.draw()
        plt.pause(self.step)
        
    
    def update(self,q1,q2):# Update the line that we are drawing with respect to q1 and q2
        
        x1 = self.l1 * np.cos(q1 - np.pi/2) # -pi/2 cause we assume that for q1 == q2 == 0 is on oy'
        y1 = self.l1 * np.sin(q1- np.pi/2)
        x2 = self.l1 * np.cos(q1- np.pi/2) + self.l2 * np.cos(q1 + q2 - np.pi/2)
        y2 = self.l1 * np.sin(q1- np.pi/2) + self.l2 * np.sin(q1 + q2 - np.pi/2)
        
        self.lineDP.set_data([self.base_coord[0,0],x1,x2],[self.base_coord[1,0],y1,y2])
        
        plt.draw()
        plt.pause(self.step)

    def _on_close(self,event):
        """A close event handler to close the terminal when visual is terminated"""
        print(f"User_Interacted ended on on_close, event_triggered: {event}")
        x = int(input("Close: Yes = 1"))
        if x:
            exit(1)
        
# Creating a sinusoidal control signal
def q(t,A = 1,f = 5,phi = np.pi/2):
    omega = 2*np.pi*f
    return A*np.sin(omega*t + phi)

def main():
    
    # Initial position
    x_init = np.array([[0.,0.,0.,0.]]).T
    u_init = np.array([[0.,0.]]).T
    
    dt = 0.01
    
    double_pendulum = DoublePendulum()
    # # x_dot = double_pendulum.dynamics(x_init,u_init)
    # x = double_pendulum.runge_kutta4(x_init,u_init)
    # print(x)
    
    # Creating a visual instance
    visual = Visualization(double_pendulum.base_coord,7,double_pendulum.l1,double_pendulum.l2,dt)
    # Attach the event handler of the closing window
    visual.fig.canvas.mpl_connect('close_event',visual._on_close)
    

    u_controls = [np.copy(u_init)]
    states = [np.copy(x_init)]
    u = np.copy(u_init)
    x = np.copy(x_init)
    t = 0
    for i in range(350):
        t += dt
        # Creating a sinusoidal control signal
        # for servo q1
        u[0,0] = q(t,A = 4.5,f = 1,phi = 0)
        
        x = double_pendulum.runge_kutta4(x,u,dt)
        
        print(i)
        print(x)
        print()
        
        visual.update(q1 = x[0,0],q2 = x[1,0])
        u_controls.append(np.copy(u))
        states.append(np.copy(x))


    plt.ioff()

    # Plotting the u_controls, q and q_dot of the above simulation
    fig2,ax2 = plt.subplots(3,1,figsize = (8,10))

    # Plotting the u_controls u1 :red , u2 :blue
    ax2[0].step([i*dt for i in range(351)],[u[0,0] for u in u_controls],"r-",where = "post",label = "u1")
    ax2[0].step([i*dt for i in range(351)],[u[1,0] for u in u_controls],"b-",where = "post",label = "u2")
    ax2[0].set_title("U_controls",loc = "left")
    ax2[0].set_xlabel("time")

    ax2[0].legend()

    # Plotting the q  q1 :red , q2 :blue
    ax2[1].plot([i*dt for i in range(351)],[state[0,0] for state in states],"r-",label = "q1")
    ax2[1].plot([i*dt for i in range(351)],[state[1,0] for state in states],"b-",label = "q2")
    ax2[1].set_title("q angles",loc = "left")
    ax2[1].set_xlabel("time")


    ax2[1].legend()

    # Plotting the q_dot  q1_dot :red , q2_dot :blue
    ax2[2].plot([i*dt for i in range(351)],[state[2,0] for state in states],"r-",label = "q1_dot")
    ax2[2].plot([i*dt for i in range(351)],[state[3,0] for state in states],"b-",label = "q2_dot")
    ax2[2].set_title("q_dot angular velocities",loc = "left")
    ax2[2].set_xlabel("time")


    ax2[2].legend()
    plt.show()


    return 0

if __name__ == "__main__":
    main()
