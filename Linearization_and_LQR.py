import numpy as np
from numpy import ndarray
import Modeling as m
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import time 


class LQR:
    
    def __init__(self,Double_Pendulum :object,N_state_dim :int = 4, M_action_dim :int = 2, total_sim_time :float = 28.,total_points = 50000,dt :float = 0.05):
        self.dp = Double_Pendulum
        self.N = N_state_dim # Dimensions of state
        self.M = M_action_dim # Dimensions of action
        self.total_sim_time = total_sim_time # Overall simulation time
        self.dt = dt # Time step
        self.Points :int = int(self.total_sim_time/self.dt) + 1
        self.Points_infinite_horizon :int  = total_points
        self.control_limits_barrier :ndarray = np.array([[-10.0,10.0]]).T 
    
    def linearize(self,x_bar = jnp.array([[np.pi,0,0,0]]).T, u_bar = jnp.array([[0.0,0.0]]).T):
        
        # Discretization of the continues dynamics
        f_discrete = lambda x_bar,u_bar:(self.dp.runge_kutta4(x_bar,u_bar,self.dt)).reshape(4,)
        # print(f_discrete(x_bar,u_bar),jnp.shape(f_discrete(x_bar,u_bar)))
        
        # Tensor derivative of f_discrete with respect to x:state
        df_discrete_tensor_wrt_x = jax.jit(jax.jacobian(f_discrete,0))
        
        # The Jacobian of discrete dynamics wrt x == Ad
        df_discrete_wrt_x = lambda x_bar,u_bar: df_discrete_tensor_wrt_x(x_bar,u_bar).reshape(4,4)
        # print(df_discrete_wrt_x(x_bar,u_bar),jnp.shape(df_discrete_wrt_x(x_bar,u_bar)))
        
        # Tensor derivative of f_discrete with respect to u:controls
        df_discrete_tensor_wrt_u = jax.jit(jax.jacobian(f_discrete,1))
        
        # The Jacobian of discrete dynamics wrt u == bd
        df_discrete_wrt_u = lambda x_bar,u_bar: df_discrete_tensor_wrt_u(x_bar,u_bar).reshape(4,2)
        # print(df_discrete_wrt_u(x_bar,u_bar),jnp.shape(df_discrete_wrt_u(x_bar,u_bar)))
        
        # Ad and Bd respectively
        # print(type(df_discrete_wrt_x(x_bar,u_bar)))
        return df_discrete_wrt_x(x_bar,u_bar),df_discrete_wrt_u(x_bar,u_bar)
    
    def lqr(self,Ad :ndarray,Bd :ndarray,QN :ndarray,Q :ndarray,R :ndarray):
        # Calculating the P and K from LQR-Riccati
        
        P = [np.zeros((self.N,self.N))] * self.Points_infinite_horizon
        K = [np.zeros((self.M,self.N))] * (self.Points_infinite_horizon - 1)
        
        P[self.Points_infinite_horizon-1] = QN
        
        for i in range(self.Points_infinite_horizon - 2, -1, -1):
            
            K[i] = np.linalg.inv(R + Bd.T @ P[i+1] @ Bd) @ Bd.T @ P[i+1] @ Ad
            P[i] = Q + Ad.T @ P[i+1] @ (Ad - Bd @ K[i])
        
        return P,K
    
    def rollout(self,x_init :ndarray,K : list[ndarray],x_ref :ndarray,u_ref :ndarray) -> list[ndarray]:
        
        x = np.copy(x_init)
        
        states = [np.copy(x)]
        u_controls :list = []
        
        for _ in range(self.Points-1): # Kinf = K[0]
            u = -K[0] @ (x - x_ref) + u_ref
            
            u = self.limit_u_controls(u)
            
            x = self.dp.runge_kutta4(x,u,self.dt)
            states.append(np.copy(x))
            u_controls.append(u)
            
        return states,u_controls
    
    def limit_u_controls(self,u) -> ndarray:
        u = jax_np_to_np(u)
        for i in range(self.M):
            if u[i,0] < self.control_limits_barrier[0,0]:
                u[i,0] = self.control_limits_barrier[0,0]
                
            if u[i,0] > self.control_limits_barrier[1,0]:
                u[i,0] = self.control_limits_barrier[1,0]
                
        return u
    
    def L_cost(self,states :list, u_controls :list,QN :ndarray,Q :ndarray,R :ndarray) -> int:
        total_cost :int  = 0.5 * states[-1].T @ QN @ states[-1]
        
        for i in range(self.Points - 1):
            total_cost += 0.5 * states[i].T @ Q @ states[i]
            total_cost += 0.5 * u_controls[i].T @ R @ u_controls[i]
        
        return total_cost[0,0]
    
def jax_np_to_np(Matrix_jax) -> ndarray:
    rows, cols = jnp.shape(Matrix_jax)

    Output_Matrix = np.zeros((rows,cols))
    
    for row in range(rows):
        for col in range(cols):
            
            Output_Matrix[row,col] = Matrix_jax[row,col]
            
    return Output_Matrix


def main():
    # Defining essential variables for the system
    dp = m.DoublePendulum()
    N_state_dim = 4
    M_action_dim = 2
    
    # Linearize the system around x:state = [np.pi,0,0,0].T and u:controls = [0.0,0.0].T
    x_bar = np.array([[np.pi,0.0,0.0,0.0]]).T # Fixed point
    u_bar = np.array([[0.0,0.0]]).T
    
    sim_lqr = LQR(dp,N_state_dim,M_action_dim)
    Ad_jax,Bd_jax = sim_lqr.linearize(x_bar = jnp.array([[np.pi,0.0,0.0,0.0]]).T, u_bar = jnp.array([[0.0,0.0]]).T)
    
    # Converting jax.numpy to numpy
    Ad = jax_np_to_np(Ad_jax)
    Bd = jax_np_to_np(Bd_jax)
    
    print("Linearized matrices of the system\n")
    print("Ad = \n",Ad,"\n","Bd = \n",Bd)
    
    # Initializing the Qk and Rk matrices of LQR
    # QN = 11e5*np.eye(N_state_dim) 
    QN = Q = 3000*np.eye(N_state_dim)
    
    # Q = 18e2*np.eye(N_state_dim) #+ np.array([[0,0,0,0],[0,0,0,0],[0,0,450,0],[0,0,0,1000]])
    R = 0.005*np.eye(M_action_dim)
    
    # Initial state
    x_init = np.array([[0.0,0.0,0.0,0.0]]).T 
    u_init = np.array([[0.0,0.0]]).T
    # x_init = np.array([[np.pi+0.3,-0.2,0.0,0.0]]).T 
    # x_init = np.array([[np.pi,-0.1,0.0,0.0]]).T 
    
    u_controls_no_action = [np.zeros((M_action_dim,1)) for _ in range(sim_lqr.Points - 1)]
    
    P, K = sim_lqr.lqr(Ad,Bd,QN,Q,R)
    
    
    states,u_controls = sim_lqr.rollout(x_init,K,x_bar,u_bar)
    print("\n\nThe initial value of cost function with no action")
    print(sim_lqr.L_cost(states,u_controls_no_action,QN,Q,R))
    
    print("\n\nThe final value of cost function")
    print(sim_lqr.L_cost(states,u_controls,QN,Q,R))
    
    print("Simulation will start in 1 sec")
    time.sleep(1)
    
    # Offline simulation
    visual = m.Visualization(dp.base_coord,7,dp.l1,dp.l2,0.05)
    # visual.fig.canvas.mpl_connect('close_event',visual._on_close)
    
    i = 0
    for x in states:
        
        print(f"State :\n{x}\n")
        print(f"U_controls:\n {u_controls[i]}\n")
        visual.update(x[0,0],x[1,0])
        
        if i < len(states)-2:
            i += 1 
    
    plt.ioff()
    # plt.show()
    
    # exit(1)
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    ax.plot([i*0.05 for i in range(sim_lqr.Points)],[states[i][0,0] for i in range(sim_lqr.Points)],label = "q1")
    ax.plot([i* 0.05 for i in range(sim_lqr.Points)],[states[i][1,0] for i in range(sim_lqr.Points)],label = "q2")
    ax.plot([i*0.05 for i in range(sim_lqr.Points)],[states[i][2,0] for i in range(sim_lqr.Points)],label = "q1_dot")
    ax.plot([i* 0.05 for i in range(sim_lqr.Points)],[states[i][3,0] for i in range(sim_lqr.Points)],label = "q2_dot")
    u_controls = [u_init] + u_controls
    ax.plot([i* 0.05 for i in range(sim_lqr.Points)],[u_controls[i][0,0] for i in range(sim_lqr.Points)],label = "u1")
    ax.plot([i* 0.05 for i in range(sim_lqr.Points)],[u_controls[i][1,0] for i in range(sim_lqr.Points)],label = "u2")
    plt.xlabel("Time")
    plt.legend()
    plt.show()
    
    
if __name__ == "__main__":
    main()