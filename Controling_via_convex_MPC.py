import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import jax.numpy as jnp
import Modeling as m
import Linearization_and_LQR as L_LQP
import proxsuite
import time

class MPC:
    
    def __init__(self,N :int,M :int,Ad :ndarray,Bd :ndarray):
        self.N = N
        self.M = M
        self.Ad = Ad
        self.Bd = Bd
        self.H :int = 20
        
        self.qp = None
        self.H_matrix = None
        self.G_matrix = None
        self.C_matrix = None
        self.h = None
        self.d = None 
        self.lower = None
        self.upper = None
    
    def mpc_controller(self,x :ndarray, x_target :ndarray,Q :ndarray,R :ndarray,P_inf :ndarray):
        
        self.d[:self.N,:] = -self.Ad @ x
        
        for i in range(self.H-2):
            self.h[i * (self.N + self.M) + self.M:i * (self.N + self.M) + self.M + self.N] = - Q @ x_target
        self.h[-self.N:] = - P_inf @ x_target
        
        
        pre_x = self.qp.results.x
        pre_eq = self.qp.results.y
        pre_in = self.qp.results.z
        
        
        self.qp.update(self.H_matrix,self.h,self.G_matrix,self.d,self.C_matrix,self.lower,self.upper)
        
        self.qp.solve(pre_x,pre_eq,pre_in)
        
        pre_x = self.qp.results.x
        pre_eq = self.qp.results.y
        pre_in = self.qp.results.z
        
        u1 = pre_x[:self.M].reshape((self.M,-1))
        return u1
    
    def qp_initialize(self,x_init :ndarray, Q :ndarray, R :ndarray, P_inf :ndarray):
            
        N_state_dim = self.N
        M_action_dim = self.M
        Ad = self.Ad
        Bd = self.Bd
        # Initializing the QP Problem
        
        # Horizon, the points that it look ahead.
        H :int = self.H
        
        # Initializing the H matrix
        H_matrix = np.zeros(((H-1)*(N_state_dim+M_action_dim),(H-1)*(N_state_dim+M_action_dim)))
        
        # The first diagonal element of H_matrix
        H_matrix[:M_action_dim,:M_action_dim] = R
        
        # The last diagonal element of H_matrix
        H_matrix[-N_state_dim:,-N_state_dim:] = P_inf
        
        # Filling the rest diagonal elements
        
        for i in range(H-2):
            
            start_id = M_action_dim + i*(M_action_dim+N_state_dim)
            end_id = start_id + N_state_dim
            H_matrix[start_id:end_id,start_id:end_id] = Q
            
            start_id = end_id
            end_id = end_id + M_action_dim
            H_matrix[start_id:end_id,start_id:end_id] = R
            
            
        # Initializing the vector of the QP problem
        h = np.zeros(((H-1)*(N_state_dim+M_action_dim),1))
        # it is actually the cost vector
        
        # Initializing the equality constrains of the problem
        # Which is actually the system dynamics
        # Initializing the G_matrix
        G_matrix = np.zeros(((H-1)*(N_state_dim),(H-1)*(N_state_dim+M_action_dim)))
        
        # Initializing the first diagonal and the above element of the G_matrix
        G_matrix[:N_state_dim,:M_action_dim] = Bd
        G_matrix[:N_state_dim,M_action_dim:N_state_dim+M_action_dim] = -np.eye(N_state_dim)
        
        # Initializing the rest elements
        for i in range(H-2):
            id1 = (i+1) * N_state_dim 
            id2 = i * (N_state_dim + M_action_dim) + M_action_dim
            
            G_matrix[id1:id1+N_state_dim, id2:id2+N_state_dim] = Ad
            G_matrix[id1:id1+N_state_dim, id2+N_state_dim:id2+N_state_dim+M_action_dim] = Bd
            G_matrix[id1:id1+N_state_dim, id2+N_state_dim+M_action_dim:id2+N_state_dim+M_action_dim+N_state_dim] = -np.eye(N_state_dim)
            
        # Initializing the vector of the equality constrains
        d = np.zeros(((H-1)*(N_state_dim),1))
        # Set up the first element
        d[:N_state_dim,:] = - Ad @ x_init
        
        # Initializing the inequality constrains
        # Actually setting up the control limits
        
        # Initializing the C matrix 
        max_force :float = 10.0
        C_matrix = np.zeros(((H-1)*M_action_dim,(H-1)*(N_state_dim+M_action_dim)))
        
        for i in range(H-1):
            C_matrix[i,i*(M_action_dim+N_state_dim)] = 1
        
        lower = -max_force*np.ones(((H-1)*M_action_dim,1))
        upper = max_force*np.ones(((H-1)*M_action_dim,1))
        
        # Setting up the QP problem using the proxsuite library
        
        qp_dim = H_matrix.shape[0]
        qp_dim_eq = G_matrix.shape[0]
        qp_dim_in = C_matrix.shape[0]
        
        qp = proxsuite.proxqp.dense.QP(qp_dim,qp_dim_eq,qp_dim_in)
        
        qp.init(H_matrix,h,G_matrix,d,C_matrix,lower,upper)
        
        # Warm Up
        qp.solve()
        
        self.qp = qp
        self.H_matrix = H_matrix
        self.G_matrix = G_matrix
        self.C_matrix = C_matrix
        self.h = h        
        self.d = d
        self.lower = lower
        self.upper = upper
        
        # Optimal states
        sol_x = qp.results.x # The solution of states with the minimum cost
        # Equality multipliers
        sol_eq = qp.results.y # That respect the system dynamics
        # Inequality multipliers 
        sol_in = qp.results.z # and does not exceed the force limits
    
        return sol_x,sol_eq,sol_in
    
    
    def riccati_backward_recursion(self,QN :ndarray,Q :ndarray,R :ndarray,POINTS :int = 50000):
        
        Ad = self.Ad
        Bd = self.Bd
                
        P = [np.zeros((self.N,self.N))] * POINTS
        K = [np.zeros((self.M,self.N))] * (POINTS - 1)
        
        P[POINTS-1] = QN
        
        for i in range(POINTS - 2, -1, -1):
            
            K[i] = np.linalg.inv(R + Bd.T @ P[i+1] @ Bd) @ Bd.T @ P[i+1] @ Ad
            P[i] = Q + Ad.T @ P[i+1] @ (Ad - Bd @ K[i])
        
        return P[0],K[0] # Return the Infinite gains
        
    
    def L_cost(self,states :list, u_controls :list,QN :ndarray,Q :ndarray,R :ndarray, Points) -> int:
        total_cost :int  = 0.5 * states[-1].T @ QN @ states[-1]
        
        for i in range(Points - 1):
            total_cost += 0.5 * states[i].T @ Q @ states[i]
            total_cost += 0.5 * u_controls[i].T @ R @ u_controls[i]
        
        return total_cost[0,0]
    
    
def main():
    # Defining essential variables for the system
    dp = m.DoublePendulum()
    N_state_dim = 4
    M_action_dim = 2
    
    # Linearize the system around x:state = [np.pi,0,0,0].T and u:controls = [0.0,0.0].T
    x_barj = jnp.array([[np.pi,0.0,0.0,0.0]]).T # Fixed point
    u_barj = jnp.array([[0.0,0.0]]).T
    
    # Converting the fixed point to ndarray
    x_bar = L_LQP.jax_np_to_np(x_barj)
    u_bar = L_LQP.jax_np_to_np(u_barj)
    
    
    sim_lqr = L_LQP.LQR(dp,N_state_dim,M_action_dim)
    Ad_jax,Bd_jax = sim_lqr.linearize(x_barj, u_barj)
    
    # Converting jax.numpy to numpy
    Ad = L_LQP.jax_np_to_np(Ad_jax)
    Bd = L_LQP.jax_np_to_np(Bd_jax)
    
    QN = Q = 3000*np.eye(N_state_dim)
    
    R = 0.005*np.eye(M_action_dim)
    mpc_obj = MPC(N_state_dim,M_action_dim,Ad,Bd)
    
    P_inf, K_inf = mpc_obj.riccati_backward_recursion(QN,Q,R)

    # Initializing the starting and target state
    x_init = np.array([[2.5,0.0,0.0,0.0]]).T
    u_init = np.array([[0.0,0.0]]).T

    x_target = np.array([[np.pi,0.0,0.0,0.0]]).T
    
    # sol_x := states and controls (Optimal)
    mpc_obj.qp_initialize(x_init - x_bar,Q,R,P_inf)
    
    
    # Real World Simulation
    total_time = 8
    dt = 0.05
    Points = round(total_time/dt) + 1
    
    u_controls_no_action = [np.zeros((M_action_dim,1))*Points]
    
    u_controls = []
    states = [np.copy(x_init)]
    x = np.copy(x_init)
    
    for i in range(Points - 1):
        u_new = mpc_obj.mpc_controller(x - x_bar ,x_target - x_bar,Q,R,P_inf) + u_bar
        u_new = np.maximum(np.minimum(10,u_new),-10)
        x = dp.runge_kutta4(x,u_new,dt)
        
        states.append(np.copy(x))
        u_controls.append(np.copy(u_new))
    
    # print("\n\nThe initial value of cost function with no action")
    # print(mpc_obj.L_cost(states,u_controls_no_action,QN,Q,R,Points))
    
    # print("\n\nThe final value of cost function")
    # print(mpc_obj.L_cost(states,u_controls,QN,Q,R,Points))
    
    time.sleep(5)
    
    # Offline simulation
    visual = m.Visualization(dp.base_coord,7,dp.l1,dp.l2,0.05)
    # visual.fig.canvas.mpl_connect('close_event',visual._on_close)
    time.sleep(1)
    
    i = 0
    for x in states:
        
        print(f"State :\n{x}\n")
        print(f"U_controls:\n {u_controls[i]}\n")
        visual.update(x[0,0],x[1,0])
        
        
        if i < len(states)-2:
            i += 1 
    
    plt.ioff()
    plt.show()
    
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1)
    
    # ax.plot([i*0.05 for i in range(sim_lqr.Points)],[states[i][0,0] for i in range(sim_lqr.Points)],label = "q1")
    # ax.plot([i* 0.05 for i in range(sim_lqr.Points)],[states[i][1,0] for i in range(sim_lqr.Points)],label = "q2")
    # ax.plot([i*0.05 for i in range(sim_lqr.Points)],[states[i][2,0] for i in range(sim_lqr.Points)],label = "q1_dot")
    # ax.plot([i* 0.05 for i in range(sim_lqr.Points)],[states[i][3,0] for i in range(sim_lqr.Points)],label = "q2_dot")
    # u_controls = [u_init] + u_controls
    # ax.plot([i* 0.05 for i in range(sim_lqr.Points)],[u_controls[i][0,0] for i in range(sim_lqr.Points)],label = "u1")
    # ax.plot([i* 0.05 for i in range(sim_lqr.Points)],[u_controls[i][1,0] for i in range(sim_lqr.Points)],label = "u2")
    # plt.xlabel("Time")
    # plt.legend()
    # plt.show()
    
    
if __name__ == '__main__':
    main()
