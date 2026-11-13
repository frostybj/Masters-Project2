from numba import njit
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import k1
import time

# @njit
# def temp_force(T,Tau,dt):
#     K_B = 1.38e-23
#     p = dt/Tau
#     v_squared = (2*K_B*T*dt/p)*np.random.chisquare(df=2)
#     v = np.sqrt(v_squared)
#     q = np.random.rand()
#     theta = np.random.rand()*2*np.pi

#     Kick = -1
#     if q>p:
#         Kick = 0
#     else:
#         Kick = 1

#     F_x = Kick*v*np.cos(theta)
#     F_y = Kick*v*np.sin(theta)
#     return F_x,F_y

# print(temp_force(0.011,0.1,0.01)[0])



from numba import njit
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import k1
import time





#making traingular lattice
lattice = np.empty((420,3))
align = False
n = -0.5
pi = np.pi

for i in range(420):
    
    if i % 20== 0:
        index = 0
        if align == False:
            align = True
            x = 0
            
        else:
            align = False
            x = 0.5
        n = n+0.5
    if n ==5:
            free = 1
    else:
            free = 0
    lattice[i]= [x+index,n*np.sqrt(3),free]
    index = index + 1




# part = lattice[lattice[:,2]==1][:,:2] 
# G = np.array([2*np.pi,0.0])
# N_v = len(part)
# C_T_part = 0
# for r in part:
#     G_dot_r = r[0]*G[0] + r[1]*G[1]
#     C_T_part = C_T_part +  np.exp(1j*G_dot_r)
# Sum = (1/N_v)*C_T_part
# C_T = np.abs(Sum)**2
# #force from temp
@njit
def temp_force(T):
    dt = 0.01
    K_B = 1
    v_squared = (2*K_B*T*dt)*np.random.chisquare(df=2)
    v = np.sqrt(v_squared)
    theta = np.random.rand()*2*np.pi
    F_x = v*np.cos(theta)
    F_y = v*np.sin(theta)
    return F_x,F_y


@njit
def velocity_calc(lattice,F_d,k1_list,r_list,T):
    #first ignoring the non free particles
    F_d_array = np.array([F_d,0])
    def move(part,full,F_d,k1_list,r_list,T):
        l = len(part)
        new_part = np.zeros((l, 2))
        
        dt = 0.01
        for i in range(l):

            #from temp
            F_x_temp, F_y_temp = temp_force(T)
            new_part[i] = part[i] + np.array([F_x_temp, F_y_temp])
        
        
            #boundary conditions
            if new_part[i][0]> 20:
                new_part[i][0] = new_part[i][0]-20
            if new_part[i][0]< 0:
                new_part[i][0] =  20 + new_part[i][0]
            if new_part[i][1] >17.5:
                new_part[i][1] = new_part[i][1]-17.5
            if new_part[i][1]< 0:
                new_part[i][1] =  17.5 + new_part[i][1]
        return new_part

    part = lattice[lattice[:,2]==1][:,:2]  # Select only the free particles
    

    steps = 400000
    history = np.empty((steps + 1, part.shape[0], 2))
    history[0] = part
    for i in range(steps):
        part = move(part, lattice,F_d_array,k1_list,r_list,T)
        history[i+1] = (part.copy())
      

    

    last = history[-100000:]    
    disp = last[1:] - last[:-1] 
    disp[..., 0] -= 20   * np.round(disp[..., 0] / 20)
    disp[..., 1] -= 17.5 * np.round(disp[..., 1] / 17.5)
    velocity = disp / 0.01
    avg_velocity = velocity.sum(axis=0) / velocity.shape[0]          
    avg_group_velocity = avg_velocity.sum(axis=0) / avg_velocity.shape[0]  
    return avg_group_velocity

# T = 0.011
# K_B = 1.38e-23
# F_crit_found = False
# F_crit = 0
# F_crit_poss = 0
# F_d_0 = np.linspace(0.0,0.1,100)
# velocity = []
# for i in F_d_0:
#     print(i)
#     vel = velocity_calc(lattice,i,0,0,T)[0]
#     velocity.append(vel)
    
# velocity = np.array(velocity)   
# pred = (velocity/(K_B*T))*np.exp(-1*(velocity**2)/(2*K_B*T))
# print(velocity)
# plt.hist(velocity, 100,density = True)
# plt.plot(velocity,pred,label = "P(v) = (v/K_BT)e^(-v^2/2K_BT)")
# plt.legend()
# plt.show()



def Jon_Temp(T,dt):
     U_x = np.random.normal()
     U_y = np.random.normal()
     F_x = U_x*np.sqrt(2*T*dt)
     F_y = U_y*np.sqrt(2*T*dt)
     return F_x,F_y

@njit
def My_temp_calc(T,dt):
     u = np.random.rand()
     v = np.sqrt(-1*4*np.log(u)*T*dt)
     return v

T = 0.005
dt = 0.01
vel_list = []
for i in range(50000):
     vel_list.append(My_temp_calc(T,dt))
vel_list = np.array(vel_list)
pred = (vel_list/(2*T*dt))*np.exp(-1*(vel_list**2)/(4*T*dt))
plt.hist(vel_list,1000,density=True) 
plt.plot(vel_list,pred,'.',label = "P(v)")  
plt.legend()
plt.show()   