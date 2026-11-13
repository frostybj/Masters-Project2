from numba import njit
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import k1
import time
#I want to check when the velocities converge


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


#bessel func


r_list  = np.linspace(0.00001, 9.0, 90000)   # spacing ~0.001
k1_list = k1(r_list) 

@njit
def besselfunc(k1_list,r_list,r):
    return np.interp(r,r_list,k1_list)

@njit
def force (x,y,n):
    Q = np.sqrt(1 + (2*pi*n)**2)
    alpha = 1/(Q*(1-((-1)**n)*np.exp(-(np.sqrt(3)*Q)/2)))
    F_x = alpha*n*np.sin(2*n*pi*x)*np.exp(-y*Q)
    return F_x



@njit
def velocity_calc(lattice,F_d,k1_list,r_list,j):
    #first ignoring the non free particles
    F_d_array = np.array([F_d,0])
    def move(part,full,F_d,k1_list,r_list):
        l = len(part)
        new_part = np.zeros((l, 2))
        
        dt = 0.01
        for i in range(l):
            F_int = np.array([0.0, 0.0])
            #other free particles
            for j in range(l):
                if j != i:
                    d = part[i] - part[j]
                    d[0] -= 20*np.round(d[0]/20)
                    d[1] -= 17.5*np.round(d[1]/17.5)  # Apply periodic boundary conditions in y
                    r = np.linalg.norm(d)
                    if r >9:
                        continue
                   
                    F_int = F_int + besselfunc(k1_list,r_list,r) * d / r
                    
                   

            #non free particles
            Force = 0
            for k in range(5):
                x = part[i][0] 
                Force = Force + force(x-0.5,(np.sqrt(3)*0.5),k)
            Force = Force*8*pi**2
            Force_non_free = Force.real

            # for k in range(len(full)):
            #     if full[k][2] == 0:  # only consider non-free particles
            #         d = part[i] - full[k][:2]
            #         d[0] -= 20*np.round(d[0]/20)  # Apply periodic boundary conditions in x
            #         d[1] -= 17.5*np.round(d[1]/17.5)  # Apply periodic boundary conditions in y
            #         r = np.linalg.norm(d)
            #         if r >9:
            #             continue
            #         F_int = F_int + besselfunc(k1_list,r_list,r) * d / r
        
            new_part[i] = part[i] + dt*(F_d + F_int + Force_non_free)
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
        part = move(part, lattice,F_d_array,k1_list,r_list)
        history[i+1] = (part.copy())
      

    
    # last = history[len(history)//2:]  
    if j == 0:
        last = history
    if j == 1 :
        last = history[len(history)//4:]
    if j == 2:
       last =  history[len(history)//2:]
    if j == 3:
        last = history[-100000:]  
        
    disp = last[1:] - last[:-1] 
    disp[..., 0] -= 20   * np.round(disp[..., 0] / 20)
    disp[..., 1] -= 17.5 * np.round(disp[..., 1] / 17.5)
    velocity = disp / 0.01
    avg_velocity = velocity.sum(axis=0) / velocity.shape[0]          
    avg_group_velocity = avg_velocity.sum(axis=0) / avg_velocity.shape[0] 
    return avg_group_velocity

velocity = []
four_vel = np.empty((4, 10))
j_list = np.arange(0,4)
for k in j_list:
    F_d_0 = np.linspace(0.044,0.055,10)
    velocity = []
    for i in F_d_0:
        print(i)
        vel = velocity_calc(lattice,i,k1_list,r_list,k)[0]
        velocity.append(vel)
    four_vel[k] = velocity.copy()

vel_0 = four_vel[0]
vel_1 = four_vel[1]
vel_2 = four_vel[2]
vel_3 = four_vel[3]





plt.plot(F_d_0,vel_0,label = "Full")
plt.plot(F_d_0,vel_1, label = "Remove first 1/4")
plt.plot(F_d_0,vel_2, label = "Remove first 1/2")
plt.plot(F_d_0,vel_3, label = "Only keep last 100,000")
plt.ylabel("Particle Velocity")
plt.xlabel("Driving Force Strength")
plt.title("Driving Force vs Velocity")
plt.legend()
plt.show()