from numba import njit
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import k1
import time



#making traingular lattice
lattice = np.empty((1260,3))
align = False
n = -0.5
pi = np.pi

for i in range(1260):
    
    if i % 60== 0:
        index = 0
        if align == False:
            align = True
            x = 0
            
        else:
            align = False
            x = 0.5
        n = n+0.5
    if n == 5:
            free = 1
    else:
            free = 0
    lattice[i]= [x+index,n*np.sqrt(3),free]
    index = index + 1
part = lattice[lattice[:,2]==1][:,:2]

bottom_row_y =part[0][1]-np.sqrt(3)/2
if part[0][0] == 0.0:
    bottom_row_x = 0.5
else:
    bottom_row_x = 0.0

#bessel func


r_list  = np.linspace(0.00001, 9.0, 90000)   # spacing ~0.001
k1_list = k1(r_list) 

#insert a particle into the free moving channel - in the middle of the channel
lattice_extra_particle = np.vstack([lattice, [30.5, 5*np.sqrt(3), 1]])
rows_free = 1



@njit
def besselfunc(k1_list,r_list,r):
    dr = r_list[1] - r_list[0]
    s = (r - r_list[0]) / dr          # how many grid steps along the table r is
    k = int(s)                        # the grid point just below r
    f = s - k                         # fraction of the way to the next point
    return k1_list[k]*(1 - f) + k1_list[k + 1]*f

@njit
def force (x,y,n,rows_free):
    Q = np.sqrt(1 + (2*pi*n)**2)
    alpha = 1/(Q*(1-((-1)**n)*np.exp(-(np.sqrt(3)*Q)/2)))
    if rows_free%2 == 0:
        Tau = (-1)**n
    else:
        Tau = 1
    w = (rows_free + 1)*np.sqrt(3)/2
    F_x = alpha*n*np.sin(2*n*pi*x)*(np.exp(-y*Q)+(Tau*np.exp(-(w-y)*Q)))
    F_y = alpha*np.cos(2*n*pi*x)*Q*(np.exp(-y*Q)-(Tau*np.exp(-(w-y)*Q)))
    return F_x, F_y

@njit
def My_temp_calc(T,dt):
     u = np.random.rand()
     v = np.sqrt(-1*4*np.log(u)*T*dt)
     theta = np.random.rand()*2*np.pi
     F_x = v*np.cos(theta)
     F_y = v*np.sin(theta)
     F_temp = np.array([F_x,F_y])
     return F_temp

@njit
def velocity_calc(lattice,F_d,k1_list,r_list,T):
    #first ignoring the non free particles
    F_d_array = np.array([F_d,0])
    def move(part,full,F_d,k1_list,r_list,T):
        l = len(part)
        new_part = np.zeros((l, 2))
        
        dt = 0.01
        for i in range(l):
            F_int = np.array([0.0, 0.0])
            #other free particles
            for j in range(l):
                if j != i:
                    d = part[i] - part[j]
                    d[0] -= 60*np.round(d[0]/60)
                    d[1] -= 17.5*np.round(d[1]/17.5)  # Apply periodic boundary conditions in y
                    r = np.linalg.norm(d)
                    if r >9:
                        continue
                   
                    F_int = F_int + besselfunc(k1_list,r_list,r) * d / r
                    
                   

            #non free particles
            F_x_non = 0
            F_y_non = 0
                       
            for k in range(5):
                x = part[i][0]
                if x-bottom_row_x !=0:
                    x = x -0.5
                y = part[i][1] - bottom_row_y
                Force_non =force(x,y,k,rows_free)
                F_x_non += Force_non[0]
                F_y_non += Force_non[1]
            F_x = F_x_non*4*pi**2
            F_y = F_y_non*2*pi
            Force_non_free = np.array([F_x, F_y])

           
        
            new_part[i] = part[i] + dt*(F_d + F_int + Force_non_free) 
            #boundary conditions
            if new_part[i][0]> 60:
                new_part[i][0] = new_part[i][0]-60
            if new_part[i][0]< 0:
                new_part[i][0] =  60 + new_part[i][0]
            if new_part[i][1] >17.5:
                new_part[i][1] = new_part[i][1]-17.5
            if new_part[i][1]< 0:
                new_part[i][1] =  17.5 + new_part[i][1]
        return new_part

    part = lattice[lattice[:,2]==1][:,:2]  # Select only the free particles
    

    steps = 400000
    save_every = 100
    history = np.empty((steps // save_every, part.shape[0], 2))
    history[0] = part
    for i in range(steps):
        part = move(part, lattice,F_d_array,k1_list,r_list,T)
        if (i + 1) % save_every == 0:
            history[(i + 1)//save_every - 1] = part
      

    

    last = history[-1000:]    
    disp = last[1:] - last[:-1] 
    disp[..., 0] -= 60   * np.round(disp[..., 0] / 60)
    disp[..., 1] -= 17.5 * np.round(disp[..., 1] / 17.5)
    velocity = disp/(save_every*0.01)
    avg_velocity = velocity.sum(axis=0) / velocity.shape[0]           
    avg_group_velocity = avg_velocity.sum(axis=0) / avg_velocity.shape[0]  
    return avg_group_velocity

from joblib import Parallel, delayed, cpu_count
from numba import njit, set_num_threads
if __name__ == "__main__":
    print(f"Using {cpu_count()} CPU cores for parallel processing.")
    F_crit_found = False
    F_crit = 0
    F_crit_poss = 0
    F_d_0 = np.linspace(0.0,0.06,10)
    T=0.0

    lattice_cases = (lattice_extra_particle,)
    results = Parallel(n_jobs=-1)(
        delayed(velocity_calc)(case_lattice, drive, k1_list, r_list, T)
        for drive in F_d_0
        for case_lattice in lattice_cases
    )

    velocity = []
    velocity_extra = []
    for drive_index, drive in enumerate(F_d_0):
        print(drive)
        vel = results[drive_index * 2][0]
        vel_extra = results[drive_index * 2 + 1][0]
        velocity.append(vel)
        velocity_extra.append(vel_extra)
        tol = 0.01  # choose relative to your typical sliding velocity
        if vel < tol:
            F_crit_poss = drive
        else:
            if F_crit_found == False:
                F_crit = F_crit_poss
                F_crit_found = True
            else:
                continue

    print(velocity)
    print(velocity_extra)
    print("Critical value is ",F_crit)

    F_theory = 0.05005
    v_check = np.sqrt(np.maximum(0.0, F_d_0**2 - F_theory**2))

    plt.plot(F_d_0,velocity,label = "Original lattice")
    plt.plot(F_d_0,velocity_extra,label = "Lattice with extra particle")
    plt.plot(F_d_0,v_check,label = "Theory")
    plt.ylabel("Particle Velocity")
    plt.xlabel("Driving Force Strength")
    plt.title("Driving Force vs Velocity")
    plt.legend()
    plt.show()


