from numba import njit
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import k1
import time
from scipy.spatial import Voronoi
from matplotlib.patches import Polygon


#making traingular lattice
lattice = np.empty((1020,2))
align = False
n = -0.5
pi = np.pi
rows_free = 2

for i in range(1020):
    
    if i % 30== 0:
        index = 0
        if align == False:
            align = True
            x = 0
            
        else:
            align = False
            x = 0.5
        n = n+0.5
  
    lattice[i]= [x+index,n*np.sqrt(3)]
    index = index + 1
#bessel func
r_list  = np.linspace(0.00001, 9.1, 91000)   
k1_list = k1(r_list) 

@njit
def besselfunc(k1_list,r_list,r):
    dr = r_list[1] - r_list[0]
    s = (r - r_list[0]) / dr          # how many grid steps along the table r is
    k = int(s)                        # the grid point just below r
    f = s - k                         # fraction of the way to the next point
    return k1_list[k]*(1 - f) + k1_list[k + 1]*f


#force function from fixed particles
@njit
def force (x,y,n,rows_free):
    Q = np.sqrt(1 + (2*pi*n)**2)
    alpha = 1/(Q*(1-((-1)**n)*np.exp(-(np.sqrt(3)*Q)/2)))
    if rows_free%2 == 0:
        Tau = (-1)**n
    else:
        Tau = 1
    F_x = alpha*n*np.sin(2*n*pi*x)*(np.exp(-y*Q)+(Tau*np.exp(-(1.5*np.sqrt(3)-y)*Q)))
    F_y = alpha*np.cos(2*n*pi*x)*Q*(np.exp(-y*Q)-(Tau*np.exp(-(1.5*np.sqrt(3)-y)*Q)))
    return F_x, F_y

#force from temp
@njit
def temp_force(T,dt):
    #K_B = 1.38e-23
    K_B = 1
    v_squared = (2*K_B*T*dt)*np.random.chisquare(df=2)
    v = np.sqrt(v_squared)
    theta = np.random.rand()*2*np.pi
    F_x = v*np.cos(theta)
    F_y = v*np.sin(theta)
    return F_x,F_y


@njit
def Jon_Temp(T,dt):
     K_B = 1
     U_x = np.random.normal()
     U_y = np.random.normal()
     F_x = U_x*np.sqrt(2*K_B*T*dt)
     F_y = U_y*np.sqrt(2*K_B*T*dt)
     return F_x,F_y

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
            #F_int = np.array([0.0, 0.0])
            F_int = np.zeros(2)
            #other free particles
            for j in range(l):
                if j != i:
                    d = part[i] - part[j]
                    d[0] -= 30*np.round(d[0]/30)
                    d[1] -= 34*np.sqrt(3)/2*np.round(d[1]/(34*np.sqrt(3)/2))  # Apply periodic boundary conditions in y
                    r = np.linalg.norm(d)
                    if r < 1e-12 or r > 9:
                        continue
                   
                    F_int = F_int + besselfunc(k1_list,r_list,r) * d / r
            
                    
                   

            
           

            #from temp
            
            F_temp = My_temp_calc(T,dt)
            #F_temp  = np.array([F_x_temp,F_y_temp])
         
            new_part[i] = part[i] + dt*(F_d + F_int) + F_temp
            #boundary conditions
            if new_part[i][0]> 30:
                new_part[i][0] = new_part[i][0]-30
            if new_part[i][0]< 0:
                new_part[i][0] =  30 + new_part[i][0]
            if new_part[i][1] >34*np.sqrt(3)/2:
                new_part[i][1] = new_part[i][1]-34*np.sqrt(3)/2
            if new_part[i][1]< 0:
                new_part[i][1] =  34*np.sqrt(3)/2 + new_part[i][1]
        return new_part

    part = lattice
    

    steps = 10000
    #history = np.empty((steps + 1, part.shape[0], 2))
    save_every = 100
    history = np.empty((steps // save_every, part.shape[0], 2))
    history[0] = part
    G = np.array([2*np.pi, 2*np.pi/np.sqrt(3)])
    N_v = len(part)

    Lx = 30.0                          # box sizes, needed for the minimum image
    Ly = 34*np.sqrt(3)/2              

    C_T_array = np.empty(steps, dtype=np.complex128)

    cm = np.zeros(2)                   # total drift of the centre of mass so far

    for i in range(steps):
        new = move(part, lattice, F_d_array, k1_list, r_list, T)  

        #how far each particle moved this step, undoing any jump from wrapping
        dd = new - part
        dd[:, 0] -= Lx*np.round(dd[:, 0]/Lx)
        dd[:, 1] -= Ly*np.round(dd[:, 1]/Ly)
        cm += dd.sum(axis=0)/N_v       # centre of mass moves by the average step

        part = new                    

        if (i + 1) % save_every == 0:
            history[(i + 1)//save_every - 1] = part

        C_T_part = 0
        for r in part:
            # positions measured relative to the centre of mass
            G_dot_r = (r[0] - cm[0])*G[0] + (r[1] - cm[1])*G[1]
            C_T_part = C_T_part + np.exp(1j*G_dot_r)

        C_T_array[i] = (1/N_v)*C_T_part

        

      

    

    last = history[-100000:]    
    disp = last[1:] - last[:-1] 
    disp[..., 0] -= 30   * np.round(disp[..., 0] / 30)
    disp[..., 1] -= 34*np.sqrt(3)/2 * np.round(disp[..., 1] / (34*np.sqrt(3)/2))
    #velocity = disp / 0.01
    velocity = disp / (0.01*save_every) 
    avg_velocity = velocity.sum(axis=0) / velocity.shape[0]          
    avg_group_velocity = avg_velocity.sum(axis=0) / avg_velocity.shape[0] 
    
    avg_CT = np.abs(np.mean(C_T_array[steps//4:]))**2
    
    return avg_group_velocity,avg_CT,history


def hex_voronoi(part, Lx):
    N = len(part)
    shifts = [(sx*Lx, sy*Ly) for sx in (0, -1, 1) for sy in (0, -1, 1)]  # (0,0) first
    pts = np.vstack([part + np.array(s) for s in shifts])
    vor = Voronoi(pts)

#each touching particle is a neighbour, so we can use ridge_points to find the neighbours of each particle
    nbrs = [[] for _ in range(N)]
    for a, b in vor.ridge_points:                         
        if a < N: nbrs[a].append(b)
        if b < N: nbrs[b].append(a)

    total, n_mis = 0j, 0
    for j in range(N):
        d = []
        for k in nbrs[j]:
            d.append(pts[k] - pts[j])
        d = np.array(d)                        # bond vectors j -> neighbours
        theta = np.arctan2(d[:, 1], d[:, 0])
        total += np.exp(6j * theta).mean()                # (1/z_j) sum_k
        n_mis += (len(nbrs[j]) != 6)
    return total / N, n_mis / N

#T = 0.0001

F_crit_found = False
F_crit = 0
F_crit_poss = 0
#setting Fd = 0
dt = 0.01
#F_d_0 = np.linspace(0.0,0.1,10)
F_d_0 = 0.0
avg_vel = []

avg_CT = []

avg_CH = []

T = np.linspace(0,0.02,10)
N_Ratio = []

from joblib import Parallel, delayed, cpu_count
from numba import njit, set_num_threads


set_num_threads(1)

def run_job(T_val, lattice, k1_list, r_list):
    
    avg_vel, avg_CT, frames= velocity_calc(lattice, 0.0, k1_list, r_list, T_val)
    
    
    frames = frames[len(frames)//4:]                 # drop the equilibration part
    out = [hex_voronoi(p, 30) for p in frames]
    psi = np.array([o[0] for o in out])
    C_H = np.abs(psi.mean())**2                      
    N_mis = np.mean([o[1] for o in out])
    return avg_vel, avg_CT,C_H,N_mis


# if __name__ == "__main__":
#     avg_vel, avg_CT, C_H, N_mis, frames, fixed = run_job(0.0, lattice, k1_list, r_list)
#     print(avg_CT, C_H, N_mis)

#     plt.scatter(*fixed.T, c='grey', s=10)
#     plt.scatter(*frames[-1].T, c='red', s=15)
#     plt.gca().set_aspect('equal'); plt.show()
if __name__ == "__main__":
    print(f"Using {cpu_count()} CPU cores for parallel processing.")
    
    T = np.linspace(0, 0.02, 10)
    
    # Run the simulations in parallel across all available cores (-2 leaves 1 core free for your OS)
    results = Parallel(n_jobs=-2)(
        delayed(run_job)(t, lattice, k1_list, r_list) for t in T
        )

    # Unpack the results back into your plotting lists
    avg_vel = [r[0][0] for r in results]
    avg_CT  = [r[1] for r in results]
    avg_CH  = [r[2] for r in results]
    N_Ratio = [r[3] for r in results]

    
    plt.plot(T, avg_CT, label="C_T")
    plt.plot(T, avg_CH, label="C_H")
    plt.plot(T, N_Ratio, label="N_mis/N_v")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Order Parameter")
    plt.legend()
    # plt.scatter(*fixed.T, c='grey', s=10)
    # plt.scatter(*frames[-1].T, c='red', s=15)
    # plt.gca().set_aspect('equal'); plt.show()   
    plt.show()
