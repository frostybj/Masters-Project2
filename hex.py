from numba import njit
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import k1
import time
from scipy.spatial import Voronoi
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation
#bessel func
r_list  = np.linspace(0.00001, 9.0, 90000)   # spacing ~0.001
k1_list = k1(r_list) 

@njit
def besselfunc(k1_list,r_list,r):
    return np.interp(r,r_list,k1_list)

#force function from fixed particles
@njit
def force (x,y,n):
    pi = np.pi
    Q = np.sqrt(1 + (2*pi*n)**2)
    alpha = 1/(Q*(1-((-1)**n)*np.exp(-(np.sqrt(3)*Q)/2)))
    F = alpha*n*np.sin(2*n*pi*x)*np.exp(-y*Q)
    return F

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

#finding nearest neighbours
part = lattice[lattice[:,2]==1][:,:2]  # Select only the free particles


N_v = len(part)
Psi_H_sum = 0
print(N_v)
for i in part:
    neighbour = []

    for j in lattice[:,:2]:
         distance = j - i
         distance[0] -= 20 * np.round(distance[0] / 20)
         distance[1] -= 19*np.round(distance[1]/19)
         r = np.hypot(distance[0], distance[1])
         if r < np.sqrt(7)/2:
            if r == 0:
                 continue
            neighbour.append(j)
    neighbour = np.array(neighbour)
    Z_j = len(neighbour)
    print(neighbour)
    print(Z_j)
    theta = np.array([np.arctan2(d[1], d[0]) for d in neighbour])
    Psi_H_exp = 0
    for k in theta:
         Psi_H_exp = Psi_H_exp + np.exp(6j*k)
    Psi_H_sum = Psi_H_sum + Z_j*Psi_H_exp

print(Psi_H_sum)
    
              

@njit
def My_temp_calc(T,dt):
     u = np.random.rand()
     v = np.sqrt(-1*4*np.log(u)*T*dt)
     theta = np.random.rand()*2*np.pi
     F_x = v*np.cos(theta)
     F_y = v*np.sin(theta)
     return F_x,0.0

@njit
def velocity_calc(lattice,F_d,k1_list,r_list,T,which_temp):
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
            Force_non_free = np.array([Force.real,0])
            

            #from temp
            
            F_x_temp, F_y_temp = My_temp_calc(T,dt)
            F_temp  = np.array([F_x_temp,F_y_temp])
            
            
            
        
            new_part[i] = part[i] + dt*(F_d + F_int + Force_non_free) + F_temp
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
    #history = np.empty((steps + 1, part.shape[0], 2))
    save_every = 100
    history = np.empty((steps // save_every, part.shape[0], 2))
    history[0] = part
    G = np.array([2*np.pi, 2*np.pi/np.sqrt(3)])
    N_v = len(part)

    C_T_array = np.empty(steps,dtype=np.complex128)
    C_H_array = np.empty(steps,dtype=np.complex128)
    # void_CH_array = np.empty(steps,dtype = np.complex128)
    # Ratio_array = np.empty(steps)

    fixed = lattice[(lattice[:, 2] == 0) &
                (np.abs(lattice[:, 1] - 5*np.sqrt(3)) < 1.0)][:, :2].copy()

    for i in range(steps):
        part = move(part, lattice,F_d_array,k1_list,r_list,T,which_temp)
        # void= hex_voronoi(fixed,part,20)
        # void_CH_array[i] = void[0]
        # Ratio_array[i] = void[1]
        all_pts = np.concatenate((fixed, part))
        if (i + 1) % save_every == 0:
            history[(i + 1)//save_every - 1] = part
        
        C_T_part = 0
        Psi_H_sum = 0

        for r in part:
            G_dot_r = r[0]*G[0] + r[1]*G[1]
            C_T_part = C_T_part +  np.exp(1j*G_dot_r)


#finding the nearest neighbours
            neighbour = []
            for j in all_pts:
                distance = j - r
                distance[0] -= 20 * np.round(distance[0] / 20)
                distance[1] -= 19*np.round(distance[1]/19)
                radius = np.hypot(distance[0], distance[1])

                if radius < np.sqrt(7)/2:
                    if radius == 0:
                        continue
                    neighbour.append(distance)
            #print(neighbour)       
            #neighbour = np.array(neighbour)
            Z_j = len(neighbour)
            theta = np.array([np.arctan2(d[1], d[0]) for d in neighbour])
            Psi_H_exp = 0
            for k in theta:
                Psi_H_exp = Psi_H_exp + np.exp(6j*k)
            if Z_j > 0:
                Psi_H_sum = Psi_H_sum + (1/Z_j)*Psi_H_exp
            if Z_j == 0:
                print("no neighbours, step", i, "y =", r[1])



        Sum = (1/N_v)*C_T_part
        C_T_array[i] = Sum

        Sum_H = (1/N_v)*Psi_H_sum
        C_H_array[i] = Sum_H



        

# for i in part:
#     neighbour = []
#     for j in lattice[:,:2]:
#          distance = j - i
#          distance[0] -= 20 * np.round(distance[0] / 20)
#          distance[1] -= 19*np.round(distance[1]/19)
#          r = np.hypot(distance[0], distance[1])
#          if r < np.sqrt(7)/2:
#             if r == 0:
#                  continue
#             neighbour.append(j)
#     neighbour = np.array(neighbour)
#     Z_j = len(neighbour)
#     print(neighbour)
#     print(Z_j)
#     theta = np.array([np.arctan2(d[1], d[0]) for d in neighbour])
#     Psi_H_exp = 0
#     for k in theta:
#          Psi_H_exp = Psi_H_exp + np.exp(6j*k)
#     Psi_H_sum = Psi_H_sum + Z_j*Psi_H_exp
      

    

    last = history[-100000:]    
    disp = last[1:] - last[:-1] 
    disp[..., 0] -= 20   * np.round(disp[..., 0] / 20)
    disp[..., 1] -= 17.5 * np.round(disp[..., 1] / 17.5)
   # velocity = disp / 0.01
    velocity = disp / (0.01*save_every) 
    avg_velocity = velocity.sum(axis=0) / velocity.shape[0]          
    avg_group_velocity = avg_velocity.sum(axis=0) / avg_velocity.shape[0] 
    avg_CT = np.abs(np.mean(C_T_array))**2
    avg_CH = np.abs(np.mean(C_H_array))**2
    # avg_void_CH = np.abs(np.mean(void_CH_array))**2
    # avg_ratio = np.mean(Ratio_array)
    return avg_group_velocity,avg_CT,avg_CH,history

#T = 0.0001

def hex_voronoi(part, fixed, Lx=20):
    
    N = len(part)
    #fixes boundary conditions error
    base = np.vstack([part, fixed])                     
    pts = np.vstack([base, base + [Lx, 0], base - [Lx, 0]])
    vor = Voronoi(pts)

    nbrs = [[] for _ in range(N)]
    for a, b in vor.ridge_points:                         # each pair of touching cells
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



F_crit_found = False
F_crit = 0
F_crit_poss = 0
#setting Fd = 0
dt = 0.01
#F_d_0 = np.linspace(0.0,0.1,10)
F_d_0 = 0.0

T = np.linspace(0,0.2,10)


from joblib import Parallel, delayed, cpu_count
from numba import njit, set_num_threads

# CRITICAL: Prevent Numba from trying to multi-thread *inside* each parallel process,
# which causes CPU core fighting and slows things down.
set_num_threads(1)

def run_job(T_val, lattice, k1_list, r_list):
    # Call your simulation function for a single temperature
   avg_vel, avg_CT, avg_CH, frames = velocity_calc(lattice, 0.0, k1_list, r_list, T_val, 0)
   fixed = lattice[(lattice[:, 2] == 0) &
                    (np.abs(lattice[:, 1] - 5 * np.sqrt(3)) < 1.0)][:, :2].copy()

   frames = frames[len(frames)//4:]               # drop the equilibration part
   out = [hex_voronoi(p, fixed, 20) for p in frames]
   psi = np.array([o[0] for o in out])
   C_H = np.abs(psi.mean())**2                      
   N_mis = np.mean([o[1] for o in out])
   return avg_vel, avg_CT, avg_CH, C_H,N_mis


if __name__ == "__main__":
    print(f"Using {cpu_count()} CPU cores for parallel processing.")
    
    T = np.linspace(0, 0.2, 10)
    
    # Run the simulations in parallel across all available cores (-2 leaves 1 core free for your OS)
    results = Parallel(n_jobs=-2)(
        delayed(run_job)(t, lattice, k1_list, r_list) for t in T
    )

    # Unpack the results back into your plotting lists
    avg_vel = [r[0][0] for r in results]
    avg_CT  = [r[1] for r in results]
    avg_CH  = [r[2] for r in results]
    avg_void_CH = [r[3] for r in results]
    N_Ratio = [r[4] for r in results]

    # --- Plotting Code ---
    plt.plot(T, avg_CT, label="C_T")
    plt.plot(T, avg_CH, label="C_H")
    plt.plot(T,avg_void_CH,label="C_H using Voronoi")
    plt.plot(T, N_Ratio, label="N_mis/N_v")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Order Parameter")
    plt.legend()
    plt.show()
#from joblib import Parallel, delayed, cpu_count

# @njit
# def seed_numba(s):
#     np.random.seed(s)

# def run_job(T, which_temp, seed):
#     seed_numba(seed)
#     avg_vel, avg_CT, avg_CH = velocity_calc(lattice, 0.0, k1_list, r_list, T, which_temp)
#     return avg_vel[0], avg_CT, avg_CH
   

# if __name__ == "__main__":
#     print(run_job(0.01, 0, 1))
#     print(f"Using {cpu_count()} cores for parallel processing.")
#     T = np.linspace(0, 0.2, 10)
#     n = len(T)

#     jobs = [(t, w, 1000*w + i) for w in (0, 1) for i, t in enumerate(T)]
#     results = Parallel(n_jobs=-2)(delayed(run_job)(*j) for j in jobs)

#     avg_CT_Rich = [r[1] for r in results[:n]]
#     avg_CT_Jon  = [r[1] for r in results[n:]]
#     avg_CH_Rich = [r[2] for r in results[:n]]
#     avg_CH_Jon  = [r[2] for r in results[n:]





# avg_vel, avg_CT, avg_CH, frames = velocity_calc(lattice, 0.0, k1_list, r_list, 0.04, 0)
# fixed = lattice[(lattice[:, 2] == 0) &
#                 (np.abs(lattice[:, 1] - 5*np.sqrt(3)) < 1.0)][:, :2].copy()
# # frames: (n_frames, 20, 2) from step 1;  fixed: the (40, 2) array from before
# Lx = 20
# colors = {4: 'tab:blue', 5: 'tab:green', 6: 'white', 7: 'tab:orange', 8: 'tab:red'}

# fig, ax = plt.subplots(figsize=(10, 3))
# ax.set_xlim(0, Lx)
# ax.set_ylim(7.6, 9.7)
# ax.scatter(fixed[:, 0], fixed[:, 1], c='k', s=15)
# dots, = ax.plot([], [], 'ro', ms=6)
# patches = []

# def update(f):
#     for p in patches:
#         p.remove()
#     patches.clear()

#     part = frames[f]
#     N = len(part)
#     base = np.vstack([part, fixed])                    # free particles are indices 0..N-1
#     pts = np.vstack([base, base + [Lx, 0], base - [Lx, 0]])
#     vor = Voronoi(pts)

#     for k in range(N):
#         region = vor.regions[vor.point_region[k]]
#         if len(region) == 0 or -1 in region:           # skip unbounded cells
#             continue
#         z = len(region)                                # number of sides = coordination number
#         patch = Polygon(vor.vertices[region], closed=True,
#                         fc=colors.get(z, 'grey'), ec='k', alpha=0.6)
#         ax.add_patch(patch)
#         patches.append(patch)

#     dots.set_data(part[:, 0], part[:, 1])
#     ax.set_title(f"frame {f}")
#     return patches + [dots]

# anim = FuncAnimation(fig, update, frames=len(frames), interval=100, blit=False)
# plt.show()
# anim.save('voronoi.gif', writer='pillow')  