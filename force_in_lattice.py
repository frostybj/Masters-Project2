from numba import njit
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import k1
import time
from scipy.spatial import Voronoi
from matplotlib.patches import Polygon


#making traingular lattice
lattice = np.empty((420,3))
align = False
n = -0.5
pi = np.pi
rows_free = 2

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
  
    lattice[i]= [x+index,n*np.sqrt(3),0]
    index = index + 1
rows = lattice.shape[0]//20
start = (rows-rows_free)//2
lattice[start*20:(start+rows_free)*20, 2] = 1  # Set the middle rows to be free
part = lattice[lattice[:,2]==1][:,:2]
print(part)
bottom_row_y =part[0][1]-np.sqrt(3)/2
if part[0][0] == 0.0:
    bottom_row_x = 0.5
else:
    bottom_row_x = 0.0



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
                    d[0] -= 20*np.round(d[0]/20)
                    d[1] -= 17.5*np.round(d[1]/17.5)  # Apply periodic boundary conditions in y
                    r = np.linalg.norm(d)
                    if r < 1e-12 or r > 9:
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
            F_x = F_x_non*8*pi**2
            F_y = F_y_non*2*pi
            Force_non_free = np.array([F_x, F_y])

            #from temp
            
            F_temp = My_temp_calc(T,dt)
            #F_temp  = np.array([F_x_temp,F_y_temp])
         
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
    
    fixed = lattice[lattice[:,2]==0][:,:2]
    for i in range(steps):
        part = move(part, lattice,F_d_array,k1_list,r_list,T)
        all_pts = np.concatenate((fixed, part))
        if (i + 1) % save_every == 0:
            history[(i + 1)//save_every - 1] = part
        
        C_T_part = 0
        Psi_H_sum = 0

        N_m_count = 0
        
        for r in part:
            G_dot_r = r[0]*G[0] + r[1]*G[1]
            C_T_part = C_T_part +  np.exp(1j*G_dot_r)






        Sum = (1/N_v)*C_T_part
        C_T_array[i] = Sum

        

      

    

    last = history[-100000:]    
    disp = last[1:] - last[:-1] 
    disp[..., 0] -= 20   * np.round(disp[..., 0] / 20)
    disp[..., 1] -= 17.5 * np.round(disp[..., 1] / 17.5)
    #velocity = disp / 0.01
    velocity = disp / (0.01*save_every) 
    avg_velocity = velocity.sum(axis=0) / velocity.shape[0]          
    avg_group_velocity = avg_velocity.sum(axis=0) / avg_velocity.shape[0] 
    
    avg_CT = np.abs(np.mean(C_T_array))**2
    
    return avg_group_velocity,avg_CT,history


def hex_voronoi(part, fixed, Lx):
    N = len(part)
    #fixes boundary conditions error
    base = np.vstack([part, fixed])                     
    pts = np.vstack([base, base + [Lx, 0], base - [Lx, 0]])
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

T = np.linspace(0,0.18,10)
N_Ratio = []

from joblib import Parallel, delayed, cpu_count
from numba import njit, set_num_threads


set_num_threads(1)

def run_job(T_val, lattice, k1_list, r_list):
    
    avg_vel, avg_CT, frames= velocity_calc(lattice, 0.0, k1_list, r_list, T_val)
    fixed = lattice[(lattice[:, 2] == 0)][:, :2]
    
    frames = frames[len(frames)//4:]                 # drop the equilibration part
    out = [hex_voronoi(p, fixed, 20) for p in frames[::100]]
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
    
    T = np.linspace(0, 0.18, 10)
    
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




#     Calc = velocity_calc(lattice,F_d_0,k1_list,r_list,i,1)
#     vel = Calc[0][0]
#     velocity_Jon.append(vel)
#     avg_CT_Jon.append(Calc[1])



    # tol = 0.01  # choose relative to your typical sliding velocity
    # if vel < tol:
    #     F_crit_poss = i
    # else:
    #     if F_crit_found == False:
    #         F_crit = F_crit_poss
    #         F_crit_found = True
    #     else:
    #         continue
 
# print(velocity)
# print("Critical value is ",F_crit)

# v_check = []
# #from the paper
# F_theory = 0.05005 
# for i in range(len(F_d_0)):
#     F = F_d_0[i]
#     v_check.append(np.sqrt(max(0.0, F**2 - F_theory**2)))
    


# plt.plot(F_d_0,velocity,label = "From the Simulation")
# #plt.plot(F_d_0,v_check, label = "√(F^2-F*^2)")
# plt.ylabel("Particle Velocity")
# plt.xlabel("Driving Force Strength")
# plt.title("Driving Force vs Velocity")
# plt.legend()
# plt.show()

# plt.plot(T,avg_CT,label = "C_T")
# plt.plot(T,avg_CH,label = "C_H")
# plt.plot(T,N_Ratio,label = "N_mis/N_v")

# #plt.plot(T,-np.log(avg_CT_Jon),label = "-logCT")
# plt.xlabel("Temperature (K)")
# plt.ylabel("Order Paramtere")
# plt.legend()
# plt.show()










#animation from claude lol 
# from matplotlib.animation import FuncAnimation

#  N = history.shape[1]               # 20 particles
#     parts = [history[:, k, :] for k in range(N)] 
# markers = ['o', 's', '^','D','v','<','>','p','h','*', '+','x','1','2','3','4','8','H','|','_']  # 20 markers

# allx = np.concatenate([p[:, 0] for p in parts])
# ally = np.concatenate([p[:, 1] for p in parts])

# fig, ax = plt.subplots()
# ax.set_xlim(lattice[:,0].min() - 1, lattice[:,0].max() + 1)
# ax.set_ylim(lattice[:,1].min() - 1, lattice[:,1].max() + 1)
# #ax.set_aspect('equal', adjustable='box')
# ax.set_xlabel('x'); ax.set_ylabel('y')

# # one dot and one trail per particle
# dots = [ax.plot([], [], markers[k], color=f'C{k}', markersize=10,
#                 label=f'particle {k+1}')[0] for k in range(N)]

# label = ax.text(0.02, 0.95, '', transform=ax.transAxes)
# #ax.legend(loc='lower right')

# def update(frame):
#     for k, p in enumerate(parts):
#         dots[k].set_data([p[frame, 0]], [p[frame, 1]])
#     label.set_text(f'step {frame}')
#     return (*dots, label)
# #trails = [ax.plot([], [], '-', color=f'C{k}', alpha=0)[0] for k in range(N)]

# anim = FuncAnimation(fig, update, frames=len(parts[0]), interval=500, blit=True)
# non_free_particles = lattice[lattice[:,2]==0][:,:2]
# plt.scatter(non_free_particles[:,0], non_free_particles[:,1], c='black')
# plt.show()