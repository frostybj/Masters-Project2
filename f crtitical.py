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
    F = alpha*n*np.sin(2*n*pi*x)*np.exp(-y*Q)
    return F



@njit
def velocity_calc(lattice,F_d,k1_list,r_list):
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
      

    

    last = history[-100000:]    
    disp = last[1:] - last[:-1] 
    disp[..., 0] -= 20   * np.round(disp[..., 0] / 20)
    disp[..., 1] -= 17.5 * np.round(disp[..., 1] / 17.5)
    velocity = disp / 0.01
    avg_velocity = velocity.sum(axis=0) / velocity.shape[0]           
    avg_group_velocity = avg_velocity.sum(axis=0) / avg_velocity.shape[0]  
    return avg_group_velocity


F_crit_found = False
F_crit = 0
F_crit_poss = 0
F_d_0 = np.linspace(0.04,0.06,10)
velocity = []
for i in F_d_0:
    print(i)
    vel = velocity_calc(lattice,i,k1_list,r_list)[0]
    velocity.append(vel)
    tol = 0.01  # choose relative to your typical sliding velocity
    if vel < tol:
        F_crit_poss = i
    else:
        if F_crit_found == False:
            F_crit = F_crit_poss
            F_crit_found = True
        else:
            continue
 
print(velocity)
print("Critical value is ",F_crit)

v_check = []
#from the paper
F_theory = 0.05005 
for i in range(len(F_d_0)):
    F = F_d_0[i]
    v_check.append(np.sqrt(max(0.0, F**2 - F_theory**2)))
    if F == F_crit:
        Next_F = F_d_0[i+1]

F_crit_range = np.linspace(F_crit,Next_F, 10)

v_check = np.array(v_check)
velocity_2= []
F_crit_2 = 0
F_crit_found_2 = False
for i in F_crit_range:
    print(i)
    vel_2 = velocity_calc(lattice,i,k1_list,r_list)[0]
    velocity_2.append(vel_2)
    tol = 0.001  # choose relative to your typical sliding velocity
    if vel_2 < tol:
        F_crit_poss = i
    else:
        if F_crit_found_2 == False:
            F_crit_2 = F_crit_poss
            F_crit_found_2 = True
        else:
            continue

print(velocity_2)
print("More specific critical value is ",F_crit_2)

plt.plot(F_d_0,velocity,label = "From the Simulation")
plt.plot(F_d_0,v_check, label = "√(F^2-F*^2)")
plt.ylabel("Particle Velocity")
plt.xlabel("Driving Force Strength")
plt.title("Driving Force vs Velocity")
plt.legend()
plt.show()








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