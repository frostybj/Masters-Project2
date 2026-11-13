from numba import njit
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import k1,k0
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
 
 
lattice_extra_particle = np.vstack([lattice, [30.5, 5*np.sqrt(3), 1]])
rows_free = 1
 
 
 
 
r_list  = np.linspace(0.00001, 9.0, 90000)   # spacing ~0.001
k_0_list = k0(r_list) 
k1_list = k1(r_list) 
 
 
 
 
@njit
def besselfunc(k_list,r_list,r):
    dr = r_list[1] - r_list[0]
    s = (r - r_list[0]) / dr          # how many grid steps along the table r is
    k = int(s)                        # the grid point just below r
    f = s - k                         # fraction of the way to the next point
    return k_list[k]*(1 - f) + k_list[k + 1]*f
 
     
 
@njit
def Potential_non_interacting(x,y,n,rows_free):
    Q = np.sqrt(1 + (2*pi*n)**2)
    alpha = 1/(Q*(1-((-1)**n)*np.exp(-(np.sqrt(3)*Q)/2)))
    if rows_free%2 == 0:
        Tau = (-1)**n
    else:
        Tau = 1
    w = (rows_free + 1)*np.sqrt(3)/2
    V = alpha*(np.exp(1j*2*n*pi*x)*(np.exp(-y*Q)+(Tau*np.exp(-(w-y)*Q))))
    return V
 
 
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
def history_potential_calc(lattice,F_d,k1_list,r_list,T,k_0_list):
    #first ignoring the non free particles
    F_d_array = np.array([F_d,0.0])
    def move(part,F_d,k1_list,r_list,T,k_0_list):
        l = len(part)
        new_part = np.zeros((l, 2))
        potential = np.zeros(l)   # one number per particle
        
        dt = 0.01
        for i in range(l):
            F_int = np.array([0.0, 0.0])
            V_int = 0.0
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
                    V_int = V_int + besselfunc(k_0_list,r_list,r)
                    
                   
 
            #non free particles
            F_x_non = 0.0
            F_y_non = 0.0
            V_non = 0.0
                       
            for k in range(5):
                x = part[i][0]
                if x-bottom_row_x !=0:
                    x = x -0.5
                y = part[i][1] - bottom_row_y
                Force_non =force(x,y,k,rows_free)
                V_non += Potential_non_interacting(x,y,k,rows_free).real   # V is complex; keep the real part
                F_x_non += Force_non[0]
                F_y_non += Force_non[1]
               
                
 
            F_x = F_x_non*4*pi**2
            F_y = F_y_non*2*pi
            V_nonon = pi*V_non
            Force_non_free = np.array([F_x, F_y])
            V_non_free = V_nonon
 
           
        
            new_part[i] = part[i] + dt*(F_d + F_int + Force_non_free) 
            potential[i] = V_non_free + V_int - F_d[0]*part[i][0]   # F_d is the [F_d, 0] array here
 
 
        
 
 
 
        
            #boundary conditions
            if new_part[i][0]> 60:
                new_part[i][0] = new_part[i][0]-60
            if new_part[i][0]< 0:
                new_part[i][0] =  60 + new_part[i][0]
            if new_part[i][1] >17.5:
                new_part[i][1] = new_part[i][1]-17.5
            if new_part[i][1]< 0:
                new_part[i][1] =  17.5 + new_part[i][1]
        return new_part,potential
 
    part = lattice[lattice[:,2]==1][:,:2]  # Select only the free particles
    
 
    steps = 40000
    save_every = 100
    history = np.empty((steps // save_every, part.shape[0], 2))
    potential = np.empty((steps // save_every, part.shape[0]))
    history[0] = part
    for i in range(steps):
        print(i)
        part,V = move(part, F_d_array, k1_list, r_list, T, k_0_list)
        if (i + 1) % save_every == 0:
            history[(i + 1)//save_every - 1] = part
            potential[(i + 1)//save_every - 1] = V
      
 
    return history,potential
 
F_drive = 0.06
T = 0.0
Values = history_potential_calc(lattice_extra_particle,F_drive,k1_list,r_list,T,k_0_list)
x = Values[0]
potential = Values[1]



length = len(x)
print(x)
d = np.zeros(length)
X = np.zeros(length)
for i in range(length):
    d[i]= x[i+1]-x[i] - 1
    X[i] = (x[i+1]+x[i])/2

plt.plot(X,d)
plt.xlabel("X")
plt.ylabel("d")
plt.show()


# #claude code again lol
#`animation
# from matplotlib.animation import FuncAnimation, PillowWriter

# from matplotlib.animation import FuncAnimation, PillowWriter

# V_plot = potential + F_drive * x[:, :, 0]        # remove the drive tilt
# n_frames = x.shape[0]

# fig, ax = plt.subplots(figsize=(7, 4))
# line, = ax.plot([], [], 'o-', ms=3)
# ax.set_xlim(0, 60)
# ax.set_ylim(V_plot.min() - 0.01, V_plot.max() + 0.01)
# #ax.set_ylim(V_plot[5:].min() - 0.01, V_plot[5:].max() + 0.01)   # skip the initial spike from adding the particle
# ax.set_xlabel('x')
# ax.set_ylabel('potential (drive removed)')
# title = ax.set_title('')

# def update(f):
#     order = np.argsort(x[f, :, 0])
#     line.set_data(x[f, order, 0], V_plot[f, order])
#     title.set_text(f'frame {f}   t = {f*100*0.01:.0f}')
#     return line, title

# anim = FuncAnimation(fig, update, frames=n_frames, interval=50)
# from matplotlib.animation import FFMpegWriter
# anim.save('soliton0.016spike.mp4', writer=FFMpegWriter(fps=20), dpi=150)
# plt.show()
# print("done")


# f = -1                                   # which frame (-1 = last)
# x_f = x[f, :, 0]                         # x positions, shape (61,)
# V_f = potential[f]                       # potentials,  shape (61,)

# order = np.argsort(x_f)                  # sort left to right
# plt.plot(x_f[order], V_f[order], 'o-', ms=4)
# plt.xlabel('x')
# plt.ylabel('potential')
# plt.show()
# for f in [0, 100, 200, 300, -1]:
#     order = np.argsort(x[f, :, 0])
#     plt.plot(x[f, order, 0], potential[f, order] + F_drive*x[f, order, 0], 'o-', ms=3, label=f'frame {f}')
# plt.legend(); plt.xlabel('x'); plt.ylabel('potential (drive removed)')
# plt.show()