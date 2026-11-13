import numpy as np
from scipy.special import k1, kv
import matplotlib.pyplot as plt

#checking convergence

pi = np.pi
def force (x,y,n):
    Q = np.sqrt(1 + (2*pi*n)**2)
    alpha = 1/(Q*(1-((-1)**n)*np.exp(-(np.sqrt(3)*Q)/2)))
    F = alpha*n*np.sin(2*n*pi*x)*np.exp(-y*Q)
    return F

#for particle x= 10,5*sqrt(3)


limits = np.linspace(1,10,10)
print(limits)


x = np.linspace(0,20,100)
y = np.sqrt(3)*0.5
#for comparison




#making traingular lattice
lattice = np.empty((420,3))
align = False
n = -0.5

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

def acc_force(pos):
    r_list  = np.linspace(0.00001, 9.0, 90000)   # spacing ~0.001
    k1_list = k1(r_list) 
    full = lattice

    def besselfunc(k1_list,r_list,r):
        return np.interp(r,r_list,k1_list)
                        
                    

                #non free particles
    F_int = np.zeros(2)
    for k in range(len(full)):
        if full[k][2] == 0:  # only consider non-free particles
            d = pos - full[k][:2]
            d[0] -= 20*np.round(d[0]/20)  # Apply periodic boundary conditions in x
            d[1] -= 17.5*np.round(d[1]/17.5)  # Apply periodic boundary conditions in y
            r = np.linalg.norm(d)
            if r >9:
                continue
            F_int = F_int + besselfunc(k1_list,r_list,r) * d / r

    return F_int
       
        








x_force = []
F_acc_val = []
#cont with this, just check that they are the same thing approx
#then implement it 
#to do this weekend- sum to 4
sum_indeces = np.linspace(0,4,5)
F_acc_val = []
x = np.linspace(0,20,100)
y = np.sqrt(3)*0.5

for k in x:
    # pos = [k,5*np.sqrt(3)]
    # F_acc_val.append(acc_force(pos))

   
    F_val = []
    for i in limits:
        Force = 0 
        print(i)
        sum_indeces = np.arange(i+1) 
        print(sum_indeces)  
        for j in sum_indeces:
            j = int(j)
        #why minus 0.5?
            Force = Force + force(k-0.5,y,j)
        Force = Force*8*pi**2
        F_val.append(Force.real)
   
    x_force.append(F_val)
    




# plt.plot(x,x_force,label="Calc Force")
# plt.plot(x,np.array(F_acc_val)[:, 0] ,label = "Force from Full System")
# plt.xlabel("x position")
# plt.ylabel("Force")
# plt.legend()
# plt.show()

for a in [0, 12, 25, 37, 50]:
    plt.plot(list(limits), x_force[a], label=f"x = {x[a]:.2f}")
plt.xlabel("Sum limit N"); plt.ylabel("Fx ")
plt.legend(); plt.show()