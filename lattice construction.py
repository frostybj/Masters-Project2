import numpy as np
import matplotlib.pyplot as plt

#making triangular array
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
print(lattice)
#need to scatter diagram it
#circle
centre = (0,5*np.sqrt(3))
radius = 9
theta = np.linspace(0, 2*np.pi, 100)
x = centre[0] + radius * np.cos(theta)
y = centre[1] + radius * np.sin(theta)


#check right particles are slected
slected = np.array(centre).reshape(1,2)
for k in range(len(lattice)):
    if lattice[k][2] == 0:  # only consider non-free particles
        d = [0,5*np.sqrt(3)] - lattice[k][:2]
        d[0] -= 20*np.round(d[0]/20)  # Apply periodic boundary conditions in x
        d[1] -= 17.5*np.round(d[1]/17.5)  # Apply periodic boundary conditions in y
        r = np.linalg.norm(d)
        if r >9:
            continue
        slected = np.vstack([slected, lattice[k][:2]])

part = lattice[lattice[:,2]==1][:,:2]
for j in range(len(part)):
    if j != i:
        d = [0,5*np.sqrt(3)] - part[j]
        d[0] -= 20*np.round(d[0]/20)
        d[1] -= 17.5*np.round(d[1]/17.5)  # Apply periodic boundary conditions in y
        r = np.linalg.norm(d)
        if r >9:
            continue
        slected = np.vstack([slected, part[j]])
       



plt.scatter(lattice[:,0],lattice[:,1],c=lattice[:,2],cmap='viridis')
plt.plot(x, y, color='red')  
plt.plot(slected[:,0],slected[:,1],'ro')
#plt.plot([centre[0]], [centre[1]], marker='o', color='red')
plt.show() 

    


   