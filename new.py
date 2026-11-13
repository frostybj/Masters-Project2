import numpy as np
from scipy.special import k1, kv

#for now, make Fd = 2 and Fint = - K1

import numpy as np
from scipy.special import k1
import matplotlib.pyplot as plt

particles = np.array([[0.0, 1.0], [1.0, 1.0]])
F_d = np.array([0.5, 0.0])
new_particles = np.zeros_like(particles)

# particle 0 movement
d = particles[0] - particles[1]          
r = np.linalg.norm(d)                    
F_int = k1(r) * d / r                   
new_particles[0] = particles[0] + F_d + F_int

#particle 1 movement
d = particles[1] - particles[0]
r = np.linalg.norm(d)
F_int = k1(r) * d / r
new_particles[1] = particles[1] + F_d + F_int

x_pos = np.array([particles[0][0], particles[1][0]])
y_pos = np.array([particles[0][1], particles[1][1]])

x_new_pos = np.array([new_particles[0][0], new_particles[1][0]])
y_new_pos = np.array([new_particles[0][1], new_particles[1][1]])

plt.plot(x_pos[0], y_pos[0], 'ro', label='Particle 1')
plt.plot(x_pos[1], y_pos[1], 'bo', label='Particle 2')
plt.plot(x_new_pos[0], y_new_pos[0], 'r*', label='New Particle 1')
plt.plot(x_new_pos[1], y_new_pos[1], 'b*', label='New Particle 2')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Particle Positions')
plt.legend()
plt.show()

