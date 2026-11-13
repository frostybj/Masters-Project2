import numpy as np
from scipy.special import k1, kv
import matplotlib.pyplot as plt


def move(part,full):
    l = len(part)
    new_part = np.zeros((l, 2))
    for i in range(l):
        F_int = np.array([0.0, 0.0])
       
        F_d = np.array([0.5, 0.0])
      
        for j in range(len(full)):
            if j != i:
                d = part[i] - full[j][:2]
                r = np.linalg.norm(d)
                F_int = F_int + k1(r) * d / r
        new_part[i] = part[i] + F_d + F_int
    return new_part
       




lattice = np.array([[0.0, 1.0], [1.0, 1.0],[2.0,1.0],[1.0,2.0],[0.0,3.0],[1.0,3.0],[2.0,3.0]])
part = np.array([1.0,2.0])
history = part

#for five steps
for i in range(5):
    part = move(part, lattice)
    history = np.vstack([history, part])

N = 4
part_1 = history[0::N]   
part_2 = history[1::N]   
part_3 = history[2::N] 
part_4 = history[3::N]


print(part_1)
print(part_2)
print(part_3)





#animation from claude lol 
from matplotlib.animation import FuncAnimation

parts = [part_1, part_2, part_3,part_4]            # list of (6, 2) arrays
markers = ['o', 's', '^','D']

allx = np.concatenate([p[:, 0] for p in parts])
ally = np.concatenate([p[:, 1] for p in parts])

fig, ax = plt.subplots()
ax.set_xlim(allx.min() - 1, allx.max() + 1)
ax.set_ylim(ally.min() - 1, ally.max() + 1)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('x'); ax.set_ylabel('y')

# one dot and one trail per particle
dots = [ax.plot([], [], markers[k], color=f'C{k}', markersize=10,
                label=f'particle {k+1}')[0] for k in range(N)]
trails = [ax.plot([], [], '-', color=f'C{k}', alpha=0.4)[0] for k in range(N)]
label = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ax.legend(loc='lower right')

def update(frame):
    for k, p in enumerate(parts):
        dots[k].set_data([p[frame, 0]], [p[frame, 1]])
        trails[k].set_data(p[:frame + 1, 0], p[:frame + 1, 1])
    label.set_text(f'step {frame}')
    return (*dots, *trails, label)

anim = FuncAnimation(fig, update, frames=len(part_1), interval=500, blit=True)
plt.show()