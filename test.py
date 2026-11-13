import numpy as np
from scipy.special import k1, kv
import matplotlib.pyplot as plt


def move(part_1,part_2,part_3,F_d):
   d_12 = part_1 - part_2
   r_12 = np.linalg.norm(d_12)
   F_int_12 =  k1(r_12) * d_12 / r_12
   d_13 = part_1 - part_3
   r_13 = np.linalg.norm(d_13)
   F_int_13 =  k1(r_13) * d_13 / r_13
   F_int_1 = F_int_12 + F_int_13
   new_part_1 = part_1 + F_d + F_int_1

   d_21 = part_2 - part_1
   r_21 = np.linalg.norm(d_21)
   F_int_21 = k1(r_21) * d_21 / r_21
   d_23 = part_2 - part_3
   r_23 = np.linalg.norm(d_23)
   F_int_23 = k1(r_23) * d_23 / r_23
   F_int_2 = F_int_21 + F_int_23
   new_part_2 = part_2 + F_d + F_int_2

   d_31 = part_3 - part_1
   r_31 = np.linalg.norm(d_31)
   F_int_31 = k1(r_31) * d_31 / r_31
   d_32 = part_3 - part_2
   r_32 = np.linalg.norm(d_32)
   F_int_32 = k1(r_32) * d_32 / r_32
   F_int_3 = F_int_31 + F_int_32
   new_part_3 = part_3 + F_d + F_int_3

   return new_part_1, new_part_2, new_part_3


F_d = np.array([0.5, 0.0])

part_1 = np.array([0.0, 1.0])      
part_2 = np.array([1.0, 1.0])
part_3 = np.array([0.5, 1+(np.sqrt(3)/2)])

history_1 = [part_1]               
history_2 = [part_2]
history_3 = [part_3]

#for five steps
for i in range(5):
    part_1, part_2, part_3 = move(part_1, part_2, part_3, F_d)
    history_1.append(part_1)
    history_2.append(part_2)
    history_3.append(part_3)

history_1 = np.array(history_1)    
history_2 = np.array(history_2)
history_3 = np.array(history_3)

print(history_1)
print(history_2)
print(history_3)
# steps = np.arange(len(history_1))

# plt.scatter(history_1[:, 0], history_1[:, 1], c=steps, cmap='viridis', marker='o', label='particle 1')
# plt.scatter(history_2[:, 0], history_2[:, 1], c=steps, cmap='viridis', marker='s', label='particle 2')
# plt.colorbar(label='step')
# plt.xlabel('x'); plt.ylabel('y')
# plt.legend()
# plt.show()
# plt.show()



from matplotlib.animation import FuncAnimation

# set the axis limits once, to cover every position in both histories
allx = np.concatenate([history_1[:, 0], history_2[:, 0], history_3[:, 0]])
ally = np.concatenate([history_1[:, 1], history_2[:, 1], history_3[:, 1]])

fig, ax = plt.subplots()
ax.set_xlim(allx.min() - 1, allx.max() + 1)
ax.set_ylim(ally.min() - 0.5, ally.max() + 0.5)
#ax.set_aspect('equal')
ax.set_xlabel('x'); ax.set_ylabel('y')

# empty artists that get updated each frame
dot_1, = ax.plot([], [], 'o', color='C0', markersize=10, label='particle 1')
dot_2, = ax.plot([], [], 's', color='C1', markersize=10, label='particle 2')
dot_3, = ax.plot([], [], '^', color='C2', markersize=10, label='particle 3')
trail_1, = ax.plot([], [], '-', color='C0', alpha=0.4)
trail_2, = ax.plot([], [], '-', color='C1', alpha=0.4)
trail_3, = ax.plot([], [], '-', color='C2', alpha=0.4)
label = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ax.legend(loc='lower right')

def update(frame):
    dot_1.set_data([history_1[frame, 0]], [history_1[frame, 1]])
    dot_2.set_data([history_2[frame, 0]], [history_2[frame, 1]])
    dot_3.set_data([history_3[frame, 0]], [history_3[frame, 1]])
    trail_1.set_data(history_1[:frame + 1, 0], history_1[:frame + 1, 1])
    trail_2.set_data(history_2[:frame + 1, 0], history_2[:frame + 1, 1])
    trail_3.set_data(history_3[:frame + 1, 0], history_3[:frame + 1, 1])
    label.set_text(f'step {frame}')
    return dot_1, dot_2, dot_3, trail_1, trail_2, trail_3, label

anim = FuncAnimation(fig, update, frames=len(history_1), interval=500, blit=True)
plt.show()

