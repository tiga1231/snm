from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np
from server import initData


fig = plt.figure(figsize=[8,8])
fig.set_tight_layout(True)

ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')
'''
ax.set_xticks(np.arange(-1.0, 1.5, 0.5))
ax.set_yticks(np.arange(-1.0, 1.5, 0.5))
ax.set_zticks(np.arange(-1.0, 1.5, 0.5))'''

plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

b = 1.1
ax.set_xlim(-b, b)
ax.set_ylim(-b, b)
ax.set_zlim(-b, b)

# load some test data for demonstration and plot a wireframe

x = initData()
k = x.shape[0]/2
ax.scatter(x[:k,0], x[:k,1], x[:k,2],s=900)
ax.scatter(x[k:,0], x[k:,1], x[k:,2],s=900)

def update(i):
    ax.view_init(i**0.8 *np.sin(0.2*i / (2.0 *np.pi)), i)

anim = FuncAnimation(fig, update, frames=np.arange(0, 360, 1), interval=1)
anim.save('data.gif', 
        fps=60, dpi=80, 
        writer='imagemagick')
#plt.show()
