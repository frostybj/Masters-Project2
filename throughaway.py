

import numpy as np
points = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2],
                   [2, 0], [2, 1], [2, 2],[3,0],[3,1],[3,2]])
from scipy.spatial import Voronoi, voronoi_plot_2d
list = [[1,2],[1,1],[1,0],[2,0],[2,1],[2,2]]
list = np.array(list)
list = np.vstack([list, [3,0],[3,1],[3,2]])
print(list)

    
