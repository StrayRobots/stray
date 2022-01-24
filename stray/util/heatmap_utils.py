import numpy as np
import math


def rbf(x, y, lengthscale):
    return np.exp(-np.linalg.norm(x - y)**2.0/(2.0 * lengthscale**2.0))

def paint_heatmap(heatmap, points, lengthscale):
    radius = math.ceil(4.5 * lengthscale)
    for point in points:
        point_x, point_y = point
        for y in range(heatmap.shape[0]):
            for x in range(heatmap.shape[1]):
                if np.sqrt(np.linalg.norm(y-point_y)**2+np.linalg.norm(x-point_x)**2) <= radius:
                    coordinate = np.array([x + 0.5, y + 0.5], dtype=point.dtype)
                    heatmap[y, x] = rbf(point, coordinate, lengthscale)


def get_heatmap(center, width, height, lengthscale):
    x = np.arange(0, width, 1, float)
    y = np.arange(0, height, 1, float)
    y = y[:,np.newaxis]
    x_c = center[0]
    y_c = center[1]
    return np.exp(-4*np.log(2) * ((x-x_c)**2 + (y-y_c)**2) / lengthscale**2)
