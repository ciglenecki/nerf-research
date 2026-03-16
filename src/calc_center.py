import numpy as np


def closest_point_to_rays(origins, directions):
    # \sum_i (I - d_i d_i^\top)(p - o_i)
    # v_\perp = v - (v\cdot d)d
    # v \cdot d = v @ d.T
    # (v \cdot d)d = (v @ d.T) @ d
    # v_\perp = v - (v @ d.T) @ d
    # v_\perp = (I - d @ d.T) @ v

    # v = (1,3) # point
    # d = (0,1) # direction
    # d / sqrt(v @ v.T)
    # d / sqrt(v @ v.T)
    # d^ = d / sqrt(1^2 + 3^2)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

    # P = I - d^ @ d^T
    P = np.eye(3) - directions[:, :, None] * directions[:, None, :]
    p1 = P.sum(axis=0)  # sum of P_i over all rays
    # p2 = sum(P_i @ o_i) over all rays
    p2 = (P @ origins[:, :, None]).sum(axis=0)
    return np.linalg.solve(p1, p2).ravel()


rng = np.random.default_rng(0)

true_center = np.array([0.4, -0.3, 2.1])
origins = rng.normal(size=(8, 3))
origins[:, 2] *= 0.3  # put camera in front a bit

directions = true_center - origins
directions += 0.05 * rng.normal(
    size=directions.shape
)  # make rays miss exact intersection

p = closest_point_to_rays(origins, directions)

print("true_center:     ", true_center)
print("estimated_center:", p)
# true_center:      [ 0.4 -0.3  2.1]
# estimated_center: [ 0.41133847 -0.30165227  2.11700861]
