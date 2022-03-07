
def transform_points(T, points):
    R = T[:3, :3]
    t = T[:3, 3]
    return (R @ points[..., :, None])[..., :, 0] + t

