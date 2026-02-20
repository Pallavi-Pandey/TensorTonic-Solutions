import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    T = np.asarray(T, dtype=float)
    pts = np.asarray(points, dtype=float)

    # Ensure points shape (N, 3)
    single = False
    if pts.ndim == 1:
        pts = pts[np.newaxis, :]
        single = True

    # Convert to homogeneous coordinates
    ones = np.ones((pts.shape[0], 1))
    pts_h = np.hstack([pts, ones])   # (N, 4)

    # Apply transform
    transformed_h = pts_h @ T.T      # (N, 4)

    # Convert back to 3D
    transformed = transformed_h[:, :3]

    return transformed[0] if single else transformed