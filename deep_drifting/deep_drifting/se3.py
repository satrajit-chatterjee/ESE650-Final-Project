from typing import Union
import numpy as np

__all__ = ["from_rotation_translation", "inverse", "transform_points"]

def from_rotation_translation(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Creates an SE3 (4x4 matrix) from a rotation and a translation

    Parameters
    ----------
    R : np.ndarray
        3x3 Rotation SO(3) matrix
    t : np.ndarray
        3 element translation vector

    Returns
    -------
    np.ndarray
        4x4 SE(3) matrix representation
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def inverse(T: np.ndarray) -> np.ndarray:
    """Fast inverse of an SE3 (4x4 matrix representation)

    Parameters
    ----------
    T : np.ndarray
        4x4 matrix representing an SE3

    Returns
    -------
    np.ndarray
        T⁻¹, the inverse of T
    """
    R = T[:3, :3]
    t = T[:3, [3]]

    return np.block([[R.T, -R.T @ t], [0, 0, 0, 1]])

def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Uses an SE3 matrix to transform an array of points

    Parameters
    ----------
    T : np.ndarray
        4x4 matrix representing an SE3
    points : np.ndarray
        (N, 2), (N, 3) or (N, 4) array of points

    Returns
    -------
    np.ndarray
        the transformed points with equivalent dimensions to `points`
    """
    N, D = points.shape
    if D == 2:
        points = np.column_stack((points, np.zeros(N), np.ones(N)))
    elif D == 3:
        points = np.column_stack((points, np.ones(N)))

    return (points @ T.T)[:, :D]
