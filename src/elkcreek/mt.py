"""
Moment tensor utilities.
"""

from __future__ import annotations

import numpy as np
from pyrocko.moment_tensor import MomentTensor


def crack_decomposition(
    moment_tensor: MomentTensor, azimuth, plunge, poissons_ratio=0.25
) -> tuple[MomentTensor, MomentTensor]:
    """
    Decompose a moment tensor into a crack closing and residual moment tensor.

    If the overall moment tensor falls in the CDC area (see Rigby 2024) this
    will result in a crack closing/double couple solution.

    Parameters
    ----------
    moment_tensor : MomentTensor
        The input moment tensor to decompose.
    azimuth
        The azimuth of the crack closing P axis.
    plunge
        The plunge of the crack closing P axis.
    poissons_ratio
        The poisson's ratio.

    Returns
    -------
    A tuple of crack closing and residual moment tensors.
    """
    # First convert to the east north up coordinate system.
    enu_tensor = moment_tensor.m_east_north_up()
    swd_tensor = enu_to_swd_tensor(enu_tensor)

    # Then get the unit crush mt and project onto momement tensor
    mt_crush_raw = get_crush(1, azimuth, plunge, poissons_ratio=poissons_ratio)
    mt_crush_raw = np.trace(swd_tensor) * mt_crush_raw / np.trace(mt_crush_raw)

    # Remove the crush from original MT
    mt_residual_raw = swd_tensor - mt_crush_raw

    # Package up and return.
    mt_crush = MomentTensor(m=swd_to_ned_tensor(mt_crush_raw))
    mt_dc = MomentTensor(m=swd_to_ned_tensor(mt_residual_raw))
    return mt_crush, mt_dc


def get_rot_z(angle):
    """Get rotation matrix for rotating around z axis."""
    r = np.zeros((3, 3))
    r[0, 0] = np.cos(angle)
    r[1, 1] = np.cos(angle)
    r[0, 1] = -np.sin(angle)
    r[1, 0] = np.sin(angle)
    r[2, 2] = 1
    return r


def get_rot_y(angle):
    """Get rotation matrix for rotating around Y axis"""
    r = np.zeros((3, 3))
    r[0, 0] = np.cos(angle)
    r[2, 2] = np.cos(angle)
    r[2, 0] = -np.sin(angle)
    r[0, 2] = np.sin(angle)
    r[1, 1] = 1
    return r


def get_crush(scale, p_azimuth_deg, p_plunge_deg, poissons_ratio=0.25):
    """Get a crack closing MT for a given aximuth and plunge."""
    mt = get_unit_crush(poissons_ratio=poissons_ratio)
    mt = get_rotated(mt, get_rot_y(np.radians(90 + p_plunge_deg)))
    mt = get_rotated(mt, get_rot_z(np.radians(p_azimuth_deg)))
    return scale * get_normalised_mt(mt)


def swd_to_ned_tensor(mt_swd):
    """Convert from south west down to north east down."""
    A = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]).T
    return A @ mt_swd @ np.linalg.inv(A)


def enu_to_swd_tensor(mt_enu):
    """Convert from east north down to south west down."""
    A = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]).T
    return A @ mt_enu @ np.linalg.inv(A)


def get_normalised_mt(mt):
    """Get the normalized moment tensor."""
    return mt / get_scalar_moment(mt)


def get_scalar_moment(mt):
    """Get scalar moment from tensor."""
    return get_l2_norm(mt) / np.sqrt(2)


def get_l2_norm(mt):
    """Get the L2 norm of the tensor."""
    return np.sqrt(np.sum(mt.flatten() ** 2))


def get_unit_crush(poissons_ratio=0.25, epsilon=1e-6):
    """
    Return crush moment tensor with P axis in z.

    Also apply a minor epsilon factor to ensure eigen value ratios don't get
    too close to allowable limits (then can't be used in Pyrocko).
    """
    if poissons_ratio == 0:
        return -np.diag([0, 0, 1])
    mt = -np.diag([1 - epsilon, 1 - epsilon, ((1 / poissons_ratio) - 1 + epsilon)])
    return get_normalised_mt(mt)


def get_rotated(mt, rot):
    """Rotate the moment tensors."""
    return rot @ mt @ rot.T


def eigen_decom(array):
    """Decompose a moment tensor into eigen values and vectors."""
    evals, evecs = np.linalg.eigh(array)
    assert evals[0] <= evals[1] <= evals[2]
    # need to re-arrange such that M1 >= M2 >= M3
    eval_sorted = evals[::-1]
    # and need to match eigenvectors
    evec_sorted = evecs[:, ::-1]
    return eval_sorted, evec_sorted


def project(moment_tensor):
    """
    Calculate Hudson's (u, v) coordinates for a given moment tensor.
    """
    if not moment_tensor.size == 3:  # need to perform decomp
        eig_m = eigen_decom(moment_tensor)[0]
    else:
        eig_m = moment_tensor
    m3, m2, m1 = eig_m / np.max(np.abs(eig_m))

    u = -2.0 / 3.0 * (m1 + m3 - 2.0 * m2)
    v = 1.0 / 3.0 * (m1 + m2 + m3)
    return u, v
