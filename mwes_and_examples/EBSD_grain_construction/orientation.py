"""Orientation algebra shared by the EBSD pipeline: quaternions, the cubic
symmetry group, and the disorientation angle.

Nothing here reads or writes a file, and nothing here imports another module of
the pipeline, so this is the bottom of the import graph:

    orientation  <-  segmentation_error  <-  ctf_to_tesr
                 <-  ctf_to_tesr
                 <-  ebsd_gb_diffusion

Conventions
-----------
Quaternions are (w, x, y, z), unit, with w >= 0 where a canonical sign matters.
Euler angles are Bunge (phi1, Phi, phi2) and are converted under Neper's
`passive` convention, the rotation carrying the sample coordinate system onto
the crystal one, which is the standard reading of Bunge angles and what a .ctf
stores. `self_test` pins all of this against values Neper publishes and should
be called once by whatever is driving the conversion.

The crystal symmetry multiplies on the right: q maps sample to crystal, so an
operator S, which relabels crystal axes, composes as q * S. See
`crystal_equivalents`.
"""

import numpy as np


# --- quaternion helpers ------------------------------------------------------
def cubic_symmetry_quaternions():
    """The 24 rotations of the cubic group, as unit quaternions.

    Nine 90/180/270-degree rotations about the <100> axes, six 180-degree
    rotations about the <110> axes, and eight 120/240-degree rotations about
    the <111> axes, plus the identity.
    """
    r = np.sqrt(0.5)
    q = [(1.0, 0.0, 0.0, 0.0)]
    for axis in range(3):
        for w, s in ((r, r), (0.0, 1.0), (r, -r)):
            v = [0.0, 0.0, 0.0]
            v[axis] = s
            q.append((w, *v))
    for i, j in ((0, 1), (0, 2), (1, 2)):
        for sign in (1.0, -1.0):
            v = [0.0, 0.0, 0.0]
            v[i], v[j] = r, sign * r
            q.append((0.0, *v))
    for sx in (0.5, -0.5):
        for sy in (0.5, -0.5):
            for sz in (0.5, -0.5):
                q.append((0.5, sx, sy, sz))
    out = np.array(q, dtype=float)
    assert out.shape == (24, 4), out.shape
    return out


def qmul(a, b):
    """Hamilton product, broadcasting over leading axes."""
    a0, a1, a2, a3 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    b0, b1, b2, b3 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        (
            a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
            a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
            a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
            a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
        ),
        axis=-1,
    )


def qconj(q):
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def euler_bunge_to_quat(phi1, Phi, phi2, degrees=True):
    """Bunge Euler angles -> unit quaternion, passive convention.

    Passive here means what Neper means by it: the rotation carrying the sample
    coordinate system onto the crystal coordinate system, which is the standard
    reading of Bunge angles and what a .ctf stores.
    """
    if degrees:
        phi1, Phi, phi2 = np.radians(phi1), np.radians(Phi), np.radians(phi2)
    sigma = 0.5 * (phi1 + phi2)
    delta = 0.5 * (phi1 - phi2)
    c, s = np.cos(0.5 * Phi), np.sin(0.5 * Phi)
    q = np.stack(
        (c * np.cos(sigma), s * np.cos(delta), s * np.sin(delta), c * np.sin(sigma)),
        axis=-1,
    )
    # a quaternion and its negative are the same rotation; fix the sign so that
    # averaging and fundamental-zone reduction are well defined
    return np.where(q[..., :1] < 0, -q, q)


def crystal_equivalents(q, sym):
    """All symmetry-equivalent descriptions of the orientations q, (n, 24, 4).

    The crystal symmetry multiplies on the *right* in this quaternion
    convention: q maps sample to crystal (Bunge, passive), so a symmetry
    operator S, which relabels crystal axes, composes as q * S. Multiplying on
    the left, S * q, would instead rotate the sample frame and yield a
    physically different orientation. Checked against Neper: for a cell pair
    (q, q*S) `-statedge theta` under -crysym cubic is 0; for (q, S*q) it is
    17 degrees.
    """
    return qmul(q[:, None, :], sym[None, :, :])


def to_fundamental_zone(q, sym, chunk=50_000):
    """Pick, for each orientation, the symmetry equivalent closest to identity.

    Any equivalent is as correct as any other -- the crysym is declared in the
    tesr and Neper applies the symmetry itself. The point of choosing this one
    is that its rotation angle is at most ~62.8 degrees for cubic, so the
    scalar part never approaches zero and the Rodrigues vector stays finite.
    """
    out = np.empty_like(q)
    for lo in range(0, len(q), chunk):
        blk = q[lo : lo + chunk]
        cand = crystal_equivalents(blk, sym)  # (n, 24, 4)
        best = np.argmax(np.abs(cand[..., 0]), axis=1)
        picked = cand[np.arange(len(blk)), best]
        out[lo : lo + chunk] = np.where(picked[..., :1] < 0, -picked, picked)
    return out


def quat_to_rodrigues(q):
    """Rodrigues vector = (q1, q2, q3) / q0. Requires q already in the FZ."""
    q0 = q[..., :1]
    if np.any(np.abs(q0) < 1e-8):
        raise ValueError("scalar part near zero; reduce to the fundamental zone first")
    return q[..., 1:] / q0


def cubic_disorientation_angle(m):
    """Disorientation angle (degrees) of a cubic misorientation quaternion.

    Closed form rather than a search over 24 x 24 symmetry pairs: with the
    absolute components sorted descending as a >= b >= c >= d, the largest
    attainable cos(omega/2) over the cubic group is

        max( a, (a + b)/sqrt(2), (a + b + c + d)/2 )

    which is the standard result for the cubic misorientation function (Grimmer,
    Acta Cryst. A36 (1980) 382). Checked in the self-test against two cases with
    known answers: a 90-degree rotation about <100> (a symmetry operation, so
    zero) and a 60-degree rotation about <111> (the Sigma-3 twin).
    """
    s = np.sort(np.abs(m), axis=-1)[..., ::-1]
    a, b, c, d = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    best = np.maximum.reduce([a, (a + b) / np.sqrt(2.0), 0.5 * (a + b + c + d)])
    return np.degrees(2.0 * np.arccos(np.clip(best, -1.0, 1.0)))


# --- self-test ---------------------------------------------------------------
def self_test():
    """Check the conventions against values Neper publishes.

    ctf_to_tesr.convert() calls this before it does anything else, so a
    drifted convention is caught before a file is written rather than after.

    Neper's orientation-convention table gives, for a 30-degree rotation about
    the sample x axis under the passive convention, Bunge angles (0, 30, 0),
    quaternion (0.965925826, 0.258819045, 0, 0) and Rodrigues vector
    (0.267949192, 0, 0). If this assertion ever fires, the Euler conversion has
    drifted away from what the tesr reader will assume.
    """
    sym = cubic_symmetry_quaternions()
    q = euler_bunge_to_quat(np.array([0.0]), np.array([30.0]), np.array([0.0]))
    assert np.allclose(q[0], [0.965925826, 0.258819045, 0, 0], atol=1e-8), q
    r = quat_to_rodrigues(to_fundamental_zone(q, sym))
    assert np.allclose(r[0], [0.267949192, 0, 0], atol=1e-8), r

    # 90 degrees about z is a cubic symmetry operation -> disorientation 0.
    # The tolerance is loose because arccos has an infinite derivative at 1, so
    # a rounding error of 1e-16 in the argument surfaces as ~1e-6 degrees. That
    # amplification is harmless at a 10-degree segmentation threshold but it is
    # the reason not to compare disorientations to exact zero anywhere.
    q90 = np.array([[np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]])
    assert cubic_disorientation_angle(q90)[0] < 1e-4
    # R(q) = g^T for Bunge (0, 30, 0): the IPF-Z direction is the third row
    w, x, y, z = q[0]
    third_row = np.array(
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
    )
    assert np.allclose(third_row, [0.0, np.sin(np.radians(30)), np.cos(np.radians(30))])

    # 60 degrees about <111> is the Sigma-3 twin -> disorientation 60
    v = np.sin(np.pi / 6) / np.sqrt(3)
    q60 = np.array([[np.cos(np.pi / 6), v, v, v]])
    assert abs(cubic_disorientation_angle(q60)[0] - 60.0) < 1e-6

    # symmetry operators are unit quaternions and closed under multiplication
    assert np.allclose(np.linalg.norm(sym, axis=1), 1.0)

    # The crystal symmetry acts on the right (see crystal_equivalents): every
    # equivalent must be at zero disorientation from the original, the
    # fundamental-zone representative must be the same orientation, and the
    # left-multiplied version must NOT be (it is a sample-frame rotation). A
    # regression of the S*q bug that used to corrupt **oridata and *ori.
    qq = euler_bunge_to_quat(np.array([37.0]), np.array([52.0]), np.array([131.0]))
    equiv = crystal_equivalents(qq, sym)[0]
    d = cubic_disorientation_angle(qmul(qconj(np.repeat(qq, 24, 0)), equiv))
    assert d.max() < 1e-4, d.max()
    assert (
        cubic_disorientation_angle(qmul(qconj(qq), to_fundamental_zone(qq, sym)))[0]
        < 1e-4
    )
    wrong = qmul(sym[13][None], qq)
    assert cubic_disorientation_angle(qmul(qconj(qq), wrong))[0] > 10.0
