import numpy as np

from ptmipf.analysis import quaternions_to_matrices


def test_identity_quaternion():
    assert np.allclose(quaternions_to_matrices([[0, 0, 0, 1]]), np.eye(3))


def test_quaternions_give_proper_rotations():
    rng = np.random.default_rng(2)
    q = rng.normal(size=(500, 4))
    m = quaternions_to_matrices(q)
    assert np.allclose(np.einsum("nij,nkj->nik", m, m), np.eye(3), atol=1e-10)
    assert np.allclose(np.linalg.det(m), 1.0)


def test_quaternion_matches_axis_angle():
    axis = np.array([1.0, 2.0, 3.0])
    axis /= np.linalg.norm(axis)
    angle = np.radians(64.0)
    q = np.concatenate([axis * np.sin(angle / 2), [np.cos(angle / 2)]])
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    expected = np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k
    assert np.allclose(quaternions_to_matrices(q[None])[0], expected)


def test_zero_quaternion_does_not_produce_nan():
    assert np.isfinite(quaternions_to_matrices([[0.0, 0.0, 0.0, 0.0]])).all()
