import numpy as np
import pytest

from ptmipf.symmetry import LAUE_GROUPS, get_laue_group

EXPECTED_ORDERS = {"m-3m": 24, "6/mmm": 12, "4/mmm": 8, "-3m": 6, "mmm": 4}


@pytest.mark.parametrize("name,order", EXPECTED_ORDERS.items())
def test_operator_count(name, order):
    assert get_laue_group(name).n_operators == order


@pytest.mark.parametrize("name", list(LAUE_GROUPS))
def test_operators_are_proper_rotations(name):
    ops = get_laue_group(name).operators
    assert np.allclose(np.linalg.det(ops), 1.0)
    assert np.allclose(np.einsum("nij,nkj->nik", ops, ops), np.eye(3), atol=1e-10)


@pytest.mark.parametrize("name", list(LAUE_GROUPS))
def test_group_is_closed(name):
    ops = get_laue_group(name).operators
    products = np.einsum("aij,bjk->abik", ops, ops).reshape(-1, 3, 3)
    for product in products:
        assert np.any(np.all(np.isclose(ops, product, atol=1e-8), axis=(1, 2)))


@pytest.mark.parametrize("name", list(LAUE_GROUPS))
def test_reduction_lands_in_sector(name):
    laue = get_laue_group(name)
    rng = np.random.default_rng(4)
    v = rng.normal(size=(2000, 3))
    reduced = laue.reduce(v)
    assert laue.in_sector(reduced, tol=1e-8).all()
    assert np.allclose(np.linalg.norm(reduced, axis=1), 1.0)


@pytest.mark.parametrize("name", list(LAUE_GROUPS))
def test_reduction_is_idempotent(name):
    laue = get_laue_group(name)
    rng = np.random.default_rng(5)
    reduced = laue.reduce(rng.normal(size=(500, 3)))
    assert np.allclose(laue.reduce(reduced), reduced, atol=1e-9)


@pytest.mark.parametrize("name", list(LAUE_GROUPS))
def test_sector_vertices_are_in_sector(name):
    laue = get_laue_group(name)
    assert laue.in_sector(laue.sector_vertices, tol=1e-8).all()
    assert laue.in_sector(laue.center[None], tol=1e-8).all()


def test_reduction_chunking_matches_single_pass():
    laue = get_laue_group("hexagonal")
    rng = np.random.default_rng(6)
    v = rng.normal(size=(1000, 3))
    assert np.allclose(laue.reduce(v, chunk_size=97), laue.reduce(v), atol=1e-12)


def test_unknown_group_raises():
    with pytest.raises(KeyError):
        get_laue_group("not-a-group")
