# Copyright 2021 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

import numpy as np
from scipy.sparse import spdiags, coo_matrix
from scipy.sparse.linalg import norm as spnorm
import math
import pytest
import hitmix
np.set_printoptions(precision=4)

def test_prolongation_restriction_matvec(sample_problem):
    # problem setup
    (A, D, omega, omega_c) = sample_problem

    # test all 1's
    x0 = np.array([1, 1, 1, 1, 1])
    y0 = hitmix.prolongation_restriction_matvec(D, A, omega_c) * x0
    y0_true = np.array([0, 1, 0, 0, 0])

    # test all 1:n
    x1 = np.array(range(A.shape[0]))
    y1 = hitmix.prolongation_restriction_matvec(D, A, omega_c) * x1
    y1_true = np.array([0, -2, -1, 1, 3])

    assert (y0 == y0_true).all()
    assert (y1 == y1_true).all()


def test_augmented_adjacency_disconnected(sample_problem_disconnected):

    # problem setup
    (A, D, omega, omega_c, disconnected_vertices) = sample_problem_disconnected

    # test
    A_aug, omega_aug, disconnected_vertices = hitmix.augmented_adjacency(A, omega)
    omega_aug_true = [0]
    disconnected_vertices_true = np.array([0, 1])
    v_true_comp = np.array([2, 3, 4])
    A_aug_true = A[v_true_comp, :][:, v_true_comp]

    # check augmented hitting set vertices
    assert (omega_aug == omega_aug_true).all()

    # check that hitting time moments for hitting set are all zero
    assert (disconnected_vertices == disconnected_vertices_true).all()

    # check all other hitting times moments
    assert spnorm(A_aug - A_aug_true, 'fro') < 1e-8


def test_hitting_time_moments(sample_problem):
    # problem setup
    (A, D, omega, omega_c) = sample_problem

    # test
    ETm, cg_info = hitmix.hitting_time_moments(A, omega)
    ETm_true = np.array([[0.0,  0.0],
                         [9.0,  10.583005244258363],
                         [12.0, 10.954451150103322],
                         [12.0, 10.954451150103322],
                         [13.0, 10.954451150103322]])

    # check that hitting time moments for hitting set are all zero
    assert np.linalg.norm(ETm[omega, :] - np.zeros_like(ETm[omega, :])) < 1e-8

    # check all other hitting times moments
    assert np.linalg.norm(ETm[omega_c, :] - ETm_true[omega_c, :]) < 1e-8

def test_hitting_time_moments_small(sample_problem_small):
    # problem setup
    (A, D, omega, omega_c) = sample_problem_small

    # test
    ETm, cg_info = hitmix.hitting_time_moments(A, omega)
    ETm_true = np.array([[0.0,  0.0],
                         [3.0, 2.82842712474619],
                         [4.0, 2.82842712474619]])

    # check that hitting time moments for hitting set are all zero
    assert np.linalg.norm(ETm[omega, :] - np.zeros_like(ETm[omega, :])) < 1e-8

    # check all other hitting times moments
    assert np.linalg.norm(ETm[omega_c, :] - ETm_true[omega_c, :]) < 1e-8


def test_hitting_time_moments_disconnected(sample_problem_disconnected):
    # problem setup
    (A, D, omega, omega_c, disconnected_vertices) = sample_problem_disconnected

    # test
    ETm, cg_info = hitmix.hitting_time_moments(A, omega)

    ETm_true = np.array([[math.inf, math.inf],
                         [math.inf, math.inf],
                         [0.0,  0.0],
                         [3.0, 2.82842712474619],
                         [4.0, 2.82842712474619]])

    # check that hitting time moments for hitting set are all zero
    assert np.linalg.norm(ETm[omega, :] - np.zeros_like(ETm[omega, :])) < 1e-8

    # check all other hitting times moments
    assert np.linalg.norm(ETm[omega_c, :] - ETm_true[omega_c, :]) < 1e-8

    # check disconnected vertices
    assert np.isinf(ETm[disconnected_vertices, :]).all()

@pytest.fixture()
def sample_problem():
    # adjacency matrix
    n = 5
    row = np.array([0, 1, 1, 2, 3])
    col = np.array([1, 2, 3, 4, 4])
    data = np.array([1, 1, 1, 1, 1])
    A = coo_matrix((data, (row, col)), shape=(n, n))
    A = A + A.T

    # diagonal degree matrix
    D = spdiags(A.sum(axis=0), 0, n, n)

    # hitting set indices
    omega = np.array([0])

    # complement of hitting set indices
    omega_c = np.setdiff1d(np.array(range(n)), np.array(omega))

    return A, D, omega, omega_c


@pytest.fixture()
def sample_problem_disconnected():
    # adjacency matrix
    n = 5
    row = np.array([0, 2, 3])
    col = np.array([1, 3, 4])
    data = np.array([1, 1, 1])
    A = coo_matrix((data, (row, col)), shape=(n, n))
    A = A + A.T

    # diagonal degree matrix
    D = spdiags(A.sum(axis=0), 0, n, n)

    # hitting set indices
    omega = [2]

    # complement of hitting set indices
    omega_c = [3, 4]

    # disconnected_vertices
    disconnected_vertices = [0, 1]

    return A, D, omega, omega_c, disconnected_vertices


@pytest.fixture()
def sample_problem_small():
    n = 3
    row = np.array([0, 1])
    col = np.array([1, 2])
    data = np.array([1, 1])
    A = coo_matrix((data, (row, col)), shape=(n, n))
    A = A + A.T

    # diagonal degree matrix
    D = spdiags(A.sum(axis=0), 0, n, n)

    # hitting set indices
    omega = [0]

    # complement of hitting set indices
    omega_c = np.setdiff1d(np.array(range(n)), np.array(omega))

    return A, D, omega, omega_c
