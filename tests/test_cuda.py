"""
GPU checks, skipped when CUDA is not available. They cover what the CPU tests
cannot: the int64 wrap-around of the NoisyParity hash, the float32 forward pass
of the MLP, and a full IKM run on the device.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blackbox import Classifier, NoisyParity, Parity
from distribution import ProductDistribution
from ikm import IKM
from oracle import MembershipOracle
from parameters import IKMParameters

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
CUDA = "cuda:0"


def test_noisy_parity_is_the_same_function_on_gpu():
    f = NoisyParity([0, 1, 2], eta=0.1, d=30, seed=7)
    X = ProductDistribution.uniform(30, 2).sample(200_000, torch.Generator().manual_seed(0))
    assert torch.equal(f.predict(X), f.predict(X.to(CUDA)).cpu())


def test_gpu_mlp_matches_sklearn():
    rng = np.random.default_rng(4)
    X = rng.integers(0, 2, size=(3000, 12)).astype(np.uint8)
    Y = np.where((X[:, 0] ^ X[:, 3]) | X[:, 7], 1, -1).astype(np.int8)
    clf = Classifier("Neural Network", seed=0, max_iter=300)
    clf.train(X, Y)
    Z = rng.integers(0, 2, size=(200_000, 12)).astype(np.uint8)
    gpu = clf.predict(torch.from_numpy(Z).to(CUDA)).cpu().numpy()
    assert np.mean(gpu == clf.model.predict(Z)) > 0.9999


def test_ikm_recovers_a_parity_on_gpu():
    d, support = 12, [2, 5, 9]
    D = ProductDistribution.uniform(d, 2, device=CUDA)
    params = IKMParameters(theta=0.25, delta=0.1, d=d, c=2)
    ikm = IKM(MembershipOracle(Parity(support)), D, params, max_degree=3)
    ikm.learn(torch.Generator(device=CUDA).manual_seed(4))
    assert {tuple(np.nonzero(s)[0]) for s in ikm.coordinates} == {tuple(support)}
    assert ikm.n_queries == params.queries
