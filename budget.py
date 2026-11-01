# -----------------------------------------------------------------------------
#
# Machine budget
#
# The cap of IKM is set in minutes per run, and converted into a number of
# queries with the throughput of the machine, measured once per (black box,
# distribution) on the very pipeline that IKM runs: sampling a batch, querying
# f twice and evaluating the basis functions of a few prefixes. The measured
# rate and the resulting cap are written to the CSV, so that the paper can
# state both.
#
# -----------------------------------------------------------------------------

from timeit import default_timer as timer

import torch

from blackbox import BlackBox
from distribution import ProductDistribution
from ikm import IKM
from oracle import MembershipOracle
from parameters import IKMParameters


def measure_throughput(blackbox: BlackBox, D: ProductDistribution, seconds: float = 5.0,
                       batch: int = 1 << 20, n_prefixes: int = 8, seed: int = 0) -> float:
    """Queries per second of IKM's weight estimation on this black box and this device."""
    oracle = MembershipOracle(blackbox)
    params = IKMParameters(theta=1.0, delta=0.5, d=D.d, c=D.c)
    ikm = IKM(oracle, D, params)
    ikm.params = type("P", (), {"b": 1, "n": batch})()
    p = max(1, D.d // 2)
    prefixes = torch.zeros((n_prefixes, p), dtype=torch.uint8, device=D.device)
    prefixes[:, :min(p, 3)] = 1  # support of size 3, a typical degree
    generator = torch.Generator(device=D.device).manual_seed(seed)

    ikm._weights(prefixes, generator)  # warm-up (kernels, caches)
    if D.device.type == "cuda":
        torch.cuda.synchronize(D.device)
    oracle.reset()
    start = timer()
    while timer() - start < seconds:
        ikm._weights(prefixes, generator)
    if D.device.type == "cuda":
        torch.cuda.synchronize(D.device)
    return oracle.n_queries / (timer() - start)


def query_cap(rate: float, minutes: float) -> int:
    return int(rate * minutes * 60)
