"""torchrun self-check: the synced gradient equals the mean of every rank's gradient, None stays None.

    torchrun --nproc_per_node=2 experiments/paper/scripts/check_gradient_sync.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.paper.common import distributed  # noqa: E402


def gradients(model, rank):
    generator = torch.Generator().manual_seed(100 + rank)
    x = torch.randn(4, 3, generator=generator).cuda()
    for p in model.parameters():
        p.grad = None
    # model[1] never receives a gradient; model[2] holds a complex weight as CUT-4DFlow does.
    (model[0](x).square().mean() + (model[2].weight * x[0, 0]).abs().sum()).backward()
    return [None if p.grad is None else p.grad.clone() for p in model.parameters()]


def main():
    world, rank, local_rank = distributed.environment()
    assert world > 1, "launch with torchrun --nproc_per_node>=2"
    distributed.initialize(world, rank, local_rank)
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 2), torch.nn.Linear(2, 2, bias=False, dtype=torch.complex64)).cuda()
    distributed.all_ranks_equal(distributed.fingerprint(model), "initial weights")
    expected = [gradients(model, r) for r in range(world)]
    gradients(model, rank)
    distributed.sync_gradients(model.parameters(), world)
    for index, p in enumerate(model.parameters()):
        if expected[0][index] is None:
            assert p.grad is None, "unused parameter must keep grad None"
        else:
            mean = torch.stack([e[index] for e in expected]).mean(0)
            torch.testing.assert_close(p.grad, mean, rtol=1e-6, atol=1e-7)
    assert distributed.any_rank(rank == world - 1, False) == [True, False]
    try:
        distributed.all_ranks_equal([rank], "rank id")
        raise AssertionError("all_ranks_equal must detect a mismatch")
    except RuntimeError:
        pass
    print(f"GRADIENT_SYNC_OK rank={rank}/{world}", flush=True)


if __name__ == "__main__":
    main()
