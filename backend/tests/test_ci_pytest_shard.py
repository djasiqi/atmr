"""Le découpage CI doit couvrir chaque test une fois, sans trou."""

from __future__ import annotations

import pytest

from tests.ci_shard import partition


def test_partition_couvre_chaque_nodeid_une_seule_fois() -> None:
    nodeids = [f"tests/test_mod.py::test_{index}" for index in range(17)]
    total = 4
    parts = [partition(nodeids, shard, total) for shard in range(1, total + 1)]
    flat = [nodeid for part in parts for nodeid in part]
    assert sorted(flat) == sorted(nodeids)
    assert len(flat) == len(set(flat))
    assert all(parts)


def test_partition_refuse_un_shard_hors_bornes() -> None:
    with pytest.raises(ValueError, match="shard invalide"):
        partition(["tests/test_mod.py::test_a"], 0, 4)
    with pytest.raises(ValueError, match="shard invalide"):
        partition(["tests/test_mod.py::test_a"], 5, 4)
