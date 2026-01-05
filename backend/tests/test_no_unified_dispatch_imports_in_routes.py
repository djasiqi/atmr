from __future__ import annotations

from pathlib import Path


def test_routes_have_no_direct_unified_dispatch_imports() -> None:
    """Garde-fou: les routes ne doivent pas importer directement services.unified_dispatch.*.

    Objectif:
        - Forcer l'utilisation des adapters Infrastructure (`backend/infrastructure/dispatch/*`).
        - Éviter le couplage direct routes -> legacy services.unified_dispatch.
    """
    root = Path(__file__).resolve().parents[1]  # backend/
    routes_dir = root / "routes"
    assert routes_dir.exists()

    offenders: list[str] = []
    for p in routes_dir.rglob("*.py"):
        text = p.read_text(encoding="utf-8")
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("#"):
                continue
            if "services.unified_dispatch" in s:
                offenders.append(f"{p.relative_to(root)}: {s}")

    assert offenders == [], (
        "Direct imports of services.unified_dispatch detected in routes:\n"
        + "\n".join(offenders)
    )
