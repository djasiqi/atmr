#!/usr/bin/env python3
"""Garde-fou architecture — autorité unique de création de réservation client.

Usage: python scripts/architecture/check_booking_create_authority.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
ROUTES = BACKEND / "routes"

CANONICAL_CREATE = (
    BACKEND / "application" / "bookings" / "create_booking.py"
).relative_to(ROOT).as_posix()
FACADE_CREATE = (
    BACKEND / "bookings" / "application" / "use_cases" / "create_booking.py"
).relative_to(ROOT).as_posix()
FACADE_INIT = (
    BACKEND / "bookings" / "application" / "use_cases" / "__init__.py"
).relative_to(ROOT).as_posix()
ADAPTER = (
    BACKEND
    / "bookings"
    / "infrastructure"
    / "adapters"
    / "booking_service_adapter.py"
).relative_to(ROOT).as_posix()
BOOKINGS_ROUTES = (ROUTES / "bookings.py").relative_to(ROOT).as_posix()
CLIENTS_ROUTES = (ROUTES / "clients.py").relative_to(ROOT).as_posix()

# Inventaire AST vérifié : seul ctor Booking() autorisé sous routes/
ALLOWED_ROUTE_BOOKING_CONSTRUCTORS: dict[tuple[str, str, str], dict[str, str]] = {
    ("backend/routes/companies.py", "TriggerReturnBooking", "post"): {
        "reason": "création du retour depuis une réservation existante",
        "created_via_context": "trigger-return / manual return leg",
    },
}

LEGACY_IMPORT_ALLOWLIST_PROD = frozenset({FACADE_CREATE, FACADE_INIT})

CANONICAL_PACKAGE_INIT = (
    BACKEND / "application" / "bookings" / "__init__.py"
).relative_to(ROOT).as_posix()

CANONICAL_IMPORT_ALLOWLIST_PROD = frozenset(
    {ADAPTER, FACADE_CREATE, CANONICAL_PACKAGE_INIT}
)

TEST_CANONICAL_IMPORT_ALLOWLIST = frozenset(
    {
        "backend/tests/services/test_booking_create_use_case.py",
    }
)

TEST_ADAPTER_ALLOWLIST = frozenset(
    {
        "backend/tests/services/test_booking_create_use_case.py",
    }
)

TEST_FACADE_IMPORT_ALLOWLIST = frozenset(
    {
        "backend/tests/services/test_booking_create_use_case.py",
    }
)

CLIENT_POST_HANDLERS = (
    (BOOKINGS_ROUTES, "CreateBooking", "post"),
    (CLIENTS_ROUTES, "ClientBookings", "post"),
    (CLIENTS_ROUTES, "ClientMyBookings", "post"),
)


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _parse(path: Path) -> ast.AST | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        # Fichiers parasites / BOM : ignorés (pas des chemins d'autorité booking).
        return None


_SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        ".local",
        ".cursor-server",
        ".cursor",
        "__pycache__",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        "htmlcov",
        "dist",
        "build",
        "migrations",
    }
)


def _iter_py_files(base: Path) -> list[Path]:
    if not base.is_dir():
        return []
    out: list[Path] = []
    for path in base.rglob("*.py"):
        if not path.is_file():
            continue
        if any(part in _SKIP_DIR_NAMES for part in path.parts):
            continue
        out.append(path)
    return sorted(out)


def _is_test_path(rel: str) -> bool:
    return "/tests/" in f"/{rel}" or rel.startswith("backend/tests/")


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


class _WalkCollector(ast.NodeVisitor):
    """Collecte imports/appels sans empiler chaque nœud (évite RecursionError)."""

    def __init__(self) -> None:
        self.class_stack: list[str] = []
        self.func_stack: list[str] = []
        self.class_defs: list[tuple[str, int]] = []
        self.imports: list[tuple[str, list[str], int]] = []
        self.calls: list[tuple[str | None, str | None, str, int]] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_defs.append((node.name, node.lineno))
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.func_stack.append(node.name)
        self.generic_visit(node)
        self.func_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.func_stack.append(node.name)
        self.generic_visit(node)
        self.func_stack.pop()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append((alias.name, [], node.lineno))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        names = [alias.name for alias in node.names]
        self.imports.append((module, names, node.lineno))

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node.func)
        if name:
            cls = self.class_stack[-1] if self.class_stack else None
            func = self.func_stack[-1] if self.func_stack else None
            self.calls.append((cls, func, name, node.lineno))
        self.generic_visit(node)


def _collect(path: Path) -> _WalkCollector | None:
    tree = _parse(path)
    if tree is None:
        return None
    collector = _WalkCollector()
    collector.visit(tree)
    return collector


def inventory_route_booking_ctors() -> list[tuple[str, str, str, int]]:
    found: list[tuple[str, str, str, int]] = []
    for path in _iter_py_files(ROUTES):
        rel = _rel(path)
        data = _collect(path)
        if data is None:
            continue
        for cls, func, name, lineno in data.calls:
            if name == "Booking":
                found.append((rel, cls or "", func or "", lineno))
    return found


def check_booking_ctors() -> list[str]:
    errors: list[str] = []
    found = inventory_route_booking_ctors()
    for rel, cls, func, lineno in found:
        key = (rel, cls, func)
        if key not in ALLOWED_ROUTE_BOOKING_CONSTRUCTORS:
            errors.append(
                f"Booking() interdit sous routes/ hors allowlist: "
                f"{rel}:{lineno} dans {cls}.{func}"
            )
    # Allowlist entries must still exist (drift detection)
    found_keys = {(rel, cls, func) for rel, cls, func, _ in found}
    for key in ALLOWED_ROUTE_BOOKING_CONSTRUCTORS:
        if key not in found_keys:
            errors.append(
                f"Allowlist Booking() obsolète (introuvable): "
                f"{key[0]} {key[1]}.{key[2]}"
            )
    return errors


def check_create_booking_use_case_defs() -> list[str]:
    errors: list[str] = []
    defs: list[tuple[str, int]] = []
    for path in _iter_py_files(BACKEND):
        rel = _rel(path)
        data = _collect(path)
        if data is None:
            continue
        for name, lineno in data.class_defs:
            if name == "CreateBookingUseCase":
                defs.append((rel, lineno))
    if len(defs) != 1 or defs[0][0] != CANONICAL_CREATE:
        errors.append(
            "Exactement une ClassDef CreateBookingUseCase attendue dans "
            f"{CANONICAL_CREATE}; trouvé: {defs}"
        )
    return errors


def check_facade_purity() -> list[str]:
    errors: list[str] = []
    path = ROOT / FACADE_CREATE
    if not path.is_file():
        return [f"Façade absente: {FACADE_CREATE}"]
    tree = _parse(path)
    if tree is None:
        return [f"Façade illisible: {FACADE_CREATE}"]

    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # docstring module
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = {alias.name for alias in node.names}
            if module != "application.bookings.create_booking":
                errors.append(
                    f"Façade: import module interdit {module!r} "
                    f"(attendu application.bookings.create_booking)"
                )
            if names != {"CreateBookingUseCase"}:
                errors.append(
                    f"Façade: noms importés invalides {sorted(names)} "
                    "(attendu CreateBookingUseCase uniquement)"
                )
            continue
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if not isinstance(node.value, (ast.List, ast.Tuple)):
                        errors.append("Façade: __all__ doit être une liste/tuple")
                    else:
                        vals = [
                            elt.value
                            for elt in node.value.elts
                            if isinstance(elt, ast.Constant)
                        ]
                        if vals != ["CreateBookingUseCase"]:
                            errors.append(
                                f"Façade: __all__ invalide {vals} "
                                "(attendu ['CreateBookingUseCase'])"
                            )
                    break
            else:
                errors.append("Façade: assignation non autorisée")
            continue
        errors.append(
            f"Façade: nœud AST interdit {type(node).__name__} "
            "(docstring + import + __all__ uniquement)"
        )
    return errors


def check_imports_and_calls() -> list[str]:
    errors: list[str] = []
    adapter_calls_create = False
    helper_calls_via = False
    via_callers_prod: list[str] = []

    for path in _iter_py_files(BACKEND):
        rel = _rel(path)
        data = _collect(path)
        if data is None:
            continue
        is_test = _is_test_path(rel)

        for module, names, lineno in data.imports:
            # Legacy module / package CreateBookingUseCase
            legacy_module = module in {
                "bookings.application.use_cases.create_booking",
                "bookings.application.use_cases",
            }
            imports_create = "CreateBookingUseCase" in names or (
                not names and legacy_module
            )
            if legacy_module and (
                "CreateBookingUseCase" in names
                or module.endswith(".create_booking")
                or (module == "bookings.application.use_cases" and imports_create)
            ):
                if module == "bookings.application.use_cases" and (
                    "CreateBookingUseCase" not in names
                ):
                    continue
                if is_test:
                    if rel not in TEST_FACADE_IMPORT_ALLOWLIST:
                        errors.append(
                            f"Import legacy CreateBookingUseCase hors allowlist test: "
                            f"{rel}:{lineno}"
                        )
                elif rel not in LEGACY_IMPORT_ALLOWLIST_PROD:
                    errors.append(
                        f"Import legacy CreateBookingUseCase interdit en prod: "
                        f"{rel}:{lineno}"
                    )

            # Import absolu du module canonique, ou relatif depuis application/bookings/
            is_canonical_import = (
                module == "application.bookings.create_booking"
                and "CreateBookingUseCase" in names
            ) or (
                rel == CANONICAL_PACKAGE_INIT
                and module in {".create_booking", "create_booking"}
                and "CreateBookingUseCase" in names
            )
            if is_canonical_import and rel != CANONICAL_PACKAGE_INIT:
                if is_test:
                    if rel not in TEST_CANONICAL_IMPORT_ALLOWLIST:
                        errors.append(
                            f"Import canonique CreateBookingUseCase hors allowlist "
                            f"test: {rel}:{lineno}"
                        )
                elif rel not in CANONICAL_IMPORT_ALLOWLIST_PROD:
                    errors.append(
                        f"Import canonique CreateBookingUseCase interdit: "
                        f"{rel}:{lineno}"
                    )

            if "SqlAlchemyBookingWriter" in names and rel.startswith("backend/routes/"):
                errors.append(
                    f"Import SqlAlchemyBookingWriter interdit sous routes/: "
                    f"{rel}:{lineno}"
                )

            if (
                module.endswith("booking_service_adapter")
                or module == "bookings.infrastructure.adapters.booking_service_adapter"
            ):
                if rel.startswith("backend/routes/"):
                    if rel != BOOKINGS_ROUTES:
                        errors.append(
                            f"Import adapter interdit sous routes/ "
                            f"(sauf bookings.py): {rel}:{lineno}"
                        )
                elif is_test and rel not in TEST_ADAPTER_ALLOWLIST:
                    errors.append(
                        f"Import adapter hors allowlist test: {rel}:{lineno}"
                    )
                elif (
                    not is_test
                    and rel != BOOKINGS_ROUTES
                    and rel != ADAPTER
                    and not rel.endswith("booking_service_adapter.py")
                ):
                    # prod hors helper : interdit (adapter file itself ok)
                    if "create_booking_via_use_case" in names or (
                        "create_booking_use_case" in names
                    ):
                        errors.append(
                            f"Import adapter create_booking_* interdit: "
                            f"{rel}:{lineno}"
                        )

            if (
                "CreateBookingUseCase" in names
                and rel.startswith("backend/routes/")
            ):
                errors.append(
                    f"Import CreateBookingUseCase interdit sous routes/: "
                    f"{rel}:{lineno}"
                )

        for cls, func, name, lineno in data.calls:
            if name == "create_booking_via_use_case":
                loc = f"{rel}:{lineno} ({cls}.{func})"
                if is_test:
                    if rel not in TEST_ADAPTER_ALLOWLIST:
                        errors.append(
                            f"Appel create_booking_via_use_case hors allowlist "
                            f"test: {loc}"
                        )
                else:
                    via_callers_prod.append(loc)
                    if rel == BOOKINGS_ROUTES and func == "execute_client_booking_creation":
                        helper_calls_via = True
                    elif rel != BOOKINGS_ROUTES or func != "execute_client_booking_creation":
                        errors.append(
                            "Unique appelant prod de create_booking_via_use_case: "
                            f"execute_client_booking_creation; trouvé {loc}"
                        )

        if rel == ADAPTER:
            src = path.read_text(encoding="utf-8")
            if "company_creation_gate_fn" not in src:
                errors.append(
                    "Adapter doit injecter company_creation_gate_fn"
                )
            if "assert_company_not_platform_suspended" not in src:
                errors.append(
                    "Adapter doit passer assert_company_not_platform_suspended"
                )
            if "application.bookings.create_booking" not in src:
                errors.append("Adapter doit importer le module canonique")
            adapter_calls_create = "CreateBookingUseCase(" in src

    if not helper_calls_via:
        errors.append(
            "execute_client_booking_creation doit appeler create_booking_via_use_case"
        )
    if not adapter_calls_create:
        errors.append("Adapter doit instancier CreateBookingUseCase")
    if via_callers_prod and not any(
        "execute_client_booking_creation" in c for c in via_callers_prod
    ):
        errors.append(
            f"Aucun appel prod attendu à create_booking_via_use_case: "
            f"{via_callers_prod}"
        )
    return errors


def check_client_post_handlers() -> list[str]:
    errors: list[str] = []
    for rel, class_name, method_name in CLIENT_POST_HANDLERS:
        path = ROOT / rel
        tree = _parse(path)
        if tree is None:
            errors.append(f"Route illisible: {rel}")
            continue
        found = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or node.name != class_name:
                continue
            for item in node.body:
                if (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == method_name
                ):
                    src = ast.get_source_segment(
                        path.read_text(encoding="utf-8"), item
                    ) or ""
                    if "execute_client_booking_creation" in src:
                        found = True
                    else:
                        errors.append(
                            f"{rel} {class_name}.{method_name} doit déléguer à "
                            "execute_client_booking_creation"
                        )
        if not found and not any(
            e.startswith(f"{rel} {class_name}") for e in errors
        ):
            errors.append(
                f"Handler introuvable ou sans délégation: "
                f"{rel} {class_name}.{method_name}"
            )
    return errors


def main() -> int:
    errors: list[str] = []
    print("Inventaire Booking() sous backend/routes/:")
    for rel, cls, func, lineno in inventory_route_booking_ctors():
        key = (rel, cls, func)
        status = "ALLOW" if key in ALLOWED_ROUTE_BOOKING_CONSTRUCTORS else "DENY"
        print(f"  [{status}] {rel}:{lineno} {cls}.{func}")

    errors.extend(check_booking_ctors())
    errors.extend(check_create_booking_use_case_defs())
    errors.extend(check_facade_purity())
    errors.extend(check_imports_and_calls())
    errors.extend(check_client_post_handlers())

    if errors:
        print("Booking create authority check: FAIL")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("Booking create authority check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
