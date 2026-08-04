"""CLI Flask — control plane LIRIE (CP-PR1)."""

from __future__ import annotations

import json

import click
from flask import Flask
from flask.cli import with_appcontext


def register_control_plane_cli(app: Flask) -> None:
    @app.cli.group("control-plane")
    def control_plane() -> None:
        """Commandes control plane partenaires / identités."""

    @control_plane.command("seed")
    @with_appcontext
    def seed_cmd() -> None:
        from services.control_plane.seed import seed_control_plane_catalogs

        out = seed_control_plane_catalogs(commit=True)
        click.echo(json.dumps(out, ensure_ascii=False))

    @control_plane.command("reconcile")
    @click.option("--dry-run", is_flag=True, default=False)
    @click.option("--apply", "do_apply", is_flag=True, default=False)
    @with_appcontext
    def reconcile_cmd(dry_run: bool, do_apply: bool) -> None:
        from services.control_plane.reconcile import reconcile_control_plane

        if do_apply and dry_run:
            raise click.ClickException("Utiliser --dry-run OU --apply, pas les deux.")
        if not do_apply and not dry_run:
            dry_run = True
        out = reconcile_control_plane(dry_run=dry_run, apply_projection=True)
        click.echo(json.dumps(out, ensure_ascii=False, default=str))

    @control_plane.command("backfill")
    @click.option("--dry-run", is_flag=True, default=False)
    @with_appcontext
    def backfill_cmd(dry_run: bool) -> None:
        from services.control_plane.backfill import backfill_control_plane

        out = backfill_control_plane(dry_run=dry_run)
        click.echo(json.dumps(out, ensure_ascii=False, default=str))

    @control_plane.command("cutover-status")
    @with_appcontext
    def cutover_status_cmd() -> None:
        """Vérifie si CONTROL_PLANE_ORGANIZATIONS_READ_MODE=control_plane est prêt."""
        from services.control_plane.cutover import control_plane_cutover_status

        out = control_plane_cutover_status()
        click.echo(json.dumps(out, ensure_ascii=False, default=str))
        if not out.get("ready"):
            raise SystemExit(1)
