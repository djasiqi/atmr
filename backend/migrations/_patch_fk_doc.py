from pathlib import Path

p = Path(__file__).parent / "DEPLOY_RUNBOOK.md"
s = p.read_text(encoding="utf-8")

old = (
    "- **49ff index/contraintes** : blocs assignment, booking, refresh_token, user protégés par _index_exists / _constraint_exists. "
    "Downgrade : drop seulement si existant ; ordre = dépendants d'abord (user, refresh_token, booking, assignment), puis tables (transport_voucher_files avant transport_vouchers)."
)
new = (
    "- **49ff index/contraintes** : index via _index_exists ; FK booking.billing_locked_by_user_id→user via _fk_exists / _get_fk_constraint_name (détection par définition, pas par nom). "
    "Downgrade : drop FK via nom trouvé ; ordre = dépendants d'abord, puis tables."
)

if old in s:
    s = s.replace(old, new, 1)
    print("ok")
else:
    old_u = old.replace("'", "\u2019")
    if old_u in s:
        s = s.replace(old_u, new.replace("'", "\u2019"), 1)
        print("ok unicode")
    else:
        print("not found", repr(s[3500:4000]))

# also add fk_exists line to helpers list
old_help = """  - `column_exists(bind, table_name, column_name, schema="public")`\n\n- **49ff"""
new_help = """  - `column_exists(bind, table_name, column_name, schema="public")`\n  - `get_fk_constraint_name(...)` / `fk_exists(...)` : FK par définition (table.col → referred_table.id).\n\n- **49ff"""
if old_help in s:
    s = s.replace(old_help, new_help, 1)
    print("helpers ok")
else:
    old_help_u = old_help.replace("'", "\u2019")
    if old_help_u in s:
        s = s.replace(old_help_u, new_help.replace("'", "\u2019"), 1)
        print("helpers ok unicode")

p.write_text(s, encoding="utf-8")
