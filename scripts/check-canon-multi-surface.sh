#!/usr/bin/env bash
# Vérifie qu'une PR qui touche plusieurs zones sensibles (natif / chauffeur web / dispatch web)
# référence explicitement le canon multi-surface ou une ADR.
# Utilisé par .github/workflows/canon-multi-surface.yml ; exécutable en local :
#   BASE_SHA=$(git merge-base origin/main HEAD) HEAD_SHA=HEAD PR_TITLE="t" PR_BODY="LIRIE_MOBILE_WEB_CANON" bash scripts/check-canon-multi-surface.sh
set -eu

: "${BASE_SHA:?BASE_SHA is required (e.g. github.event.pull_request.base.sha)}"
: "${HEAD_SHA:?HEAD_SHA is required (e.g. github.event.pull_request.head.sha)}"
PR_TITLE="${PR_TITLE:-}"
PR_BODY="${PR_BODY:-}"
PR_COMBINED="${PR_TITLE}${PR_BODY}"

FILES="$(git diff --name-only "$BASE_SHA" "$HEAD_SHA" 2>/dev/null || true)"

touch_ops=false
touch_driver=false
touch_dispatch=false

while IFS= read -r f; do
  [[ -z "${f:-}" ]] && continue
  if [[ "$f" == mobile/unified-app/* ]]; then touch_ops=true; fi
  if [[ "$f" == frontend/src/pages/driver/* ]]; then touch_driver=true; fi
  if [[ "$f" == frontend/src/pages/company/Dispatch/* ]] || [[ "$f" == frontend/src/pages/company/*/Dispatch/* ]]; then
    touch_dispatch=true
  fi
done <<< "$FILES"

zones=0
if $touch_ops; then zones=$((zones + 1)); fi
if $touch_driver; then zones=$((zones + 1)); fi
if $touch_dispatch; then zones=$((zones + 1)); fi

if [[ "$zones" -lt 2 ]]; then
  echo "check-canon-multi-surface: une seule zone (ou aucun fichier) — OK (skip)."
  exit 0
fi

shopt -s nocasematch
if [[ "$PR_COMBINED" =~ LIRIE_MOBILE_WEB_CANON|docs/LIRIE_MOBILE_WEB_CANON|docs/adr/|API_CANON_RULES|013-dispatch|013-dispatch-api-boundary|ADR[[:space:]._-]*013|ADR013 ]]; then
  echo "check-canon-multi-surface: référence canon / ADR / API_CANON / ADR013 — OK."
  exit 0
fi

echo "check-canon-multi-surface: échec — cette PR modifie au moins deux zones parmi:" >&2
echo "  - mobile/unified-app" >&2
echo "  - frontend/src/pages/driver" >&2
echo "  - frontend/src/pages/company/.../Dispatch" >&2
echo "" >&2
echo "Ajoutez dans le titre ou la description de la PR une référence explicite à :" >&2
echo "  - docs/LIRIE_MOBILE_WEB_CANON.md (ex. LIRIE_MOBILE_WEB_CANON)" >&2
echo "  - docs/adr/ (ADR)" >&2
echo "  - docs/API_CANON_RULES.md" >&2
echo "  - ADR 013 (ex. 013-dispatch, 013-dispatch-api-boundary, ADR 013)" >&2
echo "" >&2
echo "Ou découpez la PR par surface (cf. canon §5)." >&2
exit 1
