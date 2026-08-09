# Phase 2 Device Proof Matrix

Template de certification device — **ne pas marquer `pass` sans preuve réelle**.

Le gate strict (`npm run phase2:close-auth:strict`) bloque tant qu’une ligne
Android/iOS vaut `pending` ou est vide. Ce gate vit dans le workflow GitHub
`Mobile Phase 2 close authorization` (manuel / PR docs), **pas** dans
`Mobile unified-app (Lint + Jest)`.

Statuts attendus : `pass` | `fail` | `pending` | `n/a`.

| Feature | Android | iOS | Offline | Background | Evidence |
|---|---|---|---|---|---|
| mission notification | pending | pending | pending | pending | pending |
| silent refresh | pending | pending | pending | pending | pending |
| chat attachment | pending | pending | pending | n/a | pending |
| deep link routing | pending | pending | n/a | n/a | pending |
| mission bar | pending | pending | n/a | pending | pending |
| transfer flow | pending | pending | n/a | n/a | pending |

## Sign-off

- QA Device Owner:
- Mobile Runtime Owner:
- Release Owner:

## Close Authorization Checklist

- [ ] Contracts validated (`notification`, `deep_link`, `quick_actions`, `chat_attachment`, `ota_policy`)
- [ ] Runtime rollback validated in less than 15 minutes
- [ ] Runtime observability dashboard active with KPI alerts
- [ ] Cold-start routing matrix completed and reviewed
- [ ] Transfer flow conflict handling (`409`) validated in staging
