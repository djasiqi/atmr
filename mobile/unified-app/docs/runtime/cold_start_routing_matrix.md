# Cold Start Routing Matrix

| Source | Expected Route | Android | iOS | Notes |
|---|---|---|---|---|
| push mission | `/(app)/(driver)/missions/[missionId]` | pending | pending | killed app |
| push chat | `/(app)/(company)/chat` | pending | pending | context-aware |
| quick action complete | `/quick-action` -> mission detail | pending | pending | action applied |
| transfer deep link | `/(app)/(company)/ride-details` | pending | pending | requires rideId |
| OTA reopen | update gate screen or login | pending | pending | depends on policy |
