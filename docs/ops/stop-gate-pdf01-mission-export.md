# STOP GATE PDF-01 — Export PDF mission institution (Bon + Rapport)

```txt
Status: PASS
Date: 2026-06-12
Prérequis: TransportRequestDisplayModel v1 (TR-01 PASS), timeline institution
Bloque: implémentation V1 mission PDF
```

## Objectif

Certifier que toutes les sections du Bon de transport et du Rapport de mission peuvent être alimentées depuis le code existant, avec fallbacks documentés pour les 6 scénarios métier PDF-01a…f.

## Inventaire sources de données par section

| Section PDF | Sources primaires (code) | Helper V1 |
|---|---|---|
| En-tête / références | `TransportRequest.id`, `created_at`, `public_id`, `booking_id` | `format_transport_reference`, `format_request_number`, `format_booking_number` |
| Statut final | `TransportRequest.status`, `Booking.status`, `is_round_trip` | `build_mission_status_label` |
| §1 Patient | `InstitutionPatient` via `tr.patient` | `build_patient_block` |
| §2 Client + institution | `build_transport_request_display_blocks`, `contact_on_site`, `Institution`, `created_by_name` | `build_client_identity_block`, `build_institution_snapshot` |
| §3 Transporteur | `accepted_by_company`, `Booking.company` | `build_carrier_block` |
| §4 Mission | `mission_type`, `created_at`, `accepted_at`, `billing_intent` | `build_mission_info` |
| §5 Déroulement | `TransportRequestLeg[]`, display blocks, `booking._get_route_journey()`, `boarded_at`/`completed_at` | `build_route_steps` |
| §6 Historique | `TransportTimelineEvent` via `list_timeline_events(limit=500)` | `resolve_timeline_actor`, `resolve_timeline_channel` |
| §7 Communication | `BookingMessage` query par `booking_id` | `collect_messages(limit=200)` |
| §8 Traçabilité | `public_id`, `generated_at`, SHA-256 | `compute_document_hash` |
| §9 Besoins médicaux | `TransportRequest.get_mobility()`, `notes`, `floor_elevator_info` | `build_medical_block` |
| §10 Facturation | `billing_intent`, `booking.billed_to_type`, `amount`, `invoice_line_id` | `build_billing_block` |
| §11 Pièces jointes | — (V1) | `attachments: []` |
| Classification | `trip_flags`, `mission_type`, `mobility` | `build_request_classification` |

## Champs absents + fallback

**Règle** : jamais d'exception métier pour une donnée absente.

| Situation | Fallback V1 |
|---|---|
| `booking_id` null | Pas de « Réservation # » ; §5 réel partiel ; §7 « Aucun message » |
| `patient` null | Nom depuis `external_reference` ou `#id` |
| `patient.dob` / `external_reference` null | `"—"` |
| Transporteur non assigné | « Non assignée » |
| `contact_on_site.requester_service` null | `"—"` |
| `created_by_name` null | `contact_on_site.requester_name` ou `"—"` |
| `boarded_at` / `completed_at` null | heure réelle `"—"` |
| `route_journey` null | réel `"—"` par étape |
| `legs[]` vide | fallback `pickup_location` / `dropoff_location` sur TR |
| `amount` null | `"—"` |
| Timeline vide | « Aucun événement enregistré » |
| Messages sans booking | « Aucun message » |
| `attachments` V1 | « Aucune pièce jointe » |

## Limites de volume V1

| Collection | Limite | Comportement |
|---|---|---|
| Historique timeline | 500 | Tronquer ; flag `timeline_truncated` |
| Messages | 200 | Tronquer ; flag `messages_truncated` |
| Legs trajet | 20 | Tronquer ; flag `route_legs_truncated` |

## Scénarios métier — verdict

| ID | Scénario | Verdict | Notes audit code |
|---|---|---|---|
| PDF-01a | Mission simple A→B | **PASS** | Legs vides OK via pickup/dropoff TR ; booking_summary + timeline + messages |
| PDF-01b | Multi-destinations | **PASS** | `TransportRequestLeg[]` + `route_journey` multi-leg via `_collect_journey_legs` |
| PDF-01c | Aller-retour institution | **PASS** | `is_round_trip` + `RETURN_COMPLETED` → libellé « Réalisé (aller-retour) » |
| PDF-01d | Mission annulée | **PASS** | `CANCELLED` / `CANCELED` + `cancellation_display_label` booking_summary |
| PDF-01e | Sans booking | **PASS** | DRAFT/SENT/ACCEPTED sans `booking_id` ; timeline request-only |
| PDF-01f | Stabilité rendu | **PASS** | ReportLab platypus + `Paragraph` word wrap ; test auto long contenu |

## Blockers identifiés

Aucun blocker bloquant V1. Points de vigilance implémentation :

1. Timeline servie en DESC par `list_timeline_events` → tri ASC côté contexte.
2. Chauffeur/véhicule : données via `booking.driver` — ✅ **V1.1 implémenté**.
3. Snapshot institution : persisté à `request_converted` — ✅ **V2 implémenté**.

## Enrichissements post-V1

✅ **Implémenté V1.1** : chauffeur, tél. pro, véhicule, jalons opérationnels §5 bis — `mission_report_context.py`, `mission_report_pdf.py`.

✅ **Implémenté V2** : certificat de réalisation, snapshot institution persisté (`accept_offer.py`), chambre patient (étage + unité).

✅ **Implémenté V3 (partiel)** : preuve GPS Redis, référence d'archivage, note signatures numériques — pièces jointes réelles et archivage WORM restent à venir.

✅ **Implémenté** : document STOP GATE validé ; implémentation V1 autorisée.
