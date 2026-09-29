# Budget Google Geocoding (projet `lirie-app`)

Le SKU Geocoding est facturé par Google Maps Platform. Un budget Cloud prévient, il ne coupe pas les appels. La coupure réelle est dans l'application, complétée par les quotas de la console.

## Barrière applicative

Tous les appels HTTP vers `https://maps.googleapis.com/maps/api/geocode/json` passent par `backend/services/geolocation/google_geocoding_gate.py`.

| Protection | Valeur |
| --- | ---: |
| Plafond quotidien (heure de Zurich) | 200 |
| Débit | 20 / minute |
| Cache d'une réponse OK | 29 jours |
| Alerte journal | 100, 150 et 180 appels / jour |
| Appel direct depuis l'app chauffeur | interdit |

Compteurs Redis :

- `geocoding:google:day:YYYY-MM-DD` : total du jour
- `geocoding:google:day:YYYY-MM-DD:<source>` : total par origine
- `geocoding:google:raw:<hash>` : réponse mise en cache

Origines enregistrées sur `geocoding_requests_total{provider="google",source,outcome}` :

`booking_creation`, `booking_async`, `pricing`, `institution_route`, `company_address`, `client_address`, `reverse_geocode`, `api_geocode`, `autocomplete_enrichment`, `driver_map`, `dispatch`, `distance_fallback`.

`outcome=google` correspond à un appel envoyé. `cache_hit` n'est pas facturable. `budget_blocked` et `rate_limited` sont refusés avant l'appel.

Si le plafond est atteint, la création de réservation continue avec les coordonnées de repli déjà prévues. Le refus n'est pas mis en cache, pour que le lendemain les adresses puissent à nouveau être géocodées.

L'app chauffeur (`geocodeMissionAddress`) interroge `GET /geocode/geocode`. Elle ne contient plus de clé ni d'appel Geocoding.

Variables d'environnement optionnelles : `GEOCODING_DAILY_HARD_LIMIT`, `GEOCODING_PER_MINUTE_LIMIT`, `GOOGLE_MAPS_CACHE_TTL` (défaut 29 jours).

## À régler dans la console Google Cloud

Ces deux réglages ne sont pas dans le dépôt.

1. Projet `lirie-app` → Google Maps Platform → Quotas → Geocoding API : **200 / jour** et **20 / minute**.
2. Clés Android et iOS : retirer **Geocoding API**. Elles ne doivent servir qu'à l'affichage des cartes (Maps SDK). La clé serveur conserve Geocoding, et Directions ou Distance Matrix si ces SKU sont utilisés.

Le palier gratuit (10 000 requêtes / mois) est calculé au niveau du compte de facturation. Tout autre projet du même compte qui appelle Geocoding entre dans le même décompte.

Quotas à saisir tels quels :

```text
Geocoding API
v3 requests per minute      = 20
v3 requests per minute/user = 20
v3 requests per day         = 200
```

Clés : Android et iOS sans Geocoding API (Maps SDK seulement). La clé serveur garde Geocoding, avec restriction serveur.

## Alerte autocomplete

Photon doit rester le chemin normal. Google n'est qu'un enrichissement. Journal d'erreur dès le 51e appel du jour pour `source=autocomplete_enrichment`.

Alerte Prometheus suggérée :

```text
sum(increase(geocoding_requests_total{source="autocomplete_enrichment",outcome="google"}[1d])) > 50
```

Le ratio `geocoding_google_call_ratio` (et `google / (google + cache_hit + budget_blocked + rate_limited)`) doit baisser quand les adresses récurrentes sont servies par le cache.

## Gate

| Critère | Statut |
| --- | --- |
| Aucun appel Geocoding direct depuis l'app chauffeur | PASS (code) |
| Un seul point de sortie facturable | PASS (code) |
| Cache : HIT = aucun appel Google | PASS (code) |
| 10 requêtes identiques simultanées = 1 appel Google | PASS (code) |
| 201e appel du jour = budget_blocked | PASS (code) |
| 21e appel de la minute = rate_limited | PASS (code) |
| Réservation créée si le quota Geocoding bloque la distance | PASS (code, distance 0, coordonnées de repli) |
| Coordonnées déjà présentes = pas de re-géocodage | PASS (code) |
| Reverse geocoding hors tick GPS | PASS (code) |
| Quota Google Cloud 20/min et 200/jour | À configurer dans la console |
| Retrait Geocoding des clés Android et iOS | À configurer dans la console |
| `outcome=google` observé en production | En attente de déploiement |
