# CodeQL `py/weak-sensitive-data-hashing`

Classification individuelle des 5 findings ouverts (alertes 245–249).
Aucune migration cosmétique `MD5→SHA256` / `SHA1→SHA256`.

Règle : un hash n’est pas vulnérable parce qu’il s’appelle MD5 ou SHA-1,
et n’est pas sûr parce qu’il s’appelle SHA-256. La décision dépend du
rôle (donnée → menace → propriété requise → primitive → compatibilité).

Stockage des mots de passe applicatifs : `User.set_password` →
`werkzeug.security.generate_password_hash` (`pbkdf2:` / `scrypt:`).
Ne pas toucher ce chemin. Ne pas remplacer un SHA-1/SHA-256 de mot de
passe par un autre hash simple.

---

## Tableau

| ID | Algorithme | Usage | Frontière sécu | Verdict | Action |
|----|------------|-------|----------------|---------|--------|
| 245 | SHA-256 | empreinte de version d’un hash déjà stocké | NON (pas un password hash) | FALSE_POSITIVE | NO_CHANGE |
| 246 | SHA-256 | OTP passwordless (dev only) | OUI en dev / NON en prod | SAFE_NON_SECURITY_USE* | NO_CHANGE + DOCUMENT |
| 247 | HMAC-SHA256 | lookup clé API haute entropie | OUI (stockage secret) | FALSE_POSITIVE | NO_CHANGE |
| 248 | MD5 | clé cache Nominatim | NON | SAFE_NON_SECURITY_USE | NO_CHANGE |
| 249 | SHA-1 | protocole HIBP k-anonymity | NON (pas un password hash) | SAFE_NON_SECURITY_USE | NO_CHANGE |

\*L’OTP 6 chiffres a une faible entropie : SHA-256 seul ne résiste pas à
un dump Redis. La frontière réelle est le **404 hors development**
(SEC-03) + TTL + 5 tentatives + rate limit + `compare_digest`.

---

## 245 — empreinte `pwd_hash` (refresh token)

```text
ID : 245
fichier : backend/routes/auth.py
ligne : 2202
fonction : _get_password_hash_version

INPUT :
source des données = user.password déjà hashé (KDF werkzeug / bcrypt legacy)
contrôlé par utilisateur ? = non (sortie du KDF)
secret ? = non (c’est un hash stocké)
password ? = non (plus le mot de passe en clair)
PII ? = non
faible entropie ? = non (préfixe d’un hash salé)

USAGE DU HASH : NON_SECURITY_IDENTIFIER
ALGORITHM : SHA256
SECURITY BOUNDARY : NO
PROPERTY REQUIRED :
collision resistance = faible (changement de KDF ⇒ changement d’extrait)
preimage resistance = none
second-preimage resistance = none
unpredictability = none
password stretching = none (déjà fait par le KDF)
none = identifier de version

ATTACKER BENEFIT IF COLLISION/PREIMAGE :
une collision SHA-256 n’ouvre pas de login. En pratique l’extrait
16 caractères d’un hash werkzeug est le préfixe d’algorithme
(`pbkdf2:sha256:6`) : l’empreinte est identique pour tous les
comptes au même schéma. Ce contrôle ne discrimine donc pas un
changement de mot de passe. L’invalidation réelle est
`token_version` + révocation Redis / sessions (Lot 0 SEC-02).

PERSISTENCE / COMPATIBILITY :
DB = non (claim JWT `pwd_hash`, 16 hex)
cache = non
external protocol = non
API contract = claim refresh existant — hasher le digest KDF
complet changerait la sémantique sans changer le format
(migration silencieuse interdite) et forcerait un logout de masse.

VERDICT : FALSE_POSITIVE
ACTION : NO_CHANGE
```

CodeQL voit le champ `password` et conclut « password hashing ».
L’entrée est un **extrait du digest déjà stocké**, pas le secret.
Le préfixe 16 caractères rend l’empreinte constante pour werkzeug ;
ce n’est pas une raison de remplacer SHA-256 par bcrypt.

```text
FINDING : 245
ALGORITHM : SHA-256
INPUT : préfixe du hash KDF stocké (16 caractères)
USE : versionner les refresh tokens (invalidation après changement MDP)
SECURITY BOUNDARY : NO
WHY COLLISION/PREIMAGE DOES NOT GRANT SECURITY BENEFIT :
SHA-256 n’est pas la primitive qui protège le mot de passe (KDF
déjà appliqué). L’extrait 16 caractères d’un hash werkzeug est
un préfixe d’algorithme constant : ce n’est pas non plus le
contrôle d’invalidation. `token_version` + révocation sessions
couvrent le changement de mot de passe.
COMPATIBILITY REASON : claim JWT déjà émis ; migration silencieuse
casserait les sessions.
CONCLUSION : FALSE_POSITIVE / SAFE_NON_SECURITY_USE
```

---

## 246 — OTP passwordless

```text
ID : 246
fichier : backend/routes/auth.py
ligne : 4548 (et 4607 à la vérification)
fonction : PasswordlessOtpRequest.post / PasswordlessOtpVerify.post

INPUT :
source des données = secrets.randbelow(1000000) formaté 6 chiffres
contrôlé par utilisateur ? = non à la création ; oui à la vérif
secret ? = secret court temporaire
password ? = non
PII ? = non
faible entropie ? = OUI (1 000 000 valeurs)

USAGE DU HASH : OTHER (auth passwordless, environnement development)
ALGORITHM : SHA256 + hmac.compare_digest
SECURITY BOUNDARY : YES en development seulement
PROPERTY REQUIRED :
collision resistance = none
preimage resistance = souhaitable contre dump cache, insuffisante
  avec SHA-256 seul sur 6 chiffres
second-preimage resistance = none
unpredictability = fournie par secrets.randbelow, pas par SHA-256
password stretching = non applicable (OTP, pas MDP)
none = (voir modèle en couches)

ATTACKER BENEFIT IF COLLISION/PREIMAGE :
dump Redis/dev cache → brute-force offline 1e6 SHA-256 → code OTP
si la session est encore vivante. En production l’endpoint répond 404.

PERSISTENCE / COMPATIBILITY :
DB = non
cache = Redis / dict processus, clé auth:passwordless_otp:*, TTL
external protocol = non
API contract = {otp_session_id, code} — le hash n’est pas exposé

VERDICT : SAFE_NON_SECURITY_USE (modèle en couches, hors prod)
ACTION : NO_CHANGE + DOCUMENT
```

Modèle complet (ne pas transformer le protocole) :

| Contrôle | Valeur |
|----------|--------|
| Environnement | 404 si `ENVIRONMENT != development` (Lot 0 SEC-03) |
| TTL | `PASSWORDLESS_OTP_TTL_SECONDS` défaut 600, plancher 120 |
| Tentatives | 5 puis 429 |
| Rate limit | 40/h request, 60/h verify |
| Comparaison | `hmac.compare_digest` |
| Secret serveur dans le digest | non |
| Persisté DB | non |
| Password storage | non (bcrypt/werkzeug inchangé) |

HMAC(server_secret, otp) améliorerait un dump Redis **en
development**. CodeQL flaggerait encore HMAC-SHA256 (cf. alerte 247).
Hors prod + TTL + tentatives + rate limit : pas de migration.

```text
FINDING : 246
ALGORITHM : SHA-256
INPUT : OTP 6 chiffres (faible entropie)
USE : cacher le clair dans un cache éphémère (dev only)
SECURITY BOUNDARY : NO en production (404)
WHY COLLISION/PREIMAGE DOES NOT GRANT SECURITY BENEFIT :
en production il n’y a pas d’OTP. En dev, le dump cache est déjà un
compromis hôte ; la fenêtre est le TTL ; l’online est borné à 5 essais.
COMPATIBILITY REASON : sessions Redis en vol ; pas de table à migrer,
mais aucun gain prod.
CONCLUSION : SAFE_NON_SECURITY_USE
```

---

## 247 — HMAC des clés API institution

```text
ID : 247
fichier : backend/models/institution_api_key.py
ligne : 92
fonction : hash_api_key

INPUT :
source des données = lir_ + secrets.token_hex(32) (256 bits)
contrôlé par utilisateur ? = non à la génération
secret ? = oui (clé API)
password ? = non
PII ? = non
faible entropie ? = non

USAGE DU HASH : SECRET_DERIVATION / lookup (pepper HMAC)
ALGORITHM : HMAC-SHA256 (hmac.new, pas hash(secret+message))
SECURITY BOUNDARY : YES (ne pas stocker la clé brute)
PROPERTY REQUIRED :
collision resistance = oui (unicité lookup)
preimage resistance = oui (haute entropie suffit)
second-preimage resistance = oui
unpredictability = fournie par secrets
password stretching = non (inapproprié pour 256 bits)
none =

ATTACKER BENEFIT IF COLLISION/PREIMAGE :
préimage d’un HMAC-SHA256 sur 256 bits : inatteignable.
bcrypt n’apporte rien et casserait le lookup (64 hex).

PERSISTENCE / COMPATIBILITY :
DB = institution_api_keys.key_hash VARCHAR(64) unique
cache = non
external protocol = header X-API-Key
API contract = hash jamais exposé (serialize sans brute)

VERDICT : FALSE_POSITIVE
ACTION : NO_CHANGE
```

HMAC réel : `hmac.new(API_KEY_HMAC_SECRET, raw_key, hashlib.sha256)`.
Lookup par égalité du digest (index unique), pas comparaison du secret
en clair. Clé brute jamais stockée.

Risque accepté (hors algo) : défaut `dev-secret-change-in-prod` si
`API_KEY_HMAC_SECRET` absent. Avec une clé 256 bits, un pepper connu
redescend à SHA-256(key) — toujours hors brute-force.

```text
FINDING : 247
ALGORITHM : HMAC-SHA256
INPUT : clé API 256 bits
USE : stockage / lookup
SECURITY BOUNDARY : YES — primitive déjà correcte
WHY COLLISION/PREIMAGE DOES NOT GRANT SECURITY BENEFIT :
CodeQL exige un KDF mot de passe. Un KDF lent est le mauvais outil
pour un secret aléatoire haute entropie.
COMPATIBILITY REASON : colonne key_hash 64 hex déjà en production.
CONCLUSION : FALSE_POSITIVE
```

---

## 248 — MD5 cache Nominatim

```text
ID : 248
fichier : backend/services/geolocation/maps.py
ligne : 502
fonction : geocode_address_nominatim

INPUT :
source des données = adresse normalisée (+ pays)
contrôlé par utilisateur ? = oui (adresse)
secret ? = non
password ? = non
PII ? = adresse (privée), pas un secret d’auth
faible entropie ? = variable

USAGE DU HASH : CACHE_KEY
ALGORITHM : MD5 (usedforsecurity=False)
SECURITY BOUNDARY : NO
PROPERTY REQUIRED :
collision resistance = seulement anti-collision accidentelle
preimage resistance = none
second-preimage resistance = none
unpredictability = none
password stretching = none
none = identifiant de cache

ATTACKER BENEFIT IF COLLISION/PREIMAGE :
deux adresses → même clé Redis → mauvais lat/lon temporaire
(géocode faux, miss/hit incorrect). Pas d’auth bypass, pas de
permission, pas de fuite de secret, pas de frontière d’intégrité.

PERSISTENCE / COMPATIBILITY :
DB = non
cache = Redis `nominatim:geocode:{md5}` TTL 24h + LRU local
external protocol = Nominatim
API contract = non

VERDICT : SAFE_NON_SECURITY_USE
ACTION : NO_CHANGE
```

`usedforsecurity=False` est déjà posé. Remplacer MD5 par SHA-256
invaliderait le cache sans gain de sécurité.

```text
FINDING : 248
ALGORITHM : MD5
INPUT : adresse normalisée
USE : clé de cache géocode
SECURITY BOUNDARY : NO
WHY COLLISION/PREIMAGE DOES NOT GRANT SECURITY BENEFIT :
impact = mauvaise géolocalisation temporaire. Pas de franchissement
de frontière de sécurité.
COMPATIBILITY REASON : clés Redis déjà écrites.
CONCLUSION : SAFE_NON_SECURITY_USE
```

---

## 249 — HIBP / SHA-1

```text
ID : 249
fichier : backend/security/password_policy.py
ligne : 127
fonction : PasswordPolicyService.check_hibp

INPUT :
source des données = mot de passe en clair, transitoire, non persisté
contrôlé par utilisateur ? = oui
secret ? = oui (temporaire)
password ? = oui (vérification fuite, pas stockage)
PII ? = non
faible entropie ? = selon le mot de passe

USAGE DU HASH : HIBP_K_ANONYMITY / PROTOCOL_COMPATIBILITY
ALGORITHM : SHA1 (usedforsecurity=False) — imposé par HIBP
SECURITY BOUNDARY : NO (le SHA-1 n’authentifie rien)
PROPERTY REQUIRED :
collision resistance = none (protocole externe)
preimage resistance = none
second-preimage resistance = none
unpredictability = none
password stretching = none (stockage = KDF séparément)
none = compatibilité API range/

ATTACKER BENEFIT IF COLLISION/PREIMAGE :
une collision SHA-1 ne contourne ni bcrypt/werkzeug ni l’auth.
Au pire un faux positif/négatif HIBP (fail-open réseau déjà accepté).

PERSISTENCE / COMPATIBILITY :
DB = non
cache = non
external protocol = https://api.pwnedpasswords.com/range/{5}
API contract = k-anonymity HIBP

VERDICT : SAFE_NON_SECURITY_USE
ACTION : NO_CHANGE
```

Vérifications protocole :

- password temporaire → SHA-1 → préfixe 5 caractères → GET range
- suffixe comparé localement
- SHA-1 complet : non persisté, non loggé, non envoyé en entier
- logs : compteur de fuites uniquement
- mot de passe : jamais dans l’URL ni dans le body HIBP

```text
FINDING : 249
ALGORITHM : SHA-1
INPUT : password transitoire
USE : k-anonymity Have I Been Pwned
SECURITY BOUNDARY : NO
WHY COLLISION/PREIMAGE DOES NOT GRANT SECURITY BENEFIT :
le SHA-1 n’est ni stockage, ni signature, ni preuve d’intégrité.
COMPATIBILITY REASON : l’API HIBP n’accepte que SHA-1.
CONCLUSION : SAFE_NON_SECURITY_USE / PROTOCOL_COMPATIBILITY
```

---

## HMAC (hors 247)

CSRF (`backend/services/security/csrf.py`) : HMAC-SHA256 +
`hmac.compare_digest`. Pas dans les 5 findings. Aucun changement.

---

## Compatibilité

| Surface | Décision |
|---------|----------|
| DB (`users.password`, `institution_api_keys.key_hash`) | inchangée |
| Cache Nominatim / OTP Redis | inchangé |
| HIBP | SHA-1 conservé |
| JWT `pwd_hash` | format 16 hex conservé |
| bcrypt / werkzeug | inchangé |
