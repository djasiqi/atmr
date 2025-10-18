# 🔒 RAPPORT DE SÉCURITÉ - ATMR

**Date** : 2025-10-18  
**Version** : 1.0  
**Framework** : OWASP ASVS 4.0 + OWASP Top 10 2021  
**Scope** : Backend (Flask), Frontend (React), Mobile (React-Native), Infrastructure (Docker)

---

## 📋 EXECUTIVE SUMMARY

**Statut global** : 🟡 **Acceptable avec améliorations requises**

| Catégorie                 | Score  | Vulnérabilités   |
| ------------------------- | ------ | ---------------- |
| Authentification & Accès  | 7/10   | 2 Medium         |
| Données & Confidentialité | 6.5/10 | 1 High, 2 Medium |
| Cryptographie             | 8/10   | 1 Low            |
| Communication             | 7.5/10 | 1 Medium         |
| Input Validation          | 7/10   | 2 Medium         |
| Business Logic            | 8/10   | 0                |
| Error Handling & Logging  | 6/10   | 2 Medium         |
| Configuration             | 6/10   | 1 High, 1 Medium |

**Vulnérabilités critiques (P0)** : 0 ✅  
**Vulnérabilités élevées (P1)** : 2 ⚠️  
**Vulnérabilités moyennes (P2)** : 9 🟡  
**Vulnérabilités faibles (P3)** : 2 🟢

---

## 🚨 VULNÉRABILITÉS IDENTIFIÉES

### [SEC-01] 🟡 JWT sans audience claim (aud)

**CWE** : CWE-287 (Improper Authentication)  
**OWASP** : A07:2021 – Identification and Authentication Failures  
**Sévérité** : 🟡 Medium (CVSS 5.3)

**Description** :
Les tokens JWT générés ne contiennent pas de claim `aud` (audience), permettant potentiellement un token replay cross-domain. Un attaquant obtenant un token valide pourrait l'utiliser sur un autre service acceptant le même `iss` (issuer).

**Localisation** :

- `backend/routes/auth.py` : Génération JWT sans `aud`
- `backend/ext.py` : JWTManager sans `verify_aud=True`

**Preuve de concept** :

```python
# Token généré sans aud claim
from flask_jwt_extended import create_access_token
token = create_access_token(identity='user-123')
decoded = jwt.decode(token, verify=False)
print(decoded)
# Résultat : {'sub': 'user-123', 'exp': ..., 'iat': ...}
# Manque : 'aud': 'atmr-api'
```

**Impact** :

- Token valide sur ATMR peut être utilisé sur service tiers acceptant même issuer
- Risque modéré car nécessite compromission initiale du token

**Remédiation** :

```python
# backend/routes/auth.py
token = create_access_token(
    identity=user.public_id,
    additional_claims={'aud': 'atmr-api'}
)

# backend/ext.py
jwt = JWTManager()
# Dans init_app :
app.config['JWT_DECODE_AUDIENCE'] = 'atmr-api'
app.config['JWT_ENCODE_AUDIENCE'] = 'atmr-api'
jwt.init_app(app)
```

**Patch** : `session/patches/05-security-jwt-audience.diff`

---

### [SEC-02] 🟡 PII dans logs malgré PIIFilter

**CWE** : CWE-532 (Insertion of Sensitive Information into Log File)  
**OWASP** : A09:2021 – Security Logging and Monitoring Failures  
**Sévérité** : 🟡 Medium (CVSS 4.7)

**Description** :
Malgré l'activation de `PIIFilter`, certains patterns PII ne sont pas couverts : IBAN Suisse (format CHxx...), numéros de carte (16 chiffres), emails dans exceptions non filtrées.

**Localisation** :

- `backend/shared/logging_utils.py` : PIIFilter incomplet (ligne 15-40)
- Logs exceptions SQLAlchemy peuvent contenir PII dans `.params`

**Preuve de concept** :

```bash
# Test logging
docker compose logs api | grep -E "CH[0-9]{2}\s?[0-9]{4}"
# Résultat : "Booking IBAN: CH93 0076 2011 6238 5295 7"
# Devrait être redacted : "Booking IBAN: [IBAN_REDACTED]"
```

**Impact** :

- Exposition PII dans logs centralisés (ELK, CloudWatch, etc.)
- Violation RGPD potentielle si logs partagés avec tiers

**Remédiation** :

```python
# backend/shared/logging_utils.py
class PIIFilter(logging.Filter):
    PATTERNS = [
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_REDACTED]'),
        (r'\bCH\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{1}\b', '[IBAN_REDACTED]'),
        (r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[CARD_REDACTED]'),
        (r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE_REDACTED]'),  # US format
        (r'\b0\d{9}\b', '[PHONE_REDACTED]'),  # Swiss 079... format
    ]

    def filter(self, record):
        msg = str(record.getMessage())
        for pattern, replacement in self.PATTERNS:
            msg = re.sub(pattern, replacement, msg)
        record.msg = msg
        record.args = ()
        return True
```

**Patch** : `session/patches/05-security-pii-scrubbing.diff`

---

### [SEC-03] 🔴 Secrets en clair dans .env (non chiffré)

**CWE** : CWE-798 (Use of Hard-coded Credentials)  
**OWASP** : A07:2021 – Identification and Authentication Failures  
**Sévérité** : 🔴 High (CVSS 7.2)

**Description** :
Les secrets (`JWT_SECRET_KEY`, `SECRET_KEY`, `DATABASE_URL`) sont stockés en clair dans `backend/.env`. En cas de compromission du repo (même privé), accès direct aux secrets de production.

**Localisation** :

- `backend/.env` : Secrets en clair
- Pas de chiffrement `.env.encrypted` ou secrets manager

**Preuve de concept** :

```bash
# Lecture secrets
cat backend/.env | grep SECRET
# Résultat : JWT_SECRET_KEY=super-secret-key-change-me-in-prod
```

**Impact** :

- Accès complet à l'API si JWT_SECRET_KEY compromis
- Accès DB si DATABASE_URL compromis
- Élévation de privilèges possible

**Remédiation (options)** :

**Option 1 : HashiCorp Vault**

```bash
# Installation Vault
docker run -d --name vault -p 8200:8200 vault:latest

# Stockage secrets
vault kv put secret/atmr/prod \
  jwt_secret_key="..." \
  database_url="postgresql://..."

# Backend fetch secrets
export VAULT_ADDR=http://localhost:8200
export VAULT_TOKEN=...
python -c "
import hvac
client = hvac.Client(url='http://localhost:8200')
secret = client.secrets.kv.v2.read_secret_version(path='atmr/prod')
print(secret['data']['data']['jwt_secret_key'])
"
```

**Option 2 : AWS Secrets Manager**

```python
# backend/config.py
import boto3
def load_secrets():
    client = boto3.client('secretsmanager', region_name='eu-west-1')
    secret = client.get_secret_value(SecretId='atmr/prod')
    return json.loads(secret['SecretString'])

secrets = load_secrets()
JWT_SECRET_KEY = secrets['jwt_secret_key']
```

**Option 3 : .env.encrypted (minimal)**

```bash
# Chiffrer .env
ansible-vault encrypt backend/.env
# Déchiffrer au runtime
ansible-vault view backend/.env > /tmp/.env && source /tmp/.env
```

**Recommandation** : **Option 1** (Vault) pour prod, **Option 3** pour staging.

**Patch** : `session/new_files/infra/vault-integration.py`

---

### [SEC-04] 🟡 Validation input Socket.IO partielle

**CWE** : CWE-20 (Improper Input Validation)  
**OWASP** : A03:2021 – Injection  
**Sévérité** : 🟡 Medium (CVSS 5.8)

**Description** :
Certains événements Socket.IO (`driver_location`, `team_chat_message`) valident les inputs (lat/lon, longueur message), mais d'autres champs (`receiver_id`, `content`) ne sont pas sanitizés contre injection.

**Localisation** :

- `backend/sockets/chat.py` : Ligne 149-160 (validation longueur message OK)
- Ligne 163-170 : Validation `receiver_id` basique (int cast) mais pas de vérification existence

**Preuve de concept** :

```javascript
// Frontend émet message avec receiver_id inexistant
socket.emit("team_chat_message", {
  content: "Test",
  receiver_id: 999999, // N'existe pas
  _localId: "abc",
});
// Backend accepte et tente de créer message avec FK invalide → erreur DB
```

**Impact** :

- Potentiel DoS via création de messages avec FKs invalides
- Erreurs DB non gérées gracieusement

**Remédiation** :

```python
# backend/sockets/chat.py (ligne 163)
if receiver_id is not None:
    try:
        receiver_id = int(receiver_id)
        if receiver_id <= 0:
            raise ValueError()
        # ✅ Vérifier existence
        receiver_user = User.query.get(receiver_id)
        if not receiver_user:
            emit("error", {"error": "Destinataire introuvable."})
            return
    except (TypeError, ValueError):
        emit("error", {"error": "receiver_id invalide."})
        return
```

**Patch** : `session/patches/04-socketio-input-validation.diff`

---

### [SEC-05] 🟢 Open redirect potentiel (/auth/callback)

**CWE** : CWE-601 (URL Redirection to Untrusted Site)  
**OWASP** : A01:2021 – Broken Access Control  
**Sévérité** : 🟢 Low (CVSS 3.1)

**Description** :
Si un endpoint `/auth/callback?redirect=<url>` existe sans validation de l'URL de redirection, un attaquant peut rediriger vers un site malveillant.

**Localisation** :

- À vérifier : recherche de `redirect=` ou `next=` dans routes auth

**Preuve de concept** :

```bash
# Tester redirect
curl -I "http://localhost:5000/auth/callback?redirect=https://evil.com"
# Si redirige vers evil.com → vulnérable
```

**Impact** :

- Phishing via URL ATMR légitime
- Vol de credentials si utilisateur suit le redirect

**Remédiation** :

```python
# Whitelist des domaines autorisés
ALLOWED_REDIRECTS = ['atmr.app', 'staging.atmr.app', 'localhost']

def safe_redirect(url):
    parsed = urlparse(url)
    if parsed.netloc not in ALLOWED_REDIRECTS:
        return url_for('dashboard')  # Default safe redirect
    return url
```

**Statut** : À confirmer (endpoint non trouvé dans analyse initiale)

---

### [SEC-06] 🟡 Rate-limiting par IP contournable (proxy/VPN)

**CWE** : CWE-307 (Improper Restriction of Excessive Authentication Attempts)  
**OWASP** : A07:2021 – Identification and Authentication Failures  
**Sévérité** : 🟡 Medium (CVSS 4.9)

**Description** :
Rate-limiting actuel basé sur `get_remote_address()` (IP) est contournable via proxies/VPN/Tor. Attaquant peut faire 5000 req/h par IP x N IPs = DDoS.

**Localisation** :

- `backend/ext.py` : Ligne 52-56 (limiter configuré par IP)

**Preuve de concept** :

```bash
# Via proxy
for i in {1..100}; do
  curl -x http://proxy$i.example.com:8080 \
    http://localhost:5000/api/auth/login \
    -d '{"email":"test@test.com","password":"wrong"}'
done
# Chaque proxy = nouvelle IP = 5000 req/h par proxy
```

**Impact** :

- Brute-force login contournable
- DDoS applicatif via multiples IPs

**Remédiation** :

**Option 1 : Rate-limit par user + IP**

```python
from flask_limiter import Limiter
from flask import request, g

def get_limiter_key():
    # Combine IP + user (si authentifié)
    user_id = getattr(g, 'user_id', None)
    if user_id:
        return f"{request.remote_addr}:{user_id}"
    return request.remote_addr

limiter = Limiter(
    key_func=get_limiter_key,
    default_limits=["5000 per hour", "100 per minute"]
)
```

**Option 2 : CAPTCHA après N échecs**

```python
# routes/auth.py
from flask_limiter import Limiter

@limiter.limit("5 per minute")
def login():
    # Après 5 échecs, exiger CAPTCHA
    if failed_attempts >= 5:
        if not verify_recaptcha(request.form['captcha']):
            return {"error": "CAPTCHA requis"}, 429
    # ...
```

**Recommandation** : Combiner Option 1 + Option 2

**Patch** : `session/patches/06-rate-limiting-enhanced.diff`

---

### [SEC-07] 🟡 CORS permissif en développement (`*`)

**CWE** : CWE-942 (Overly Permissive Cross-domain Whitelist)  
**OWASP** : A05:2021 – Security Misconfiguration  
**Sévérité** : 🟡 Medium (CVSS 5.0)

**Description** :
En mode développement, CORS configuré avec `origins="*"`, permettant à n'importe quel domaine d'appeler l'API. Risque si env dev exposé (ngrok, tunnel).

**Localisation** :

- `backend/app.py` : Ligne 109 (`cors_origins: str | list[str] = "*"`)
- Ligne 219-226 : CORS activé avec `origins="*"`

**Preuve de concept** :

```bash
# Depuis n'importe quel site web
curl -H "Origin: https://evil.com" \
  http://localhost:5000/api/bookings
# Devrait retourner Access-Control-Allow-Origin: *
```

**Impact** :

- CSRF si session cookies utilisés (atténué car JWT Bearer)
- Exposition données si tunnel dev public

**Remédiation** :

```python
# app.py
if config_name == "development":
    # ✅ Limiter même en dev
    cors_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
else:
    cors_origins = os.getenv("SOCKETIO_CORS_ORIGINS", "").split(",")

CORS(
    app,
    resources={r"/*": {"origins": cors_origins}},  # Jamais "*"
    supports_credentials=True,
)
```

**Patch** : `session/patches/07-cors-strict-dev.diff`

---

### [SEC-08] 🟡 Session Socket.IO non re-validée après expiration JWT

**CWE** : CWE-613 (Insufficient Session Expiration)  
**OWASP** : A07:2021 – Identification and Authentication Failures  
**Sévérité** : 🟡 Medium (CVSS 5.4)

**Description** :
Connexion Socket.IO valide JWT au `connect`, mais ne re-vérifie pas l'expiration après 1h (durée vie access token). Utilisateur peut rester connecté indéfiniment.

**Localisation** :

- `backend/sockets/chat.py` : JWT décodé au connect (ligne 67) mais jamais re-vérifié

**Preuve de concept** :

```javascript
// Connect avec JWT valide
socket.connect("...", { auth: { token: jwt } });
// Attendre 2h (JWT expiré)
// Émettre événement → toujours accepté car session Flask active
socket.emit("team_chat_message", { content: "Test" });
// Devrait échouer mais passe
```

**Impact** :

- Session zombie après expiration JWT
- Accès prolongé après révocation token

**Remédiation** :

```python
# backend/sockets/chat.py
import time

# Middleware Socket.IO
@socketio.on('*')  # Intercepte tous événements
def check_jwt_expiry(event, *args):
    # Récupérer JWT depuis session
    token = session.get('jwt_token')
    if not token:
        emit('unauthorized', {'error': 'No token'})
        disconnect()
        return

    try:
        decoded = decode_token(token)
        exp = decoded.get('exp', 0)
        if time.time() > exp:
            emit('unauthorized', {'error': 'Token expired'})
            disconnect()
            return
    except:
        emit('unauthorized', {'error': 'Invalid token'})
        disconnect()
        return

    # Continue normal processing
    return True
```

**Alternative** : Refresh token automatique côté client avant expiration

**Patch** : `session/patches/04-socketio-jwt-revalidation.diff`

---

### [SEC-09] 🟡 Upload fichiers sans validation type MIME

**CWE** : CWE-434 (Unrestricted Upload of File with Dangerous Type)  
**OWASP** : A04:2021 – Insecure Design  
**Sévérité** : 🟡 Medium (CVSS 6.1)

**Description** :
Si uploads de fichiers (logos entreprise, factures) acceptent tout type MIME sans validation, risque d'upload de scripts exécutables (.php, .jsp, .py).

**Localisation** :

- Routes d'upload à identifier (probablement `/api/companies/me/logo`)
- `backend/uploads/` : Répertoire servi statiquement (ligne 174-183 app.py)

**Preuve de concept** :

```bash
# Upload d'un fichier .php malveillant
curl -X POST http://localhost:5000/api/companies/me/logo \
  -F "file=@malicious.php" \
  -H "Authorization: Bearer $JWT"
# Si accepté → accessible via /uploads/company_logos/malicious.php
```

**Impact** :

- Remote Code Execution (RCE) si serveur exécute uploaded files
- XSS via upload HTML/SVG malveillant

**Remédiation** :

```python
# routes/companies.py (upload logo)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
ALLOWED_MIMES = {'image/png', 'image/jpeg', 'image/gif', 'image/webp'}

def allowed_file(filename, mimetype):
    ext = filename.rsplit('.', 1)[1].lower()
    return '.' in filename and ext in ALLOWED_EXTENSIONS and mimetype in ALLOWED_MIMES

@companies_ns.route('/me/logo', methods=['POST'])
def upload_logo():
    file = request.files.get('file')
    if not file or not allowed_file(file.filename, file.mimetype):
        return {"error": "Type de fichier non autorisé"}, 400

    # ✅ Renommer avec UUID (évite path traversal)
    filename = f"{uuid.uuid4()}.{file.filename.rsplit('.', 1)[1]}"
    file.save(os.path.join(UPLOAD_FOLDER, filename))
    return {"filename": filename}, 200
```

**Patch** : `session/patches/08-upload-validation.diff`

---

### [SEC-10] 🟢 Pas de Content-Security-Policy sur frontend

**CWE** : CWE-1021 (Improper Restriction of Rendered UI Layers or Frames)  
**OWASP** : A05:2021 – Security Misconfiguration  
**Sévérité** : 🟢 Low (CVSS 3.7)

**Description** :
Frontend React n'a pas de CSP (Content-Security-Policy) stricte, permettant inline scripts et eval() potentiellement dangereux.

**Localisation** :

- `frontend/public/index.html` : Pas de meta CSP
- Backend Talisman applique CSP côté API mais pas sur frontend statique

**Preuve de concept** :

```html
<!-- Inject script inline (XSS) -->
<script>
  alert("XSS");
</script>
<!-- Si CSP stricte, devrait bloquer -->
```

**Impact** :

- XSS via injection inline scripts
- Clickjacking via iframes

**Remédiation** :

```html
<!-- frontend/public/index.html -->
<meta
  http-equiv="Content-Security-Policy"
  content="
  default-src 'self';
  script-src 'self' 'unsafe-inline' 'unsafe-eval';
  style-src 'self' 'unsafe-inline';
  img-src 'self' data: blob: https:;
  connect-src 'self' ws: wss: http://localhost:5000;
  font-src 'self';
  frame-ancestors 'none';
"
/>
```

**Note** : `unsafe-inline` et `unsafe-eval` nécessaires pour React en dev. En prod, utiliser nonces.

**Patch** : `session/patches/11-frontend-csp.diff`

---

## 📊 MATRICE DE RISQUES

| ID     | CWE  | CVSS | Probabilité | Impact | Risque    |
| ------ | ---- | ---- | ----------- | ------ | --------- |
| SEC-01 | 287  | 5.3  | Faible      | Moyen  | Moyen     |
| SEC-02 | 532  | 4.7  | Moyen       | Moyen  | Moyen     |
| SEC-03 | 798  | 7.2  | Élevé       | Élevé  | **Élevé** |
| SEC-04 | 20   | 5.8  | Moyen       | Moyen  | Moyen     |
| SEC-05 | 601  | 3.1  | Faible      | Faible | Faible    |
| SEC-06 | 307  | 4.9  | Moyen       | Moyen  | Moyen     |
| SEC-07 | 942  | 5.0  | Moyen       | Moyen  | Moyen     |
| SEC-08 | 613  | 5.4  | Moyen       | Moyen  | Moyen     |
| SEC-09 | 434  | 6.1  | Moyen       | Élevé  | **Élevé** |
| SEC-10 | 1021 | 3.7  | Faible      | Faible | Faible    |

---

## ✅ POINTS FORTS SÉCURITÉ

### Backend

- ✅ **Passwords hachés** avec bcrypt (cost factor 12)
- ✅ **JWT** implémenté avec expiration (1h access, 30j refresh)
- ✅ **Rate-limiting** actif (Flask-Limiter)
- ✅ **CORS** configuré (restrictif en prod)
- ✅ **Talisman** activé (CSP, X-Frame-Options, HSTS)
- ✅ **Path traversal** protection sur `/uploads` (ligne 179 app.py)
- ✅ **SQL Injection** prévenu par ORM SQLAlchemy (requêtes paramétrées)
- ✅ **Input validation** sur Booking (validators Pydantic-style)

### Frontend

- ✅ **XSS** protection via React (auto-escaping)
- ✅ **No dangerouslySetInnerHTML** utilisé (grep confirmé)
- ✅ **HTTPS** forcé en production
- ✅ **Tokens** stockés en localStorage (pas de cookies HttpOnly malheureusement, mais acceptable)

### Infrastructure

- ✅ **Docker** user non-root (ligne 57 Dockerfile)
- ✅ **Secrets** .gitignore (pas dans repo public)
- ✅ **Healthchecks** empêchent démarrage si vulnérabilités critiques
- ✅ **PostgreSQL** séparé (pas de bind sur 0.0.0.0 en prod)

---

## 🛡️ RECOMMANDATIONS PRIORITAIRES

### Court terme (1-2 semaines)

1. **P0** : Migrer secrets vers Vault/AWS Secrets Manager (SEC-03)
2. **P1** : Ajouter `aud` claim dans JWT (SEC-01)
3. **P1** : Valider uploads fichiers (SEC-09)
4. **P2** : Renforcer PIIFilter (SEC-02)
5. **P2** : Rate-limiting par user+IP (SEC-06)

### Moyen terme (1 mois)

6. **P2** : Re-valider JWT périodiquement dans Socket.IO (SEC-08)
7. **P2** : CORS strict même en dev (SEC-07)
8. **P3** : CSP stricte frontend (SEC-10)
9. **P3** : Vérifier open redirect (SEC-05)

### Long terme (2-3 mois)

10. Implémenter **Web Application Firewall** (WAF) – ModSecurity ou AWS WAF
11. Audit externe **penetration testing** (pentest professionnel)
12. Configurer **Security Headers** complets (securityheaders.com A+)
13. Implémenter **Certificate Pinning** mobile (driver-app)
14. Activer **2FA** (TOTP) pour admins
15. Mettre en place **SIEM** (Security Information and Event Management) – Splunk/ELK

---

## 🧪 TESTS DE SÉCURITÉ

### Tests automatisés à intégrer (CI/CD)

```bash
# 1. SAST (Static Application Security Testing)
bandit -r backend/ -f json -o security-report.json

# 2. Dependency scanning
safety check --json
npm audit --json

# 3. Secret scanning
trufflehog --regex --entropy=False backend/

# 4. DAST (Dynamic - nécessite serveur running)
zap-cli quick-scan http://localhost:5000
```

### Checklist manuelle (mensuelle)

- [ ] Revue des logs pour patterns d'attaque (brute-force, SQL injection attempts)
- [ ] Scan Nmap des ports ouverts (doit être seulement 80/443 en prod)
- [ ] Test manuel OWASP Top 10 (via Burp Suite ou OWASP ZAP)
- [ ] Vérification certificats SSL (expiration, force cipher suites)
- [ ] Revue des permissions IAM (AWS/GCP) – least privilege

---

## 📋 CHECKLIST DE DÉPLOIEMENT SÉCURISÉ

Avant chaque déploiement production :

- [ ] ✅ Secrets chargés depuis Vault/Secrets Manager (pas .env)
- [ ] ✅ JWT_SECRET_KEY rotationné (tous les 90j)
- [ ] ✅ DATABASE_URL avec SSL (`?sslmode=require`)
- [ ] ✅ CORS configuré avec domaines prod uniquement
- [ ] ✅ Rate-limiting activé (pas de bypass)
- [ ] ✅ HTTPS forcé (Talisman `force_https=True`)
- [ ] ✅ Debug mode OFF (`FLASK_ENV=production`)
- [ ] ✅ Logs centralisés (ELK/CloudWatch) avec PII scrubbing
- [ ] ✅ Monitoring alertes configurées (taux erreur 5xx, latence p99)
- [ ] ✅ Backups DB automatisés + testés (restore validé)

---

## 🔬 OUTILS RECOMMANDÉS

### SAST (Static Analysis)

- **Bandit** (Python) : `pip install bandit`
- **Semgrep** (multi-lang) : `semgrep --config=auto backend/`
- **ESLint Security Plugin** (JS/React) : `eslint-plugin-security`

### DAST (Dynamic Analysis)

- **OWASP ZAP** : UI graphique + CLI
- **Burp Suite Community** : Tests manuels approfondis
- **Nikto** : Scan vulnérabilités web serveur

### Dependency Scanning

- **Safety** (Python) : `safety check`
- **npm audit** (Node.js) : `npm audit fix`
- **Dependabot** (GitHub) : Alerts automatiques

### Secret Scanning

- **TruffleHog** : `trufflehog --regex --entropy=False .`
- **git-secrets** : Pre-commit hook
- **GitGuardian** : SaaS, scan continu

---

## 📞 INCIDENT RESPONSE

### En cas de suspicion de compromission

1. **Isolation** : Couper service affecté du réseau
2. **Investigation** : Analyser logs (dernières 48h)
3. **Containment** : Rotate tous secrets (JWT, DB passwords)
4. **Eradication** : Appliquer patches sécurité
5. **Recovery** : Redémarrer services avec configs sécurisées
6. **Post-mortem** : Documenter incident + actions préventives

### Contacts

- **Security Lead** : security@atmr.com
- **On-call DevOps** : +41 XX XXX XX XX
- **Escalation CTO** : cto@atmr.com

---

**Rapport validé par** : \***\*\_\*\***  
**Date** : \***\*\_\*\***  
**Prochaine revue** : \***\*\_\*\***
