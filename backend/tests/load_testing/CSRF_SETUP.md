# 🔐 Configuration CSRF pour Tests de Charge

## Problème Identifié

Lors de l'exécution des tests Locust, l'erreur suivante se produit :

```
403 FORBIDDEN: Token CSRF manquant
```

Le backend Flask requiert un token CSRF pour toutes les requêtes POST, ce qui bloque les tests de charge.

## Solutions Disponibles

### Option 1 : Désactiver CSRF pour Tests (Recommandé)

Créer un endpoint spécial `/api/auth/login-test` sans protection CSRF.

**Fichier :** `backend/routes/auth.py`

```python
@auth_ns.route("/login-test")
class LoginTest(Resource):
    """Login endpoint pour tests (sans CSRF)."""

    @auth_ns.expect(login_model)
    @auth_ns.response(200, "Connexion réussie")
    def post(self):
        """Login sans validation CSRF (environnement test uniquement)."""
        # ⚠️ Activer UNIQUEMENT en environnement test/dev
        if os.getenv("FLASK_ENV") not in ["development", "testing"]:
            abort(403, "Endpoint disponible uniquement en dev/test")

        # ... (même logique que /login normal)
        data = request.json
        email = data.get("email")
        password = data.get("password")

        user = User.query.filter_by(email=email).first()
        if not user or not user.check_password(password):
            return {"error": "Identifiants invalides"}, 401

        access_token = create_access_token(identity=user.id)
        return {
            "access_token": access_token,
            "user": user.to_dict()
        }, 200
```

**Puis modifier les tests Locust :**

```python
# dispatch_load_test.py, ligne 68
response = self.client.post(
    "/api/auth/login-test",  # ← Utiliser endpoint test
    json={
        "email": "admin@test.com",
        "password": "test123",
    },
    name="[AUTH] Login",
)
```

### Option 2 : Gérer CSRF dans Locust

Obtenir et envoyer le token CSRF dans chaque requête.

**Étapes :**

1. **Obtenir le token CSRF** (GET sur une page qui le fournit)
2. **Stocker le token** dans les cookies/headers
3. **Envoyer le token** avec chaque requête POST

**Exemple :**

```python
class DispatchLoadTest(HttpUser):
    csrf_token: str | None = None

    def on_start(self) -> None:
        # 1. Obtenir CSRF token
        response = self.client.get("/api/csrf-token")
        if response.status_code == 200:
            self.csrf_token = response.json().get("csrf_token")

        # 2. Login avec CSRF token
        self._login()

    def _login(self) -> None:
        headers = {"X-CSRF-TOKEN": self.csrf_token} if self.csrf_token else {}

        response = self.client.post(
            "/api/auth/login",
            json={"email": "admin@test.com", "password": "test123"},
            headers=headers,
            name="[AUTH] Login",
        )
        # ...
```

**Nécessite backend :**

```python
# backend/routes/csrf.py
@csrf_ns.route("/token")
class CSRFToken(Resource):
    def get(self):
        """Obtenir un token CSRF."""
        token = generate_csrf()
        return {"csrf_token": token}, 200
```

### Option 3 : Variable d'Environnement CSRF_DISABLED

Désactiver CSRF globalement en mode test.

**Fichier :** `backend/config.py`

```python
class Config:
    # ...
    WTF_CSRF_ENABLED = os.getenv("CSRF_ENABLED", "true").lower() == "true"
```

**Puis :**

```bash
# Lancer backend sans CSRF pour tests
export CSRF_ENABLED=false
docker-compose up -d api

# Lancer tests Locust
python -m locust -f tests/load_testing/dispatch_load_test.py ...

# Réactiver CSRF après tests
export CSRF_ENABLED=true
docker-compose restart api
```

## Recommandation

**Option 1** est la plus propre :

- ✅ CSRF reste actif en production
- ✅ Endpoint test isolé et protégé (env check)
- ✅ Pas besoin de modifier la config globale
- ✅ Tests Locust simples et lisibles

## Implémentation Rapide (Option 1)

### 1. Ajouter endpoint test dans `backend/routes/auth.py` :

```python
@auth_ns.route("/login-test")
class LoginTest(Resource):
    """Login sans CSRF pour tests de charge."""

    @auth_ns.expect(login_model)
    def post(self):
        """Login test (dev/test uniquement)."""
        if os.getenv("FLASK_ENV") not in ["development", "testing"]:
            abort(403)

        data = request.json
        user = User.query.filter_by(email=data.get("email")).first()

        if not user or not user.check_password(data.get("password")):
            return {"error": "Identifiants invalides"}, 401

        access_token = create_access_token(identity=user.id)
        return {"access_token": access_token}, 200
```

### 2. Modifier les 3 fichiers de test Locust :

```bash
# Dans dispatch_load_test.py, multi_company_test.py, slow_osrm_test.py
# Remplacer "/api/auth/login" par "/api/auth/login-test"
```

### 3. Relancer tests :

```bash
python -m locust -f tests/load_testing/dispatch_load_test.py \
    --host=http://localhost:5000 \
    --users=1 \
    --spawn-rate=1 \
    --run-time=2m \
    --headless
```

## Tests de Validation

Une fois la solution implémentée, valider avec :

```bash
# Test manuel
curl -X POST http://localhost:5000/api/auth/login-test \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@test.com","password":"test123"}'

# Test Locust rapide
python -m locust -f tests/load_testing/dispatch_load_test.py \
    --host=http://localhost:5000 \
    --users=1 \
    --run-time=30s \
    --headless
```

---

**Date :** 7 janvier 2025  
**Status :** Documentation créée - Implémentation requise  
**Priority :** P0 (bloquant pour C2-J3-4)
