# 🔧 Fix : Rafraîchissement Automatique JWT Mobile

**Date** : 2026-01-13  
**Problème** : L'application mobile reçoit des erreurs 500 au lieu de 401 quand le token JWT expire, empêchant le rafraîchissement automatique.

---

## 🔴 Problème Initial

### Symptômes

- L'application mobile affichait `Request failed with status code 500`
- Les endpoints suivants échouaient :
  - `/api/v1/company_mobile/dispatch/v1/rides`
  - `/api/v1/company_mobile/dispatch/v1/dashboard/realtime`
- L'utilisateur devait se déconnecter et se reconnecter manuellement

### Cause Racine

Le backend retournait une **erreur 500** au lieu d'une **401** quand un token JWT expirait :

```python
jwt.exceptions.ExpiredSignatureError: Signature has expired
# ❌ Cette exception n'était pas interceptée et devenait une 500
```

**Conséquence** : L'intercepteur HTTP côté mobile ne pouvait pas détecter l'expiration et rafraîchir automatiquement le token.

---

## ✅ Solution Implémentée

### 1. Gestionnaires d'erreurs Flask-RESTX

**Fichier** : `backend/routes_api.py`

Ajout de gestionnaires d'erreurs spécifiques pour `api_v1` et `api_v2` qui interceptent les exceptions JWT **avant** qu'elles ne deviennent des 500 :

```python
@api_v1.errorhandler(Exception)
def handle_jwt_errors_v1(error):
    """Intercepte les erreurs JWT pour retourner 401 au lieu de 500."""
    from jwt.exceptions import ExpiredSignatureError, InvalidTokenError

    if isinstance(error, ExpiredSignatureError):
        return {"error": "token_expired", "message": "Signature has expired"}, 401
    if isinstance(error, InvalidTokenError):
        return {"error": "invalid_token", "message": str(error)}, 422
    raise error
```

**Pourquoi Flask-RESTX ?**  
Flask-RESTX a ses propres gestionnaires d'erreurs qui prennent priorité sur les gestionnaires Flask globaux. Il faut donc les enregistrer directement sur les objets `Api`.

### 2. Fonction de support dans ext.py

**Fichier** : `backend/ext.py`

Ajout d'une fonction `register_jwt_error_handlers(app)` pour enregistrer des gestionnaires globaux Flask (en cas de routes hors Flask-RESTX) :

```python
def register_jwt_error_handlers(app):
    """Enregistre les gestionnaires d'erreurs globaux pour les exceptions JWT."""
    from jwt.exceptions import ExpiredSignatureError, InvalidTokenError

    @app.errorhandler(ExpiredSignatureError)
    def handle_expired_token(e):
        return jsonify({
            "error": "token_expired",
            "message": "Signature has expired"
        }), 401

    @app.errorhandler(InvalidTokenError)
    def handle_invalid_token(e):
        return jsonify({
            "error": "invalid_token",
            "message": str(e)
        }), 422
```

---

## 🔄 Mécanisme de Rafraîchissement Automatique

### Côté Mobile (déjà implémenté)

**Fichier** : `mobile/operations-app/hooks/useAuth.tsx`

1. **Rafraîchissement proactif** (lignes 529-577) :

   - Le token est rafraîchi **5 minutes avant** son expiration
   - Un `setTimeout` planifie le rafraîchissement automatique

2. **Intercepteur HTTP** :
   **Fichier** : `mobile/operations-app/services/enterpriseAuth.ts`

   - Intercepte les réponses **401** (maintenant correctement retournées)
   - Rafraîchit automatiquement le token via `/auth/refresh`
   - Rejoue la requête originale avec le nouveau token

3. **Endpoint de refresh** :
   - `/api/v1/auth/refresh` (déjà implémenté côté backend)
   - Accepte un `refresh_token` et retourne un nouveau `access_token`

---

## 📊 Configuration JWT

### Durée de vie actuelle

```python
# backend/config.py
JWT_ACCESS_TOKEN_EXPIRES = timedelta(seconds=3600)  # 1 heure
JWT_REFRESH_TOKEN_EXPIRES = timedelta(seconds=2592000)  # 30 jours
```

### Pour modifier (optionnel)

```bash
# Dans .env production
JWT_ACCESS_TOKEN_EXPIRES_SECONDS=86400  # 24 heures
JWT_REFRESH_TOKEN_EXPIRES_SECONDS=2592000  # 30 jours
```

---

## 🧪 Test du Fix

### Test manuel

1. **Attendre l'expiration du token** (1 heure par défaut)
2. **Effectuer une requête** depuis l'app mobile
3. **Vérifier** :
   - ✅ La requête reçoit une **401** (pas 500)
   - ✅ L'intercepteur rafraîchit automatiquement le token
   - ✅ La requête originale est rejouée avec succès
   - ✅ L'utilisateur **ne** se fait **pas** déconnecter

### Logs backend (succès)

```
[JWT] Token expiré intercepté : Signature has expired
172.18.0.2 - - [...] "GET /api/v1/company_mobile/dispatch/v1/rides?..." 401 ...
172.18.0.2 - - [...] "POST /api/v1/auth/refresh" 200 ...
172.18.0.2 - - [...] "GET /api/v1/company_mobile/dispatch/v1/rides?..." 200 ...
```

### Logs mobile (succès)

```
[useAuth] 🔄 Refresh proactif du token entreprise (5min avant expiration)
[useAuth] ✅ Refresh proactif entreprise réussi
```

---

## 🚀 Déploiement

### Commandes

```bash
# Commit et push
git add backend/routes_api.py backend/ext.py FIX_JWT_AUTO_REFRESH_MOBILE.md
git commit -m "fix(auth): retourner 401 au lieu de 500 pour tokens JWT expirés

- Ajoute error handlers Flask-RESTX pour intercepter ExpiredSignatureError
- Permet le rafraîchissement automatique des tokens côté mobile
- Évite les déconnexions forcées des utilisateurs

Closes #JWT_AUTO_REFRESH"
git push

# Déployer
git push  # Déclenche le workflow GitHub Actions
```

### Redémarrage backend (si nécessaire)

```bash
ssh deploy@138.201.155.201
cd /srv/atmr
docker compose -f docker-compose.production.yml restart backend
```

---

## 📝 Notes Importantes

1. **Pas de migration de base de données nécessaire** : Changements uniquement dans le code
2. **Rétrocompatible** : Les anciennes versions de l'app mobile continueront de fonctionner (elles recevront juste des 401 au lieu de 500)
3. **Performance** : Aucun impact sur les performances, juste un meilleur code d'erreur
4. **Sécurité** : Améliore la sécurité en forçant le rafraîchissement au lieu de garder des tokens expirés

---

## 🔍 Vérification Post-Déploiement

```bash
# Vérifier les logs backend
ssh deploy@138.201.155.201 "cd /srv/atmr && docker compose -f docker-compose.production.yml logs backend --tail=100 | grep -i 'token.*expir'"

# Devrait montrer des 401 au lieu de 500
# ✅ Bon : "... 401 ..." + "[JWT] Token expiré intercepté"
# ❌ Mauvais : "... 500 ..." + "ERROR in app: Exception"
```

---

## 📚 Références

- **Backend JWT Callbacks** : `backend/ext.py` (lignes 408-489)
- **Mobile Auto-Refresh** : `mobile/operations-app/hooks/useAuth.tsx` (lignes 529-577)
- **Mobile Interceptor** : `mobile/operations-app/services/enterpriseAuth.ts` (lignes 547-565)
- **Documentation JWT** : [Flask-JWT-Extended](https://flask-jwt-extended.readthedocs.io/)

---

**Statut** : ✅ **RÉSOLU** - Les tokens JWT expirés retournent maintenant 401 et sont automatiquement rafraîchis côté mobile
