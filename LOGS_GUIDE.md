# Guide pour voir les logs du serveur en continu

## Commandes principales

### 1. Logs du service API (en continu)

```bash
ssh deploy@138.201.155.201
cd /srv/atmr/backend
docker compose logs -f api
```

### 2. Logs de tous les services (en continu)

```bash
docker compose logs -f
```

### 3. Logs d'un service spécifique (dernières 200 lignes)

```bash
docker compose logs --tail 200 api
```

### 4. Logs avec filtrage par mot-clé

```bash
docker compose logs -f api | grep -i "invoice\|error\|exception"
```

### 5. Logs depuis un timestamp spécifique

```bash
docker compose logs --since 10m api
```

## Commandes utiles

### Voir les logs d'erreur uniquement

```bash
docker compose logs api | grep -i error
```

### Voir les logs en temps réel avec timestamps

```bash
docker compose logs -f --timestamps api
```

### Voir les logs de plusieurs services

```bash
docker compose logs -f api celery-worker
```

## Pour déboguer l'erreur de génération de facture

1. **Voir les logs récents de l'API :**

```bash
ssh deploy@138.201.155.201
cd /srv/atmr/backend
docker compose logs --tail 100 api | grep -i "invoice\|generate"
```

2. **Suivre les logs en temps réel :**

```bash
docker compose logs -f api
```

Puis déclenchez la génération de facture depuis le frontend.

3. **Voir les erreurs complètes avec stack trace :**

```bash
docker compose logs api | grep -A 20 "Erreur lors de la génération de facture"
```
