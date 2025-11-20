# Configuration et Utilisation de Semgrep

## 📋 Table des matières

1. [Qu'est-ce que Semgrep ?](#quest-ce-que-semgrep-)
2. [Pourquoi utiliser Semgrep ?](#pourquoi-utiliser-semgrep-)
3. [Configuration](#configuration)
4. [Gestion des faux positifs](#gestion-des-faux-positifs)
5. [Meilleures pratiques](#meilleures-pratiques)

## Qu'est-ce que Semgrep ?

**Semgrep** est un analyseur de code statique (SAST - Static Application Security Testing) qui :

- 🔍 Scanne votre code pour détecter des vulnérabilités de sécurité
- 🛡️ Applique des règles de sécurité (OWASP Top 10, CWE, etc.)
- 🎯 Identifie des anti-patterns et des problèmes de qualité de code
- ⚡ S'intègre facilement dans CI/CD pour un feedback rapide

### À quoi sert Semgrep ?

Semgrep est particulièrement utile pour :

1. **Détecter des vulnérabilités courantes** :
   - Injection SQL
   - Désérialisation non sécurisée (pickle, yaml.load, etc.)
   - Mots de passe non validés
   - Secrets en clair dans le code
   - Utilisation de fonctions dangereuses (eval, exec, etc.)

2. **Appliquer des standards de sécurité** :
   - Règles OWASP Top 10
   - Règles CWE (Common Weakness Enumeration)
   - Bonnes pratiques de sécurité

3. **Prévenir les problèmes avant la production** :
   - Blocage des PR contenant des vulnérabilités
   - Rapports automatisés dans CI/CD
   - Feedback immédiat aux développeurs

## Pourquoi utiliser Semgrep ?

### ✅ Avantages

- **Gratuit et open-source** : Pas de coût pour l'utilisation de base
- **Rapide** : Analyse des projets en quelques secondes
- **Précis** : Moins de faux positifs que certains outils SAST
- **Configurable** : Règles personnalisables et exclusions ciblées
- **Intégration CI/CD** : Facile à intégrer dans GitHub Actions, GitLab CI, etc.
- **Multi-langages** : Support de Python, JavaScript, Java, Go, etc.

### ⚠️ Limitations

- **Faux positifs possibles** : Nécessite du triage (ex: utilisation légitime de pickle pour ML)
- **Ne remplace pas l'audit manuel** : Complémentaire aux revues de code
- **Configuration nécessaire** : Besoin d'ajuster les règles selon le contexte

### 💡 Quand utiliser Semgrep ?

Semgrep est particulièrement utile pour :

- ✅ Projets avec beaucoup de code (détection automatisée)
- ✅ Équipes qui veulent appliquer des standards de sécurité
- ✅ CI/CD où vous voulez bloquer les vulnérabilités automatiquement
- ✅ Projets qui doivent respecter des normes de sécurité (ISO 27001, SOC 2, etc.)

## Configuration

### Structure actuelle du projet

```
atmr/
├── .semgrep.yml              # Configuration principale Semgrep
├── backend/
│   ├── .semgrepignore        # Fichiers à ignorer
│   └── ...                   # Code source
└── .github/
    └── workflows/
        └── backend-tests.yml # CI/CD avec Semgrep
```

### Fichier de configuration principal (`.semgrep.yml`)

```yaml
# Configuration Semgrep pour le projet ATMR
rules:
  - p/ci                    # Règles de qualité de code
  - p/security-audit        # Règles de sécurité OWASP

exclude:
  # Patterns de fichiers à ignorer globalement
```

### Gestion des exclusions (`.semgrepignore`)

Pour ignorer des fichiers spécifiques :

```
# Fichiers à ignorer
**/__pycache__/**
**/*.pyc
**/.pytest_cache/**
```

### Suppressions inline dans le code

Pour ignorer une règle spécifique sur une ligne :

```python
# nosemgrep: python.lang.security.deserialization.pickle.avoid-pickle
joblib.dump(model_data, f)
```

**Format** :
- `# nosemgrep: rule-id` - Ignorer une règle spécifique
- `# nosemgrep` - Ignorer toutes les règles (non recommandé)

## Gestion des faux positifs

### Cas courant : Utilisation de pickle/joblib pour ML

**Problème** : Semgrep détecte l'utilisation de pickle comme vulnérable.

**Solution** : Utiliser `joblib` (recommandé pour scikit-learn) avec suppression inline :

```python
# Utilisation de joblib (recommandé pour scikit-learn) au lieu de pickle direct
# joblib utilise pickle en interne mais avec des optimisations pour numpy/scipy
joblib.dump(model_data, f)  # nosemgrep: python.lang.security.deserialization.pickle.avoid-pickle
```

### Cas courant : Validation de mots de passe personnalisée

**Problème** : Semgrep détecte que `set_password()` n'utilise pas la validation Django.

**Solution** : Valider explicitement avant d'appeler `set_password()` :

```python
from routes.utils import validate_password_or_raise

# Validation explicite du mot de passe avant set_password (sécurité)
validate_password_or_raise(new_password, _user=user)
user.set_password(new_password)  # nosemgrep: python.django.security.audit.unvalidated-password.unvalidated-password
```

### Quand ignorer une règle ?

✅ **Bon** : Ignorer quand :
- L'utilisation est justifiée et documentée
- Il y a une alternative sécurisée mais Semgrep ne la reconnaît pas
- C'est un faux positif évident

❌ **Mauvais** : Ignorer quand :
- Vous ne comprenez pas le problème
- Vous voulez simplement faire passer le CI
- Le code présente un vrai risque de sécurité

## Meilleures pratiques

### 1. Intégration CI/CD

```yaml
- name: Run Semgrep
  run: |
    cd backend
    # Générer rapport JSON (toujours créer le rapport)
    semgrep --config p/ci --config p/security-audit . --json -o semgrep.json || true
    # Bloquer sur les findings
    semgrep --config p/ci --config p/security-audit . --error
```

### 2. Tri des résultats

1. **Vérifier chaque finding** : Ne pas ignorer automatiquement
2. **Documenter les suppressions** : Expliquer pourquoi vous ignorez
3. **Réviser régulièrement** : Revoir les suppressions lors des audits

### 3. Configuration par projet

- **Utiliser `.semgrep.yml`** : Configuration centralisée
- **Utiliser `.semgrepignore`** : Exclusions par fichier
- **Suppressions inline** : Pour des cas spécifiques documentés

### 4. Rapports et monitoring

- **Toujours générer un rapport JSON** : Pour l'analyse et le suivi
- **Afficher un résumé** : Pour une visibilité immédiate
- **Archiver les rapports** : Pour suivre l'évolution

## Références

- [Documentation officielle Semgrep](https://semgrep.dev/docs/)
- [Règles disponibles](https://semgrep.dev/r)
- [Configuration avancée](https://semgrep.dev/docs/configuration-files/)
- [Gestion des suppressions](https://semgrep.dev/docs/ignoring-findings/)

## Questions fréquentes

### Q: Semgrep bloque mon CI, que faire ?

**R:** 
1. Vérifiez si c'est un vrai problème de sécurité
2. Si c'est un faux positif, ajoutez une suppression inline documentée
3. Si c'est un vrai problème, corrigez-le avant de merger

### Q: Dois-je ignorer les fichiers de test ?

**R:** Non recommandé. Les tests peuvent aussi contenir des vulnérabilités (mocks, fixtures, etc.). Il est préférable de scanner tous les fichiers.

### Q: Semgrep est-il suffisant pour la sécurité ?

**R:** Non. Semgrep est un outil parmi d'autres :
- ✅ Analyse statique (Semgrep, Bandit)
- ✅ Analyse de dépendances (Safety, Snyk)
- ✅ Revues de code manuelles
- ✅ Tests de sécurité (penetration testing)
- ✅ Audit de sécurité réguliers

### Q: Comment améliorer la précision de Semgrep ?

**R:**
1. Configurez des règles spécifiques à votre projet
2. Triagez et documentez les faux positifs
3. Réutilisez les patterns dans des règles personnalisées
4. Formez l'équipe sur la gestion des suppressions

