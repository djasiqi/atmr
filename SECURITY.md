# 🔐 Security Policy

## 📋 Versions Supportées

Nous fournissons des mises à jour de sécurité pour les versions suivantes :

| Version  | Supportée          |
| -------- | ------------------ |
| `main`   | :white_check_mark: |
| `latest` | :white_check_mark: |

**Note** : Les versions de branches de développement peuvent ne pas recevoir de mises à jour de sécurité. Nous recommandons d'utiliser la branche `main` en production.

---

## 🚨 Signaler une Vulnérabilité

Nous prenons la sécurité de notre système ATMR très au sérieux. Si vous découvrez une vulnérabilité de sécurité, nous apprécions votre aide pour la divulguer de manière responsable.

### 📧 Comment Signaler

**NE PAS** créer une issue publique sur GitHub pour les vulnérabilités de sécurité.

Veuillez signaler les vulnérabilités de sécurité par email à :

**Email de sécurité** : `info@lirie.ch` (avec le sujet `[SECURITY]`)

### 📝 Informations à Inclure

Pour nous aider à comprendre et reproduire la vulnérabilité, veuillez inclure :

1. **Description détaillée** de la vulnérabilité
2. **Étapes pour reproduire** (avec exemples de code si possible)
3. **Impact potentiel** (confidentialité, intégrité, disponibilité)
4. **Suggestions de correctif** (si vous en avez)
5. **Version affectée** (si applicable)
6. **Preuve de concept** (si disponible, mais de manière sécurisée)

### ⏱️ Processus de Réponse

1. **Accusé de réception** : Nous confirmerons la réception de votre rapport dans les **48 heures**
2. **Évaluation** : Nous évaluerons la vulnérabilité dans les **7 jours**
3. **Mise à jour** : Nous vous tiendrons informé de l'avancement tous les **7 jours** jusqu'à résolution
4. **Correction** : Nous travaillerons sur un correctif et vous informerons avant la publication
5. **Divulgation** : Après correction, nous publierons un avis de sécurité (avec votre accord)

### ✅ Ce qui est Couvert

- Vulnérabilités d'authentification et d'autorisation
- Injections (SQL, XSS, commande, etc.)
- Expositions de données sensibles
- Problèmes de configuration de sécurité
- Vulnérabilités dans les dépendances critiques
- Problèmes de chiffrement ou de gestion des secrets

### ❌ Ce qui n'est PAS Couvert

- Problèmes de disponibilité (DoS/DDoS) - à moins qu'ils ne soient critiques
- Vulnérabilités nécessitant un accès physique
- Vulnérabilités dans des services tiers non maintenus par nous
- Problèmes de sécurité nécessitant des privilèges administrateur locaux
- Problèmes de sécurité dans des versions non supportées

### 🎁 Reconnaissance

Nous reconnaissons volontiers les chercheurs en sécurité qui nous aident à améliorer la sécurité de notre système. Si vous souhaitez être reconnu publiquement, indiquez-le dans votre rapport.

---

## 🔒 Bonnes Pratiques de Sécurité

### Pour les Contributeurs

- Ne jamais commiter de secrets, tokens, ou mots de passe
- Utiliser des variables d'environnement pour les configurations sensibles
- Vérifier les dépendances avec `npm audit` et `pip-audit`
- Suivre les principes de sécurité par défaut (secure by default)

### Pour les Utilisateurs

- Maintenir vos dépendances à jour
- Utiliser HTTPS en production
- Configurer correctement les variables d'environnement
- Ne pas exposer les services internes publiquement
- Utiliser des mots de passe forts pour les comptes administrateur

---

## 📚 Ressources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE Top 25](https://cwe.mitre.org/top25/)
- [GitHub Security Advisories](https://github.com/advisories)

---

## 📅 Historique des Avis de Sécurité

Les avis de sécurité publiés seront listés ici après leur divulgation.

---

**Dernière mise à jour** : 2025-01-XX

**Contact** : info@lirie.ch
