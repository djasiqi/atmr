# 🧪 Guide de Test - Nouvelle Page Settings

**Date :** 14 octobre 2025

---

## 🎯 Objectif

Tester la nouvelle page Settings avec ses 5 onglets et toutes les nouvelles fonctionnalités.

---

## 🚀 Préparation

### 1. Vérifier que le backend est redémarré

```bash
docker compose ps
```

✅ `atmr-api-1` doit être "Up (healthy)"

---

## 📋 Tests par Onglet

### 🏢 Onglet 1 : Général

**Navigation :**

1. Allez sur : `http://localhost:3000/dashboard/company/{public_id}/settings`
2. Vous devriez voir :
   - ✅ Header gradient teal "⚙️ Paramètres de l'entreprise"
   - ✅ 5 onglets : 🏢 🚗 💰 📧 🔐
   - ✅ Onglet "Général" actif (background teal)

**Tests :**

| Action                   | Résultat Attendu               |
| ------------------------ | ------------------------------ |
| Voir le logo             | Logo 160×160 avec hover effect |
| Cliquer "✏️ Modifier"    | Mode édition activé            |
| Modifier le nom          | Input éditable                 |
| Cliquer "Annuler"        | Retour aux valeurs initiales   |
| Cliquer "💾 Enregistrer" | Message de succès              |
| Upload logo (fichier)    | Prévisualisation + upload      |
| Supprimer logo           | Confirmation + suppression     |

---

### 🚗 Onglet 2 : Opérations

**Navigation :**

1. Cliquer sur l'onglet "🚗 Opérations"
2. Animation fade-in
3. Nouvelle page se charge

**Tests :**

| Action                | Résultat Attendu                                  |
| --------------------- | ------------------------------------------------- |
| Voir zone de service  | Champ avec placeholder "Genève, Vaud, Valais"     |
| Voir limite courses   | Nombre (défaut: 50)                               |
| Toggle dispatch auto  | Switch moderne animé                              |
| Activer dispatch      | Switch devient vert (gradient teal)               |
| Cliquer "📍 Détecter" | Demande permission GPS + auto-remplissage         |
| Sauvegarder           | Message "✅ Paramètres opérationnels enregistrés" |

**Vérification API :**

```bash
# Dans la console DevTools
fetch('/api/company-settings/operational', {
  headers: { 'Authorization': 'Bearer ' + localStorage.getItem('token') }
})
.then(r => r.json())
.then(console.log)
```

---

### 💰 Onglet 3 : Facturation

**Navigation :**

1. Cliquer sur l'onglet "💰 Facturation"

**Tests :**

| Action                  | Résultat Attendu                                   |
| ----------------------- | -------------------------------------------------- |
| Voir délai paiement     | Input numérique (défaut: 10 jours)                 |
| Voir frais retard       | Input (défaut: 15 CHF)                             |
| Toggle rappels auto     | Switch moderne                                     |
| Activer rappels         | Affiche 3 sections (1er, 2e, 3e rappel)            |
| Modifier 1er rappel     | Délai + frais configurables                        |
| Changer préfixe         | Input "EM" → voir preview mise à jour              |
| Changer format          | Dropdown → preview change automatiquement          |
| Modifier template email | Textarea avec variables                            |
| Sauvegarder             | Message "✅ Paramètres de facturation enregistrés" |

**Preview Numéro :**

- Préfixe "EM" + Format "{PREFIX}-{YYYY}-{MM}-{SEQ4}"
- Résultat : `EM-2025-10-0001`

---

### 📧 Onglet 4 : Notifications

**Navigation :**

1. Cliquer sur l'onglet "📧 Notifications"

**Tests :**

| Action               | Résultat Attendu                                                       |
| -------------------- | ---------------------------------------------------------------------- |
| Voir 6 toggles       | Nouvelle réservation, confirmée, annulée, dispatch, retards, analytics |
| Toggle notification  | Switch moderne animé                                                   |
| Activer notification | Switch gradient teal                                                   |
| Ajouter emails       | Input avec placeholder                                                 |
| Sauvegarder          | Message "✅ Paramètres de notifications enregistrés"                   |

---

### 🔐 Onglet 5 : Sécurité

**Navigation :**

1. Cliquer sur l'onglet "🔐 Sécurité"

**Tests :**

| Action                | Résultat Attendu                         |
| --------------------- | ---------------------------------------- |
| Voir infos compte     | Dernière connexion, IP, sessions         |
| Voir logs activité    | Tableau avec 3 logs factices             |
| Hover sur ligne       | Background change                        |
| Cliquer "📥 Exporter" | Alert "en cours de développement" (TODO) |
| Voir infos système    | Version API, environnement, DB           |

---

## 📱 Tests Responsive

### Desktop (1920px)

- [ ] Onglets : tous visibles avec labels
- [ ] Formulaires : 2 colonnes
- [ ] Logo : 160×160px

### Tablet (768px)

- [ ] Onglets : scroll horizontal si nécessaire
- [ ] Formulaires : 1 colonne
- [ ] Logo : 140×140px

### Mobile (375px)

- [ ] Onglets : icônes uniquement (🏢 🚗 💰 📧 🔐)
- [ ] Formulaires : 1 colonne
- [ ] Boutons : pleine largeur
- [ ] Logo : 120×120px

---

## 🔍 Vérifications API

### Test 1 : Operational Settings

```bash
# Dans PowerShell ou terminal
docker compose exec api python -c "
from app import create_app
from models import Company
app = create_app()
with app.app_context():
    c = Company.query.first()
    print(f'Zone: {c.service_area}')
    print(f'Max: {c.max_daily_bookings}')
    print(f'Dispatch: {c.dispatch_enabled}')
    print(f'GPS: {c.latitude}, {c.longitude}')
"
```

### Test 2 : Billing Settings

```bash
docker compose exec api python -c "
from app import create_app
from models import CompanyBillingSettings
app = create_app()
with app.app_context():
    b = CompanyBillingSettings.query.first()
    if b:
        print(f'Payment terms: {b.payment_terms_days} days')
        print(f'Prefix: {b.invoice_prefix}')
        print(f'Auto reminders: {b.auto_reminders_enabled}')
    else:
        print('No billing settings yet')
"
```

---

## ✅ Checklist Complète

### Navigation

- [ ] Header gradient teal s'affiche
- [ ] Subtitle visible
- [ ] 5 onglets visibles
- [ ] Clic sur chaque onglet fonctionne
- [ ] Animation fade-in lors du changement

### Onglet Général

- [ ] Logo preview 160×160
- [ ] Upload fichier fonctionne
- [ ] Upload URL fonctionne
- [ ] Suppression fonctionne
- [ ] Mode édition/lecture
- [ ] Validation formulaire
- [ ] Sauvegarde fonctionne

### Onglet Opérations

- [ ] Champs affichés correctement
- [ ] Toggle dispatch animé
- [ ] Détection GPS fonctionne
- [ ] Sauvegarde API OK

### Onglet Facturation

- [ ] Tous les champs affichés
- [ ] Toggle rappels fonctionne
- [ ] 3 sections rappels visibles quand activé
- [ ] Preview numéro facture correct
- [ ] Templates emails éditables
- [ ] Sauvegarde API OK

### Onglet Notifications

- [ ] 6 toggles affichés
- [ ] Switches animés
- [ ] Email destinataires éditable
- [ ] Sauvegarde (temporaire)

### Onglet Sécurité

- [ ] Infos compte affichées
- [ ] Tableau logs affiché
- [ ] Hover sur lignes
- [ ] Export button visible

### Design

- [ ] Cohérent avec Analytics/Dispatch
- [ ] Hover effects sur sections
- [ ] Boutons avec gradients
- [ ] Focus teal sur inputs
- [ ] Responsive (tester 3 tailles)

---

## 🐛 Si Problème...

### "Erreur 404 sur /api/company-settings"

→ Redémarrez l'API : `docker compose restart api`

### "Cannot find module TabNavigation"

→ Vérifiez que les fichiers sont bien créés dans `components/ui/`

### "Onglets ne changent pas"

→ Vérifiez la console pour des erreurs JS

### "API 422 Unprocessable"

→ Vérifiez le token JWT dans localStorage

---

## 🎉 Résultat Attendu

Après tous les tests, vous devriez avoir :

✅ Une page Settings **magnifique et professionnelle**  
✅ **5 onglets** parfaitement fonctionnels  
✅ **Tous les paramètres** de l'entreprise configurables  
✅ Un design **100% cohérent** avec le reste de l'app  
✅ Une UX **de classe mondiale** 🌟

---

**Bonne découverte de votre nouvelle page Settings ! 🛠️✨**
