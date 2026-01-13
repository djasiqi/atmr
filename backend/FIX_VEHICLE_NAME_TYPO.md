# 🚗 Correction nom véhicule + Fix erreur VehiclesTab

**Date**: 2026-01-13  
**Version**: 1.0  
**Fichiers concernés**:

- `frontend/src/pages/company/Settings/tabs/VehiclesTab.jsx`

---

## 🎯 Problème 1 : Erreur JavaScript

**Erreur** :

```
TypeError: l.year.trim is not a function
    at onSubmit (VehiclesTab.jsx:285:42)
```

**Cause** :

- `formData.year` est un **nombre** (2024) et non une **chaîne**
- Le code appelait `.trim()` sur un nombre ❌

**Solution** :

- Convertir en string avec `String(formData.year)` avant d'appeler `.trim()`

---

## ✅ Correction JavaScript

**Fichier** : `frontend/src/pages/company/Settings/tabs/VehiclesTab.jsx`

**Avant** (ligne 285-293) :

```javascript
// Ajouter les champs optionnels seulement s'ils ont une valeur
if (formData.year && formData.year.trim()) {
  // ❌ Erreur si year est un nombre
  payload.year = parseInt(formData.year);
}
if (formData.vin && formData.vin.trim()) {
  payload.vin = formData.vin.trim();
}
if (formData.seats && formData.seats.trim()) {
  // ❌ Erreur si seats est un nombre
  payload.seats = parseInt(formData.seats);
}
```

**Après** :

```javascript
// Ajouter les champs optionnels seulement s'ils ont une valeur
if (formData.year && String(formData.year).trim()) {
  // ✅ Convertit en string d'abord
  payload.year = parseInt(formData.year);
}
if (formData.vin && String(formData.vin).trim()) {
  // ✅ Sécurisé aussi
  payload.vin = String(formData.vin).trim();
}
if (formData.seats && String(formData.seats).trim()) {
  // ✅ Convertit en string d'abord
  payload.seats = parseInt(formData.seats);
}
```

---

## 🎯 Problème 2 : Nom véhicule incorrect

**Véhicule** :

- Nom actuel : **"FPRD Tourneo Connect"** ❌ (typo)
- Nom correct : **"FORD Tourneo Connect"** ✅
- Plaque : GE963826
- Année : 2024
- Sièges : 5

---

## ✅ Correction nom véhicule

### Option 1 : Via l'interface web (recommandé)

Maintenant que le bug JavaScript est corrigé :

```bash
1. Se connecter au dashboard entreprise
2. Aller dans Paramètres → Véhicules
3. Cliquer sur le véhicule "FPRD Tourneo Connect"
4. Modifier le nom → "FORD Tourneo Connect"
5. Cliquer sur Enregistrer
```

---

### Option 2 : Via SQL (si accès direct à la base)

```sql
-- Vérifier d'abord le véhicule actuel
SELECT
    id,
    model,
    license_plate,
    year,
    seats,
    is_active
FROM vehicle
WHERE model LIKE '%FPRD%Tourneo%';

-- Mettre à jour le nom
UPDATE vehicle
SET model = 'FORD Tourneo Connect'
WHERE model = 'FPRD Tourneo Connect';

-- Vérifier la mise à jour
SELECT
    id,
    model,
    license_plate,
    year,
    seats,
    is_active
FROM vehicle
WHERE model LIKE '%FORD%Tourneo%';
```

---

### Option 3 : Via Docker (production)

```bash
# Se connecter au conteneur PostgreSQL
docker compose -f docker-compose.production.yml exec -it postgres psql -U atmr -d atmr

# Exécuter les commandes SQL
atmr=# UPDATE vehicle SET model = 'FORD Tourneo Connect' WHERE model = 'FPRD Tourneo Connect';
atmr=# SELECT id, model, license_plate FROM vehicle WHERE model LIKE '%FORD%Tourneo%';
atmr=# \q
```

---

## 🧪 Tests de validation

### Test 1 : Vérifier la correction JavaScript

```bash
1. Ouvrir le dashboard entreprise
2. Aller dans Paramètres → Véhicules
3. Cliquer sur "Ajouter un véhicule"
4. Remplir le formulaire avec :
   - Modèle : FORD Tourneo Connect
   - Plaque : TEST123
   - Année : 2024 (nombre)
   - Sièges : 5 (nombre)
5. Cliquer sur Enregistrer
6. Vérifier qu'il n'y a PAS d'erreur ✅
```

### Test 2 : Vérifier la correction du nom

```bash
1. Ouvrir le dashboard entreprise
2. Aller dans Paramètres → Véhicules
3. Chercher "FORD Tourneo Connect"
4. Vérifier que le nom est correct (pas "FPRD") ✅
```

---

## 📊 Détails techniques

### Pourquoi l'erreur se produisait ?

**Types JavaScript** :

- `formData.year` peut être de type `number` ou `string` selon la source
- Si c'est un nombre, `.trim()` n'existe pas
- `String()` convertit n'importe quel type en chaîne

**Exemple** :

```javascript
// Cas 1 : year est une string
formData.year = "2024";
formData.year.trim(); // ✅ Fonctionne

// Cas 2 : year est un nombre
formData.year = 2024;
formData.year.trim(); // ❌ Erreur: trim is not a function

// Solution : Convertir d'abord
formData.year = 2024;
String(formData.year).trim(); // ✅ "2024"
```

---

## 🔧 Déploiement

### Frontend

```bash
# Build et déployer le frontend
cd frontend
npm run build

# Ou si utilisez Vercel
vercel --prod
```

### Base de données (si correction SQL nécessaire)

```bash
# Se connecter au serveur
ssh deploy@138.201.155.201

# Aller dans le dossier atmr
cd /srv/atmr

# Exécuter la correction SQL
docker compose -f docker-compose.production.yml exec -T postgres psql -U atmr -d atmr << 'EOF'
UPDATE vehicle SET model = 'FORD Tourneo Connect' WHERE model = 'FPRD Tourneo Connect';
EOF

# Vérifier
docker compose -f docker-compose.production.yml exec -T postgres psql -U atmr -d atmr -c "SELECT id, model, license_plate FROM vehicle WHERE model LIKE '%FORD%Tourneo%';"
```

---

## 📌 Points importants

### ✅ Pourquoi `String()` et pas `toString()` ?

```javascript
// String() est plus sûr
String(null);       // "null"
String(undefined);  // "undefined"
String(2024);       // "2024"

// toString() peut échouer
null.toString();      // ❌ Erreur: Cannot read property 'toString' of null
undefined.toString(); // ❌ Erreur: Cannot read property 'toString' of undefined
2024.toString();      // ✅ "2024"
```

**Conclusion** : `String()` est plus robuste pour des données de formulaire.

---

## 🔮 Évolutions possibles

### 1. Validation des types en amont

```javascript
// Dans le composant, s'assurer que year et seats sont toujours des strings
const [formData, setFormData] = useState({
  model: "",
  license_plate: "",
  year: "", // Toujours string
  vin: "",
  seats: "", // Toujours string
  wheelchair_accessible: false,
  is_active: true,
});
```

### 2. Fonction de sanitization

```javascript
// Créer une fonction utilitaire
const sanitizeFormData = (data) => {
  return {
    ...data,
    year: data.year ? String(data.year).trim() : "",
    seats: data.seats ? String(data.seats).trim() : "",
    vin: data.vin ? String(data.vin).trim() : "",
  };
};

// Utiliser dans onSubmit
const payload = sanitizeFormData(formData);
```

---

**Version**: 1.0  
**Dernière mise à jour**: 2026-01-13  
**Auteur**: Assistant IA  
**Status**: ✅ Fix JavaScript implémenté, correction SQL documentée
