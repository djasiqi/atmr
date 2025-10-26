# ⚡ Guide Rapide d'Optimisation Cursor

## Ce qui a été fait

✅ Fichier `.cursorignore` créé  
✅ Paramètres optimisés prêts  
✅ Script d'application fourni

---

## 🚀 APPLICATION EN 2 MINUTES

### Option A : Automatique (Recommandé)

```powershell
# Dans PowerShell, depuis le dossier du projet
.\appliquer-parametres-cursor.ps1
```

### Option B : Manuel

1. **Ouvrez les paramètres Cursor** : `Ctrl + ,`

2. **Cliquez sur l'icône en haut à droite** (voir image ci-dessous)

3. **Copiez-collez le contenu** de `cursor-settings.json` dans `settings.json`

4. **Sauvegardez** : `Ctrl + S`

---

## 📋 APRES APPLICATION

### Étape 1 : Recharger Cursor

```
Ctrl + Shift + P
→ Tapez "Reload Window"
→ Entrée
```

### Étape 2 : Reconstruire l'Index

```
Ctrl + Shift + P
→ Tapez "Rebuild Index"
→ Entrée
```

### Étape 3 : Vérifier

Cliquez sur l'icône en bas à gauche → Voir "Codebase indexed" en vert

---

## 🎯 RÉSULTATS ATTENDUS

### Performance Avant

- Indexation : 5-10 minutes
- Recherche : 2-5 secondes
- Autocomplétion : 1-3 secondes

### Performance Après

- Indexation : 1-2 minutes
- Recherche : < 1 seconde
- Autocomplétion : instantané

---

## ❓ PROBLÈMES ?

### Cursor lent après configuration

```powershell
# Windows
Ctrl + Shift + P → "Clear Cursor Cache"
```

### Indexation toujours lente

1. Vérifiez que `.cursorignore` est dans la racine du projet
2. Redémarrez Cursor complètement
3. Réinstallez si nécessaire

### Autocomplétion ne fonctionne pas

1. `Ctrl + Shift + P` → "Python: Select Interpreter"
2. Choisissez votre venv Python
3. Rechargez la fenêtre

---

## 📚 DOCUMENTATION COMPLÈTE

Pour les détails techniques, consultez : `CURSOR_OPTIMISATION.md`

---

## ✅ CHECKLIST FINALE

- [ ] Script PowerShell exécuté ou paramètres collés manuellement
- [ ] Cursor rechargé (Reload Window)
- [ ] Index reconstruit (Rebuild Index)
- [ ] "Codebase indexed" visible en vert
- [ ] Performances améliorées testées

---

**🎉 C'est tout ! Votre Cursor est maintenant optimisé !**
