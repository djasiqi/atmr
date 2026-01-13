# 🔍 Débogage : Modal Client ne s'ouvre pas

## 🎯 Problème

Le bouton "Créer un nouveau client" s'affiche mais le modal ne s'ouvre pas au clic.

## ✅ Vérifications

### 1. Le bouton s'affiche-t-il ?

- ✅ OUI : Le bouton apparaît quand aucun client n'est trouvé
- Condition : recherche >= 2 caractères ET aucun résultat

### 2. Chaîne d'appel correcte

```typescript
// ClientSelector.tsx (ligne 129)
onPress={onNewClient}
    ↓
// RideCreateModal.tsx (ligne 313-315)
onNewClient={() => {
    if (onOpenClientCreate) {
        onOpenClientCreate();
    }
}}
    ↓
// rides.tsx (ligne 621)
onOpenClientCreate={() => setClientCreateModalVisible(true)}
    ↓
// rides.tsx (ligne 630)
visible={clientCreateModalVisible}
```

## 🔧 Solutions possibles

### Solution 1 : Clear Metro Cache

```bash
cd mobile/operations-app

# Arrêter Metro Bundler
# Puis:
npm start -- --reset-cache
# ou
npx expo start --clear
```

### Solution 2 : Rebuild l'application

```bash
# iOS
eas build --platform ios --profile development --local

# Android
eas build --platform android --profile development --local
```

### Solution 3 : Vérifier les logs

Quand vous cliquez sur "Créer un nouveau client", ouvrez la console et cherchez:

```
[ClientSelector] onNewClient called
[RideCreateModal] onOpenClientCreate called
[rides.tsx] setClientCreateModalVisible(true)
```

## 🐛 Ajout de logs de debug

### Dans `ClientSelector.tsx`

```typescript
onNewClient={() => {
    console.log('[ClientSelector] onNewClient called');
    onNewClient?.();
}}
```

### Dans `RideCreateModal.tsx`

```typescript
onNewClient={() => {
    console.log('[RideCreateModal] onNewClient triggered');
    if (onOpenClientCreate) {
        console.log('[RideCreateModal] Calling onOpenClientCreate');
        onOpenClientCreate();
    } else {
        console.warn('[RideCreateModal] onOpenClientCreate is undefined!');
    }
}}
```

### Dans `rides.tsx`

```typescript
onOpenClientCreate={() => {
    console.log('[rides.tsx] Opening client create modal');
    setClientCreateModalVisible(true);
}}
```

## 📱 Test rapide

Ajoutez temporairement un bouton de test dans `rides.tsx` :

```typescript
<TouchableOpacity
  onPress={() => {
    console.log("[TEST] Forcing modal open");
    setClientCreateModalVisible(true);
  }}
  style={{ padding: 20, backgroundColor: "red" }}
>
  <Text style={{ color: "white" }}>TEST: Ouvrir Modal Client</Text>
</TouchableOpacity>
```

Si ce bouton ouvre le modal → Le problème est dans la chaîne d'appel  
Si ce bouton n'ouvre pas le modal → Le problème est dans le Modal lui-même

## 🔍 Vérification du state

Ajoutez un `useEffect` dans `rides.tsx` :

```typescript
useEffect(() => {
  console.log(
    "[rides.tsx] clientCreateModalVisible changed:",
    clientCreateModalVisible
  );
}, [clientCreateModalVisible]);
```

## ⚡ Fix immédiat (workaround)

Si rien ne fonctionne, essayez de forcer le re-render :

```typescript
const [modalKey, setModalKey] = useState(0);

<ClientCreateModal
    key={modalKey}  // Force re-render
    visible={clientCreateModalVisible}
    onClose={() => {
        setClientCreateModalVisible(false);
        setModalKey(k => k + 1);  // Increment key
    }}
    onSuccess={...}
/>
```
