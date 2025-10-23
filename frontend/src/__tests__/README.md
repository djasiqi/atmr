# 🧪 Tests Frontend ATMR

Ce dossier contient tous les tests frontend du projet ATMR.

## 📁 Structure

```
tests/
├── services/           # Tests des services API
│   ├── authService.test.js
│   ├── bookingService.test.js
│   ├── companyService.test.js
│   ├── driverService.test.js
│   └── clientService.test.js
├── hooks/             # Tests des hooks React
│   ├── useAuthToken.test.js
│   ├── useCompanyData.test.js
│   ├── useDispatchDelays.test.js
│   └── useDriver.test.js
├── components/        # Tests des composants React
│   ├── ManualBookingForm.test.jsx
│   ├── CompanyDashboard.test.jsx
│   ├── UnifiedDispatch.test.jsx
│   ├── ClientDashboard.test.jsx
│   ├── ReservationsPage.test.jsx
│   └── AddressAutocomplete.test.jsx
├── setupTests.js      # Configuration Jest globale
└── README.md          # Ce fichier
```

## 🚀 Exécution des tests

### Mode développement (watch)

```bash
cd frontend
npm test
```

### Run once avec coverage

```bash
npm run test:coverage
```

### CI (non-interactif)

```bash
npm run test:ci
```

### Tests par catégorie

```bash
# Services seulement
npm test -- tests/services

# Hooks seulement
npm test -- tests/hooks

# Composants seulement
npm test -- tests/components
```

## 📊 Coverage

Les rapports de coverage sont générés dans `frontend/coverage/`.

**Objectifs** :

- Services : ≥80% ✅
- Hooks : ≥80% ✅
- Composants critiques : ≥60% 🔄

**Atteints** :

- Services : **~80%** ✅
- Hooks : **~88%** ✅
- Composants : **~45%** (en progression)

## 🎯 Tests créés

### Services (5 fichiers - 51 tests)

- ✅ `authService.test.js` - 10 tests (login, register, logout, resetPassword)
- ✅ `bookingService.test.js` - 7 tests (fetch, cancel, export PDF)
- ✅ `companyService.test.js` - 17 tests (réservations, chauffeurs, dispatch)
- ✅ `driverService.test.js` - 11 tests (profil, localisation, bookings)
- ✅ `clientService.test.js` - 6 tests (profil client)

### Hooks (4 fichiers - 36 tests)

- ✅ `useAuthToken.test.js` - 8 tests (décodage JWT, expiration, rôles)
- ✅ `useCompanyData.test.js` - 8 tests (données entreprise, reload)
- ✅ `useDispatchDelays.test.js` - 10 tests (retards, monitoring, auto-refresh)
- ✅ `useDriver.test.js` - 10 tests (CRUD chauffeurs, état optimiste)

### Composants (6 fichiers - 35 tests)

- ✅ `ManualBookingForm.test.jsx` - 8 tests (formulaire réservation)
- ✅ `CompanyDashboard.test.jsx` - 6 tests (dashboard entreprise)
- ✅ `UnifiedDispatch.test.jsx` - 5 tests (dispatch automatique)
- ✅ `ClientDashboard.test.jsx` - 5 tests (dashboard client)
- ✅ `ReservationsPage.test.jsx` - 10 tests (liste, filtres, annulation)
- ✅ `AddressAutocomplete.test.jsx` - 7 tests (recherche, navigation clavier)

**Total** : **122 tests** couvrant les fonctionnalités critiques

## 🛠️ Configuration

### setupTests.js

Configuration globale Jest :

- Import `@testing-library/jest-dom`
- Mock `window.matchMedia` (Material-UI)
- Mock `localStorage`
- Suppress console warnings

### Dépendances de test

```json
{
  "@testing-library/react": "^16.3.0",
  "@testing-library/jest-dom": "^6.6.4",
  "@testing-library/user-event": "^14.6.1",
  "msw": "^2.11.5"
}
```

### Mocks Courants

Les composants/services complexes sont mockés :

- `AddressAutocomplete`, `EstablishmentSelect`, `ServiceSelect`
- `react-select/async-creatable`
- `react-leaflet` (cartes)
- `apiClient`
- Layout components (Sidebar, Header, Footer)

## 📝 Convention de nommage

- Tests services : `*.test.js`
- Tests hooks : `*.test.js`
- Tests composants : `*.test.jsx`
- Fichiers de tests miroir de la structure `src/`

## 🎨 Best Practices

### Testing Library Rules

✅ Pas de multiple assertions dans `waitFor`
✅ Pas de side effects dans `waitFor`
✅ Utiliser `findBy` pour attendre les éléments
✅ Utiliser `userEvent` pour interactions réalistes

### Hooks Testing

✅ Utiliser `renderHook` de `@testing-library/react`
✅ Wrapper avec `act()` pour actions async
✅ Mock fake timers pour intervals
✅ Tester état optimiste

### Mocks

✅ Mock au niveau module avec `jest.mock()`
✅ Mock localStorage avant chaque test
✅ Clear mocks dans `beforeEach()`
✅ Mock console.error pour tests propres

## 📊 Statistiques

### Par Phase

| Phase     | Catégorie         | Tests   | Temps     | Coverage |
| --------- | ----------------- | ------- | --------- | -------- |
| Semaine 2 | Foundation        | 38      | 7h        | ~40%     |
| Phase 1   | Services          | 34      | 4h        | ~80%     |
| Phase 2   | Composants Client | 22      | 3h30      | ~55%     |
| Phase 3   | Hooks Business    | 28      | 3h        | ~88%     |
| **Total** | **4 phases**      | **122** | **17h30** | **~70%** |

### Par Catégorie

- **Services** : 51 tests (~80% coverage) ✅
- **Hooks** : 36 tests (~88% coverage) ✅
- **Composants** : 35 tests (~45% coverage) 🔄

## 🔗 Liens utiles

- [Jest Documentation](https://jestjs.io/)
- [React Testing Library](https://testing-library.com/react)
- [Jest-DOM Matchers](https://github.com/testing-library/jest-dom)
- [Testing Library Best Practices](https://kentcdodds.com/blog/common-mistakes-with-react-testing-library)

## 🎯 Roadmap Tests

### ✅ Complété

- [x] Foundation (Semaine 2)
- [x] Phase 1 - Services Critiques
- [x] Phase 2 - Composants Client
- [x] Phase 3 - Hooks Business

### 🔄 En cours / À venir

- [ ] Phase 4 - Utils & Helpers
- [ ] Composants UI réutilisables
- [ ] Tests E2E Cypress (Semaine 3)

---

**Date de création** : 16 octobre 2025  
**Dernière mise à jour** : 16 octobre 2025  
**Version** : 3.0  
**Tests** : 122  
**Coverage Moyen** : ~70%
