# 📚 Exemples Pratiques d'Utilisation

Ce fichier contient des exemples concrets d'utilisation du système de design.

## 🎯 Exemple 1 : Page de Dashboard Simple

```jsx
import React from 'react';
import CompanyHeader from '../components/layout/Header/CompanyHeader';
import CompanySidebar from '../components/layout/Sidebar/CompanySidebar';

function SimpleDashboard() {
  return (
    <div className="min-h-screen bg-secondary">
      <CompanyHeader />

      <div className="flex">
        <CompanySidebar />

        <main className="flex-1 p-lg m-lg">
          {/* Header de page */}
          <div className="bg-gradient-brand rounded-xl p-lg mb-lg shadow-brand">
            <h1 className="text-3xl font-bold text-white mb-sm">Tableau de bord</h1>
            <p className="text-white opacity-75">Vue d'ensemble de votre activité</p>
          </div>

          {/* Stats Cards */}
          <div className="grid grid-cols-3 gap-lg mb-xl">
            <StatsCard title="Courses aujourd'hui" value="24" trend="+12%" type="success" />
            <StatsCard title="Revenus du mois" value="15,430 CHF" trend="+8%" type="success" />
            <StatsCard title="Chauffeurs actifs" value="12" trend="-1" type="warning" />
          </div>

          {/* Tableau récent */}
          <div className="card-container">
            <div className="card-header">
              <h2 className="card-title">Courses récentes</h2>
            </div>
            <div className="card-body p-0">
              <RecentBookingsTable />
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}

// Composant de carte statistique
function StatsCard({ title, value, trend, type }) {
  const trendColor =
    type === 'success' ? 'text-success' : type === 'warning' ? 'text-warning' : 'text-error';

  return (
    <div className="card card-hover">
      <div className="flex-between mb-md">
        <p className="text-sm text-secondary font-medium">{title}</p>
        <span className={`badge badge-${type}`}>{trend}</span>
      </div>
      <p className="text-3xl font-bold text-primary">{value}</p>
    </div>
  );
}
```

**Styles utilisés :**

- ✅ Layout : `flex`, `grid-cols-3`, `gap-lg`
- ✅ Spacing : `p-lg`, `m-lg`, `mb-xl`
- ✅ Composants : `card`, `badge`
- ✅ Couleurs : `bg-gradient-brand`, `text-success`
- ✅ **Aucun CSS Module nécessaire !**

---

## 🎯 Exemple 2 : Formulaire de Création

```jsx
import React, { useState } from 'react';

function CreateBookingForm({ onSubmit, onCancel }) {
  const [formData, setFormData] = useState({
    clientName: '',
    pickupAddress: '',
    dropoffAddress: '',
    date: '',
    time: '',
    serviceType: 'standard',
    notes: '',
  });

  const [errors, setErrors] = useState({});

  const handleSubmit = (e) => {
    e.preventDefault();
    // Validation...
    onSubmit(formData);
  };

  return (
    <div className="modal-overlay">
      <div className="modal-content modal-lg">
        {/* Header */}
        <div className="modal-header">
          <h2 className="modal-title">Nouvelle course</h2>
          <button className="modal-close" onClick={onCancel}>
            ×
          </button>
        </div>

        {/* Body avec formulaire */}
        <form onSubmit={handleSubmit}>
          <div className="modal-body">
            {/* Alerte si erreurs */}
            {Object.keys(errors).length > 0 && (
              <div className="alert alert-error mb-lg">
                <div className="alert-title">Erreurs de validation</div>
                <div>Veuillez corriger les champs en rouge</div>
              </div>
            )}

            {/* Champs du formulaire */}
            <div className="grid grid-cols-2 gap-md">
              {/* Nom du client */}
              <div className="form-group">
                <label className="form-label required">Nom du client</label>
                <input
                  type="text"
                  className={`form-input ${errors.clientName ? 'error' : ''}`}
                  value={formData.clientName}
                  onChange={(e) => setFormData({ ...formData, clientName: e.target.value })}
                  placeholder="Jean Dupont"
                />
                {errors.clientName && <span className="form-error">{errors.clientName}</span>}
              </div>

              {/* Type de service */}
              <div className="form-group">
                <label className="form-label">Type de service</label>
                <select
                  className="form-select"
                  value={formData.serviceType}
                  onChange={(e) => setFormData({ ...formData, serviceType: e.target.value })}
                >
                  <option value="standard">Standard</option>
                  <option value="medical">Médical</option>
                  <option value="emergency">Urgence</option>
                </select>
                <span className="form-hint">Les urgences sont prioritaires</span>
              </div>
            </div>

            {/* Adresses */}
            <div className="form-group">
              <label className="form-label required">Adresse de départ</label>
              <input
                type="text"
                className="form-input"
                value={formData.pickupAddress}
                onChange={(e) => setFormData({ ...formData, pickupAddress: e.target.value })}
                placeholder="Rue de la Gare 10, 1003 Lausanne"
              />
            </div>

            <div className="form-group">
              <label className="form-label required">Adresse d'arrivée</label>
              <input
                type="text"
                className="form-input"
                value={formData.dropoffAddress}
                onChange={(e) => setFormData({ ...formData, dropoffAddress: e.target.value })}
                placeholder="Avenue du Théâtre 1, 1005 Lausanne"
              />
            </div>

            {/* Date et heure */}
            <div className="grid grid-cols-2 gap-md">
              <div className="form-group">
                <label className="form-label required">Date</label>
                <input
                  type="date"
                  className="form-input"
                  value={formData.date}
                  onChange={(e) => setFormData({ ...formData, date: e.target.value })}
                />
              </div>

              <div className="form-group">
                <label className="form-label required">Heure</label>
                <input
                  type="time"
                  className="form-input"
                  value={formData.time}
                  onChange={(e) => setFormData({ ...formData, time: e.target.value })}
                />
              </div>
            </div>

            {/* Notes */}
            <div className="form-group">
              <label className="form-label">Notes supplémentaires</label>
              <textarea
                className="form-textarea"
                value={formData.notes}
                onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
                placeholder="Informations complémentaires..."
                rows={4}
              />
            </div>
          </div>

          {/* Footer avec boutons */}
          <div className="modal-footer">
            <button type="button" className="btn btn-secondary" onClick={onCancel}>
              Annuler
            </button>
            <button type="submit" className="btn btn-primary">
              Créer la course
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
```

**Styles utilisés :**

- ✅ Modal : `modal-overlay`, `modal-content`, `modal-header`
- ✅ Formulaire : `form-group`, `form-input`, `form-error`
- ✅ Alerte : `alert`, `alert-error`
- ✅ Grid : `grid-cols-2`, `gap-md`
- ✅ **Aucun CSS Module nécessaire !**

---

## 🎯 Exemple 3 : Tableau avec Actions

```jsx
function DriversTable({ drivers, onEdit, onDelete }) {
  return (
    <div className="card-container">
      <div className="card-header">
        <div className="flex-between">
          <h2 className="card-title">Chauffeurs</h2>
          <button className="btn btn-primary btn-sm">+ Ajouter un chauffeur</button>
        </div>
      </div>

      <div className="table-container">
        <table className="table">
          <thead>
            <tr>
              <th>Nom</th>
              <th>Email</th>
              <th>Téléphone</th>
              <th>Statut</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {drivers.map((driver) => (
              <tr key={driver.id}>
                <td>
                  <div className="flex gap-sm">
                    <div className="rounded-full bg-brand text-white w-10 h-10 flex-center">
                      {driver.initials}
                    </div>
                    <div>
                      <div className="font-medium">{driver.name}</div>
                      <div className="text-sm text-tertiary">ID: {driver.id}</div>
                    </div>
                  </div>
                </td>
                <td>{driver.email}</td>
                <td>{driver.phone}</td>
                <td>
                  <span className={`badge ${driver.active ? 'badge-success' : 'badge-error'}`}>
                    {driver.active ? 'Actif' : 'Inactif'}
                  </span>
                </td>
                <td>
                  <div className="flex gap-xs">
                    <button className="btn btn-secondary btn-sm" onClick={() => onEdit(driver)}>
                      ✏️ Éditer
                    </button>
                    <button className="btn btn-danger btn-sm" onClick={() => onDelete(driver)}>
                      🗑️ Supprimer
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      <div className="card-footer">
        <div className="pagination">
          <button className="pagination-item disabled">← Précédent</button>
          <button className="pagination-item active">1</button>
          <button className="pagination-item">2</button>
          <button className="pagination-item">3</button>
          <button className="pagination-item">Suivant →</button>
        </div>
      </div>
    </div>
  );
}
```

**Styles utilisés :**

- ✅ Card : `card-container`, `card-header`, `card-footer`
- ✅ Table : `table-container`, `table`
- ✅ Badge : `badge-success`, `badge-error`
- ✅ Pagination : `pagination`, `pagination-item`
- ✅ **Aucun CSS Module nécessaire !**

---

## 🎯 Exemple 4 : Loading States

```jsx
function DataLoader({ loading, error, data, children }) {
  if (loading) {
    return (
      <div className="loading-overlay">
        <div className="flex-col flex-center gap-md">
          <div className="spinner spinner-lg"></div>
          <p className="text-tertiary">Chargement en cours...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="alert alert-error">
        <div className="alert-title">❌ Erreur de chargement</div>
        <div>{error.message}</div>
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <div className="card text-center py-2xl">
        <div className="text-6xl mb-lg">📭</div>
        <h3 className="text-xl font-semibold mb-sm">Aucune donnée</h3>
        <p className="text-tertiary">Il n'y a rien à afficher pour le moment</p>
      </div>
    );
  }

  return children(data);
}

// Utilisation
function MyComponent() {
  const { loading, error, data } = useFetchData();

  return (
    <DataLoader loading={loading} error={error} data={data}>
      {(data) => (
        <div className="grid grid-cols-3 gap-lg">
          {data.map((item) => (
            <ItemCard key={item.id} item={item} />
          ))}
        </div>
      )}
    </DataLoader>
  );
}
```

---

## 🎯 Exemple 5 : Composant avec Styles Spécifiques

Parfois, vous avez **vraiment** besoin d'un CSS Module pour des styles complexes :

```jsx
// SpecialMap.jsx
import styles from './SpecialMap.module.css';

function SpecialMap({ markers, routes }) {
  return (
    <div className="card-container">
      <div className="card-header">
        <h2 className="card-title">Carte en temps réel</h2>
      </div>

      {/* Utilise le CSS Module pour le layout complexe de la carte */}
      <div className={styles.mapContainer}>
        <div className={styles.mapCanvas} id="map"></div>

        <div className={styles.mapControls}>
          <button className="btn btn-secondary btn-sm">🔍 Zoom +</button>
          <button className="btn btn-secondary btn-sm">🔍 Zoom -</button>
        </div>

        <div className={styles.mapLegend}>
          <div className="flex gap-sm">
            <span className="badge badge-success">Actif</span>
            <span className="badge badge-warning">En pause</span>
            <span className="badge badge-error">Hors ligne</span>
          </div>
        </div>
      </div>
    </div>
  );
}
```

```css
/* SpecialMap.module.css */

/* ✅ Styles vraiment spécifiques à la carte */
.mapContainer {
  position: relative;
  width: 100%;
  height: 600px;
  background: var(--bg-tertiary);
}

.mapCanvas {
  width: 100%;
  height: 100%;
  border-radius: var(--radius-lg);
}

.mapControls {
  position: absolute;
  top: var(--spacing-md);
  right: var(--spacing-md);
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
}

.mapLegend {
  position: absolute;
  bottom: var(--spacing-md);
  left: var(--spacing-md);
  padding: var(--spacing-md);
  background: var(--bg-primary);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-lg);
}
```

**Points importants :**

- ✅ Utilise card-container pour l'enveloppe
- ✅ Utilise btn pour les boutons de contrôle
- ✅ Utilise badge pour la légende
- ✅ CSS Module **uniquement** pour le layout complexe de la carte
- ✅ Même dans le CSS Module, utilise les **variables globales** (`var(--spacing-md)`)

---

## 💡 Règles d'Or

1. **Essayez d'abord les classes communes** avant de créer un CSS Module
2. **Combinez les classes** : `<div className="flex-between gap-md p-lg shadow-sm">`
3. **Utilisez les variables CSS** même dans les CSS Modules
4. **Créez un CSS Module** seulement si vraiment nécessaire
5. **Restez cohérent** : Si un composant similaire existe, utilisez le même pattern

---

**Bon développement ! 🚀**
