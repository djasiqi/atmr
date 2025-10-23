import React, { useState, useEffect } from 'react';
import styles from './Filters.module.css';
import { fetchCompanyClients } from '../../../../../services/companyService';

const Filters = ({ filters, onFilterChange, companyId }) => {
  const [clients, setClients] = useState([]);
  const [loadingClients, setLoadingClients] = useState(false);

  // Charger la liste des clients
  useEffect(() => {
    const loadClients = async () => {
      if (!companyId) return;

      try {
        setLoadingClients(true);
        const clientsData = await fetchCompanyClients();
        setClients(clientsData);
      } catch (error) {
        console.error('Erreur lors du chargement des clients:', error);
      } finally {
        setLoadingClients(false);
      }
    };

    loadClients();
  }, [companyId]);

  const handleFilterChange = (key, value) => {
    onFilterChange({ [key]: value });
  };

  const handleReset = () => {
    onFilterChange({
      status: '',
      client_id: '',
      year: new Date().getFullYear(),
      month: '',
      q: '',
      with_balance: false,
      with_reminders: false,
      page: 1,
    });
  };

  const currentYear = new Date().getFullYear();
  const years = Array.from({ length: 5 }, (_, i) => currentYear - i);
  const months = [
    { value: '', label: 'Tous les mois' },
    { value: 1, label: 'Janvier' },
    { value: 2, label: 'Février' },
    { value: 3, label: 'Mars' },
    { value: 4, label: 'Avril' },
    { value: 5, label: 'Mai' },
    { value: 6, label: 'Juin' },
    { value: 7, label: 'Juillet' },
    { value: 8, label: 'Août' },
    { value: 9, label: 'Septembre' },
    { value: 10, label: 'Octobre' },
    { value: 11, label: 'Novembre' },
    { value: 12, label: 'Décembre' },
  ];

  const statusOptions = [
    { value: '', label: 'Tous les statuts' },
    { value: 'draft', label: 'Brouillon' },
    { value: 'sent', label: 'Envoyée' },
    { value: 'partially_paid', label: 'Partiellement payée' },
    { value: 'paid', label: 'Payée' },
    { value: 'overdue', label: 'En retard' },
    { value: 'cancelled', label: 'Annulée' },
  ];

  return (
    <>
      <div className={styles.filtersGrid}>
        {/* Recherche textuelle */}
        <div className={styles.filterGroup}>
          <label className={styles.filterLabel}>Recherche</label>
          <input
            type="text"
            className={styles.filterInput}
            placeholder="N° facture, client..."
            value={filters.q || ''}
            onChange={(e) => handleFilterChange('q', e.target.value)}
          />
        </div>

        {/* Statut */}
        <div className={styles.filterGroup}>
          <label className={styles.filterLabel}>Statut</label>
          <select
            className={styles.filterSelect}
            value={filters.status || ''}
            onChange={(e) => handleFilterChange('status', e.target.value)}
          >
            {statusOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>

        {/* Client */}
        <div className={styles.filterGroup}>
          <label className={styles.filterLabel}>Client</label>
          <select
            className={styles.filterSelect}
            value={filters.client_id || ''}
            onChange={(e) => handleFilterChange('client_id', e.target.value)}
            disabled={loadingClients}
          >
            <option value="">Tous les clients</option>
            {clients.map((client) => (
              <option key={client.id} value={client.id}>
                {`${client.first_name || ''} ${client.last_name || ''}`.trim() || client.username}
              </option>
            ))}
          </select>
        </div>

        {/* Année */}
        <div className={styles.filterGroup}>
          <label className={styles.filterLabel}>Année</label>
          <select
            className={styles.filterSelect}
            value={filters.year || ''}
            onChange={(e) => handleFilterChange('year', e.target.value)}
          >
            {years.map((year) => (
              <option key={year} value={year}>
                {year}
              </option>
            ))}
          </select>
        </div>

        {/* Mois */}
        <div className={styles.filterGroup}>
          <label className={styles.filterLabel}>Mois</label>
          <select
            className={styles.filterSelect}
            value={filters.month || ''}
            onChange={(e) => handleFilterChange('month', e.target.value)}
          >
            {months.map((month) => (
              <option key={month.value} value={month.value}>
                {month.label}
              </option>
            ))}
          </select>
        </div>

        {/* Éléments par page */}
        <div className={styles.filterGroup}>
          <label className={styles.filterLabel}>Par page</label>
          <select
            className={styles.filterSelect}
            value={filters.per_page || 20}
            onChange={(e) => handleFilterChange('per_page', parseInt(e.target.value))}
          >
            <option value={10}>10</option>
            <option value={20}>20</option>
            <option value={50}>50</option>
            <option value={100}>100</option>
          </select>
        </div>
      </div>

      {/* Filtres avancés */}
      <div className={styles.advancedFilters}>
        <div className={styles.filterCheckbox}>
          <input
            type="checkbox"
            id="with_balance"
            checked={filters.with_balance || false}
            onChange={(e) => handleFilterChange('with_balance', e.target.checked)}
          />
          <label htmlFor="with_balance">Avec solde &gt; 0</label>
        </div>

        <div className={styles.filterCheckbox}>
          <input
            type="checkbox"
            id="with_reminders"
            checked={filters.with_reminders || false}
            onChange={(e) => handleFilterChange('with_reminders', e.target.checked)}
          />
          <label htmlFor="with_reminders">Avec rappels en cours</label>
        </div>
      </div>

      {/* Actions */}
      <div className={styles.filterActions}>
        <button
          className={`${styles.filterBtn} ${styles.filterBtnSecondary}`}
          onClick={handleReset}
        >
          Réinitialiser
        </button>
        <button
          className={`${styles.filterBtn} ${styles.filterBtnPrimary}`}
          onClick={() => {
            /* Les filtres sont appliqués automatiquement */
          }}
        >
          Appliquer
        </button>
      </div>
    </>
  );
};

export default Filters;
