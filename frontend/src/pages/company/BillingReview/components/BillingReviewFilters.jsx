// frontend/src/pages/company/BillingReview/components/BillingReviewFilters.jsx
import React, { useState, useEffect } from 'react';
import { fetchBillingParties, fetchClinicBillingMappings } from '../../../../services/settingsService';
import styles from './BillingReviewFilters.module.css';

const BillingReviewFilters = ({
  filters,
  onFiltersChange,
  companyId,
  sourceOptions = [],
  localFilters,
  onLocalFiltersChange,
  searchInputRef,
}) => {
  const [billingParties, setBillingParties] = useState([]);
  const [loadingParties, setLoadingParties] = useState(false);
  const [clinics, setClinics] = useState([]);
  const [loadingClinics, setLoadingClinics] = useState(false);

  useEffect(() => {
    if (companyId) {
      loadBillingParties();
      loadClinics();
    }
  }, [companyId]);

  const loadBillingParties = async () => {
    try {
      setLoadingParties(true);
      const response = await fetchBillingParties();
      setBillingParties(response.data || []);
    } catch (err) {
      console.error('Erreur lors du chargement des tiers payeurs:', err);
    } finally {
      setLoadingParties(false);
    }
  };

  const loadClinics = async () => {
    try {
      setLoadingClinics(true);
      const response = await fetchClinicBillingMappings();
      const mappings = response.data || [];
      // Extraire les cliniques uniques depuis les mappings
      const uniqueClinics = Array.from(
        new Map(
          mappings.map((m) => [m.clinic_company_id, { id: m.clinic_company_id, name: m.clinic_company_name }])
        ).values()
      );
      setClinics(uniqueClinics);
    } catch (err) {
      console.error('Erreur lors du chargement des cliniques:', err);
    } finally {
      setLoadingClinics(false);
    }
  };

  const handleFilterChange = (key, value) => {
    onFiltersChange({ ...filters, [key]: value || null });
  };

  const handleLocalFilterChange = (key, value) => {
    onLocalFiltersChange({
      ...localFilters,
      [key]: value,
    });
  };

  const currentYear = new Date().getFullYear();
  const years = Array.from({ length: 5 }, (_, i) => currentYear - 2 + i);

  return (
    <div className={styles.filtersContainer}>
      <div className={styles.filterGroup}>
        <label htmlFor="year">Année</label>
        <select
          id="year"
          value={filters.year}
          onChange={(e) => handleFilterChange('year', parseInt(e.target.value))}
        >
          {years.map((year) => (
            <option key={year} value={year}>
              {year}
            </option>
          ))}
        </select>
      </div>

      <div className={styles.filterGroup}>
        <label htmlFor="month">Mois</label>
        <select
          id="month"
          value={filters.month}
          onChange={(e) => handleFilterChange('month', parseInt(e.target.value))}
        >
          {Array.from({ length: 12 }, (_, i) => i + 1).map((month) => (
            <option key={month} value={month}>
              {new Date(2000, month - 1).toLocaleString('fr-FR', { month: 'long' })}
            </option>
          ))}
        </select>
      </div>

      <div className={styles.filterGroup}>
        <label htmlFor="status">Statut</label>
        <select
          id="status"
          value={filters.status || ''}
          onChange={(e) => handleFilterChange('status', e.target.value || null)}
        >
          <option value="">Tous</option>
          <option value="draft">Brouillon</option>
          <option value="needs_review">À vérifier</option>
          <option value="ready">Prêt</option>
          <option value="locked">Verrouillé</option>
        </select>
      </div>

      <div className={styles.filterGroup}>
        <label htmlFor="billing_party">Tiers payeur</label>
        <select
          id="billing_party"
          value={filters.billing_party_id || ''}
          onChange={(e) =>
            handleFilterChange('billing_party_id', e.target.value ? parseInt(e.target.value) : null)
          }
          disabled={loadingParties}
        >
          <option value="">Tous</option>
          {billingParties.map((party) => (
            <option key={party.id} value={party.id}>
              {party.display_name}
            </option>
          ))}
        </select>
      </div>

      <div className={styles.filterGroup}>
        <label htmlFor="clinic">Clinique</label>
        <select
          id="clinic"
          value={filters.clinic_id || ''}
          onChange={(e) =>
            handleFilterChange('clinic_id', e.target.value ? parseInt(e.target.value) : null)
          }
          disabled={loadingClinics}
        >
          <option value="">Toutes</option>
          {clinics.map((clinic) => (
            <option key={clinic.id} value={clinic.id}>
              {clinic.name}
            </option>
          ))}
        </select>
      </div>

      <div className={styles.filterGroup}>
        <label htmlFor="source">Source</label>
        <select
          id="source"
          value={localFilters.source || ''}
          onChange={(e) => handleLocalFilterChange('source', e.target.value || null)}
        >
          <option value="">Toutes</option>
          {sourceOptions.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>

      <div className={styles.filterGroup}>
        <label htmlFor="search">Recherche client</label>
        <input
          id="search"
          type="text"
          placeholder="Nom du patient"
          value={localFilters.search}
          onChange={(e) => handleLocalFilterChange('search', e.target.value)}
          ref={searchInputRef}
        />
      </div>

      <div className={styles.filterGroup}>
        <label className={styles.checkboxLabel}>
          <input
            type="checkbox"
            checked={!!localFilters.needs_review_only}
            onChange={(e) => handleLocalFilterChange('needs_review_only', e.target.checked)}
          />
          <span>À vérifier uniquement</span>
        </label>
      </div>
    </div>
  );
};

export default BillingReviewFilters;
