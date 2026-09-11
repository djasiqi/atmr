// pages/institution/Settings/components/AddCompanyModal.jsx
/**
 * Modal pour ajouter une entreprise de transport aux préférences.
 * Affiche le cache immédiatement ; jamais de « Chargement... » infini.
 */

import React, { useState, useMemo } from 'react';
import { FaTimes, FaSearch, FaPlus } from 'react-icons/fa';
import { useEligibleCompanies } from '../../../../hooks/useInstitutionData';
import {
  ELIGIBLE_CARRIERS_COPY,
  resolveEligibleCarriersView,
} from '../../../../utils/institutionEligibleCarriers';
import styles from '../InstitutionSettings.module.css';

const AddCompanyModal = ({ currentPreferences, onAdd, onClose }) => {
  const {
    data: eligibleData,
    isPending,
    isError,
    error,
    refetch,
    isFetching,
  } = useEligibleCompanies();
  const [search, setSearch] = useState('');

  const companies = useMemo(
    () => eligibleData?.companies || [],
    [eligibleData],
  );
  const currentIds = useMemo(
    () => new Set((currentPreferences || []).map((p) => p.company_id)),
    [currentPreferences],
  );

  const notYetAdded = useMemo(
    () => companies.filter((c) => !currentIds.has(c.id)),
    [companies, currentIds],
  );

  const availableCompanies = useMemo(() => {
    if (!search.trim()) return notYetAdded;
    const q = search.toLowerCase();
    return notYetAdded.filter((c) => (
      c.name?.toLowerCase().includes(q)
      || c.address?.toLowerCase().includes(q)
      || c.contact_email?.toLowerCase().includes(q)
    ));
  }, [notYetAdded, search]);

  const view = resolveEligibleCarriersView({
    isPending: isPending && !eligibleData,
    isError,
    error,
    availableCount: availableCompanies.length,
    notYetAddedCount: notYetAdded.length,
    catalogTotal: companies.length,
    search,
  });

  const handleAdd = (company) => {
    onAdd(company);
    onClose();
  };

  return (
    <div className={styles.modal}>
      <div className={styles.modalContent}>
        <div className={styles.modalHeader}>
          <h3>Ajouter un transporteur</h3>
          <button type="button" onClick={onClose}><FaTimes /></button>
        </div>
        <div className={styles.modalBody}>
          <p style={{ fontSize: 12, color: '#888', marginBottom: 12 }}>
            Seules les entreprises autorisées et compatibles avec votre institution sont affichées.
          </p>

          <div className={styles.searchField}>
            <FaSearch className={styles.searchIcon} />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Rechercher par nom, adresse ou email..."
              autoFocus
            />
          </div>

          {view.kind === 'success' && (
            <p style={{ fontSize: 12, color: '#888', margin: '8px 0' }}>
              {search
                ? `${availableCompanies.length} résultat${availableCompanies.length !== 1 ? 's' : ''}`
                : `${notYetAdded.length} entreprise${notYetAdded.length !== 1 ? 's' : ''} disponible${notYetAdded.length !== 1 ? 's' : ''}`
              }
              {isFetching ? ' · Mise à jour…' : ''}
            </p>
          )}

          {view.kind === 'loading' && (
            <p style={{ textAlign: 'center', padding: 20, color: '#666' }}>
              {ELIGIBLE_CARRIERS_COPY.loading}
            </p>
          )}

          {(view.kind === 'error' || view.kind === 'forbidden') && (
            <div style={{ textAlign: 'center', padding: 20 }}>
              <p style={{ color: '#666', marginBottom: 12 }}>
                {view.kind === 'forbidden'
                  ? ELIGIBLE_CARRIERS_COPY.forbidden
                  : ELIGIBLE_CARRIERS_COPY.error}
              </p>
              {view.kind === 'error' && (
                <button
                  type="button"
                  className={styles.addKeyBtn}
                  onClick={() => refetch()}
                  style={{ width: 'auto', margin: '0 auto' }}
                >
                  {ELIGIBLE_CARRIERS_COPY.retry}
                </button>
              )}
            </div>
          )}

          {view.kind === 'empty' && (
            <p style={{ textAlign: 'center', padding: 20, color: '#666' }}>
              {ELIGIBLE_CARRIERS_COPY.empty}
            </p>
          )}

          {view.kind === 'already_added' && (
            <p style={{ textAlign: 'center', padding: 20, color: '#666' }}>
              {ELIGIBLE_CARRIERS_COPY.alreadyAdded}
            </p>
          )}

          {view.kind === 'search_empty' && (
            <p style={{ textAlign: 'center', padding: 20, color: '#666' }}>
              {ELIGIBLE_CARRIERS_COPY.searchEmpty}
            </p>
          )}

          {view.kind === 'success' && (
            <div className={styles.companiesList}>
              {availableCompanies.map((company) => (
                <div key={company.id} className={styles.companyRow}>
                  <div className={styles.companyInfo}>
                    <span className={styles.companyRowName}>{company.name}</span>
                    {company.address && (
                      <span className={styles.companyRowAddress}>{company.address}</span>
                    )}
                  </div>
                  <button
                    type="button"
                    className={styles.addCompanyBtn}
                    onClick={() => handleAdd(company)}
                    title="Ajouter"
                  >
                    <FaPlus />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AddCompanyModal;
