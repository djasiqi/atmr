// pages/institution/Settings/components/AddCompanyModal.jsx
/**
 * Modal pour ajouter une entreprise de transport aux préférences.
 * Charge la liste des entreprises éligibles et filtre celles déjà présentes.
 */

import React, { useState, useMemo } from 'react';
import { FaTimes, FaSearch, FaPlus } from 'react-icons/fa';
import { useEligibleCompanies } from '../../../../hooks/useInstitutionData';
import styles from '../InstitutionSettings.module.css';

const AddCompanyModal = ({ currentPreferences, onAdd, onClose }) => {
  const { data: eligibleData, isLoading } = useEligibleCompanies();
  const [search, setSearch] = useState('');

  // Filtrer : exclure celles déjà dans les préférences + recherche
  const availableCompanies = useMemo(() => {
    const companies = eligibleData?.companies || [];
    const currentIds = new Set((currentPreferences || []).map(p => p.company_id));

    return companies.filter(c => {
      if (currentIds.has(c.id)) return false;
      if (search.trim()) {
        const q = search.toLowerCase();
        return (
          c.name?.toLowerCase().includes(q) ||
          c.address?.toLowerCase().includes(q)
        );
      }
      return true;
    });
  }, [eligibleData, currentPreferences, search]);

  // Count total available (excluding already added)
  const totalEligible = useMemo(() => {
    const companies = eligibleData?.companies || [];
    const currentIds = new Set((currentPreferences || []).map(p => p.company_id));
    return companies.filter(c => !currentIds.has(c.id)).length;
  }, [eligibleData, currentPreferences]);

  const handleAdd = (company) => {
    onAdd(company);
    onClose();
  };

  return (
    <div className={styles.modal}>
      <div className={styles.modalContent}>
        <div className={styles.modalHeader}>
          <h3>Ajouter un transporteur</h3>
          <button onClick={onClose}><FaTimes /></button>
        </div>
        <div className={styles.modalBody}>
          <p style={{ fontSize: 12, color: '#888', marginBottom: 12 }}>
            Seules les entreprises autorisées et compatibles avec votre institution sont affichées.
          </p>

          {/* Barre de recherche */}
          <div className={styles.searchField}>
            <FaSearch className={styles.searchIcon} />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Rechercher par nom ou adresse..."
              autoFocus
            />
          </div>

          {/* Compteur résultats */}
          {!isLoading && (
            <p style={{ fontSize: 12, color: '#888', margin: '8px 0' }}>
              {search
                ? `${availableCompanies.length} résultat${availableCompanies.length !== 1 ? 's' : ''}`
                : `${totalEligible} entreprise${totalEligible !== 1 ? 's' : ''} disponible${totalEligible !== 1 ? 's' : ''}`
              }
            </p>
          )}

          {/* Liste */}
          {isLoading ? (
            <p style={{ textAlign: 'center', padding: 20, color: '#666' }}>Chargement...</p>
          ) : availableCompanies.length === 0 ? (
            <p style={{ textAlign: 'center', padding: 20, color: '#666' }}>
              {search ? 'Aucune entreprise ne correspond à votre recherche' : 'Toutes les entreprises éligibles sont déjà dans vos préférences'}
            </p>
          ) : (
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
