// frontend/src/pages/company/Settings/tabs/PartnershipsTab.jsx
import React, { useState, useEffect, useRef } from 'react';
import styles from '../CompanySettings.module.css';
import partnershipStyles from './PartnershipsTab.module.css';
import apiClient from '../../../../utils/apiClient';
import { showSuccess, showError } from '../../../../utils/toast';
import KPICards from './components/KPICards';
import StatementGenerationForm from './components/StatementGenerationForm';

const PartnershipsTab = () => {
  const [partnerships, setPartnerships] = useState([]);
  const [pendingRequests, setPendingRequests] = useState([]);
  const [stats, setStats] = useState(null);
  const [showRequestModal, setShowRequestModal] = useState(false);
  const [showEditModal, setShowEditModal] = useState(false);
  const [showStatementModal, setShowStatementModal] = useState(false);
  const [editingPartnership, setEditingPartnership] = useState(null);
  const [statementPartnership, setStatementPartnership] = useState(null);
  const [selectedPartnership, setSelectedPartnership] = useState(null);
  const [loading, setLoading] = useState(true);
  const [loadingStats, setLoadingStats] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [searching, setSearching] = useState(false);
  const [lastSearchQuery, setLastSearchQuery] = useState('');
  const searchDebounceRef = useRef(null);

  // Formulaire de demande
  const [requestForm, setRequestForm] = useState({
    partner_company_id: null,
    partner_company_name: '',
    default_partner_tariff_percent: 90,
    payment_terms_days: 30,
    auto_accept: false,
  });

  // Formulaire d'édition
  const [editForm, setEditForm] = useState({
    default_partner_tariff_percent: 90,
    auto_accept: false,
    auto_invoice: true,
  });

  // Charger les données
  useEffect(() => {
    loadStats();
    loadPartnerships();
  }, []);

  // Réinitialiser la recherche à l'ouverture du modal de demande
  useEffect(() => {
    if (showRequestModal) {
      setSearchQuery('');
      setSearchResults([]);
      setLastSearchQuery('');
    }
  }, [showRequestModal]);

  const loadStats = async () => {
    try {
      setLoadingStats(true);
      const { data } = await apiClient.get('/companies/me/partnerships/stats');
      setStats(data?.data || data);
    } catch (err) {
      console.error('❌ Erreur chargement stats:', err);
    } finally {
      setLoadingStats(false);
    }
  };

  const loadPartnerships = async () => {
    try {
      setLoading(true);
      const { data } = await apiClient.get('/companies/me/partnerships');
      const partnershipsList = data?.data || data || [];

      // Dédupliquer par ID (au cas où il y aurait des doublons)
      const uniquePartnerships = partnershipsList.reduce((acc, p) => {
        if (!acc.find((existing) => existing.id === p.id)) {
          acc.push(p);
        }
        return acc;
      }, []);

      // Séparer actifs et en attente
      const active = uniquePartnerships.filter(
        (p) => p.status === 'ACCEPTED' && p.is_active
      );
      const pending = uniquePartnerships.filter((p) => p.status === 'PENDING');
      
      console.log('✅ Partenariats chargés:', {
        total: partnershipsList.length,
        unique: uniquePartnerships.length,
        active: active.length,
        pending: pending.length,
        partnerships: active.map(p => ({
          id: p.id,
          partner_name: p.partner_company_name_display,
          owner_id: p.owner_company_id,
          partner_id: p.partner_company_id,
          current_id: p.current_company_id,
        })),
      });
      
      setPartnerships(active);
      setPendingRequests(pending);
    } catch (err) {
      console.error('❌ Erreur chargement partenariats:', err);
      showError('Impossible de charger les partenariats');
    } finally {
      setLoading(false);
    }
  };

  const searchCompanies = async (query) => {
    const q = typeof query === 'string' ? query.trim() : '';
    if (!q || q.length < 2) {
      setSearchResults([]);
      setLastSearchQuery('');
      return;
    }
    try {
      setSearching(true);
      const { data } = await apiClient.get('/companies/search', {
        params: { q },
      });
      const list = Array.isArray(data?.data) ? data.data : Array.isArray(data) ? data : [];
      setSearchResults(list);
      setLastSearchQuery(q);
    } catch (err) {
      console.error('Erreur recherche entreprises:', err);
      showError('Erreur lors de la recherche');
      setSearchResults([]);
      setLastSearchQuery(q);
    } finally {
      setSearching(false);
    }
  };

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('fr-CH', {
      style: 'currency',
      currency: 'CHF',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount || 0);
  };

  const getPartnerName = (p) => {
    // ✅ Utiliser le champ enrichi par le backend qui indique le nom du partenaire
    // (l'autre entreprise, pas celle qui consulte)
    return p.partner_company_name_display || p.partner_company_name || p.owner_company_name || 'Partenaire inconnu';
  };

  const getCompanyType = (_p) => {
    // Pour l'instant, on retourne "Transport" par défaut
    // TODO: Ajouter le type d'entreprise dans le modèle
    return 'Transport';
  };

  const getStatusBadge = (p) => {
    if (!p.is_active) return { text: '🔴 Suspendu', color: '#F44336' };
    // TODO: Ajouter logique pour "À régulariser"
    return { text: '🟢 Actif', color: '#4CAF50' };
  };

  const handleRequestPartnership = async () => {
    if (!requestForm.partner_company_id) {
      showError('Veuillez sélectionner une entreprise');
      return;
    }
    try {
      await apiClient.post('/partnerships', {
        partner_company_id: requestForm.partner_company_id,
        default_partner_tariff_percent: requestForm.default_partner_tariff_percent,
        payment_terms_days: requestForm.payment_terms_days,
        auto_accept_rules: requestForm.auto_accept,
      });
      showSuccess('Demande de partenariat envoyée');
      setShowRequestModal(false);
      setRequestForm({
        partner_company_id: null,
        partner_company_name: '',
        default_partner_tariff_percent: 90,
        payment_terms_days: 30,
        auto_accept: false,
      });
      loadPartnerships();
      loadStats();
    } catch (err) {
      console.error('Erreur création partenariat:', err);
      showError(err?.response?.data?.error || 'Erreur lors de la création');
    }
  };

  const handleAcceptRequest = async (partnershipId) => {
    try {
      await apiClient.post(`/partnerships/${partnershipId}/accept`);
      showSuccess('Partenariat accepté');
      loadPartnerships();
      loadStats();
    } catch (err) {
      console.error('Erreur acceptation:', err);
      showError('Erreur lors de l\'acceptation');
    }
  };

  const handleRejectRequest = async (partnershipId) => {
    try {
      await apiClient.post(`/partnerships/${partnershipId}/reject`);
      showSuccess('Demande refusée');
      loadPartnerships();
    } catch (err) {
      console.error('Erreur refus:', err);
      showError('Erreur lors du refus');
    }
  };

  const handleEditPartnership = (partnership) => {
    setEditingPartnership(partnership);
    setEditForm({
      default_partner_tariff_percent: partnership.default_partner_tariff_percent || 90,
      auto_accept: partnership.auto_accept_rules || false,
      auto_invoice: partnership.auto_invoice !== undefined ? partnership.auto_invoice : true,
    });
    setShowEditModal(true);
  };

  const handleUpdatePartnership = async () => {
    if (!editingPartnership) return;

    try {
      await apiClient.put(`/companies/me/partnerships/${editingPartnership.id}`, {
        default_partner_tariff_percent: editForm.default_partner_tariff_percent,
        auto_accept: editForm.auto_accept,
        auto_invoice: editForm.auto_invoice,
      });

      showSuccess('Partenariat mis à jour avec succès');
      setShowEditModal(false);
      setEditingPartnership(null);
      loadPartnerships();
      loadStats();
    } catch (err) {
      console.error('Erreur mise à jour partenariat:', err);
      showError(
        err?.response?.data?.error || 'Erreur lors de la mise à jour du partenariat'
      );
    }
  };

  const handleGenerateStatement = async (isConsolidated, periodType, year, month, startDate, endDate) => {
    try {
      let response;
      if (isConsolidated) {
        // Décompte consolidé
        const payload = { period_type: periodType };
        if (year) payload.year = year;
        if (month) payload.month = month;
        if (startDate) payload.start_date = startDate;
        if (endDate) payload.end_date = endDate;

        response = await apiClient.post('/companies/me/partnerships/statements/generate', payload);
      } else {
        // Décompte par partenaire
        if (!statementPartnership) {
          showError('Aucun partenariat sélectionné');
          return;
        }

        const payload = { period_type: periodType };
        if (year) payload.year = year;
        if (month) payload.month = month;
        if (startDate) payload.start_date = startDate;
        if (endDate) payload.end_date = endDate;

        response = await apiClient.post(
          `/companies/me/partnerships/${statementPartnership.id}/statements/generate`,
          payload
        );
      }

      const pdfUrl = response.data?.data?.pdf_url;
      if (pdfUrl) {
        // Ouvrir le PDF dans un nouvel onglet
        window.open(pdfUrl, '_blank');
        showSuccess('Décompte généré avec succès');
        setShowStatementModal(false);
        setStatementPartnership(null);
      } else {
        showError('URL du PDF non trouvée dans la réponse');
      }
    } catch (err) {
      console.error('Erreur génération décompte:', err);
      showError(
        err?.response?.data?.error || 'Erreur lors de la génération du décompte'
      );
    }
  };

  if (loading && loadingStats) {
    return (
      <div className={styles.loadingContainer}>
        <div className={styles.spinner}></div>
        <p>Chargement…</p>
      </div>
    );
  }

  return (
    <div className={styles.settingsForm} style={{ display: 'block' }}>
      <h2>🤝 Partenariats</h2>

      {/* KPI Cards */}
      <KPICards stats={stats} loading={loadingStats} />

      {/* Tableau Partenariats Actifs */}
      <section className={styles.section}>
        <h3>Partenariats actifs</h3>
        {partnerships.length === 0 ? (
          <p style={{ color: 'var(--text-secondary)' }}>
            Aucun partenariat actif
          </p>
        ) : (
          <div className={partnershipStyles.tableContainer}>
            <table className={partnershipStyles.partnershipsTable}>
              <thead>
                <tr>
                  <th>Partenaire</th>
                  <th>Type</th>
                  <th>Courses envoyées</th>
                  <th>Courses reçues</th>
                  <th>CA généré</th>
                  <th>À payer</th>
                  <th>À recevoir</th>
                  <th>Solde</th>
                  <th>Statut</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {partnerships.map((p) => {
                  const stats = p.stats || {};
                  const statusBadge = getStatusBadge(p);
                  const balance = stats.balance || 0;
                  return (
                    <tr
                      key={p.id}
                      onClick={() => setSelectedPartnership(p)}
                      className={partnershipStyles.tableRow}
                    >
                      <td>
                        <strong>{getPartnerName(p)}</strong>
                      </td>
                      <td>{getCompanyType(p)}</td>
                      <td>{stats.sent_transfers || 0}</td>
                      <td>{stats.received_transfers || 0}</td>
                      <td>{formatCurrency(stats.total_revenue)}</td>
                      <td>{formatCurrency(stats.amount_to_pay)}</td>
                      <td>{formatCurrency(stats.amount_to_receive)}</td>
                      <td
                        style={{
                          color: balance >= 0 ? '#4CAF50' : '#F44336',
                          fontWeight: 'bold',
                        }}
                      >
                        {formatCurrency(balance)}
                      </td>
                      <td>
                        <span style={{ color: statusBadge.color }}>
                          {statusBadge.text}
                        </span>
                      </td>
                      <td>
                        <div className={partnershipStyles.actionButtons}>
                          <button
                            type="button"
                            className={partnershipStyles.actionBtn}
                            onClick={(e) => {
                              e.stopPropagation();
                              handleEditPartnership(p);
                            }}
                            title="Modifier"
                          >
                            ✏️
                          </button>
                          <button
                            type="button"
                            className={partnershipStyles.actionBtn}
                            onClick={(e) => {
                              e.stopPropagation();
                              setStatementPartnership(p);
                              setShowStatementModal(true);
                            }}
                            title="Générer décompte"
                          >
                            📄
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* Demandes en attente */}
      <section className={styles.section}>
        <h3>Demandes de partenariat</h3>
        {pendingRequests.length === 0 ? (
          <p style={{ color: 'var(--text-secondary)' }}>
            Aucune demande en attente
          </p>
        ) : (
          <div className={partnershipStyles.tableContainer}>
            <table className={partnershipStyles.partnershipsTable}>
              <thead>
                <tr>
                  <th>Société</th>
                  <th>Type</th>
                  <th>Zone</th>
                  <th>Date</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {pendingRequests.map((p) => {
                  const date = p.created_at
                    ? new Date(p.created_at).toLocaleDateString('fr-CH')
                    : 'N/A';
                  return (
                    <tr key={p.id}>
                      <td>
                        <strong>{getPartnerName(p)}</strong>
                      </td>
                      <td>{getCompanyType(p)}</td>
                      <td>Genève</td>
                      <td>{date}</td>
                      <td>
                        <div className={partnershipStyles.actionButtons}>
                          <button
                            type="button"
                            className={`${styles.button} ${styles.primary}`}
                            onClick={() => handleAcceptRequest(p.id)}
                          >
                            ✅ Accepter
                          </button>
                          <button
                            type="button"
                            className={`${styles.button} ${styles.danger}`}
                            onClick={() => handleRejectRequest(p.id)}
                          >
                            ❌ Refuser
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* Bouton pour demander un partenariat */}
      <section className={styles.section}>
        <button
          type="button"
          className={`${styles.button} ${styles.primary}`}
          onClick={() => setShowRequestModal(true)}
        >
          ➕ Demander un partenariat
        </button>
      </section>

      {/* Modal Détail Partenariat */}
      {selectedPartnership && (
        <div
          className={partnershipStyles.modalOverlay}
          onClick={() => setSelectedPartnership(null)}
        >
          <div
            className={partnershipStyles.modalContent}
            onClick={(e) => e.stopPropagation()}
          >
            <div className={partnershipStyles.modalHeader}>
              <h3>Détails du partenariat</h3>
              <button
                type="button"
                onClick={() => setSelectedPartnership(null)}
                className={partnershipStyles.closeBtn}
              >
                ✕
              </button>
            </div>
            <div className={partnershipStyles.modalBody}>
              <div className={partnershipStyles.detailSection}>
                <h4>🏢 Infos générales</h4>
                <p>
                  <strong>Partenaire:</strong> {getPartnerName(selectedPartnership)}
                </p>
                <p>
                  <strong>Type:</strong> {getCompanyType(selectedPartnership)}
                </p>
                <p>
                  <strong>Réduction:</strong>{' '}
                  {selectedPartnership.default_partner_tariff_percent
                    ? `${100 - selectedPartnership.default_partner_tariff_percent}%`
                    : '10%'}
                </p>
                <p>
                  <strong>Délai de paiement:</strong>{' '}
                  {selectedPartnership.payment_terms_days} jours
                </p>
                <p>
                  <strong>Date de début:</strong>{' '}
                  {selectedPartnership.created_at
                    ? new Date(selectedPartnership.created_at).toLocaleDateString('fr-CH')
                    : 'N/A'}
                </p>
              </div>
              {selectedPartnership.stats && (
                <>
                  <div className={partnershipStyles.detailSection}>
                    <h4>🚗 Activité opérationnelle</h4>
                    <p>
                      <strong>Courses envoyées:</strong>{' '}
                      {selectedPartnership.stats.sent_transfers || 0}
                    </p>
                    <p>
                      <strong>Courses reçues:</strong>{' '}
                      {selectedPartnership.stats.received_transfers || 0}
                    </p>
                  </div>
                  <div className={partnershipStyles.detailSection}>
                    <h4>💰 Financier</h4>
                    <p>
                      <strong>CA généré:</strong>{' '}
                      {formatCurrency(selectedPartnership.stats.total_revenue)}
                    </p>
                    <p>
                      <strong>À payer:</strong>{' '}
                      {formatCurrency(selectedPartnership.stats.amount_to_pay)}
                    </p>
                    <p>
                      <strong>À recevoir:</strong>{' '}
                      {formatCurrency(selectedPartnership.stats.amount_to_receive)}
                    </p>
                    <p>
                      <strong>Solde:</strong>{' '}
                      <span
                        style={{
                          color:
                            (selectedPartnership.stats.balance || 0) >= 0
                              ? '#4CAF50'
                              : '#F44336',
                          fontWeight: 'bold',
                        }}
                      >
                        {formatCurrency(selectedPartnership.stats.balance)}
                      </span>
                    </p>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Modal d'édition de partenariat */}
      {showEditModal && editingPartnership && (
        <div
          className="modal-overlay"
          onClick={() => {
            setShowEditModal(false);
            setEditingPartnership(null);
          }}
        >
          <div
            className="modal-content modal-md"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="modal-header">
              <h2 className="modal-title">Modifier le partenariat</h2>
              <button
                type="button"
                onClick={() => {
                  setShowEditModal(false);
                  setEditingPartnership(null);
                }}
                className="modal-close"
              >
                ✕
              </button>
            </div>
            <div className="modal-body">
              <div className={styles.formGroup}>
                <label className={styles.label}>Partenaire</label>
                <div
                  style={{
                    padding: 'var(--spacing-sm)',
                    background: 'var(--bg-secondary)',
                    borderRadius: 'var(--radius-md)',
                    border: '1px solid var(--border-primary)',
                    color: 'var(--text-primary)',
                    fontWeight: 'var(--font-weight-medium)',
                  }}
                >
                  {getPartnerName(editingPartnership)}
                </div>
              </div>
              <div className={styles.formGroup}>
                <label
                  htmlFor="edit_partner_tariff_percent"
                  className={styles.label}
                >
                  Pourcentage pour le partenaire (%)
                </label>
                <input
                  id="edit_partner_tariff_percent"
                  type="number"
                  min="1"
                  max="100"
                  value={editForm.default_partner_tariff_percent}
                  onChange={(e) =>
                    setEditForm({
                      ...editForm,
                      default_partner_tariff_percent: parseFloat(e.target.value) || 90,
                    })
                  }
                  className={styles.input}
                />
                <small
                  style={{
                    display: 'block',
                    marginTop: 'var(--spacing-xs)',
                    color: 'var(--text-secondary)',
                    fontSize: 'var(--font-size-xs)',
                  }}
                >
                  Réduction: {100 - editForm.default_partner_tariff_percent}%
                </small>
              </div>
              <div className={styles.formGroup}>
                <label
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 'var(--spacing-sm)',
                    cursor: 'pointer',
                    padding: 'var(--spacing-sm)',
                    border: '1px solid var(--border-primary)',
                    borderRadius: 'var(--radius-md)',
                    transition: 'var(--transition-fast)',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = 'var(--bg-secondary)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = 'transparent';
                  }}
                >
                  <input
                    type="checkbox"
                    checked={editForm.auto_accept}
                    onChange={(e) =>
                      setEditForm({
                        ...editForm,
                        auto_accept: e.target.checked,
                      })
                    }
                    style={{
                      width: '18px',
                      height: '18px',
                      cursor: 'pointer',
                    }}
                  />
                  <span>Auto-accepter les transferts</span>
                </label>
              </div>
              <div className={styles.formGroup}>
                <label
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 'var(--spacing-sm)',
                    cursor: 'pointer',
                    padding: 'var(--spacing-sm)',
                    border: '1px solid var(--border-primary)',
                    borderRadius: 'var(--radius-md)',
                    transition: 'var(--transition-fast)',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = 'var(--bg-secondary)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = 'transparent';
                  }}
                >
                  <input
                    type="checkbox"
                    checked={editForm.auto_invoice}
                    onChange={(e) =>
                      setEditForm({
                        ...editForm,
                        auto_invoice: e.target.checked,
                      })
                    }
                    style={{
                      width: '18px',
                      height: '18px',
                      cursor: 'pointer',
                    }}
                  />
                  <span>Facturation automatique</span>
                </label>
              </div>
            </div>
            <div className="modal-footer">
              <button
                type="button"
                className="btn btn-danger"
                onClick={async () => {
                  if (
                    !window.confirm(
                      'Êtes-vous sûr de vouloir supprimer ce partenariat ? Cette action est irréversible et supprimera complètement le lien entre les deux entreprises.'
                    )
                  ) {
                    return;
                  }
                    try {
                      const response = await apiClient.delete(
                        `/companies/me/partnerships/${editingPartnership.id}`
                      );
                      console.log('✅ Suppression réussie:', response);
                      showSuccess('Partenariat supprimé avec succès');
                      setShowEditModal(false);
                      setEditingPartnership(null);
                      loadPartnerships();
                      loadStats();
                    } catch (err) {
                      console.error('❌ Erreur suppression partenariat:', err);
                      console.error('Response data:', err?.response?.data);
                      console.error('Response status:', err?.response?.status);
                      const errorMessage =
                        err?.response?.data?.error ||
                        err?.response?.data?.message ||
                        err?.message ||
                        'Erreur lors de la suppression du partenariat';
                      showError(errorMessage);
                    }
                }}
              >
                🗑️ Supprimer le partenariat
              </button>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => {
                  setShowEditModal(false);
                  setEditingPartnership(null);
                }}
              >
                Annuler
              </button>
              <button
                type="button"
                className="btn btn-primary"
                onClick={handleUpdatePartnership}
              >
                Enregistrer les modifications
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Modal de génération de décompte */}
      {showStatementModal && (
        <div
          className="modal-overlay"
          onClick={() => {
            setShowStatementModal(false);
            setStatementPartnership(null);
          }}
        >
          <div
            className="modal-content modal-md"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="modal-header">
              <h2 className="modal-title">Générer un décompte</h2>
              <button
                type="button"
                onClick={() => {
                  setShowStatementModal(false);
                  setStatementPartnership(null);
                }}
                className="modal-close"
              >
                ✕
              </button>
            </div>
            <div className="modal-body">
              <StatementGenerationForm
                isConsolidated={!statementPartnership}
                partnership={statementPartnership}
                onGenerate={handleGenerateStatement}
                onCancel={() => {
                  setShowStatementModal(false);
                  setStatementPartnership(null);
                }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Modal de demande de partenariat */}
      {showRequestModal && (
        <div
          className={partnershipStyles.modalOverlay}
          onClick={() => setShowRequestModal(false)}
        >
          <div
            className={partnershipStyles.modalContent}
            onClick={(e) => e.stopPropagation()}
          >
            <h3>Demander un partenariat</h3>
            <div className={styles.formGroup}>
              <label htmlFor="company_search">Rechercher une entreprise</label>
              <input
                id="company_search"
                type="text"
                value={searchQuery}
                onChange={(e) => {
                  const v = e.target.value;
                  setSearchQuery(v);
                  if (searchDebounceRef.current) clearTimeout(searchDebounceRef.current);
                  searchDebounceRef.current = setTimeout(() => searchCompanies(v), 300);
                }}
                placeholder="Nom, email ou domaine (ex: emmenez-moi.ch)..."
                className={styles.input}
              />
              {searching && <p>Recherche en cours...</p>}
              {!searching && lastSearchQuery && searchResults.length === 0 && (
                <p style={{ marginTop: '0.25rem', color: 'var(--color-text-secondary, #666)', fontSize: '0.9rem' }}>
                  Aucune entreprise trouvée pour « {lastSearchQuery} »
                </p>
              )}
              {searchResults.length > 0 && (
                <div style={{ marginTop: '0.5rem' }}>
                  {searchResults.map((company) => (
                    <div
                      key={company.id}
                      onClick={() => {
                        setRequestForm({
                          ...requestForm,
                          partner_company_id: company.id,
                          partner_company_name: company.name,
                        });
                        setSearchQuery(company.name);
                        setSearchResults([]);
                      }}
                      style={{
                        padding: '0.5rem',
                        cursor: 'pointer',
                        border: '1px solid #ddd',
                        borderRadius: '4px',
                        marginBottom: '0.25rem',
                      }}
                    >
                      {company.name}
                    </div>
                  ))}
                </div>
              )}
            </div>
            {requestForm.partner_company_name && (
              <div className={styles.formGroup}>
                <label>Entreprise sélectionnée</label>
                <div>{requestForm.partner_company_name}</div>
              </div>
            )}
            <div className={styles.formGroup}>
              <label htmlFor="partner_tariff_percent">
                Pourcentage pour le partenaire (%)
              </label>
              <input
                id="partner_tariff_percent"
                type="number"
                min="1"
                max="100"
                value={requestForm.default_partner_tariff_percent}
                onChange={(e) =>
                  setRequestForm({
                    ...requestForm,
                    default_partner_tariff_percent: parseFloat(e.target.value) || 90,
                  })
                }
                className={styles.input}
              />
              <small>
                Réduction: {100 - requestForm.default_partner_tariff_percent}%
              </small>
            </div>
            <div className={styles.formGroup}>
              <label htmlFor="payment_terms_days">Délai de paiement (jours)</label>
              <input
                id="payment_terms_days"
                type="number"
                min="1"
                value={requestForm.payment_terms_days}
                onChange={(e) =>
                  setRequestForm({
                    ...requestForm,
                    payment_terms_days: parseInt(e.target.value) || 30,
                  })
                }
                className={styles.input}
              />
            </div>
            <div className={styles.formGroup}>
              <label>
                <input
                  type="checkbox"
                  checked={requestForm.auto_accept}
                  onChange={(e) =>
                    setRequestForm({
                      ...requestForm,
                      auto_accept: e.target.checked,
                    })
                  }
                />
                Auto-accepter les transferts
              </label>
            </div>
            <div
              style={{
                display: 'flex',
                gap: '0.5rem',
                justifyContent: 'flex-end',
                marginTop: '1rem',
              }}
            >
              <button
                type="button"
                className={`${styles.button} ${styles.secondary}`}
                onClick={() => setShowRequestModal(false)}
              >
                Annuler
              </button>
              <button
                type="button"
                className={`${styles.button} ${styles.primary}`}
                onClick={handleRequestPartnership}
              >
                Envoyer la demande
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PartnershipsTab;
