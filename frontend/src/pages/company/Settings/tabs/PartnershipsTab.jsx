// frontend/src/pages/company/Settings/tabs/PartnershipsTab.jsx
import React, { useState, useEffect, useRef, useCallback, forwardRef, useImperativeHandle } from 'react';
import {
  FiUsers,
  FiFileText,
  FiCheck,
  FiX,
  FiPlus,
  FiHome,
  FiTruck,
  FiDollarSign,
  FiTrash2,
  FiBarChart2,
  FiInbox,
} from 'react-icons/fi';
import styles from '../CompanySettings.module.css';
import partnershipStyles from './PartnershipsTab.module.css';
import apiClient from '../../../../utils/apiClient';
import { showSuccess, showError } from '../../../../utils/toast';
import { ensurePdfUrlWorksInDev } from '../../../../utils/pdfUrlFallback';
import KPICards from './components/KPICards';
import StatementGenerationForm from './components/StatementGenerationForm';

const PartnershipsTab = forwardRef(({ isEditing }, ref) => {
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

  const [rowEdits, setRowEdits] = useState({});
  const [_savingRows, setSavingRows] = useState({});

  useEffect(() => {
    if (isEditing) {
      const edits = {};
      partnerships.forEach((p) => {
        edits[p.id] = {
          default_partner_tariff_percent: p.default_partner_tariff_percent ?? 90,
          auto_accept: p.auto_accept_rules ?? false,
          auto_invoice: p.auto_invoice !== undefined ? p.auto_invoice : true,
        };
      });
      setRowEdits(edits);
    } else {
      setRowEdits({});
      setSavingRows({});
    }
  }, [isEditing, partnerships]);

  const updateRowEdit = useCallback((partnershipId, field, value) => {
    setRowEdits((prev) => ({
      ...prev,
      [partnershipId]: { ...prev[partnershipId], [field]: value },
    }));
  }, []);

  const _handleSaveRow = useCallback(async (partnershipId) => {
    const edits = rowEdits[partnershipId];
    if (!edits) return;
    setSavingRows((prev) => ({ ...prev, [partnershipId]: true }));
    try {
      await apiClient.put(`/companies/me/partnerships/${partnershipId}`, {
        default_partner_tariff_percent: edits.default_partner_tariff_percent,
        auto_accept: edits.auto_accept,
        auto_invoice: edits.auto_invoice,
      });
      showSuccess('Partenariat mis à jour');
      loadPartnerships();
      loadStats();
    } catch (err) {
      console.error('Erreur mise à jour partenariat:', err);
      showError(err?.response?.data?.error || 'Erreur lors de la mise à jour');
    } finally {
      setSavingRows((prev) => ({ ...prev, [partnershipId]: false }));
    }
  }, [rowEdits]);

  const handleDeleteRow = useCallback(async (partnershipId, partnerName) => {
    if (!window.confirm(`Supprimer le partenariat avec ${partnerName} ? Cette action est irréversible.`)) return;
    try {
      await apiClient.delete(`/companies/me/partnerships/${partnershipId}`);
      showSuccess('Partenariat supprimé');
      loadPartnerships();
      loadStats();
    } catch (err) {
      console.error('Erreur suppression partenariat:', err);
      showError(err?.response?.data?.error || 'Erreur lors de la suppression');
    }
  }, []);

  useImperativeHandle(ref, () => ({
    async save() {
      const ids = Object.keys(rowEdits);
      if (ids.length === 0) return;
      const errors = [];
      for (const id of ids) {
        const edits = rowEdits[id];
        try {
          await apiClient.put(`/companies/me/partnerships/${id}`, {
            default_partner_tariff_percent: edits.default_partner_tariff_percent,
            auto_accept: edits.auto_accept,
            auto_invoice: edits.auto_invoice,
          });
        } catch (err) {
          errors.push(err);
        }
      }
      if (errors.length > 0) {
        throw errors[0];
      }
      loadPartnerships();
      loadStats();
    },
  }), [rowEdits]);

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
      console.error('Erreur chargement stats:', err);
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
      
      console.log('Partenariats charges:', {
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
      const status = err?.response?.status;
      const data = err?.response?.data;
      console.error('Erreur chargement partenariats:', status, data, err);
      const isServerError =
        status === 500 || data?.error_code === 'internal_error';
      const message = isServerError
        ? 'Erreur serveur lors du chargement des partenariats'
        : data?.error ?? 'Impossible de charger les partenariats';
      showError(message);
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
      setLastSearchQuery(''); // Ne pas afficher "Aucune entreprise trouvée" en cas d'erreur réseau/API
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
    // Utiliser le champ enrichi par le backend qui indique le nom du partenaire
    // (l'autre entreprise, pas celle qui consulte)
    return p.partner_company_name_display || p.partner_company_name || p.owner_company_name || 'Partenaire inconnu';
  };

  const getCompanyType = (_p) => {
    // Pour l'instant, on retourne "Transport" par défaut
    // TODO: Ajouter le type d'entreprise dans le modèle
    return 'Transport';
  };

  const getStatusBadge = (p) => {
    if (!p.is_active) return { text: 'Suspendu', className: partnershipStyles.badgeSuspended };
    // TODO: Ajouter logique pour "À régulariser"
    return { text: 'Actif', className: partnershipStyles.badgeActive };
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
      const status = err?.response?.status;
      const data = err?.response?.data;
      const msg =
        data?.error ?? data?.message ?? 'Erreur lors de la création';
      let display = msg;
      if (status === 409) {
        display = msg || 'Un partenariat existe déjà ou une demande est en attente.';
      } else if (status === 404) {
        display = msg || 'Entreprise partenaire introuvable.';
      } else if (status === 400) {
        display = msg || 'Données invalides (vérifiez l’entreprise sélectionnée).';
      }
      console.error('Erreur création partenariat:', status, data, err);
      showError(display);
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

  const _handleEditPartnership = (partnership) => {
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
        // Ouvrir le PDF dans un nouvel onglet (fallback localhost→127.0.0.1 en dev)
        window.open(ensurePdfUrlWorksInDev(pdfUrl), '_blank');
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
    <div className={`${styles.settingsForm} ${styles.blockDisplay}`}>
      {/* Carte 1 : Vue d'ensemble */}
      <div className={styles.card}>
        <div className={styles.cardHeader}>
          <div className={styles.cardIcon}><FiBarChart2 size={16} /></div>
          <div className={styles.cardHeaderText}>
            <h3 className={styles.cardTitle}>Vue d'ensemble</h3>
            <p className={styles.cardHint}>Activite et solde du mois en cours</p>
          </div>
        </div>
        <KPICards stats={stats} loading={loadingStats} />
      </div>

      {/* Carte 2 : Partenariats actifs */}
      <div className={styles.card}>
        <div className={styles.cardHeader}>
          <div className={styles.cardIcon}><FiUsers size={16} /></div>
          <div className={styles.cardHeaderText}>
            <h3 className={styles.cardTitle}>Partenariats actifs</h3>
          </div>
          <button
            type="button"
            className={`${styles.button} ${styles.primary}`}
            onClick={() => setShowRequestModal(true)}
          >
            <FiPlus size={14} aria-hidden />
            Nouveau
          </button>
        </div>
        {partnerships.length === 0 ? (
          <p className={partnershipStyles.textSecondary}>
            Aucun partenariat actif
          </p>
        ) : (
          <div className={partnershipStyles.tableContainer}>
            <table className={partnershipStyles.partnershipsTable}>
              <thead>
                <tr>
                  <th>Partenaire</th>
                  {isEditing ? (
                    <>
                      <th>Part %</th>
                      <th>Auto-accept</th>
                      <th>Fact. auto</th>
                    </>
                  ) : (
                    <>
                      <th>Type</th>
                      <th>Courses envoyées</th>
                      <th>Courses reçues</th>
                      <th>CA généré</th>
                      <th>À payer</th>
                      <th>À recevoir</th>
                      <th>Solde</th>
                    </>
                  )}
                  <th>Statut</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {partnerships.map((p) => {
                  const pStats = p.stats || {};
                  const statusBadge = getStatusBadge(p);
                  const balance = pStats.balance || 0;
                  const edits = rowEdits[p.id];
                  return (
                    <tr
                      key={p.id}
                      onClick={() => !isEditing && setSelectedPartnership(p)}
                      className={isEditing ? undefined : partnershipStyles.tableRow}
                    >
                      <td>
                        <strong>{getPartnerName(p)}</strong>
                      </td>
                      {isEditing && edits ? (
                        <>
                          <td>
                            <input
                              type="number"
                              min="1"
                              max="100"
                              value={edits.default_partner_tariff_percent}
                              onChange={(e) =>
                                updateRowEdit(p.id, 'default_partner_tariff_percent', parseFloat(e.target.value) || 90)
                              }
                              onClick={(e) => e.stopPropagation()}
                              className={partnershipStyles.inlineInput}
                            />
                          </td>
                          <td className={partnershipStyles.inlineCheckboxCell}>
                            <div className={partnershipStyles.inlineToggle}>
                              <label className={partnershipStyles.inlineCheckbox} onClick={(e) => e.stopPropagation()}>
                                <input
                                  type="checkbox"
                                  checked={edits.auto_accept}
                                  onChange={(e) => updateRowEdit(p.id, 'auto_accept', e.target.checked)}
                                />
                                <span className={partnershipStyles.inlineSlider} />
                              </label>
                              <span className={partnershipStyles.inlineToggleLabel}>
                                {edits.auto_accept ? 'Oui' : 'Non'}
                              </span>
                            </div>
                          </td>
                          <td className={partnershipStyles.inlineCheckboxCell}>
                            <div className={partnershipStyles.inlineToggle}>
                              <label className={partnershipStyles.inlineCheckbox} onClick={(e) => e.stopPropagation()}>
                                <input
                                  type="checkbox"
                                  checked={edits.auto_invoice}
                                  onChange={(e) => updateRowEdit(p.id, 'auto_invoice', e.target.checked)}
                                />
                                <span className={partnershipStyles.inlineSlider} />
                              </label>
                              <span className={partnershipStyles.inlineToggleLabel}>
                                {edits.auto_invoice ? 'Oui' : 'Non'}
                              </span>
                            </div>
                          </td>
                        </>
                      ) : (
                        <>
                          <td>{getCompanyType(p)}</td>
                          <td>{pStats.sent_transfers || 0}</td>
                          <td>{pStats.received_transfers || 0}</td>
                          <td>{formatCurrency(pStats.total_revenue)}</td>
                          <td>{formatCurrency(pStats.amount_to_pay)}</td>
                          <td>{formatCurrency(pStats.amount_to_receive)}</td>
                          <td
                            className={
                              balance >= 0
                                ? partnershipStyles.balancePositive
                                : partnershipStyles.balanceNegative
                            }
                          >
                            {formatCurrency(balance)}
                          </td>
                        </>
                      )}
                      <td>
                        <span className={statusBadge.className}>
                          {statusBadge.text}
                        </span>
                      </td>
                      <td>
                        <div className={partnershipStyles.actionButtons}>
                          {isEditing && (
                            <button
                              type="button"
                              className={`${partnershipStyles.actionBtn} ${partnershipStyles.actionBtnDelete}`}
                              onClick={(e) => {
                                e.stopPropagation();
                                handleDeleteRow(p.id, getPartnerName(p));
                              }}
                              title="Supprimer"
                            >
                              <FiTrash2 size={15} aria-hidden />
                            </button>
                          )}
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
                            <FiFileText size={16} aria-hidden />
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
      </div>

      {/* Carte 3 : Demandes en attente */}
      <div className={styles.card}>
        <div className={styles.cardHeader}>
          <div className={styles.cardIcon}><FiInbox size={16} /></div>
          <div className={styles.cardHeaderText}>
            <h3 className={styles.cardTitle}>Demandes en attente</h3>
          </div>
        </div>
        {pendingRequests.length === 0 ? (
          <p className={partnershipStyles.textSecondary}>
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
                            <FiCheck size={16} aria-hidden />
                            Accepter
                          </button>
                          <button
                            type="button"
                            className={`${styles.button} ${styles.danger}`}
                            onClick={() => handleRejectRequest(p.id)}
                          >
                            <FiX size={16} aria-hidden />
                            Refuser
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
      </div>

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
                aria-label="Fermer"
              >
                <FiX size={20} aria-hidden />
              </button>
            </div>
            <div className={partnershipStyles.modalBody}>
              <div className={partnershipStyles.detailSection}>
                <h4 className={partnershipStyles.detailSectionTitle}>
                  <FiHome size={16} aria-hidden />
                  Infos générales
                </h4>
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
                    <h4 className={partnershipStyles.detailSectionTitle}>
                      <FiTruck size={16} aria-hidden />
                      Activité opérationnelle
                    </h4>
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
                    <h4 className={partnershipStyles.detailSectionTitle}>
                      <FiDollarSign size={16} aria-hidden />
                      Financier
                    </h4>
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
                        className={
                          (selectedPartnership.stats.balance || 0) >= 0
                            ? partnershipStyles.balancePositive
                            : partnershipStyles.balanceNegative
                        }
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
          className={styles.modalOverlay}
          onClick={() => {
            setShowEditModal(false);
            setEditingPartnership(null);
          }}
        >
          <div
            className={partnershipStyles.modalContentMd}
            onClick={(e) => e.stopPropagation()}
          >
            <div className={partnershipStyles.modalHeaderRow}>
              <div>
                <h2 className={partnershipStyles.modalTitleText}>
                  {getPartnerName(editingPartnership)}
                </h2>
                <p className={partnershipStyles.modalSubtitle}>Modifier les conditions du partenariat</p>
              </div>
              <button
                type="button"
                onClick={() => {
                  setShowEditModal(false);
                  setEditingPartnership(null);
                }}
                className={partnershipStyles.modalCloseBtn}
                aria-label="Fermer"
              >
                <FiX size={20} aria-hidden />
              </button>
            </div>

            <div className={partnershipStyles.modalBodyBlock}>
              <div className={styles.formGroup}>
                <label htmlFor="edit_partner_tariff_percent">
                  Part partenaire (%)
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
                <small className={partnershipStyles.inputHint}>
                  Reduction : {100 - editForm.default_partner_tariff_percent}%
                </small>
              </div>

              <label className={partnershipStyles.checkboxLabel}>
                <input
                  type="checkbox"
                  checked={editForm.auto_accept}
                  onChange={(e) =>
                    setEditForm({
                      ...editForm,
                      auto_accept: e.target.checked,
                    })
                  }
                  className={partnershipStyles.checkboxInput}
                />
                <span>Auto-accepter les transferts entrants</span>
              </label>

              <label className={partnershipStyles.checkboxLabel}>
                <input
                  type="checkbox"
                  checked={editForm.auto_invoice}
                  onChange={(e) =>
                    setEditForm({
                      ...editForm,
                      auto_invoice: e.target.checked,
                    })
                  }
                  className={partnershipStyles.checkboxInput}
                />
                <span>Facturation automatique</span>
              </label>
            </div>

            <div className={partnershipStyles.modalFooterRow}>
              <button
                type="button"
                className={partnershipStyles.dangerLink}
                onClick={async () => {
                  if (
                    !window.confirm(
                      'Supprimer ce partenariat ? Cette action est irreversible.'
                    )
                  ) {
                    return;
                  }
                  try {
                    await apiClient.delete(
                      `/companies/me/partnerships/${editingPartnership.id}`
                    );
                    showSuccess('Partenariat supprime');
                    setShowEditModal(false);
                    setEditingPartnership(null);
                    loadPartnerships();
                    loadStats();
                  } catch (err) {
                    console.error('Erreur suppression partenariat:', err);
                    showError(
                      err?.response?.data?.error ||
                      err?.response?.data?.message ||
                      'Erreur lors de la suppression'
                    );
                  }
                }}
              >
                <FiTrash2 size={13} aria-hidden />
                Supprimer
              </button>
              <div className={partnershipStyles.footerActions}>
                <button
                  type="button"
                  className={`${styles.button} ${styles.secondary}`}
                  onClick={() => {
                    setShowEditModal(false);
                    setEditingPartnership(null);
                  }}
                >
                  Annuler
                </button>
                <button
                  type="button"
                  className={`${styles.button} ${styles.primary}`}
                  onClick={handleUpdatePartnership}
                >
                  Enregistrer
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Modal de génération de décompte */}
      {showStatementModal && (
        <div
          className={styles.modalOverlay}
          onClick={() => {
            setShowStatementModal(false);
            setStatementPartnership(null);
          }}
        >
          <div
            className={partnershipStyles.modalContentMd}
            onClick={(e) => e.stopPropagation()}
          >
            <div className={partnershipStyles.modalHeaderRow}>
              <h2 className={partnershipStyles.modalTitleText}>Générer un décompte</h2>
              <button
                type="button"
                onClick={() => {
                  setShowStatementModal(false);
                  setStatementPartnership(null);
                }}
                className={partnershipStyles.modalCloseBtn}
                aria-label="Fermer"
              >
                <FiX size={20} aria-hidden />
              </button>
            </div>
            <div className={partnershipStyles.modalBodyBlock}>
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
            className={partnershipStyles.modalContentMd}
            onClick={(e) => e.stopPropagation()}
          >
            <div className={partnershipStyles.modalHeaderRow}>
              <h2 className={partnershipStyles.modalTitleText}>Nouveau partenariat</h2>
              <button
                type="button"
                onClick={() => setShowRequestModal(false)}
                className={partnershipStyles.modalCloseBtn}
                aria-label="Fermer"
              >
                <FiX size={20} aria-hidden />
              </button>
            </div>

            <div className={partnershipStyles.modalBodyBlock}>
              {/* Recherche entreprise */}
              <div className={styles.formGroup}>
                <label htmlFor="company_search">Entreprise partenaire</label>
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
                  placeholder="Rechercher par nom, email ou domaine..."
                  className={styles.input}
                />
                {searching && (
                  <small className={partnershipStyles.inputHint}>Recherche en cours...</small>
                )}
                {!searching && lastSearchQuery && searchResults.length === 0 && (
                  <small className={partnershipStyles.searchNoResults}>
                    Aucun resultat pour « {lastSearchQuery} »
                  </small>
                )}
                {searchResults.length > 0 && (
                  <div className={partnershipStyles.searchResultsList}>
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
                        className={partnershipStyles.searchResultItem}
                      >
                        {company.name}
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {requestForm.partner_company_name && (
                <div className={styles.formGroup}>
                  <label>Selectionnee</label>
                  <div className={partnershipStyles.formReadonlyBox}>
                    {requestForm.partner_company_name}
                  </div>
                </div>
              )}

              {/* Conditions */}
              <div className={partnershipStyles.formRow}>
                <div className={styles.formGroup}>
                  <label htmlFor="partner_tariff_percent">Part partenaire (%)</label>
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
                  <small className={partnershipStyles.inputHint}>
                    Reduction : {100 - requestForm.default_partner_tariff_percent}%
                  </small>
                </div>
                <div className={styles.formGroup}>
                  <label htmlFor="payment_terms_days">Delai de paiement</label>
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
                  <small className={partnershipStyles.inputHint}>jours</small>
                </div>
              </div>

              <label className={partnershipStyles.checkboxLabel}>
                <input
                  type="checkbox"
                  checked={requestForm.auto_accept}
                  onChange={(e) =>
                    setRequestForm({
                      ...requestForm,
                      auto_accept: e.target.checked,
                    })
                  }
                  className={partnershipStyles.checkboxInput}
                />
                <span>Auto-accepter les transferts entrants</span>
              </label>
            </div>

            <div className={partnershipStyles.modalFooterRow}>
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
                disabled={!requestForm.partner_company_id}
              >
                Envoyer la demande
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
});

PartnershipsTab.displayName = 'PartnershipsTab';

export default PartnershipsTab;
