import React, { useCallback, useEffect, useState } from 'react';
import apiClient from '../../../../../utils/apiClient';
import { getActiveUser } from '../../../../../utils/webAuthSession';
import styles from '../../CompanySettings.module.css';
import {
  fetchTransportVouchers,
  createTransportVoucher,
  updateTransportVoucher,
  deleteTransportVoucher,
  validateTransportVoucher,
  rejectTransportVoucher,
  uploadTransportVoucherFile,
  deleteTransportVoucherFile,
} from '../../../../../services/transportVoucherService';
import { fetchBillingParties } from '../../../../../services/settingsService';

const STATUS_LABELS = {
  draft: 'Brouillon',
  submitted: 'Soumis',
  validated: 'Validé',
  rejected: 'Rejeté',
  expired: 'Expiré',
};

const STATUS_COLORS = {
  draft: '#6b7280',
  submitted: '#3b82f6',
  validated: '#10b981',
  rejected: '#ef4444',
  expired: '#f59e0b',
};

const TYPE_LABELS = {
  clinic: 'Clinique',
  insurance: 'Assurance',
  other: 'Autre',
};

export default function TransportVouchersSection() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [vouchers, setVouchers] = useState([]);
  const [billingParties, setBillingParties] = useState([]);
  const [clients, setClients] = useState([]);

  // Filtres
  const [filters, setFilters] = useState({
    status: '',
    type: '',
    client_id: '',
  });

  // Formulaire création/édition
  const [showForm, setShowForm] = useState(false);
  const [editingVoucher, setEditingVoucher] = useState(null);
  const [formData, setFormData] = useState({
    client_id: '',
    booking_id: '',
    billing_party_id: '',
    type: 'clinic',
    status: 'draft',
    valid_from: '',
    valid_to: '',
    external_ref: '',
    notes: '',
  });

  // Modal validation/rejet
  const [showValidateModal, setShowValidateModal] = useState(false);
  const [showRejectModal, setShowRejectModal] = useState(false);
  const [selectedVoucher, setSelectedVoucher] = useState(null);
  const [validateData, setValidateData] = useState({ billing_party_id: '', notes: '' });
  const [rejectData, setRejectData] = useState({ reason: '', notes: '' });

  // Upload de fichiers
  const [uploadingFile, setUploadingFile] = useState(false);
  const [expandedVoucher, setExpandedVoucher] = useState(null);

  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      setError('');

      const [vouchersRes, billingPartiesRes] = await Promise.all([
        fetchTransportVouchers(filters),
        fetchBillingParties({ active: true }),
      ]);

      setVouchers(vouchersRes?.data || []);
      setBillingParties(billingPartiesRes?.data || []);

      // Charger les clients pour l'autocomplete
      try {
        const clientsRes = await apiClient.get('/clients', {
          params: { search: '' },
        });
        setClients(clientsRes.data || []);
      } catch (e) {
        console.warn('Erreur chargement clients:', e);
      }
    } catch (e) {
      console.error('[TransportVouchersSection] load failed:', e);
      setError(e?.response?.data?.error || e?.message || 'Erreur lors du chargement');
    } finally {
      setLoading(false);
    }
  }, [filters]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleFilterChange = (key, value) => {
    setFilters((prev) => ({ ...prev, [key]: value }));
  };

  const handleFormChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleCreate = () => {
    setEditingVoucher(null);
    setFormData({
      client_id: '',
      booking_id: '',
      billing_party_id: '',
      type: 'clinic',
      status: 'draft',
      valid_from: '',
      valid_to: '',
      external_ref: '',
      notes: '',
    });
    setShowForm(true);
  };

  const handleEdit = (voucher) => {
    setEditingVoucher(voucher);
    setFormData({
      client_id: voucher.client_id?.toString() || '',
      booking_id: voucher.booking_id?.toString() || '',
      billing_party_id: voucher.billing_party_id?.toString() || '',
      type: voucher.type || 'clinic',
      status: voucher.status || 'draft',
      valid_from: voucher.valid_from ? voucher.valid_from.split('T')[0] : '',
      valid_to: voucher.valid_to ? voucher.valid_to.split('T')[0] : '',
      external_ref: voucher.external_ref || '',
      notes: voucher.notes || '',
    });
    setShowForm(true);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      setError('');
      setMessage('');

      const payload = {
        ...formData,
        client_id: parseInt(formData.client_id),
        booking_id: formData.booking_id ? parseInt(formData.booking_id) : null,
        billing_party_id: formData.billing_party_id ? parseInt(formData.billing_party_id) : null,
        valid_from: formData.valid_from ? `${formData.valid_from}T00:00:00Z` : null,
        valid_to: formData.valid_to ? `${formData.valid_to}T23:59:59Z` : null,
      };

      if (editingVoucher) {
        await updateTransportVoucher(editingVoucher.id, payload);
        setMessage('Bon mis à jour avec succès');
      } else {
        // Récupérer company_id depuis le token/utilisateur
        const user = getActiveUser() || {};
        payload.company_id = user.company_id || user.companyId;
        await createTransportVoucher(payload);
        setMessage('Bon créé avec succès');
      }

      setShowForm(false);
      await loadData();
    } catch (e) {
      console.error('[TransportVouchersSection] submit failed:', e);
      setError(e?.response?.data?.error || e?.message || 'Erreur lors de la sauvegarde');
    }
  };

  const handleDelete = async (voucher) => {
    if (!window.confirm('Êtes-vous sûr de vouloir supprimer ce bon ?')) return;

    try {
      setError('');
      await deleteTransportVoucher(voucher.id);
      setMessage('Bon supprimé avec succès');
      await loadData();
    } catch (e) {
      console.error('[TransportVouchersSection] delete failed:', e);
      setError(e?.response?.data?.error || e?.message || 'Erreur lors de la suppression');
    }
  };

  const handleValidate = async () => {
    try {
      setError('');
      await validateTransportVoucher(selectedVoucher.id, validateData);
      setMessage('Bon validé avec succès');
      setShowValidateModal(false);
      setValidateData({ billing_party_id: '', notes: '' });
      await loadData();
    } catch (e) {
      console.error('[TransportVouchersSection] validate failed:', e);
      setError(e?.response?.data?.error || e?.message || 'Erreur lors de la validation');
    }
  };

  const handleReject = async () => {
    if (!rejectData.reason.trim()) {
      setError('La raison du rejet est obligatoire');
      return;
    }

    try {
      setError('');
      await rejectTransportVoucher(selectedVoucher.id, rejectData);
      setMessage('Bon rejeté avec succès');
      setShowRejectModal(false);
      setRejectData({ reason: '', notes: '' });
      await loadData();
    } catch (e) {
      console.error('[TransportVouchersSection] reject failed:', e);
      setError(e?.response?.data?.error || e?.message || 'Erreur lors du rejet');
    }
  };

  const handleFileUpload = async (voucherId, file) => {
    try {
      setUploadingFile(true);
      setError('');
      await uploadTransportVoucherFile(voucherId, file);
      setMessage('Fichier uploadé avec succès');
      await loadData();
    } catch (e) {
      console.error('[TransportVouchersSection] file upload failed:', e);
      setError(e?.response?.data?.error || e?.message || 'Erreur lors de l\'upload');
    } finally {
      setUploadingFile(false);
    }
  };

  const handleFileDelete = async (voucherId, fileId) => {
    if (!window.confirm('Êtes-vous sûr de vouloir supprimer ce fichier ?')) return;

    try {
      setError('');
      await deleteTransportVoucherFile(voucherId, fileId);
      setMessage('Fichier supprimé avec succès');
      await loadData();
    } catch (e) {
      console.error('[TransportVouchersSection] file delete failed:', e);
      setError(e?.response?.data?.error || e?.message || 'Erreur lors de la suppression');
    }
  };

  if (loading) {
    return <div className={styles.loading}>Chargement...</div>;
  }

  return (
    <div className={styles.section}>
      <h2>🎫 Bons de transport</h2>

      {error && <div className={styles.error}>{error}</div>}
      {message && <div className={styles.success}>{message}</div>}

      {/* Filtres */}
      <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
        <select
          value={filters.status}
          onChange={(e) => handleFilterChange('status', e.target.value)}
          style={{ padding: '0.5rem', borderRadius: '4px', border: '1px solid #ddd' }}
        >
          <option value="">Tous les statuts</option>
          {Object.entries(STATUS_LABELS).map(([key, label]) => (
            <option key={key} value={key}>
              {label}
            </option>
          ))}
        </select>

        <select
          value={filters.type}
          onChange={(e) => handleFilterChange('type', e.target.value)}
          style={{ padding: '0.5rem', borderRadius: '4px', border: '1px solid #ddd' }}
        >
          <option value="">Tous les types</option>
          {Object.entries(TYPE_LABELS).map(([key, label]) => (
            <option key={key} value={key}>
              {label}
            </option>
          ))}
        </select>

        <button onClick={handleCreate} className={styles.button}>
          + Créer un bon
        </button>
      </div>

      {/* Formulaire création/édition */}
      {showForm && (
        <div
          style={{
            background: '#f9fafb',
            padding: '1rem',
            borderRadius: '8px',
            marginBottom: '1rem',
            border: '1px solid #e5e7eb',
          }}
        >
          <h3>{editingVoucher ? 'Modifier le bon' : 'Nouveau bon de transport'}</h3>
          <form onSubmit={handleSubmit}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              <div>
                <label>Client *</label>
                <select name="client_id" value={formData.client_id} onChange={handleFormChange} required>
                  <option value="">Sélectionner un client</option>
                  {clients.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.first_name} {c.last_name} (#{c.id})
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label>Type *</label>
                <select name="type" value={formData.type} onChange={handleFormChange} required>
                  {Object.entries(TYPE_LABELS).map(([key, label]) => (
                    <option key={key} value={key}>
                      {label}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label>Booking ID (optionnel)</label>
                <input
                  type="number"
                  name="booking_id"
                  value={formData.booking_id}
                  onChange={handleFormChange}
                  placeholder="ID de la course"
                />
              </div>

              <div>
                <label>Payeur (optionnel)</label>
                <select name="billing_party_id" value={formData.billing_party_id} onChange={handleFormChange}>
                  <option value="">Aucun</option>
                  {billingParties.map((bp) => (
                    <option key={bp.id} value={bp.id}>
                      {bp.legal_name} (#{bp.id})
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label>Date de début (optionnel)</label>
                <input
                  type="date"
                  name="valid_from"
                  value={formData.valid_from}
                  onChange={handleFormChange}
                />
              </div>

              <div>
                <label>Date de fin (optionnel)</label>
                <input
                  type="date"
                  name="valid_to"
                  value={formData.valid_to}
                  onChange={handleFormChange}
                />
              </div>

              <div style={{ gridColumn: '1 / -1' }}>
                <label>Référence externe (optionnel)</label>
                <input
                  type="text"
                  name="external_ref"
                  value={formData.external_ref}
                  onChange={handleFormChange}
                  placeholder="N° dossier, n° sinistre..."
                  maxLength={255}
                />
              </div>

              <div style={{ gridColumn: '1 / -1' }}>
                <label>Notes (optionnel)</label>
                <textarea
                  name="notes"
                  value={formData.notes}
                  onChange={handleFormChange}
                  rows={3}
                  placeholder="Notes additionnelles..."
                />
              </div>
            </div>

            <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem' }}>
              <button type="submit" className={styles.button}>
                {editingVoucher ? 'Mettre à jour' : 'Créer'}
              </button>
              <button
                type="button"
                onClick={() => setShowForm(false)}
                style={{ background: '#6b7280', color: 'white', padding: '0.5rem 1rem', border: 'none', borderRadius: '4px', cursor: 'pointer' }}
              >
                Annuler
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Liste des bons */}
      {vouchers.length === 0 ? (
        <p className={styles.mutedText}>Aucun bon de transport trouvé.</p>
      ) : (
        <div className={styles.tableContainer}>
          <table className={styles.activityTable}>
            <thead>
              <tr>
                <th>ID</th>
                <th>Client</th>
                <th>Type</th>
                <th>Statut</th>
                <th>Période</th>
                <th>Référence</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {vouchers.map((v) => (
                <React.Fragment key={v.id}>
                  <tr>
                    <td>#{v.id}</td>
                    <td>Client #{v.client_id}</td>
                    <td>{TYPE_LABELS[v.type] || v.type}</td>
                    <td>
                      <span
                        style={{
                          padding: '0.25rem 0.5rem',
                          borderRadius: '4px',
                          fontSize: '0.875rem',
                          backgroundColor: `${STATUS_COLORS[v.status]}20`,
                          color: STATUS_COLORS[v.status],
                          fontWeight: '500',
                        }}
                      >
                        {STATUS_LABELS[v.status] || v.status}
                      </span>
                    </td>
                    <td>
                      {v.valid_from && v.valid_to
                        ? `${new Date(v.valid_from).toLocaleDateString()} - ${new Date(v.valid_to).toLocaleDateString()}`
                        : v.valid_from
                          ? `À partir du ${new Date(v.valid_from).toLocaleDateString()}`
                          : '—'}
                    </td>
                    <td>{v.external_ref || '—'}</td>
                    <td>
                      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                        <button
                          onClick={() => setExpandedVoucher(expandedVoucher === v.id ? null : v.id)}
                          style={{ padding: '0.25rem 0.5rem', fontSize: '0.875rem', background: '#6b7280', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}
                        >
                          {expandedVoucher === v.id ? '▼' : '▶'} Détails
                        </button>
                        <button
                          onClick={() => handleEdit(v)}
                          style={{ padding: '0.25rem 0.5rem', fontSize: '0.875rem', background: '#3b82f6', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}
                        >
                          Modifier
                        </button>
                        {v.status === 'draft' && (
                          <button
                            onClick={() => handleDelete(v)}
                            style={{ padding: '0.25rem 0.5rem', fontSize: '0.875rem', background: '#ef4444', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}
                          >
                            Supprimer
                          </button>
                        )}
                        {v.status === 'submitted' && (
                          <>
                            <button
                              onClick={() => {
                                setSelectedVoucher(v);
                                setShowValidateModal(true);
                              }}
                              style={{ padding: '0.25rem 0.5rem', fontSize: '0.875rem', background: '#10b981', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}
                            >
                              Valider
                            </button>
                            <button
                              onClick={() => {
                                setSelectedVoucher(v);
                                setShowRejectModal(true);
                              }}
                              style={{ padding: '0.25rem 0.5rem', fontSize: '0.875rem', background: '#ef4444', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}
                            >
                              Rejeter
                            </button>
                          </>
                        )}
                      </div>
                    </td>
                  </tr>
                  {expandedVoucher === v.id && (
                    <tr>
                      <td colSpan="7" style={{ padding: '1rem', background: '#f9fafb' }}>
                        <div>
                          <h4>Détails du bon #{v.id}</h4>
                          <div style={{ marginBottom: '1rem' }}>
                            <p><strong>Notes:</strong> {v.notes || 'Aucune'}</p>
                            {v.booking_id && <p><strong>Booking ID:</strong> #{v.booking_id}</p>}
                            {v.billing_party_name && <p><strong>Payeur:</strong> {v.billing_party_name}</p>}
                          </div>
                          
                          <div style={{ marginBottom: '1rem' }}>
                            <h5>Fichiers attachés ({v.files?.length || 0})</h5>
                            {v.files && v.files.length > 0 ? (
                              <ul style={{ listStyle: 'none', padding: 0 }}>
                                {v.files.map((f) => (
                                  <li key={f.id} style={{ marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                    <a href={f.file_url} target="_blank" rel="noopener noreferrer" style={{ color: '#3b82f6' }}>
                                      📎 {f.filename}
                                    </a>
                                    <button
                                      onClick={() => handleFileDelete(v.id, f.id)}
                                      style={{ padding: '0.25rem 0.5rem', fontSize: '0.75rem', background: '#ef4444', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}
                                    >
                                      Supprimer
                                    </button>
                                  </li>
                                ))}
                              </ul>
                            ) : (
                              <p style={{ color: '#6b7280' }}>Aucun fichier attaché</p>
                            )}
                            
                            <div style={{ marginTop: '1rem' }}>
                              <label>
                                <input
                                  type="file"
                                  onChange={(e) => {
                                    const file = e.target.files?.[0];
                                    if (file) {
                                      handleFileUpload(v.id, file);
                                      e.target.value = ''; // Reset input
                                    }
                                  }}
                                  disabled={uploadingFile}
                                  accept=".pdf,.png,.jpg,.jpeg,.gif,.webp"
                                  style={{ marginRight: '0.5rem' }}
                                />
                                {uploadingFile ? 'Upload en cours...' : 'Ajouter un fichier'}
                              </label>
                              <small style={{ display: 'block', marginTop: '0.25rem', color: '#6b7280' }}>
                                Formats acceptés: PDF, PNG, JPG, GIF, WEBP (max 10 Mo)
                              </small>
                            </div>
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Modal validation */}
      {showValidateModal && selectedVoucher && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0,0,0,0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 'var(--z-modal-app)',
            padding: 'var(--modal-overlay-padding)',
            boxSizing: 'border-box',
          }}
          onClick={() => setShowValidateModal(false)}
        >
          <div
            style={{
              background: 'white',
              padding: '2rem',
              borderRadius: '8px',
              maxWidth: '500px',
              width: '90%',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <h3>Valider le bon #{selectedVoucher.id}</h3>
            <div style={{ marginBottom: '1rem' }}>
              <label>Payeur (optionnel)</label>
              <select
                value={validateData.billing_party_id}
                onChange={(e) => setValidateData({ ...validateData, billing_party_id: e.target.value })}
                style={{ width: '100%', padding: '0.5rem', marginTop: '0.5rem' }}
              >
                <option value="">Aucun</option>
                {billingParties.map((bp) => (
                  <option key={bp.id} value={bp.id}>
                    {bp.legal_name} (#{bp.id})
                  </option>
                ))}
              </select>
            </div>
            <div style={{ marginBottom: '1rem' }}>
              <label>Notes (optionnel)</label>
              <textarea
                value={validateData.notes}
                onChange={(e) => setValidateData({ ...validateData, notes: e.target.value })}
                rows={3}
                style={{ width: '100%', padding: '0.5rem', marginTop: '0.5rem' }}
              />
            </div>
            <div style={{ display: 'flex', gap: '1rem' }}>
              <button onClick={handleValidate} className={styles.button}>
                Valider
              </button>
              <button
                onClick={() => setShowValidateModal(false)}
                style={{ background: '#6b7280', color: 'white', padding: '0.5rem 1rem', border: 'none', borderRadius: '4px', cursor: 'pointer' }}
              >
                Annuler
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Modal rejet */}
      {showRejectModal && selectedVoucher && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0,0,0,0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 'var(--z-modal-app)',
            padding: 'var(--modal-overlay-padding)',
            boxSizing: 'border-box',
          }}
          onClick={() => setShowRejectModal(false)}
        >
          <div
            style={{
              background: 'white',
              padding: '2rem',
              borderRadius: '8px',
              maxWidth: '500px',
              width: '90%',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <h3>Rejeter le bon #{selectedVoucher.id}</h3>
            <div style={{ marginBottom: '1rem' }}>
              <label>Raison du rejet *</label>
              <textarea
                value={rejectData.reason}
                onChange={(e) => setRejectData({ ...rejectData, reason: e.target.value })}
                rows={3}
                required
                style={{ width: '100%', padding: '0.5rem', marginTop: '0.5rem' }}
                placeholder="Expliquez pourquoi ce bon est rejeté..."
              />
            </div>
            <div style={{ marginBottom: '1rem' }}>
              <label>Notes supplémentaires (optionnel)</label>
              <textarea
                value={rejectData.notes}
                onChange={(e) => setRejectData({ ...rejectData, notes: e.target.value })}
                rows={2}
                style={{ width: '100%', padding: '0.5rem', marginTop: '0.5rem' }}
              />
            </div>
            <div style={{ display: 'flex', gap: '1rem' }}>
              <button onClick={handleReject} className={styles.button} style={{ background: '#ef4444' }}>
                Rejeter
              </button>
              <button
                onClick={() => setShowRejectModal(false)}
                style={{ background: '#6b7280', color: 'white', padding: '0.5rem 1rem', border: 'none', borderRadius: '4px', cursor: 'pointer' }}
              >
                Annuler
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
