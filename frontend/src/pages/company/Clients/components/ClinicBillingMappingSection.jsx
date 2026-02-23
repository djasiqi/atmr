// frontend/src/pages/company/Clients/components/ClinicBillingMappingSection.jsx
import React, { useEffect, useState, useCallback } from 'react';
import {
  fetchBillingParties,
  fetchClinicBillingMapping,
  upsertClinicBillingMapping,
  createBillingParty,
  updateBillingParty,
} from '../../../../services/settingsService';
import { createCompanyForInstitutionClient } from '../../../../services/companyService';
import AddressAutocomplete from '../../../../components/common/AddressAutocomplete';
import { parseAddressWithEstablishment } from '../../../../utils/addressParser';
import styles from './ClientEditForm.module.css';

/** Parse une adresse brute et retourne { billing_street, billing_postcode, billing_city }. */
function parseAddressToFields(address) {
  if (!address || !String(address).trim()) {
    return { billing_street: '', billing_postcode: '', billing_city: '' };
  }
  const parsed = parseAddressWithEstablishment(String(address).trim(), {});
  const billing_street =
    parsed.street && parsed.streetNumber
      ? `${parsed.street} ${parsed.streetNumber}`.trim()
      : (parsed.street || '').trim();
  return {
    billing_street,
    billing_postcode: (parsed.postcode || '').trim(),
    billing_city: (parsed.city || '').trim(),
  };
}

function normalizeBillingPartyName(value) {
  if (!value) return '';
  return String(value)
    .toLowerCase()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[’`´']/g, "'")
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

/**
 * Section pour gérer le mapping clinique → billing party
 * Affiche le mapping actuel et permet de le créer/modifier
 */
const ClinicBillingMappingSection = ({
  clinicCompanyId: initialClinicCompanyId,
  clientId,
  clinicCompanyName,
  onCompanyCreated,
  onClinicCompanyIdChange,
}) => {
  const [clinicCompanyId, setClinicCompanyId] = useState(initialClinicCompanyId);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [message, setMessage] = useState('');
  const [billingParties, setBillingParties] = useState([]);
  const [existingMapping, setExistingMapping] = useState(null);
  const [selectedBillingPartyId, setSelectedBillingPartyId] = useState('');
  const [creatingCompany, setCreatingCompany] = useState(false);
  const [showCreateBillingPartyForm, setShowCreateBillingPartyForm] = useState(false);
  const [creatingBillingParty, setCreatingBillingParty] = useState(false);
  const [showEditBillingPartyForm, setShowEditBillingPartyForm] = useState(false);
  const [updatingBillingParty, setUpdatingBillingParty] = useState(false);
  const [newBillingPartyData, setNewBillingPartyData] = useState({
    display_name: clinicCompanyName || '',
    billing_address: '',
    billing_street: '',
    billing_postcode: '',
    billing_city: '',
    contact_email: '',
    contact_phone: '',
  });
  const [editBillingPartyData, setEditBillingPartyData] = useState({
    display_name: '',
    billing_address: '',
    billing_street: '',
    billing_postcode: '',
    billing_city: '',
    contact_email: '',
    contact_phone: '',
  });

  // Mettre à jour clinicCompanyId si la prop change
  useEffect(() => {
    // Si initialClinicCompanyId change (après rechargement), mettre à jour l'état local
    if (initialClinicCompanyId) {
      setClinicCompanyId(initialClinicCompanyId);
    }
    // Note: on ne réinitialise pas clinicCompanyId si initialClinicCompanyId devient null
    // car cela pourrait être dû à un rechargement en cours
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialClinicCompanyId]);

  // Charger les données
  const loadData = useCallback(async () => {
    if (!clinicCompanyId) {
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);
      const [m, bp] = await Promise.all([
        fetchClinicBillingMapping(clinicCompanyId),
        fetchBillingParties({ active: true }),
      ]);

      // Le mapping peut être null si aucun mapping n'existe
      setExistingMapping(m?.data || null);
      setBillingParties(
        Array.isArray(bp?.data) ? bp.data : bp?.data?.data || []
      );
    } catch (err) {
      console.error('[ClinicBillingMappingSection] load failed:', err);
      setError(
        err?.response?.data?.error ||
          err?.message ||
          'Erreur lors du chargement'
      );
    } finally {
      setLoading(false);
    }
  }, [clinicCompanyId]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Initialiser le billing party sélectionné si un mapping existe
  useEffect(() => {
    if (existingMapping) {
      setSelectedBillingPartyId(
        String(existingMapping.billing_party_id || '')
      );
    } else {
      setSelectedBillingPartyId('');
    }
  }, [existingMapping]);

  const billingPartyOptions = React.useMemo(() => {
    const byKey = new Map();
    const mappedId = Number(existingMapping?.billing_party_id || 0);
    const entries = Array.isArray(billingParties) ? billingParties : [];
    for (const bp of entries) {
      if (!bp?.id) continue;
      const key = `${bp.type || 'other'}:${normalizeBillingPartyName(bp.display_name)}`;
      const previous = byKey.get(key);
      if (!previous) {
        byKey.set(key, bp);
        continue;
      }
      // Priorité au BillingParty déjà mappé; sinon garder l'id le plus récent.
      if (previous.id === mappedId) continue;
      if (bp.id === mappedId || bp.id > previous.id) {
        byKey.set(key, bp);
      }
    }
    return Array.from(byKey.values()).sort((a, b) =>
      String(a.display_name || '').localeCompare(String(b.display_name || ''), 'fr', {
        sensitivity: 'base',
      })
    );
  }, [billingParties, existingMapping?.billing_party_id]);

  const handleCreateCompany = async () => {
    try {
      setCreatingCompany(true);
      setError(null);
      const response = await createCompanyForInstitutionClient(clientId);
      const company = response?.data || response;
      
      if (company && company.id) {
        console.log('Company creee:', company);
        
        // Mettre à jour l'état local immédiatement
        setClinicCompanyId(company.id);
        
        // Notifier le parent du changement de clinicCompanyId
        if (onClinicCompanyIdChange) {
          onClinicCompanyIdChange(company.id);
        }
        
        // Si un BillingParty et un mapping ont été créés automatiquement
        if (company.billing_party_id) {
          setMessage(
            `Tout est configure automatiquement. Company, BillingParty et Mapping crees. ` +
            `Le destinataire "${company.billing_party_name || '—'}" est déjà configuré pour cette clinique.`
          );
          // Pré-sélectionner le BillingParty créé automatiquement
          setSelectedBillingPartyId(String(company.billing_party_id));
        } else {
          setMessage('Company creee avec succes. Rechargement des donnees...');
        }
        
        // Appeler le callback pour recharger les données du client
        if (onCompanyCreated) {
          await onCompanyCreated(company);
        }
        
        // Recharger les données immédiatement avec le nouvel ID
        // Puis recharger à nouveau après un délai pour s'assurer que le backend a tout mis à jour
        await loadData();
        setTimeout(async () => {
          await loadData();
        }, 2000);
      } else {
        setError('Company créée mais données incomplètes reçues');
      }
    } catch (err) {
      console.error('[ClinicBillingMappingSection] create company failed:', err);
      setError(
        err?.response?.data?.error ||
          err?.message ||
          'Erreur lors de la création de la Company'
      );
    } finally {
      setCreatingCompany(false);
    }
  };

  const handleCreateBillingParty = async () => {
    try {
      setCreatingBillingParty(true);
      setError(null);

      if (!newBillingPartyData.display_name.trim()) {
        setError('Le nom du destinataire est requis');
        return;
      }

      const billingAddress = (newBillingPartyData.billing_address || '').trim();
      if (!billingAddress) {
        setError('L\'adresse de facturation est requise');
        return;
      }

      // Créer le BillingParty
      const payload = {
        display_name: newBillingPartyData.display_name.trim(),
        type: 'clinic',
        billing_address: billingAddress,
        contact_email: newBillingPartyData.contact_email.trim() || null,
        contact_phone: newBillingPartyData.contact_phone.trim() || null,
        is_active: true,
      };

      const response = await createBillingParty(payload);
      const newBillingParty = response?.data?.data || response?.data;

      if (!newBillingParty || !newBillingParty.id) {
        throw new Error('BillingParty créé mais données incomplètes');
      }

      // Recharger la liste des BillingParties
      await loadData();

      // Sélectionner automatiquement le nouveau BillingParty
      setSelectedBillingPartyId(String(newBillingParty.id));

      // Créer automatiquement le mapping si clinicCompanyId existe
      if (clinicCompanyId) {
        await upsertClinicBillingMapping({
          clinic_company_id: Number(clinicCompanyId),
          billing_party_id: newBillingParty.id,
          is_active: true,
        });

        // Recharger le mapping
        await loadData();
        setMessage('Destinataire cree et mapping configure avec succes');
        setShowCreateBillingPartyForm(false);
      } else {
        setMessage('Destinataire cree avec succes. Selectionnez-le ci-dessous.');
        setShowCreateBillingPartyForm(false);
      }

      setTimeout(() => setMessage(''), 5000);
    } catch (err) {
      console.error('[ClinicBillingMappingSection] create billing party failed:', err);
      setError(
        err?.response?.data?.error ||
          err?.message ||
          'Erreur lors de la création du destinataire'
      );
    } finally {
      setCreatingBillingParty(false);
    }
  };

  const handleUpdateBillingParty = async () => {
    try {
      setUpdatingBillingParty(true);
      setError(null);

      if (!existingMapping || !existingMapping.billing_party_id) {
        setError('Aucun destinataire de facturation à modifier');
        return;
      }

      if (!editBillingPartyData.display_name.trim()) {
        setError('Le nom du destinataire est requis');
        return;
      }

      const billingAddress = (editBillingPartyData.billing_address || '').trim();
      if (!billingAddress) {
        setError('L\'adresse de facturation est requise');
        return;
      }

      await updateBillingParty(existingMapping.billing_party_id, {
        display_name: editBillingPartyData.display_name.trim(),
        billing_address: billingAddress,
        contact_email: editBillingPartyData.contact_email.trim() || null,
        contact_phone: editBillingPartyData.contact_phone.trim() || null,
      });

      // Recharger les données
      await loadData();
      setShowEditBillingPartyForm(false);
      setMessage('Destinataire de facturation mis a jour avec succes');
      setTimeout(() => setMessage(''), 5000);
    } catch (err) {
      console.error('[ClinicBillingMappingSection] update billing party failed:', err);
      setError(
        err?.response?.data?.error ||
          err?.message ||
          'Erreur lors de la mise à jour du destinataire'
      );
    } finally {
      setUpdatingBillingParty(false);
    }
  };

  const handleEditBillingParty = () => {
    const mappedBillingParty = billingParties.find(
      (bp) => bp.id === existingMapping.billing_party_id
    );
    if (mappedBillingParty) {
      const address = mappedBillingParty.billing_address || '';
      const fields = parseAddressToFields(address);
      setEditBillingPartyData({
        display_name: mappedBillingParty.display_name || '',
        billing_address: address,
        billing_street: fields.billing_street,
        billing_postcode: fields.billing_postcode,
        billing_city: fields.billing_city,
        contact_email: mappedBillingParty.contact_email || '',
        contact_phone: mappedBillingParty.contact_phone || '',
      });
      setShowEditBillingPartyForm(true);
    }
  };

  const handleSave = async () => {
    try {
      setMessage('');
      setError(null);

      if (!clinicCompanyId) {
        setError(
          'Impossible de créer le mapping : aucune entreprise associée à cette clinique.'
        );
        return;
      }

      const bpId = Number(selectedBillingPartyId);
      if (!bpId) {
        setError('Veuillez sélectionner un destinataire de facturation.');
        return;
      }

      await upsertClinicBillingMapping({
        clinic_company_id: Number(clinicCompanyId),
        billing_party_id: bpId,
        is_active: true,
      });

      // Recharger le mapping
      await loadData();
      setMessage('Mapping enregistre avec succes');
      setTimeout(() => setMessage(''), 3000);
    } catch (err) {
      console.error('[ClinicBillingMappingSection] save failed:', err);
      setError(
        err?.response?.data?.error ||
          err?.message ||
          'Erreur lors de la sauvegarde'
      );
    }
  };

  // Debug: afficher les valeurs pour comprendre le problème
  React.useEffect(() => {
    console.log('[ClinicBillingMappingSection] Debug:', {
      initialClinicCompanyId,
      clinicCompanyId,
      clientId,
      clinicCompanyName,
    });
  }, [initialClinicCompanyId, clinicCompanyId, clientId, clinicCompanyName]);

  if (!clinicCompanyId) {
    return (
      <div className={styles.accordionContent}>
        <p className={styles.hint}>
          Pour configurer le destinataire de facturation, cette clinique doit
          être enregistrée comme Company dans le système. La clinique est le payeur,
          et le mapping permet de définir où envoyer la facture (adresse, email, etc.).
        </p>
        <div style={{ marginTop: '1rem' }}>
          <button
            type="button"
            onClick={handleCreateCompany}
            className={styles.saveButton}
            disabled={creatingCompany}
            style={{ width: 'auto', padding: '0.5rem 1rem' }}
          >
            {creatingCompany
              ? 'Création...'
              : 'Enregistrer cette clinique comme Company (payeur)'}
          </button>
        </div>
        {clinicCompanyName && (
          <p className={styles.hint} style={{ marginTop: '0.5rem' }}>
            Clinique : <strong>{clinicCompanyName}</strong>
          </p>
        )}
        {/* Debug info */}
        {process.env.NODE_ENV === 'development' && (
          <p className={styles.hint} style={{ marginTop: '0.5rem', fontSize: '0.75rem', color: '#666' }}>
            Debug: initialClinicCompanyId={String(initialClinicCompanyId || 'null')}, 
            clinicCompanyId={String(clinicCompanyId || 'null')}
          </p>
        )}
      </div>
    );
  }

  if (loading) {
    return (
      <div className={styles.accordionContent}>
        <p>Chargement...</p>
      </div>
    );
  }

  // Si un mapping existe déjà, afficher un message informatif
  if (existingMapping && existingMapping.billing_party_id) {
    const mappedBillingParty = billingParties.find(
      (bp) => bp.id === existingMapping.billing_party_id
    );

    return (
      <div className={styles.accordionContent}>
        {error && <div className={styles.error}>{error}</div>}
        {message && <div className={styles.messageSuccess}>{message}</div>}

        <div className={styles.successBadge}>
          <p className={styles.successBadgeTitle}>
            Mapping configure
          </p>
          <p className={styles.successBadgeText}>
            Destinataire de facturation déjà configuré pour cette clinique.
          </p>
        </div>

        {!showEditBillingPartyForm ? (
          <>
            <div className={styles.formGroup}>
              <label htmlFor="billing_party_mapping" className={styles.label}>
                Destinataire de facturation actuel
              </label>
              <div className={styles.infoBox}>
                <span className={styles.infoBoxTitle}>
                  {existingMapping.billing_party_name || '—'}
                </span>
                {mappedBillingParty && (
                  <span className={styles.infoBoxSubtitle}>
                    ({mappedBillingParty.type})
                  </span>
                )}
              </div>
              {mappedBillingParty && (
                <div style={{ marginTop: '0.75rem' }}>
                  {mappedBillingParty.billing_address && (
                    <p className={styles.hint}>
                      <strong>Adresse :</strong> {mappedBillingParty.billing_address}
                    </p>
                  )}
                  {mappedBillingParty.contact_email && (
                    <p className={styles.hint}>
                      <strong>Email :</strong>{' '}
                      <a href={`mailto:${mappedBillingParty.contact_email}`}>
                        {mappedBillingParty.contact_email}
                      </a>
                    </p>
                  )}
                  {mappedBillingParty.contact_phone && (
                    <p className={styles.hint}>
                      <strong>Téléphone :</strong>{' '}
                      <a href={`tel:${mappedBillingParty.contact_phone}`}>
                        {mappedBillingParty.contact_phone}
                      </a>
                    </p>
                  )}
                </div>
              )}
              {clinicCompanyName && (
                <p className={styles.hint} style={{ marginTop: '0.5rem' }}>
                  Clinique : <strong>{clinicCompanyName}</strong>
                </p>
              )}
              <small className={styles.hint}>
                La clinique est le payeur. Les factures seront envoyées à ce destinataire.
              </small>
            </div>

            <div className={styles.formActions}>
              <button
                type="button"
                onClick={handleEditBillingParty}
                className={styles.saveButton}
              >
                Modifier les informations
              </button>
            </div>
          </>
        ) : (
          <div>
            <div className={styles.sectionHeader}>
              <h4 className={styles.sectionTitle}>
                Modifier le destinataire de facturation
              </h4>
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="edit_bp_display_name" className={styles.label}>
                Nom du destinataire *
              </label>
              <input
                id="edit_bp_display_name"
                type="text"
                className={styles.input}
                value={editBillingPartyData.display_name}
                onChange={(e) =>
                  setEditBillingPartyData({
                    ...editBillingPartyData,
                    display_name: e.target.value,
                  })
                }
                placeholder="Ex: Service facturation - Clinique des Grangettes"
              />
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="edit_bp_billing_address" className={styles.label}>
                Adresse de facturation *
              </label>
              <AddressAutocomplete
                name="edit_bp_billing_address"
                value={editBillingPartyData.billing_address}
                onChange={(e) => {
                  const v = e.target.value;
                  const fields = parseAddressToFields(v);
                  setEditBillingPartyData({
                    ...editBillingPartyData,
                    billing_address: v,
                    billing_street: fields.billing_street,
                    billing_postcode: fields.billing_postcode,
                    billing_city: fields.billing_city,
                  });
                }}
                onSelect={(item) => {
                  const fullAddress = item.label || '';
                  const parsed = parseAddressWithEstablishment(fullAddress, item);
                  const street =
                    parsed.street && parsed.streetNumber
                      ? `${parsed.street} ${parsed.streetNumber}`.trim()
                      : parsed.street || '';
                  setEditBillingPartyData({
                    ...editBillingPartyData,
                    billing_address: fullAddress,
                    billing_street: street,
                    billing_postcode: parsed.postcode || '',
                    billing_city: parsed.city || '',
                  });
                }}
                placeholder="Adresse complète (rue, code postal, ville)"
                disabled={updatingBillingParty}
              />
            </div>

            <div className={styles.formRow}>
              <div className={styles.formGroup}>
                <label htmlFor="edit_bp_billing_street" className={styles.label}>
                  Rue et numéro *
                </label>
                <input
                  id="edit_bp_billing_street"
                  type="text"
                  className={styles.input}
                  value={editBillingPartyData.billing_street}
                  readOnly
                  placeholder="Rempli automatiquement depuis l'adresse"
                />
              </div>
              <div className={styles.formGroup}>
                <label htmlFor="edit_bp_billing_postcode" className={styles.label}>
                  Code postal *
                </label>
                <input
                  id="edit_bp_billing_postcode"
                  type="text"
                  className={styles.input}
                  value={editBillingPartyData.billing_postcode}
                  readOnly
                  placeholder="Rempli automatiquement depuis l'adresse"
                  maxLength={10}
                />
              </div>
              <div className={styles.formGroup}>
                <label htmlFor="edit_bp_billing_city" className={styles.label}>
                  Ville *
                </label>
                <input
                  id="edit_bp_billing_city"
                  type="text"
                  className={styles.input}
                  value={editBillingPartyData.billing_city}
                  readOnly
                  placeholder="Rempli automatiquement depuis l'adresse"
                />
              </div>
            </div>

            <div className={styles.formRowTwo}>
              <div className={styles.formGroup}>
                <label htmlFor="edit_bp_contact_email" className={styles.label}>
                  Email de contact
                </label>
                <input
                  id="edit_bp_contact_email"
                  type="email"
                  className={styles.input}
                  value={editBillingPartyData.contact_email}
                  onChange={(e) =>
                    setEditBillingPartyData({
                      ...editBillingPartyData,
                      contact_email: e.target.value,
                    })
                  }
                  placeholder="facturation@clinique.ch"
                />
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="edit_bp_contact_phone" className={styles.label}>
                  Téléphone de contact
                </label>
                <input
                  id="edit_bp_contact_phone"
                  type="tel"
                  className={styles.input}
                  value={editBillingPartyData.contact_phone}
                  onChange={(e) =>
                    setEditBillingPartyData({
                      ...editBillingPartyData,
                      contact_phone: e.target.value,
                    })
                  }
                  placeholder="+41 XX XXX XX XX"
                />
              </div>
            </div>

            <div className={styles.formActions}>
              <button
                type="button"
                onClick={handleUpdateBillingParty}
                className={styles.saveButton}
                disabled={updatingBillingParty}
              >
                {updatingBillingParty ? 'Mise à jour...' : 'Enregistrer les modifications'}
              </button>
              <button
                type="button"
                onClick={() => {
                  setShowEditBillingPartyForm(false);
                  setError(null);
                }}
                className={styles.cancelButton}
                disabled={updatingBillingParty}
              >
                Annuler
              </button>
            </div>
          </div>
        )}

        <div className={styles.sectionDivider}>
          <p className={styles.hint} style={{ marginBottom: '0.75rem' }}>
            Vous pouvez modifier le destinataire si nécessaire :
          </p>
          <div className={styles.formGroup}>
            <select
              id="billing_party_mapping"
              className={styles.input}
              value={selectedBillingPartyId}
              onChange={(e) => setSelectedBillingPartyId(e.target.value)}
            >
              <option value="">— Sélectionner un autre destinataire —</option>
              {billingPartyOptions.map((bp) => (
                <option key={bp.id} value={bp.id}>
                  {bp.display_name} ({bp.type})
                </option>
              ))}
            </select>
          </div>
          {selectedBillingPartyId &&
            selectedBillingPartyId !== String(existingMapping.billing_party_id) && (
              <div className={styles.formActions}>
                <button
                  type="button"
                  onClick={handleSave}
                  className={styles.saveButton}
                >
                  Mettre à jour le mapping
                </button>
              </div>
            )}
        </div>
      </div>
    );
  }

  return (
    <div className={styles.accordionContent}>
      {error && <div className={styles.error}>{error}</div>}
      {message && (
        <div style={{ color: '#10b981', marginBottom: '0.75rem' }}>
          {message}
        </div>
      )}

      {!showCreateBillingPartyForm ? (
        <>
          <div className={styles.formGroup}>
            <label htmlFor="billing_party_mapping" className={styles.label}>
              Destinataire de facturation (BillingParty) *
            </label>
            <p className={styles.hint} style={{ marginBottom: '0.75rem', fontSize: '0.875rem' }}>
              <strong>Explication :</strong> La clinique est le <strong>payeur</strong> (elle paie les factures).
              Le BillingParty définit <strong>où envoyer la facture</strong> (adresse postale, email, contact).
              <br />
              <br />
              Sélectionnez un destinataire existant ou créez-en un nouveau ci-dessous.
            </p>
            <select
              id="billing_party_mapping"
              className={styles.input}
              value={selectedBillingPartyId}
              onChange={(e) => setSelectedBillingPartyId(e.target.value)}
            >
              <option value="">— Sélectionner un destinataire —</option>
              {billingPartyOptions.map((bp) => (
                <option key={bp.id} value={bp.id}>
                  {bp.display_name} ({bp.type})
                </option>
              ))}
            </select>
            {clinicCompanyName && (
              <p className={styles.hint} style={{ marginTop: '0.5rem' }}>
                Clinique (payeur) : <strong>{clinicCompanyName}</strong>
              </p>
            )}
          </div>

          <div style={{ marginTop: '1rem', display: 'flex', gap: '0.5rem' }}>
            <button
              type="button"
              onClick={handleSave}
              className={styles.saveButton}
              disabled={!selectedBillingPartyId}
              style={{ width: 'auto', padding: '0.5rem 1rem' }}
            >
              Enregistrer le mapping
            </button>
            <button
              type="button"
              onClick={() => {
                setShowCreateBillingPartyForm(true);
                setNewBillingPartyData({
                  display_name: clinicCompanyName || '',
                  billing_address: '',
                  billing_street: '',
                  billing_postcode: '',
                  billing_city: '',
                  contact_email: '',
                  contact_phone: '',
                });
              }}
              className={styles.cancelButton}
              style={{ width: 'auto', padding: '0.5rem 1rem' }}
            >
              + Créer un nouveau destinataire
            </button>
          </div>
        </>
      ) : (
        <div>
          <div className={styles.sectionHeader}>
            <h4 className={styles.sectionTitle}>
              Créer un nouveau destinataire de facturation
            </h4>
          </div>

          <div className={styles.formGroup}>
            <label htmlFor="bp_display_name" className={styles.label}>
              Nom du destinataire *
            </label>
            <input
              id="bp_display_name"
              type="text"
              className={styles.input}
              value={newBillingPartyData.display_name}
              onChange={(e) =>
                setNewBillingPartyData({
                  ...newBillingPartyData,
                  display_name: e.target.value,
                })
              }
              placeholder="Ex: Service facturation - Clinique des Grangettes"
            />
          </div>

          <div className={styles.formGroup}>
            <label htmlFor="bp_billing_address" className={styles.label}>
              Adresse de facturation *
            </label>
            <AddressAutocomplete
              name="bp_billing_address"
              value={newBillingPartyData.billing_address}
              onChange={(e) => {
                const v = e.target.value;
                const fields = parseAddressToFields(v);
                setNewBillingPartyData({
                  ...newBillingPartyData,
                  billing_address: v,
                  billing_street: fields.billing_street,
                  billing_postcode: fields.billing_postcode,
                  billing_city: fields.billing_city,
                });
              }}
              onSelect={(item) => {
                const fullAddress = item.label || '';
                const parsed = parseAddressWithEstablishment(fullAddress, item);
                const street =
                  parsed.street && parsed.streetNumber
                    ? `${parsed.street} ${parsed.streetNumber}`.trim()
                    : parsed.street || '';
                setNewBillingPartyData({
                  ...newBillingPartyData,
                  billing_address: fullAddress,
                  billing_street: street,
                  billing_postcode: parsed.postcode || '',
                  billing_city: parsed.city || '',
                });
              }}
              placeholder="Adresse complète (rue, code postal, ville)"
              disabled={creatingBillingParty}
            />
          </div>

          <div className={styles.formRow}>
            <div className={styles.formGroup}>
              <label htmlFor="bp_billing_street" className={styles.label}>
                Rue et numéro *
              </label>
              <input
                id="bp_billing_street"
                type="text"
                className={styles.input}
                value={newBillingPartyData.billing_street}
                readOnly
                placeholder="Rempli automatiquement depuis l'adresse"
              />
            </div>
            <div className={styles.formGroup}>
              <label htmlFor="bp_billing_postcode" className={styles.label}>
                Code postal *
              </label>
              <input
                id="bp_billing_postcode"
                type="text"
                className={styles.input}
                value={newBillingPartyData.billing_postcode}
                readOnly
                placeholder="Rempli automatiquement depuis l'adresse"
                maxLength={10}
              />
            </div>
            <div className={styles.formGroup}>
              <label htmlFor="bp_billing_city" className={styles.label}>
                Ville *
              </label>
              <input
                id="bp_billing_city"
                type="text"
                className={styles.input}
                value={newBillingPartyData.billing_city}
                readOnly
                placeholder="Rempli automatiquement depuis l'adresse"
              />
            </div>
          </div>

          <div className={styles.formRowTwo}>
            <div className={styles.formGroup}>
              <label htmlFor="bp_contact_email" className={styles.label}>
                Email de contact
              </label>
              <input
                id="bp_contact_email"
                type="email"
                className={styles.input}
                value={newBillingPartyData.contact_email}
                onChange={(e) =>
                  setNewBillingPartyData({
                    ...newBillingPartyData,
                    contact_email: e.target.value,
                  })
                }
                placeholder="facturation@clinique.ch"
              />
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="bp_contact_phone" className={styles.label}>
                Téléphone de contact
              </label>
              <input
                id="bp_contact_phone"
                type="tel"
                className={styles.input}
                value={newBillingPartyData.contact_phone}
                onChange={(e) =>
                  setNewBillingPartyData({
                    ...newBillingPartyData,
                    contact_phone: e.target.value,
                  })
                }
                placeholder="+41 XX XXX XX XX"
              />
            </div>
          </div>

          <div className={styles.formActions}>
            <button
              type="button"
              onClick={handleCreateBillingParty}
              className={styles.saveButton}
              disabled={creatingBillingParty}
            >
              {creatingBillingParty ? 'Création...' : 'Créer et enregistrer'}
            </button>
            <button
              type="button"
              onClick={() => {
                setShowCreateBillingPartyForm(false);
                setError(null);
              }}
              className={styles.cancelButton}
              disabled={creatingBillingParty}
            >
              Annuler
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default ClinicBillingMappingSection;
