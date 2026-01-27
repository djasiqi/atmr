// frontend/src/pages/company/Clients/components/ClientBillingPartiesSection.jsx
/**
 * Section de gestion des tiers payeurs pour un client
 * 
 * OBJECTIF DES TIERS PAYEURS :
 * - Un tiers payeur représente qui paie les factures pour un client (curatelle, famille, assurance, etc.)
 * - Un client peut avoir PLUSIEURS tiers payeurs (ex: curateur principal + famille secondaire)
 * - Mais UNE SEULE facture est générée par client (pas une facture par tiers payeur)
 * - Le "payeur par défaut" est utilisé automatiquement pour la facturation
 * - Les autres tiers payeurs peuvent être utilisés pour des cas spécifiques (séjours hospitaliers, etc.)
 * 
 * CAS D'USAGE :
 * - Curatelle : le curateur paie toutes les factures
 * - Famille : un membre de la famille paie
 * - Assurance : l'assurance paie (pour certains trajets)
 * - Clinique/EMS : l'établissement paie (pendant un séjour)
 * 
 * IMPORTANT : Chaque facture est toujours pour UN client, mais peut être adressée à différents tiers payeurs selon le contexte.
 */
import React, { useState, useEffect, useCallback, useLayoutEffect, useRef } from 'react';
import { fetchBillingParties, createBillingParty } from '../../../../services/settingsService';
import { fetchClientBillingParties, linkClientBillingParty, unlinkClientBillingParty, updateClientBillingPartyLink } from '../../../../services/companyService';
import AddressAutocomplete from '../../../../components/common/AddressAutocomplete';
import styles from './ClientBillingPartiesSection.module.css';

const ClientBillingPartiesSection = ({
  clientId,
  readOnly = false,
  showTitle = true,
  autoShowForm = false,
  onScrollBottomGapChange,
}) => {
  const [billingParties, setBillingParties] = useState([]);
  const [clientLinks, setClientLinks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showForm, setShowForm] = useState(autoShowForm);
  const [createMode, setCreateMode] = useState(false); // Mode création vs sélection
  const [submitting, setSubmitting] = useState(false); // État de soumission
  const [formData, setFormData] = useState({
    billing_party_id: '',
    is_default: false,
    role: '',
    contact_name: '',
    contact_email: '',
    contact_phone: '',
  });
  const [newPartyData, setNewPartyData] = useState({
    display_name: '',
    type: 'family', // Par défaut "famille"
    billing_address: '',
    contact_email: '',
    contact_phone: '',
  });
  const [_billingAddressCoords, setBillingAddressCoords] = useState({ lat: null, lon: null });
  const actionsRef = useRef(null);
  const [bottomSpacerHeight, setBottomSpacerHeight] = useState(32);

  const loadBillingParties = useCallback(async () => {
    try {
      setLoading(true);
      const response = await fetchBillingParties({ active: true });
      setBillingParties(response.data || []);
    } catch (err) {
      setError(err.response?.data?.error || 'Erreur lors du chargement des tiers payeurs');
    } finally {
      setLoading(false);
    }
  }, []);

  const loadClientLinks = useCallback(async () => {
    try {
      const response = await fetchClientBillingParties(clientId);
      setClientLinks(response.data || []);
    } catch (err) {
      // Si l'endpoint n'existe pas encore, la liste reste vide
      console.warn('Endpoint billing-parties non disponible:', err);
      setClientLinks([]);
    }
  }, [clientId]);

  useEffect(() => {
    loadBillingParties();
    loadClientLinks();
  }, [loadBillingParties, loadClientLinks]);

  // Afficher automatiquement le formulaire si autoShowForm est true
  // Le formulaire s'affiche dès que le composant est monté (quand la section s'ouvre)
  useEffect(() => {
    if (autoShowForm && !readOnly) {
      // Réinitialiser le formulaire à chaque ouverture de la section
      setShowForm(true);
      setFormData({
        billing_party_id: '',
        is_default: false,
        role: '',
        contact_name: '',
        contact_email: '',
        contact_phone: '',
      });
    }
  }, [autoShowForm, readOnly]);

  const handleCreateParty = async (e) => {
    if (e) {
      e.preventDefault();
      e.stopPropagation();
    }
    
    if (submitting) return; // Éviter les doubles soumissions
    
    try {
      setSubmitting(true);
      setError(null);
      
      if (!newPartyData.display_name.trim()) {
        setError('Le nom est requis');
        setSubmitting(false);
        return;
      }
      if (newPartyData.type !== 'patient' && !newPartyData.billing_address.trim()) {
        setError('L\'adresse est requise pour ce type de tiers payeur');
        setSubmitting(false);
        return;
      }

      const payload = {
        display_name: newPartyData.display_name.trim(),
        type: newPartyData.type,
        billing_address: newPartyData.billing_address.trim() || null,
        contact_email: newPartyData.contact_email.trim() || null,
        contact_phone: newPartyData.contact_phone.trim() || null,
        is_active: true,
      };

      console.log('📤 Création du tiers payeur:', payload);
      const response = await createBillingParty(payload);
      console.log('📥 Réponse complète:', response);
      // Le backend retourne {success: true, data: {...}}
      const newParty = response.data?.data || response.data;
      console.log('✅ Tiers payeur créé:', newParty);
      
      if (!newParty || !newParty.id) {
        throw new Error('Réponse invalide du serveur : tiers payeur non créé');
      }

      // Recharger la liste des tiers payeurs
      await loadBillingParties();

      // Sélectionner automatiquement le nouveau tiers payeur et le lier directement
      const linkPayload = {
        billing_party_id: newParty.id.toString(),
        is_default: formData.is_default,
        role: formData.role,
        contact_name: formData.contact_name,
        contact_email: formData.contact_email,
        contact_phone: formData.contact_phone,
      };
      
      // Lier automatiquement le nouveau tiers payeur au client
      try {
        await linkClientBillingParty(clientId, linkPayload);
        await loadClientLinks();
        
        // Réinitialiser les formulaires
        setFormData({
          billing_party_id: '',
          is_default: false,
          role: '',
          contact_name: '',
          contact_email: '',
          contact_phone: '',
        });
        setCreateMode(false);
        setNewPartyData({
          display_name: '',
          type: 'family',
          billing_address: '',
          contact_email: '',
          contact_phone: '',
        });
        setBillingAddressCoords({ lat: null, lon: null });
        
        // Si autoShowForm est activé, garder le formulaire ouvert
        if (!autoShowForm) {
          setShowForm(false);
        }
      } catch (linkErr) {
        // Si le lien échoue, au moins on a créé le tiers payeur, on peut juste sélectionner
        setFormData({
          billing_party_id: newParty.id.toString(),
          is_default: formData.is_default,
          role: formData.role,
          contact_name: formData.contact_name,
          contact_email: formData.contact_email,
          contact_phone: formData.contact_phone,
        });
        setCreateMode(false);
        setError('Tiers payeur créé mais erreur lors de la liaison. Vous pouvez le lier manuellement.');
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Erreur lors de la création du tiers payeur');
    }
  };

  const handleSubmit = async (e) => {
    if (e) e.preventDefault();
    try {
      setError(null);
      if (hasExistingLink) {
        await updateClientBillingPartyLink(selectedLink.id, {
          is_default: formData.is_default,
          role: formData.role,
          contact_name: formData.contact_name,
          contact_email: formData.contact_email,
          contact_phone: formData.contact_phone,
        });
        await loadClientLinks();
      } else {
        await linkClientBillingParty(clientId, formData);
        await loadClientLinks();
      }
      // Si autoShowForm est activé, garder le formulaire ouvert pour permettre l'ajout d'un autre
      if (!autoShowForm) {
        setShowForm(false);
      }
      setFormData({
        billing_party_id: '',
        is_default: false,
        role: '',
        contact_name: '',
        contact_email: '',
        contact_phone: '',
      });
      setCreateMode(false);
    } catch (err) {
      if (err.response?.status === 404) {
        setError(
          'L\'endpoint backend pour lier un client à un tiers payeur n\'est pas encore disponible. Cette fonctionnalité sera bientôt implémentée.'
        );
      } else {
        setError(err.response?.data?.error || 'Erreur lors de la sauvegarde');
      }
    }
  };

  const selectedParty = billingParties.find(
    (party) => String(party.id) === String(formData.billing_party_id)
  );
  const selectedLink = clientLinks.find(
    (link) => String(link.billing_party_id) === String(formData.billing_party_id)
  );
  const hasExistingLink = !!selectedLink?.id;
  const isCuratorshipParty = selectedParty
    ? selectedParty.type === 'curatorship' || /opad/i.test(selectedParty.display_name || '')
    : false;
  const hasLinkContact = !!(
    selectedLink?.contact_name || selectedLink?.contact_email || selectedLink?.contact_phone
  );
  const showCuratorFields = isCuratorshipParty || hasLinkContact;

  useLayoutEffect(() => {
    if (!onScrollBottomGapChange) return;

    const updateGap = () => {
      const height = actionsRef.current?.offsetHeight || 0;
      const base = error ? 56 : showCuratorFields ? 40 : 28;
      const computed = Math.min(Math.max(height + base, 32), 240);
      setBottomSpacerHeight(computed);
      onScrollBottomGapChange(computed);
    };

    updateGap();

    if (!actionsRef.current || typeof ResizeObserver === 'undefined') return undefined;
    const observer = new ResizeObserver(updateGap);
    observer.observe(actionsRef.current);

    return () => observer.disconnect();
  }, [
    onScrollBottomGapChange,
    showCuratorFields,
    formData.billing_party_id,
    createMode,
    showForm,
    error,
  ]);

  // Si showTitle est false, on est dans un accordéon, pas besoin de conteneur supplémentaire
  const content = (
    <>
      {showTitle && (
        <div className={styles.sectionHeader}>
          <h3 className={styles.sectionTitle}>💰 Tiers payeur / Curateur</h3>
          {!readOnly && !showForm && (
            <button
              type="button"
              onClick={() => {
                setShowForm(true);
                setFormData({
                  billing_party_id: '',
                  is_default: false,
                  role: '',
                  contact_name: '',
                  contact_email: '',
                  contact_phone: '',
                });
              }}
              className={styles.addButton}
            >
              ➕ Ajouter un tiers payeur
            </button>
          )}
        </div>
      )}

      {showTitle && (
        <p className={styles.sectionDescription}>
          Définissez qui paie les factures pour ce client (curatelle, famille, assurance, etc.).
          Le payeur par défaut sera utilisé automatiquement pour la facturation.
        </p>
      )}

      {!readOnly && showForm && (
        <>
          {!createMode ? (
            <div>
              <div className={styles.formGroup}>
                <label htmlFor="billing_party_id" className={styles.label}>
                  Tiers payeur *
                </label>
                <select
                  id="billing_party_id"
                  value={formData.billing_party_id}
                  onChange={(e) => {
                    const selectedId = e.target.value;
                    const existingLink = clientLinks.find(
                      (link) => String(link.billing_party_id) === String(selectedId)
                    );
                    if (existingLink) {
                      setFormData({
                        billing_party_id: selectedId,
                        is_default: !!existingLink.is_default,
                        role: existingLink.role || '',
                        contact_name: existingLink.contact_name || '',
                        contact_email: existingLink.contact_email || '',
                        contact_phone: existingLink.contact_phone || '',
                      });
                      return;
                    }
                    setFormData({
                      billing_party_id: selectedId,
                      is_default: false,
                      role: '',
                      contact_name: '',
                      contact_email: '',
                      contact_phone: '',
                    });
                  }}
                  required
                  className={styles.input}
                >
                  <option value="">-- Sélectionnez un tiers payeur --</option>
                  {billingParties
                    .filter((party) => party.type !== 'clinic')
                    .map((party) => (
                      <option key={party.id} value={party.id}>
                        {party.display_name} ({party.type})
                      </option>
                    ))}
                </select>
                <button
                  type="button"
                  onClick={() => setCreateMode(true)}
                  style={{
                    marginTop: '0.5rem',
                    padding: '0.5rem 1rem',
                    background: 'transparent',
                    border: '1px solid #3b82f6',
                    color: '#3b82f6',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    fontSize: '0.875rem',
                    fontWeight: 500,
                  }}
                >
                  ➕ Créer un nouveau tiers payeur
                </button>
              </div>

              {selectedParty && (
                <div className={styles.mappingInfo}>
                  <div className={styles.mappingTitle}>Mapping pour ce client</div>
                  <div className={styles.mappingRow}>
                    <span className={styles.mappingLabel}>Tiers payeur</span>
                    <span className={styles.mappingValue}>
                      {selectedParty.display_name} ({selectedParty.type})
                    </span>
                  </div>
                  <div className={styles.mappingRow}>
                    <span className={styles.mappingLabel}>Coordonnées OPAD</span>
                    <div className={styles.mappingValue}>
                      {selectedParty.billing_address ? (
                        <div className={styles.mappingValueMultiline}>
                          {selectedParty.billing_address}
                        </div>
                      ) : (
                        <div className={styles.mappingValueMuted}>Adresse non renseignée</div>
                      )}
                      {selectedParty.contact_email && (
                        <div className={styles.mappingValue}>
                          ✉️{' '}
                          <a href={`mailto:${selectedParty.contact_email}`}>
                            {selectedParty.contact_email}
                          </a>
                        </div>
                      )}
                      {selectedParty.contact_phone && (
                        <div className={styles.mappingValue}>
                          📞{' '}
                          <a href={`tel:${selectedParty.contact_phone}`}>
                            {selectedParty.contact_phone}
                          </a>
                        </div>
                      )}
                    </div>
                  </div>
                  {hasExistingLink ? (
                    <>
                      {formData.role && (
                        <div className={styles.mappingRow}>
                          <span className={styles.mappingLabel}>Rôle</span>
                          <span className={styles.mappingValue}>{formData.role}</span>
                        </div>
                      )}
                      {formData.is_default && (
                        <div className={styles.mappingRow}>
                          <span className={styles.mappingTag}>⭐ Payeur par défaut</span>
                        </div>
                      )}
                      {(formData.contact_name ||
                        formData.contact_email ||
                        formData.contact_phone) && (
                        <div className={styles.mappingContact}>
                          <div className={styles.mappingLabel}>Curateur</div>
                          {formData.contact_name && (
                            <div className={styles.mappingValue}>
                              👤 {formData.contact_name}
                            </div>
                          )}
                          {formData.contact_email && (
                            <div className={styles.mappingValue}>
                              ✉️{' '}
                              <a href={`mailto:${formData.contact_email}`}>
                                {formData.contact_email}
                              </a>
                            </div>
                          )}
                          {formData.contact_phone && (
                            <div className={styles.mappingValue}>
                              📞{' '}
                              <a href={`tel:${formData.contact_phone}`}>
                                {formData.contact_phone}
                              </a>
                            </div>
                          )}
                        </div>
                      )}
                    </>
                  ) : (
                    <div className={styles.mappingEmpty}>
                      Aucun mapping pour ce client. Enregistrer pour créer le lien.
                    </div>
                  )}
                </div>
              )}

          <div className={styles.formGroup}>
            <label htmlFor="role" className={styles.label}>
              Rôle (optionnel)
            </label>
            <input
              type="text"
              id="role"
              value={formData.role}
              onChange={(e) => setFormData({ ...formData, role: e.target.value })}
              className={styles.input}
              placeholder="Ex: curateur principal, payeur secondaire..."
            />
          </div>

          {showCuratorFields && (
            <>
              <div className={styles.formGroup}>
                <label htmlFor="contact_name" className={styles.label}>
                  Curateur (optionnel)
                </label>
                <input
                  type="text"
                  id="contact_name"
                  value={formData.contact_name}
                  onChange={(e) => setFormData({ ...formData, contact_name: e.target.value })}
                  className={styles.input}
                  placeholder="Ex: Curateur A"
                />
                <small className={styles.hint}>
                  Contact spécifique à ce client (ne modifie pas OPAD)
                </small>
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="contact_email" className={styles.label}>
                  Email du curateur (optionnel)
                </label>
                <input
                  type="email"
                  id="contact_email"
                  value={formData.contact_email}
                  onChange={(e) => setFormData({ ...formData, contact_email: e.target.value })}
                  className={styles.input}
                  placeholder="curateur@opad.ch"
                />
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="contact_phone" className={styles.label}>
                  Téléphone du curateur (optionnel)
                </label>
                <input
                  type="tel"
                  id="contact_phone"
                  value={formData.contact_phone}
                  onChange={(e) => setFormData({ ...formData, contact_phone: e.target.value })}
                  className={styles.input}
                  placeholder="+41 22 000 00 00"
                />
              </div>
            </>
          )}

          <div className={styles.checkboxGroup}>
            <label className={styles.checkboxLabel}>
              <input
                type="checkbox"
                checked={formData.is_default}
                onChange={(e) => setFormData({ ...formData, is_default: e.target.checked })}
                className={styles.checkbox}
              />
              <span className={styles.checkboxText}>
                <strong>Définir comme payeur par défaut</strong>
                <small>Ce payeur sera utilisé automatiquement pour la facturation</small>
              </span>
            </label>
          </div>

              {error && <div className={styles.error}>{error}</div>}

              <div ref={actionsRef} className={styles.formActions}>
                <button
                  type="button"
                  onClick={() => {
                    setShowForm(false);
                    setFormData({
                      billing_party_id: '',
                      is_default: false,
                      role: '',
                      contact_name: '',
                      contact_email: '',
                      contact_phone: '',
                    });
                    setCreateMode(false);
                  }}
                  className={styles.cancelButton}
                >
                  Annuler
                </button>
                <button type="button" onClick={handleSubmit} className={styles.saveButton}>
                  {hasExistingLink ? 'Mettre à jour le mapping' : 'Lier le tiers payeur'}
                </button>
              </div>
            </div>
          ) : (
            <div>
              <div className={styles.formGroup}>
                <label htmlFor="new_party_display_name" className={styles.label}>
                  Nom complet *
                </label>
                <input
                  type="text"
                  id="new_party_display_name"
                  value={newPartyData.display_name}
                  onChange={(e) => setNewPartyData({ ...newPartyData, display_name: e.target.value })}
                  required
                  className={styles.input}
                  placeholder="Ex: Jean Dupont"
                />
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="new_party_type" className={styles.label}>
                  Type *
                </label>
                <select
                  id="new_party_type"
                  value={newPartyData.type}
                  onChange={(e) => setNewPartyData({ ...newPartyData, type: e.target.value })}
                  required
                  className={styles.input}
                >
                  <option value="family">Famille</option>
                  <option value="curatorship">Curatelle</option>
                  <option value="lawyer">Avocat</option>
                  <option value="insurance">Assurance</option>
                  <option value="other">Autre</option>
                </select>
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="new_party_billing_address" className={styles.label}>
                  Adresse {newPartyData.type !== 'patient' && '*'}
                </label>
                <AddressAutocomplete
                  name="new_party_billing_address"
                  value={newPartyData.billing_address}
                  onChange={(e) => {
                    setBillingAddressCoords({ lat: null, lon: null });
                    setNewPartyData({ ...newPartyData, billing_address: e.target.value });
                  }}
                  onSelect={(item) => {
                    const fullAddress = item.label || '';
                    setNewPartyData({ ...newPartyData, billing_address: fullAddress });
                    setBillingAddressCoords({
                      lat: item.lat ?? null,
                      lon: item.lon ?? null,
                    });
                  }}
                  placeholder="Ex: Avenue de la Gare 5, 1003, Lausanne"
                />
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="new_party_contact_email" className={styles.label}>
                  Email
                </label>
                <input
                  type="email"
                  id="new_party_contact_email"
                  value={newPartyData.contact_email}
                  onChange={(e) => setNewPartyData({ ...newPartyData, contact_email: e.target.value })}
                  className={styles.input}
                  placeholder="exemple@email.ch"
                />
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="new_party_contact_phone" className={styles.label}>
                  Téléphone
                </label>
                <input
                  type="tel"
                  id="new_party_contact_phone"
                  value={newPartyData.contact_phone}
                  onChange={(e) => setNewPartyData({ ...newPartyData, contact_phone: e.target.value })}
                  className={styles.input}
                  placeholder="+41 22 123 45 67"
                />
              </div>

              <div className={styles.subSectionTitle}>
                Contact curateur pour ce client
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="contact_name" className={styles.label}>
                  Contact curateur (optionnel)
                </label>
                <input
                  type="text"
                  id="contact_name"
                  value={formData.contact_name}
                  onChange={(e) => setFormData({ ...formData, contact_name: e.target.value })}
                  className={styles.input}
                  placeholder="Ex: Curateur A"
                />
                <small className={styles.hint}>
                  Contact spécifique à ce client (ne modifie pas OPAD)
                </small>
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="contact_email" className={styles.label}>
                  Email du curateur (optionnel)
                </label>
                <input
                  type="email"
                  id="contact_email"
                  value={formData.contact_email}
                  onChange={(e) => setFormData({ ...formData, contact_email: e.target.value })}
                  className={styles.input}
                  placeholder="curateur@opad.ch"
                />
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="contact_phone" className={styles.label}>
                  Téléphone du curateur (optionnel)
                </label>
                <input
                  type="tel"
                  id="contact_phone"
                  value={formData.contact_phone}
                  onChange={(e) => setFormData({ ...formData, contact_phone: e.target.value })}
                  className={styles.input}
                  placeholder="+41 22 000 00 00"
                />
              </div>

              {error && <div className={styles.error}>{error}</div>}

              <div ref={actionsRef} className={styles.formActions}>
                <button
                  type="button"
                  onClick={() => {
                    setCreateMode(false);
                    setNewPartyData({
                      display_name: '',
                      type: 'family',
                      billing_address: '',
                      contact_email: '',
                      contact_phone: '',
                    });
                    setBillingAddressCoords({ lat: null, lon: null });
                    setError(null);
                  }}
                  className={styles.cancelButton}
                >
                  Annuler
                </button>
                <button 
                  type="button" 
                  onClick={handleCreateParty} 
                  className={styles.saveButton}
                  disabled={submitting}
                >
                  {submitting ? 'Création...' : 'Créer et lier'}
                </button>
              </div>
            </div>
          )}
          <div
            className={styles.bottomSpacer}
            style={{ height: bottomSpacerHeight }}
            aria-hidden="true"
          />
        </>
      )}

      {/* Afficher la liste des liens existants en dessous du formulaire si autoShowForm, ou à la place si le formulaire n'est pas affiché */}
      {(!showForm || (autoShowForm && clientLinks.length > 0)) && (
        <>
          {loading ? (
            <div className={styles.loading}>Chargement des tiers payeurs...</div>
          ) : clientLinks.length === 0 && !autoShowForm ? (
            <div className={styles.emptyState}>
              {readOnly
                ? 'Aucun tiers payeur configuré.'
                : 'Aucun tiers payeur configuré.'}
            </div>
          ) : clientLinks.length > 0 ? (
            <>
              {!readOnly && !autoShowForm && (
                <div style={{ marginBottom: '1rem' }}>
                  <button
                    type="button"
                    onClick={() => {
                      setShowForm(true);
                      setFormData({
                        billing_party_id: '',
                        is_default: false,
                        role: '',
                        contact_name: '',
                        contact_email: '',
                        contact_phone: '',
                      });
                    }}
                    className={styles.addButton}
                  >
                    ➕ Ajouter un tiers payeur
                  </button>
                </div>
              )}
              <div className={styles.linksList}>
                {clientLinks.map((link) => (
                  <div key={link.id} className={styles.linkCard}>
                    <div className={styles.linkHeader}>
                      <div className={styles.linkInfo}>
                        <strong>{link.billing_party?.display_name || 'Tiers payeur'}</strong>
                        <div className={styles.linkDetails}>
                          Type: {link.billing_party?.type || 'N/A'}
                          {link.role && ` • Rôle: ${link.role}`}
                          {link.is_default && (
                            <span className={styles.defaultBadge}>⭐ Par défaut</span>
                          )}
                        </div>
                        {(link.contact_name ||
                          link.contact_email ||
                          link.contact_phone) && (
                          <div className={styles.linkContacts}>
                            {link.contact_name && (
                              <div>👤 {link.contact_name}</div>
                            )}
                            {link.contact_email && (
                              <div>
                                ✉️{' '}
                                <a href={`mailto:${link.contact_email}`}>
                                  {link.contact_email}
                                </a>
                              </div>
                            )}
                            {link.contact_phone && (
                              <div>
                                📞{' '}
                                <a href={`tel:${link.contact_phone}`}>
                                  {link.contact_phone}
                                </a>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                      <div className={styles.linkActions}>
                        <button
                          type="button"
                          onClick={async () => {
                            if (window.confirm('Voulez-vous supprimer ce lien avec le tiers payeur ?')) {
                              try {
                                await unlinkClientBillingParty(link.id);
                                await loadClientLinks();
                              } catch (err) {
                                if (err.response?.status === 404) {
                                  alert(
                                    'L\'endpoint backend pour supprimer un lien n\'est pas encore disponible.'
                                  );
                                } else {
                                  alert(err.response?.data?.error || 'Erreur lors de la suppression');
                                }
                              }
                            }
                          }}
                          className={styles.deleteButton}
                          title="Supprimer le lien"
                        >
                          🗑️
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </>
          ) : null}
        </>
      )}
    </>
  );

  // Si showTitle est true, on ajoute le conteneur .section, sinon on retourne directement le contenu (pas de conteneur redondant)
  if (showTitle) {
    return <div className={styles.section}>{content}</div>;
  }
  return content;
};

export default ClientBillingPartiesSection;
