// pages/institution/Settings/InstitutionSettings.jsx
/**
 * Paramètres de l'institution.
 *
 * Onglets (ordre):
 * 1. Profil (infos institution) — tous voient, admin modifie
 * 2. Utilisateurs & accès — admin only (voir + modifier)
 * 3. Transporteurs préférés — tous voient, admin modifie
 * 4. Facturation — tous voient, admin + billing modifient
 * 5. Notifications — tous voient, admin modifie
 * 6. Clés API — admin only (voir + modifier)
 */

import React, { useState, useEffect } from 'react';
import { FaKey, FaTruck, FaPlus, FaTimes, FaCopy, FaArrowUp, FaArrowDown, FaSave, FaBuilding, FaTrash, FaFileInvoiceDollar, FaBell, FaUsersCog, FaUserCircle, FaUsers } from 'react-icons/fa';
import { useTransportPreferences, useUpdateTransportPreferences, useApiKeys, useCreateApiKey, useRevokeApiKey, useInstitutionMe, useInstitutionSettings, useUpdateInstitutionSettings } from '../../../hooks/useInstitutionData';
import { isAdmin, canEditBilling } from '../../../utils/institutionPermissions';
import { toast } from 'sonner';
import InstitutionProfileTab from './components/InstitutionProfileTab';
import MyProfileTab from './components/MyProfileTab';
import UsersRolesTab from './components/UsersRolesTab';
import BillingDefaultsTab from './components/BillingDefaultsTab';
import NotificationsTab from './components/NotificationsTab';
import TeamsTab from './components/TeamsTab';
import AddCompanyModal from './components/AddCompanyModal';
import styles from './InstitutionSettings.module.css';

const AVAILABLE_SCOPES = [
  { value: 'requests:read', label: 'Lecture demandes' },
  { value: 'requests:write', label: 'Création/modification demandes' },
  { value: 'requests:send', label: 'Envoi demandes' },
  { value: 'requests:cancel', label: 'Annulation demandes' },
  { value: 'patients:read', label: 'Lecture patients' },
  { value: 'patients:write', label: 'Création/modification patients' },
];

const InstitutionSettings = () => {
  const { data: meData } = useInstitutionMe();
  const { data: prefsData, isLoading: loadingPrefs } = useTransportPreferences();
  const { data: keysData, isLoading: loadingKeys } = useApiKeys();
  const { data: settingsData } = useInstitutionSettings();
  const updatePrefsMutation = useUpdateTransportPreferences();
  const updateSettingsMutation = useUpdateInstitutionSettings();
  const createKeyMutation = useCreateApiKey();
  const revokeKeyMutation = useRevokeApiKey();
  
  const institutionRole = meData?.institution_role;
  const institutionType = meData?.institution_type;
  const canAdmin = isAdmin(institutionRole);
  const isBillingRole = canEditBilling(institutionRole) && !canAdmin;
  const isCuratelle = (institutionType || '').toLowerCase() === 'curatelle';
  
  const [activeTab, setActiveTab] = useState(isBillingRole ? 'myprofile' : 'profile');
  
  // Le rôle billing accède à Mon profil + Facturation → forcer au chargement
  useEffect(() => {
    if (isBillingRole && activeTab !== 'billing' && activeTab !== 'myprofile') {
      setActiveTab('myprofile');
    }
  }, [isBillingRole]); // eslint-disable-line react-hooks/exhaustive-deps
  const [preferences, setPreferences] = useState([]);
  const [prefsLoaded, setPrefsLoaded] = useState(false);
  
  // Transport settings local state
  const [transportPickupMode, setTransportPickupMode] = useState('institution');
  const [transportDispatchMode, setTransportDispatchMode] = useState('sequential');
  const [transportEntryPoints, setTransportEntryPoints] = useState([]);
  const [transportContactPhone, setTransportContactPhone] = useState('');
  const [newEntryPoint, setNewEntryPoint] = useState('');
  const [transportSettingsLoaded, setTransportSettingsLoaded] = useState(false);
  const [transportSettingsDirty, setTransportSettingsDirty] = useState(false);

  // Load transport settings from server
  useEffect(() => {
    if (settingsData?.settings && !transportSettingsLoaded) {
      setTransportPickupMode(settingsData.settings.default_pickup_mode || 'institution');
      setTransportDispatchMode(settingsData.settings.offer_dispatch_mode || 'sequential');
      setTransportEntryPoints(settingsData.settings.entry_points || []);
      setTransportContactPhone(settingsData.settings.default_contact_phone || '');
      setTransportSettingsLoaded(true);
    }
  }, [settingsData, transportSettingsLoaded]);

  // AddCompanyModal state
  const [showAddCompany, setShowAddCompany] = useState(false);
  
  // API Key creation state
  const [showCreateKey, setShowCreateKey] = useState(false);
  const [newKeyName, setNewKeyName] = useState('');
  const [selectedScopes, setSelectedScopes] = useState([]);
  const [createdKey, setCreatedKey] = useState(null);
  
  // Revoke confirmation modal state
  const [revokeKeyId, setRevokeKeyId] = useState(null);
  
  // Load preferences when data arrives
  React.useEffect(() => {
    if (prefsData?.preferences && !prefsLoaded) {
      setPreferences(prefsData.preferences);
      setPrefsLoaded(true);
    }
  }, [prefsData, prefsLoaded]);
  
  const keys = keysData?.api_keys || [];
  
  // Preferences handlers
  const movePreference = (index, direction) => {
    const newPrefs = [...preferences];
    const newIndex = index + direction;
    if (newIndex < 0 || newIndex >= newPrefs.length) return;
    [newPrefs[index], newPrefs[newIndex]] = [newPrefs[newIndex], newPrefs[index]];
    newPrefs.forEach((p, i) => { p.priority = i + 1; });
    setPreferences(newPrefs);
  };

  // Confirmation state for removing last transporter
  const [confirmRemoveIndex, setConfirmRemoveIndex] = useState(null);

  const removePreference = (index) => {
    // Warn when removing the last transporter
    if (preferences.length === 1) {
      setConfirmRemoveIndex(index);
      return;
    }
    const newPrefs = preferences.filter((_, i) => i !== index);
    newPrefs.forEach((p, i) => { p.priority = i + 1; });
    setPreferences(newPrefs);
  };

  const confirmRemoveLast = () => {
    if (confirmRemoveIndex !== null) {
      const newPrefs = preferences.filter((_, i) => i !== confirmRemoveIndex);
      newPrefs.forEach((p, i) => { p.priority = i + 1; });
      setPreferences(newPrefs);
      setConfirmRemoveIndex(null);
    }
  };

  const handleAddCompany = (company) => {
    const newPref = {
      company_id: company.id,
      company_name: company.name,
      priority: preferences.length + 1,
    };
    setPreferences(prev => [...prev, newPref]);
    toast.success(`${company.name} ajouté`);
  };
  
  const savePreferences = async () => {
    try {
      const result = await updatePrefsMutation.mutateAsync({
        company_ids: preferences.map(p => p.company_id),
      });
      // Mettre à jour avec les données du serveur
      if (result?.preferences) {
        setPreferences(result.preferences);
      }
      toast.success('Préférences enregistrées');
    } catch (err) {
      toast.error(err?.response?.data?.error || 'Erreur');
    }
  };
  
  // API Key handlers
  const toggleScope = (scope) => {
    setSelectedScopes(prev => 
      prev.includes(scope) 
        ? prev.filter(s => s !== scope)
        : [...prev, scope]
    );
  };
  
  const handleCreateKey = async () => {
    if (!newKeyName.trim()) {
      toast.error('Nom requis');
      return;
    }
    if (selectedScopes.length === 0) {
      toast.error('Sélectionnez au moins un scope');
      return;
    }
    
    try {
      const result = await createKeyMutation.mutateAsync({
        name: newKeyName,
        scopes: selectedScopes,
      });
      setCreatedKey(result.key);
      setNewKeyName('');
      setSelectedScopes([]);
      setShowCreateKey(false);
      toast.success('Clé API créée');
    } catch (err) {
      toast.error(err?.response?.data?.error || 'Erreur');
    }
  };
  
  const handleRevokeKey = async (keyId) => {
    try {
      await revokeKeyMutation.mutateAsync(keyId);
      setRevokeKeyId(null);
      toast.success('Clé révoquée');
    } catch (err) {
      toast.error(err?.response?.data?.error || 'Erreur');
    }
  };
  
  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    toast.success('Copié !');
  };
  
  const formatDate = (dateStr) => {
    if (!dateStr) return '-';
    return new Date(dateStr).toLocaleString('fr-CH');
  };

  const isSequentialDispatch = transportDispatchMode !== 'broadcast';

  return (
    <div className={styles.container}>
      <div className={styles.tabs}>
        {/* Mon profil — visible par tous les rôles */}
        {isBillingRole && (
          <button
            className={`${styles.tab} ${activeTab === 'myprofile' ? styles.active : ''}`}
            onClick={() => setActiveTab('myprofile')}
          >
            <FaUserCircle /> Mon profil
          </button>
        )}
        {/* Profil institution — pas pour billing */}
        {!isBillingRole && (
          <button 
            className={`${styles.tab} ${activeTab === 'profile' ? styles.active : ''}`}
            onClick={() => setActiveTab('profile')}
          >
            <FaBuilding /> Profil
          </button>
        )}
        {/* Mon profil — pour admin, requester, reader (après profil institution) */}
        {!isBillingRole && (
          <button
            className={`${styles.tab} ${activeTab === 'myprofile' ? styles.active : ''}`}
            onClick={() => setActiveTab('myprofile')}
          >
            <FaUserCircle /> Mon profil
          </button>
        )}
        {canAdmin && (
          <button 
            className={`${styles.tab} ${activeTab === 'users' ? styles.active : ''}`}
            onClick={() => setActiveTab('users')}
          >
            <FaUsersCog /> Utilisateurs
          </button>
        )}
        {canAdmin && isCuratelle && (
          <button 
            className={`${styles.tab} ${activeTab === 'teams' ? styles.active : ''}`}
            onClick={() => setActiveTab('teams')}
          >
            <FaUsers /> Équipes
          </button>
        )}
        {!isBillingRole && (
          <button 
            className={`${styles.tab} ${activeTab === 'preferences' ? styles.active : ''}`}
            onClick={() => setActiveTab('preferences')}
          >
            <FaTruck /> Transport
          </button>
        )}
        <button 
          className={`${styles.tab} ${activeTab === 'billing' ? styles.active : ''}`}
          onClick={() => setActiveTab('billing')}
        >
          <FaFileInvoiceDollar /> Facturation
        </button>
        {!isBillingRole && (
          <button 
            className={`${styles.tab} ${activeTab === 'notifications' ? styles.active : ''}`}
            onClick={() => setActiveTab('notifications')}
          >
            <FaBell /> Notifications
          </button>
        )}
        {canAdmin && (
          <button 
            className={`${styles.tab} ${activeTab === 'apikeys' ? styles.active : ''}`}
            onClick={() => setActiveTab('apikeys')}
          >
            <FaKey /> Clés API
          </button>
        )}
      </div>

      {/* My Profile Tab */}
      {activeTab === 'myprofile' && (
        <MyProfileTab />
      )}

      {/* Profile Tab (institution) */}
      {activeTab === 'profile' && (
        <InstitutionProfileTab />
      )}

      {/* Users & Roles Tab (admin only) */}
      {activeTab === 'users' && canAdmin && (
        <UsersRolesTab />
      )}

      {/* Teams Tab (curatelle admin only) */}
      {activeTab === 'teams' && canAdmin && isCuratelle && (
        <TeamsTab />
      )}

      {/* Billing Tab */}
      {activeTab === 'billing' && (
        <BillingDefaultsTab />
      )}

      {/* Notifications Tab */}
      {activeTab === 'notifications' && (
        <NotificationsTab />
      )}
      
      {/* Transport Preferences */}
      {activeTab === 'preferences' && (
        <div className={styles.section}>
          <div className={styles.sectionHeader}>
            <h3>Organisation des transports</h3>
            <p style={{ color: '#666', fontSize: 13, lineHeight: 1.5 }}>
              Définissez les entreprises de transport autorisées et leur ordre de priorité.
              Les demandes sont envoyées automatiquement selon ces règles, sans intervention manuelle.
            </p>
          </div>

          {/* Mode d'attribution */}
          <div className={styles.allocationModeCard}>
            <p className={styles.allocationModeTitle}>
              Mode actuel :{' '}
              {isSequentialDispatch
                ? 'Séquentiel avec escalade automatique'
                : 'Diffusion simultanée (broadcast)'}
            </p>
            <p className={styles.allocationModeDescription}>
              {isSequentialDispatch
                ? "Les demandes sont envoyées successivement aux transporteurs selon l'ordre défini. En cas de non-réponse dans le délai imparti, la demande est automatiquement proposée au transporteur suivant."
                : 'Les demandes sont envoyées en parallèle à tous les transporteurs éligibles. Le premier à accepter remporte la demande.'}
            </p>
            <div className={styles.allocationModeBadges} role="status" aria-live="polite">
              <span className={`${styles.allocationBadge} ${styles.allocationBadgePrimary}`}>
                {isSequentialDispatch ? 'Envoi séquentiel' : 'Envoi broadcast'}
              </span>
              {isSequentialDispatch ? (
                <>
                  <span className={styles.allocationBadge}>
                    Délai jour même : {settingsData?.settings?.timeout_same_day_minutes ?? 5} min
                  </span>
                  <span className={styles.allocationBadge}>
                    Délai planifié : {settingsData?.settings?.timeout_default_minutes ?? 60} min
                  </span>
                  <span className={styles.allocationBadge}>Escalade automatique activée</span>
                </>
              ) : (
                <span className={styles.allocationBadge}>Escalade non utilisée</span>
              )}
            </div>
          </div>

          {/* ── Paramètres de demande (type de trajet, points d'accueil, contact) ── */}
          <div style={{
            background: '#f8f9fa',
            border: '1px solid #e0e0e0',
            borderRadius: 8,
            padding: '16px',
            marginBottom: 20,
          }}>
            <h4 style={{ margin: '0 0 4px', fontSize: 14, color: '#333' }}>Paramètres de demande</h4>
            <p style={{ fontSize: 12, color: '#888', margin: '0 0 12px' }}>
              Configurent le pré-remplissage du formulaire de demande de transport.
            </p>

            {/* Mode de départ par défaut */}
            <div style={{ marginBottom: 14 }}>
              <label style={{ display: 'block', fontSize: 13, fontWeight: 500, marginBottom: 6 }}>
                Mode d&apos;attribution des transporteurs
              </label>
              <div style={{ display: 'flex', gap: 16, marginBottom: 8, flexWrap: 'wrap' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 13, cursor: canAdmin ? 'pointer' : 'default' }}>
                  <input
                    type="radio"
                    name="offer_dispatch_mode"
                    value="sequential"
                    checked={transportDispatchMode === 'sequential'}
                    disabled={!canAdmin}
                    onChange={() => { setTransportDispatchMode('sequential'); setTransportSettingsDirty(true); }}
                  />
                  Séquentiel (avec escalade)
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 13, cursor: canAdmin ? 'pointer' : 'default' }}>
                  <input
                    type="radio"
                    name="offer_dispatch_mode"
                    value="broadcast"
                    checked={transportDispatchMode === 'broadcast'}
                    disabled={!canAdmin}
                    onChange={() => { setTransportDispatchMode('broadcast'); setTransportSettingsDirty(true); }}
                  />
                  Broadcast (envoi simultané)
                </label>
              </div>
              <span style={{ fontSize: 11, color: '#999', display: 'block', marginBottom: 10 }}>
                Réglage propre à votre institution, appliqué à chaque envoi de demande.
              </span>

              <label style={{ display: 'block', fontSize: 13, fontWeight: 500, marginBottom: 4 }}>
                Lieu de départ par défaut
              </label>
              <div style={{ display: 'flex', gap: 16 }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 13, cursor: canAdmin ? 'pointer' : 'default' }}>
                  <input
                    type="radio"
                    name="default_pickup_mode"
                    value="institution"
                    checked={transportPickupMode === 'institution'}
                    disabled={!canAdmin}
                    onChange={() => { setTransportPickupMode('institution'); setTransportSettingsDirty(true); }}
                  />
                  Institution (clinique/EMS)
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 13, cursor: canAdmin ? 'pointer' : 'default' }}>
                  <input
                    type="radio"
                    name="default_pickup_mode"
                    value="domicile"
                    checked={transportPickupMode === 'domicile'}
                    disabled={!canAdmin}
                    onChange={() => { setTransportPickupMode('domicile'); setTransportSettingsDirty(true); }}
                  />
                  Domicile du patient (IMAD)
                </label>
              </div>
              <span style={{ fontSize: 11, color: '#999' }}>
                Détermine le pré-remplissage du lieu de départ dans le formulaire de demande.
              </span>
            </div>

            {/* Points d'accueil (suggestions) — éditable */}
            <div style={{ marginBottom: 14 }}>
              <label style={{ display: 'block', fontSize: 13, fontWeight: 500, marginBottom: 4 }}>
                Points d'accueil / Entrées
              </label>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 6 }}>
                {transportEntryPoints.map((ep, i) => (
                  <span key={i} style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: 4,
                    padding: '3px 10px',
                    background: '#e3f2fd',
                    color: '#1565c0',
                    borderRadius: 12,
                    fontSize: 12,
                    fontWeight: 500,
                  }}>
                    {ep}
                    {canAdmin && (
                      <button
                        type="button"
                        onClick={() => {
                          setTransportEntryPoints(prev => prev.filter((_, idx) => idx !== i));
                          setTransportSettingsDirty(true);
                        }}
                        style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#1565c0', padding: 0, fontSize: 14, lineHeight: 1 }}
                        title="Retirer"
                      >
                        ×
                      </button>
                    )}
                  </span>
                ))}
                {transportEntryPoints.length === 0 && (
                  <span style={{ fontSize: 12, color: '#999', fontStyle: 'italic' }}>
                    Aucun point d'accueil configuré. Les demandeurs saisiront en texte libre.
                  </span>
                )}
              </div>
              {canAdmin && (
                <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                  <input
                    type="text"
                    value={newEntryPoint}
                    onChange={(e) => setNewEntryPoint(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && newEntryPoint.trim()) {
                        e.preventDefault();
                        if (!transportEntryPoints.includes(newEntryPoint.trim())) {
                          setTransportEntryPoints(prev => [...prev, newEntryPoint.trim()]);
                          setTransportSettingsDirty(true);
                        }
                        setNewEntryPoint('');
                      }
                    }}
                    placeholder="Ajouter un point d'accueil..."
                    style={{
                      flex: 1,
                      maxWidth: 280,
                      padding: '5px 10px',
                      borderRadius: 6,
                      border: '1px solid #ddd',
                      fontSize: 13,
                    }}
                  />
                  <button
                    type="button"
                    onClick={() => {
                      if (newEntryPoint.trim() && !transportEntryPoints.includes(newEntryPoint.trim())) {
                        setTransportEntryPoints(prev => [...prev, newEntryPoint.trim()]);
                        setTransportSettingsDirty(true);
                        setNewEntryPoint('');
                      }
                    }}
                    style={{
                      padding: '5px 12px',
                      borderRadius: 6,
                      border: '1px solid #1565c0',
                      background: '#e3f2fd',
                      color: '#1565c0',
                      fontSize: 13,
                      cursor: 'pointer',
                      fontWeight: 500,
                    }}
                  >
                    <FaPlus style={{ fontSize: 10, marginRight: 4 }} /> Ajouter
                  </button>
                </div>
              )}
              <span style={{ display: 'block', fontSize: 11, color: '#999', marginTop: 4 }}>
                Ces suggestions apparaîtront dans le formulaire de demande pour faciliter la saisie.
              </span>
            </div>

            {/* Téléphone standard */}
            <div>
              <label style={{ display: 'block', fontSize: 13, fontWeight: 500, marginBottom: 4 }}>
                Téléphone standard institution
              </label>
              <input
                type="tel"
                value={transportContactPhone}
                disabled={!canAdmin}
                onChange={(e) => { setTransportContactPhone(e.target.value); setTransportSettingsDirty(true); }}
                style={{
                  width: '100%',
                  maxWidth: 280,
                  padding: '6px 10px',
                  borderRadius: 6,
                  border: '1px solid #ddd',
                  fontSize: 13,
                  background: !canAdmin ? '#f5f5f5' : '#fff',
                }}
                placeholder="Ex: +41 22 123 45 67"
              />
              <span style={{ display: 'block', fontSize: 11, color: '#999', marginTop: 2 }}>
                Pré-rempli comme contact sur place dans les demandes de transport.
              </span>
            </div>

            {/* Bouton sauvegarder les paramètres de demande */}
            {canAdmin && transportSettingsDirty && (
              <div style={{ marginTop: 14 }}>
                <button
                  className={styles.saveBtn}
                  onClick={async () => {
                    try {
                      await updateSettingsMutation.mutateAsync({
                        offer_dispatch_mode: transportDispatchMode,
                        default_pickup_mode: transportPickupMode,
                        entry_points: transportEntryPoints,
                        default_contact_phone: transportContactPhone || null,
                      });
                      setTransportSettingsDirty(false);
                      toast.success('Paramètres de demande enregistrés');
                    } catch (err) {
                      toast.error(err?.response?.data?.error || 'Erreur lors de la sauvegarde');
                    }
                  }}
                  disabled={updateSettingsMutation.isPending}
                  style={{ width: 'auto' }}
                >
                  <FaSave /> {updateSettingsMutation.isPending ? 'Enregistrement...' : 'Enregistrer les paramètres'}
                </button>
              </div>
            )}
          </div>

          {/* Titre liste + bouton ajouter */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <div>
              <h4 style={{ margin: 0, fontSize: 15, color: '#333' }}>Ordre de priorité</h4>
              <span style={{ fontSize: 12, color: '#888' }}>
                Le premier transporteur dispose d'un délai de réponse avant escalade au suivant.
              </span>
            </div>
            {canAdmin && (
              <button
                className={styles.addKeyBtn}
                onClick={() => setShowAddCompany(true)}
                style={{ width: 'auto', marginBottom: 0, flexShrink: 0 }}
              >
                <FaPlus /> Ajouter un transporteur
              </button>
            )}
          </div>
          
          {loadingPrefs ? (
            <p>Chargement...</p>
          ) : preferences.length === 0 ? (
            <div style={{
              background: '#fff8e1',
              border: '1px solid #ffe082',
              borderRadius: 8,
              padding: '16px 20px',
              textAlign: 'center',
              color: '#5d4037',
              lineHeight: 1.6,
            }}>
              <strong>Aucun transporteur n'est actuellement configuré.</strong>
              <br />
              Tant qu'aucune entreprise n'est ajoutée, aucune demande ne pourra être envoyée automatiquement.
              <br />
              <span style={{ fontSize: 13 }}>
                Ajoutez au moins un transporteur pour activer l'attribution des transports.
              </span>
            </div>
          ) : (
            <>
              <div className={styles.preferencesList}>
                {preferences.map((pref, index) => (
                  <div key={pref.company_id} className={styles.preferenceItem}>
                    <span className={styles.priority}>{index + 1}</span>
                    <span className={styles.companyName}>
                      {pref.company_name || `Entreprise #${pref.company_id}`}
                    </span>
                    <div className={styles.moveButtons}>
                      <button
                        onClick={() => movePreference(index, -1)}
                        disabled={index === 0 || !canAdmin}
                        title="Monter dans la priorité"
                      >
                        <FaArrowUp />
                      </button>
                      <button
                        onClick={() => movePreference(index, 1)}
                        disabled={index === preferences.length - 1 || !canAdmin}
                        title="Descendre dans la priorité"
                      >
                        <FaArrowDown />
                      </button>
                      {canAdmin && (
                        <button
                          className={styles.removeBtn}
                          onClick={() => removePreference(index)}
                          title="Retirer ce transporteur"
                        >
                          <FaTrash />
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
              
              <p style={{ fontSize: 11, color: '#999', margin: '8px 0 16px' }}>
                Le retrait d'un transporteur n'affecte pas les demandes déjà envoyées.
              </p>

              {canAdmin && (
                <div className={styles.saveBtnRow}>
                  <button 
                    className={styles.saveBtn}
                    onClick={savePreferences}
                    disabled={updatePrefsMutation.isPending}
                  >
                    <FaSave /> {updatePrefsMutation.isPending ? 'Enregistrement...' : 'Enregistrer l\'ordre'}
                  </button>
                </div>
              )}
            </>
          )}

          {/* Note droits */}
          {!canAdmin && (
            <p style={{ fontSize: 12, color: '#999', marginTop: 16, fontStyle: 'italic' }}>
              Seuls les administrateurs peuvent modifier l'organisation des transports.
            </p>
          )}

          {/* Confirmation retrait dernier transporteur */}
          {confirmRemoveIndex !== null && (
            <div className={styles.modal}>
              <div className={styles.modalContent}>
                <div className={styles.modalHeader}>
                  <h3>Retirer le dernier transporteur ?</h3>
                  <button onClick={() => setConfirmRemoveIndex(null)}><FaTimes /></button>
                </div>
                <div className={styles.modalBody}>
                  <div style={{
                    background: '#fff8e1',
                    border: '1px solid #ffe082',
                    borderRadius: 8,
                    padding: '12px 16px',
                    color: '#5d4037',
                    lineHeight: 1.6,
                    fontSize: 13,
                  }}>
                    <strong>Attention :</strong> sans transporteur configuré, aucune demande ne pourra
                    être envoyée automatiquement.
                  </div>
                </div>
                <div className={styles.modalActions}>
                  <button onClick={() => setConfirmRemoveIndex(null)}>Annuler</button>
                  <button
                    className={styles.revokeBtn}
                    onClick={confirmRemoveLast}
                    style={{ padding: '10px 20px', borderRadius: '8px', fontSize: '14px' }}
                  >
                    Confirmer le retrait
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* AddCompanyModal */}
          {showAddCompany && (
            <AddCompanyModal
              currentPreferences={preferences}
              onAdd={handleAddCompany}
              onClose={() => setShowAddCompany(false)}
            />
          )}
        </div>
      )}
      
      {/* API Keys */}
      {activeTab === 'apikeys' && (
        <div className={styles.section}>
          <div className={styles.sectionHeader}>
            <h3>Clés API</h3>
            <p>Gérez les clés d'accès pour vos intégrations DPI</p>
          </div>
          
          <button 
            className={styles.addKeyBtn}
            onClick={() => setShowCreateKey(true)}
          >
            <FaPlus /> Nouvelle clé API
          </button>
          
          {/* Created key display */}
          {createdKey && (
            <div className={styles.createdKeyAlert}>
              <div className={styles.alertHeader}>
                <strong>Nouvelle clé créée</strong>
                <button onClick={() => setCreatedKey(null)}><FaTimes /></button>
              </div>
              <p>Copiez cette clé maintenant. Elle ne sera plus affichée.</p>
              <div className={styles.keyDisplay}>
                <code>{createdKey}</code>
                <button onClick={() => copyToClipboard(createdKey)}>
                  <FaCopy />
                </button>
              </div>
            </div>
          )}
          
          {/* Keys list */}
          {loadingKeys ? (
            <p>Chargement...</p>
          ) : keys.length === 0 ? (
            <p className={styles.emptyState}>Aucune clé API</p>
          ) : (
            <div className={styles.keysList}>
              {keys.map((key) => (
                <div key={key.id} className={`${styles.keyItem} ${key.revoked_at ? styles.revoked : ''}`}>
                  <div className={styles.keyInfo}>
                    <span className={styles.keyName}>{key.name}</span>
                    <span className={styles.keyPrefix}>
                      {key.key_prefix}...
                    </span>
                  </div>
                  <div className={styles.keyMeta}>
                    <span>Scopes: {key.scopes?.join(', ') || '-'}</span>
                    <span>Créée: {formatDate(key.created_at)}</span>
                    {key.last_used_at && <span>Dernière util.: {formatDate(key.last_used_at)}</span>}
                    {key.revoked_at && <span className={styles.revokedBadge}>Révoquée</span>}
                  </div>
                  {!key.revoked_at && (
                    <button 
                      className={styles.revokeBtn}
                      onClick={() => setRevokeKeyId(key.id)}
                      disabled={revokeKeyMutation.isPending}
                    >
                      Révoquer
                    </button>
                  )}
                </div>
              ))}
            </div>
          )}
          
          {/* Create key modal */}
          {showCreateKey && (
            <div className={styles.modal}>
              <div className={styles.modalContent}>
                <div className={styles.modalHeader}>
                  <h3>Nouvelle clé API</h3>
                  <button onClick={() => setShowCreateKey(false)}><FaTimes /></button>
                </div>
                <div className={styles.modalBody}>
                  <div className={styles.field}>
                    <label>Nom de la clé *</label>
                    <input
                      type="text"
                      value={newKeyName}
                      onChange={(e) => setNewKeyName(e.target.value)}
                      placeholder="Ex: Intégration DPI"
                    />
                  </div>
                  
                  <div className={styles.field}>
                    <label>Permissions *</label>
                    <div className={styles.scopesGrid}>
                      {AVAILABLE_SCOPES.map((scope) => (
                        <label key={scope.value} className={styles.scopeItem}>
                          <input
                            type="checkbox"
                            checked={selectedScopes.includes(scope.value)}
                            onChange={() => toggleScope(scope.value)}
                          />
                          <span>{scope.label}</span>
                        </label>
                      ))}
                    </div>
                  </div>
                </div>
                <div className={styles.modalActions}>
                  <button onClick={() => setShowCreateKey(false)}>Annuler</button>
                  <button 
                    className={styles.createBtn}
                    onClick={handleCreateKey}
                    disabled={createKeyMutation.isPending}
                  >
                    Créer la clé
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Revoke confirmation modal */}
          {revokeKeyId && (
            <div className={styles.modal}>
              <div className={styles.modalContent}>
                <div className={styles.modalHeader}>
                  <h3>Révoquer cette clé API ?</h3>
                  <button onClick={() => setRevokeKeyId(null)}><FaTimes /></button>
                </div>
                <div className={styles.modalBody}>
                  <p style={{ margin: 0, color: '#666' }}>
                    Cette action est irréversible. La clé ne pourra plus être utilisée pour l'authentification API.
                  </p>
                </div>
                <div className={styles.modalActions}>
                  <button onClick={() => setRevokeKeyId(null)}>Annuler</button>
                  <button 
                    className={styles.revokeBtn}
                    onClick={() => handleRevokeKey(revokeKeyId)}
                    disabled={revokeKeyMutation.isPending}
                    style={{ padding: '10px 20px', borderRadius: '8px', fontSize: '14px' }}
                  >
                    {revokeKeyMutation.isPending ? 'Révocation...' : 'Confirmer la révocation'}
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default InstitutionSettings;
