// frontend/src/pages/company/Settings/CompanySettings.jsx
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useLocation } from 'react-router-dom';
import {
  FiEdit2,
  FiSave,
  FiX,
  FiHome,
  FiActivity,
  FiUsers,
  FiTruck,
  FiFileText,
  FiBell,
  FiShield,
} from 'react-icons/fi';
import styles from './CompanySettings.module.css';
import CompanyHeader from '../../../components/layout/Header/CompanyHeader';
import CompanySidebar from '../../../components/layout/Sidebar/CompanySidebar/CompanySidebar';
import GeneralTab from './tabs/GeneralTab';
import OperationsTab from './tabs/OperationsTab';
import PartnershipsTab from './tabs/PartnershipsTab';
import BillingTab from './tabs/BillingTab';
import NotificationsTab from './tabs/NotificationsTab';
import SecurityTab from './tabs/SecurityTab';
import VehiclesTab from './tabs/VehiclesTab';

import useCompanyData from '../../../hooks/useCompanyData';
import { updateCompanyInfo, uploadCompanyLogo } from '../../../services/companyService';
import { getFreshToken } from '../../../services/authService';
import resolveLogoUrl from '../../../utils/resolveLogoUrl';

// Validations locales
const emailRx = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const phoneRx = /^\+?[0-9\s\-()]{7,20}$/;
const uidRx = /^(CHE[- ]?\d{3}\.\d{3}\.\d{3}(\s*TVA)?)$|^(CHE[- ]?\d{9}(\s*TVA)?)$/i;

function normalizeIban(value = '') {
  return value.replace(/\s+/g, '').toUpperCase();
}

function formatIbanPretty(value = '') {
  const v = normalizeIban(value);
  return v.replace(/(.{4})/g, '$1 ').trim();
}

function ibanChecksumIsValid(iban) {
  const v = normalizeIban(iban);
  if (!v) return true;
  if (v.length < 15 || v.length > 34) return false;
  if (!/^[A-Z]{2}\d{2}[A-Z0-9]+$/.test(v)) return false;
  const rearranged = v.slice(4) + v.slice(0, 4);
  const expanded = rearranged.replace(/[A-Z]/g, (ch) => (ch.charCodeAt(0) - 55).toString());
  let remainder = 0;
  for (let i = 0; i < expanded.length; i += 7) {
    remainder = parseInt(String(remainder) + expanded.slice(i, i + 7), 10) % 97;
  }
  return remainder === 1;
}

export default function CompanySettings() {
  const { company, error: loadError, loadingCompany, reloadCompany } = useCompanyData();
  const location = useLocation();

  // Onglet actif (détecte le hash dans l'URL)
  const [activeTab, setActiveTab] = useState(() => {
    const hash = location.hash.replace('#', '');
    const validTabs = ['general', 'operations', 'partnerships', 'billing', 'notifications', 'security', 'vehicles'];
    return validTabs.includes(hash) ? hash : 'general';
  });

  // Écouter les changements de hash (via React Router location)
  useEffect(() => {
    const hash = location.hash.replace('#', '');
    const validTabs = ['general', 'operations', 'partnerships', 'billing', 'notifications', 'security', 'vehicles'];
    if (validTabs.includes(hash)) {
      setActiveTab(hash);
    }
    // Si pas de hash, on reste sur l'onglet actuel (ou 'general' par défaut)
  }, [location.hash]);

  // Écouter aussi les changements de hash via l'événement hashchange (pour les navigations directes)
  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash.replace('#', '');
      const validTabs = ['general', 'operations', 'partnerships', 'billing', 'notifications', 'security', 'vehicles'];
      if (validTabs.includes(hash)) {
        setActiveTab(hash);
      }
    };

    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  const [isEditing, setIsEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [showPasswordModal, setShowPasswordModal] = useState(false);
  const [passwordInput, setPasswordInput] = useState('');
  const [pendingPayload, setPendingPayload] = useState(null);

  // -------- Logo --------
  const [logoPreview, setLogoPreview] = useState(null);
  const [logoUrlEditOpen, setLogoUrlEditOpen] = useState(false);
  const [logoUrlInput, setLogoUrlInput] = useState('');
  const [logoBusy, setLogoBusy] = useState(false);
  const fileInputRef = useRef(null);
  const billingRef = useRef(null);
  const partnershipsRef = useRef(null);
  const vehiclesRef = useRef(null);
  const operationsRef = useRef(null);
  const pendingSaveBillingRef = useRef(false);

  useEffect(() => {
    const resolved = resolveLogoUrl(company?.logo_url);
    setLogoPreview(resolved || null);
  }, [company?.logo_url]);

  // -------- Form principal (Général) --------
  const [form, setForm] = useState({
    name: '',
    address: '',
    latitude: null,
    longitude: null,
    contact_email: '',
    contact_phone: '',
    iban: '',
    uid_ide: '',
    billing_email: '',
    billing_notes: '',
    preferential_rate: '',
    domicile_address_line1: '',
    domicile_address_line2: '',
    domicile_zip: '',
    domicile_city: '',
    domicile_country: 'CH',
  });

  useEffect(() => {
    if (!company) return;
    setForm({
      name: company.name || '',
      address: company.address || '',
      latitude: company.latitude || null,
      longitude: company.longitude || null,
      contact_email: company.contact_email || company.email || '',
      contact_phone: company.contact_phone || company.phone || '',
      iban: company.iban ? formatIbanPretty(company.iban) : '',
      uid_ide: company.uid_ide || '',
      billing_email: company.billing_email || '',
      billing_notes: company.billing_notes || '',
      preferential_rate: company.preferential_rate ? company.preferential_rate.toString() : '',
      domicile_address_line1: company.domicile_address_line1 || '',
      domicile_address_line2: company.domicile_address_line2 || '',
      domicile_zip: company.domicile_zip || '',
      domicile_city: company.domicile_city || '',
      domicile_country: company.domicile_country || 'CH',
    });
    setLogoUrlInput(company.logo_url || '');
  }, [company]);

  const fieldErrors = useMemo(() => {
    if (!isEditing) return {};
    const errs = {};
    if (form.contact_email && !emailRx.test(form.contact_email))
      errs.contact_email = 'Email invalide.';
    if (form.billing_email && !emailRx.test(form.billing_email))
      errs.billing_email = 'Email de facturation invalide.';
    if (form.contact_phone && !phoneRx.test(form.contact_phone))
      errs.contact_phone = 'Téléphone invalide.';
    if (form.uid_ide && !uidRx.test(form.uid_ide.trim()))
      errs.uid_ide = 'IDE/UID invalide (ex: CHE-123.456.789).';
    if (form.iban && !ibanChecksumIsValid(form.iban)) errs.iban = 'IBAN invalide (checksum).';
    if (!form.name?.trim()) errs.name = "Le nom de l'entreprise est requis.";
    return errs;
  }, [form, isEditing]);
  const hasErrors = Object.keys(fieldErrors).length > 0;

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({
      ...prev,
      [name]: name === 'iban' ? formatIbanPretty(value) : value,
    }));
  };

  const handleAddressSelect = (selectedItem) => {
    setForm((prev) => ({
      ...prev,
      address: selectedItem?.label || selectedItem?.address || prev.address,
      latitude: selectedItem?.lat || null,
      longitude: selectedItem?.lon || null,
    }));
  };

  const handleDomicileAddressSelect = (selectedItem) => {
    let streetAddress = selectedItem?.address || selectedItem?.label || '';
    if (streetAddress.includes(',')) {
      const parts = streetAddress.split(',').map(p => p.trim());
      // Si le premier élément est un nombre, on inverse
      if (parts.length === 2 && /^\d+/.test(parts[0])) {
        // Cas "145, Route de Chevrens" -> "Route de Chevrens 145"
        streetAddress = `${parts[1]} ${parts[0]}`;
      } else if (parts.length === 2) {
        // Cas "Route de Chevrens, 145" -> "Route de Chevrens 145"
        streetAddress = `${parts[0]} ${parts[1]}`;
      }
    }
    
    setForm((prev) => ({
      ...prev,
      domicile_address_line1: streetAddress || prev.domicile_address_line1,
      domicile_zip: selectedItem?.postcode || prev.domicile_zip,
      domicile_city: selectedItem?.city || prev.domicile_city,
      domicile_country: selectedItem?.country || prev.domicile_country || 'CH',
    }));
  };

  const onClickEdit = () => {
    setMessage('');
    setError('');
    setIsEditing(true);
  };

  const onClickCancel = () => {
    if (company) {
      setForm({
        name: company.name || '',
        address: company.address || '',
        latitude: company.latitude || null,
        longitude: company.longitude || null,
        contact_email: company.contact_email || company.email || '',
        contact_phone: company.contact_phone || company.phone || '',
        iban: company.iban ? formatIbanPretty(company.iban) : '',
        uid_ide: company.uid_ide || '',
        billing_email: company.billing_email || '',
        billing_notes: company.billing_notes || '',
        preferential_rate: company.preferential_rate ? company.preferential_rate.toString() : '',
        domicile_address_line1: company.domicile_address_line1 || '',
        domicile_address_line2: company.domicile_address_line2 || '',
        domicile_zip: company.domicile_zip || '',
        domicile_city: company.domicile_city || '',
        domicile_country: company.domicile_country || 'CH',
      });
    }
    billingRef.current?.reset();
    operationsRef.current?.reset();
    setIsEditing(false);
    setError('');
    setMessage('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setMessage('');
    setError('');
    if (hasErrors) {
      setError('Veuillez corriger les erreurs du formulaire.');
      return;
    }

    if (billingRef.current?.isReady && !billingRef.current.isReady()) {
      setError('Chargement en cours, veuillez patienter...');
      return;
    }

    setSaving(true);
    try {
      // Sauvegarder GeneralTab (infos entreprise)
      const payload = {
        name: form.name || undefined,
        address: form.address || undefined,
        latitude: form.latitude || undefined,
        longitude: form.longitude || undefined,
        contact_email: form.contact_email || undefined,
        contact_phone: form.contact_phone || undefined,
        billing_email: form.billing_email || undefined,
        billing_notes: form.billing_notes || undefined,
        iban: normalizeIban(form.iban) || undefined,
        uid_ide: form.uid_ide || undefined,
        domicile_address_line1: form.domicile_address_line1 || undefined,
        domicile_address_line2: form.domicile_address_line2 || undefined,
        domicile_zip: form.domicile_zip || undefined,
        domicile_city: form.domicile_city || undefined,
        domicile_country: form.domicile_country || undefined,
      };
      const updated = await updateCompanyInfo(payload);

      // Sauvegarder BillingTab si le ref existe (onglet monte)
      let billingError = null;
      if (billingRef.current?.save) {
        try {
          await billingRef.current.save();
        } catch (billingErr) {
          billingError = billingErr;
        }
      }

      // Sauvegarder PartnershipsTab si le ref existe (onglet monte)
      let partnershipsError = null;
      if (partnershipsRef.current?.save) {
        try {
          await partnershipsRef.current.save();
        } catch (partnerErr) {
          partnershipsError = partnerErr;
        }
      }

      // Sauvegarder VehiclesTab si le ref existe (onglet monte)
      let vehiclesError = null;
      if (vehiclesRef.current?.save) {
        try {
          await vehiclesRef.current.save();
        } catch (vErr) {
          vehiclesError = vErr;
        }
      }

      // Sauvegarder OperationsTab si le ref existe (onglet monte)
      let operationsError = null;
      if (operationsRef.current?.save) {
        try {
          await operationsRef.current.save();
        } catch (operationsErr) {
          operationsError = operationsErr;
        }
      }

      await reloadCompany?.();
      setForm((prev) => ({
        ...prev,
        iban: updated?.iban ? formatIbanPretty(updated.iban) : prev.iban,
        uid_ide: updated?.uid_ide ?? prev.uid_ide,
      }));

      if (billingError || partnershipsError || vehiclesError || operationsError) {
        const parts = [];
        if (billingError) {
          parts.push(`facturation : ${billingError?.response?.data?.error || billingError?.message || ''}`);
        }
        if (partnershipsError) {
          parts.push(`partenariats : ${partnershipsError?.response?.data?.error || partnershipsError?.message || ''}`);
        }
        if (vehiclesError) {
          parts.push(`véhicules : ${vehiclesError?.response?.data?.error || vehiclesError?.message || ''}`);
        }
        if (operationsError) {
          parts.push(`opérations : ${operationsError?.response?.data?.error || operationsError?.message || ''}`);
        }
        setError(`Infos entreprise enregistrees, mais erreur ${parts.join(' / ')}`);
      } else {
        setMessage('Parametres enregistres avec succes.');
        setIsEditing(false);
      }
    } catch (err) {
      if (err?.isFreshTokenRequired) {
        const payload = {
          name: form.name || undefined,
          address: form.address || undefined,
          latitude: form.latitude || undefined,
          longitude: form.longitude || undefined,
          contact_email: form.contact_email || undefined,
          contact_phone: form.contact_phone || undefined,
          billing_email: form.billing_email || undefined,
          billing_notes: form.billing_notes || undefined,
          iban: normalizeIban(form.iban) || undefined,
          uid_ide: form.uid_ide || undefined,
          domicile_address_line1: form.domicile_address_line1 || undefined,
          domicile_address_line2: form.domicile_address_line2 || undefined,
          domicile_zip: form.domicile_zip || undefined,
          domicile_city: form.domicile_city || undefined,
          domicile_country: form.domicile_country || undefined,
        };
        setPendingPayload(payload);
        pendingSaveBillingRef.current = true;
        setShowPasswordModal(true);
        setError('');
      } else {
        const errorMsg = err?.response?.data?.error || err?.message || 'Erreur lors de la sauvegarde.';
        setError(errorMsg);
      }
    } finally {
      setSaving(false);
    }
  };

  // Gérer la soumission du mot de passe pour obtenir un token fresh
  const handlePasswordSubmit = async (e) => {
    e.preventDefault();
    if (!passwordInput.trim()) {
      setError('Veuillez entrer votre mot de passe.');
      return;
    }

    setSaving(true);
    setError('');
    try {
      // Obtenir un token fresh
      await getFreshToken(passwordInput);
      setShowPasswordModal(false);
      setPasswordInput('');
      
      // Réessayer la modification avec le nouveau token
      if (pendingPayload) {
        const updated = await updateCompanyInfo(pendingPayload);
        await reloadCompany?.();
        setForm((prev) => ({
          ...prev,
          iban: updated?.iban ? formatIbanPretty(updated.iban) : prev.iban,
          uid_ide: updated?.uid_ide ?? prev.uid_ide,
        }));
        setPendingPayload(null);
      }

      if (pendingSaveBillingRef.current && billingRef.current?.save) {
        try {
          await billingRef.current.save();
        } finally {
          pendingSaveBillingRef.current = false;
        }
      }

      setMessage('Parametres enregistres avec succes.');
      setIsEditing(false);
    } catch (err) {
      setError(err?.response?.data?.error || err?.message || 'Mot de passe incorrect ou erreur lors de l\'obtention du token.');
    } finally {
      setSaving(false);
    }
  };

  // ======== LOGO: upload fichier ========
  const onPickFile = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const allowed = ['image/png', 'image/jpeg', 'image/jpg', 'image/svg+xml'];
    if (!allowed.includes(file.type)) {
      setError('Format de logo non supporté (PNG, JPG, SVG).');
      return;
    }
    if (file.size > 2 * 1024 * 1024) {
      setError('Le fichier est trop volumineux (max 2 Mo).');
      return;
    }

    const localUrl = URL.createObjectURL(file);
    setLogoPreview(localUrl);

    setLogoBusy(true);
    setError('');
    setMessage('');
    try {
      const result = await uploadCompanyLogo(file);

      // Mettre à jour le preview immédiatement avec le résultat
      if (result?.logo_url) {
        const resolved = resolveLogoUrl(result.logo_url);
        setLogoPreview(resolved || null);
      }

      // Recharger les données de l'entreprise pour synchroniser
      await reloadCompany?.();

      // Mettre à jour le preview avec les données rechargées (au cas où)
      if (company?.logo_url) {
        const resolved = resolveLogoUrl(company.logo_url);
        setLogoPreview(resolved || null);
      }

      setMessage('Logo mis à jour avec succès.');
      setLogoUrlEditOpen(false);
    } catch (err) {
      setError(err?.response?.data?.error || err?.message || "Échec de l'upload du logo.");
      // Restaurer le logo précédent en cas d'erreur
      const resolved = resolveLogoUrl(company?.logo_url);
      setLogoPreview(resolved || null);
    } finally {
      setLogoBusy(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const onSaveLogoUrl = async () => {
    if (!logoUrlInput?.trim()) {
      setError('Veuillez saisir une URL valide.');
      return;
    }
    setLogoBusy(true);
    setError('');
    setMessage('');
    try {
      await updateCompanyInfo({ logo_url: logoUrlInput.trim() });

      // Mettre à jour le preview avec la nouvelle URL
      const resolved = resolveLogoUrl(logoUrlInput.trim());
      setLogoPreview(resolved || null);

      await reloadCompany?.();
      setMessage('Logo mis à jour via URL.');
      setLogoUrlEditOpen(false);

      // S'assurer que le preview est à jour après le reload
      if (company?.logo_url) {
        const resolvedAfterReload = resolveLogoUrl(company.logo_url);
        setLogoPreview(resolvedAfterReload || null);
      }
    } catch (err) {
      setError(
        err?.response?.data?.error || err?.message || "Impossible d'enregistrer l'URL du logo."
      );
      // Restaurer le logo précédent en cas d'erreur
      const resolved = resolveLogoUrl(company?.logo_url);
      setLogoPreview(resolved || null);
    } finally {
      setLogoBusy(false);
    }
  };

  const onRemoveLogo = async () => {
    if (!window.confirm('Supprimer le logo ?')) return;
    setLogoBusy(true);
    setError('');
    setMessage('');
    try {
      await updateCompanyInfo({ logo_url: null });
      setLogoUrlInput('');
      setLogoPreview(null);

      await reloadCompany?.();
      setMessage('Logo supprimé.');

      // S'assurer que le preview est null après le reload
      setLogoPreview(null);
    } catch (err) {
      setError(err?.response?.data?.error || err?.message || 'Impossible de supprimer le logo.');
      // Restaurer le logo précédent en cas d'erreur
      const resolved = resolveLogoUrl(company?.logo_url);
      setLogoPreview(resolved || null);
    } finally {
      setLogoBusy(false);
    }
  };

  // ======== Configuration des onglets (V5) ========
  const tabs = [
    { id: 'general', label: 'General', Icon: FiHome },
    { id: 'operations', label: 'Operations', Icon: FiActivity },
    { id: 'partnerships', label: 'Partenariats', Icon: FiUsers },
    { id: 'vehicles', label: 'Vehicules', Icon: FiTruck },
    { id: 'billing', label: 'Facturation', Icon: FiFileText },
    { id: 'notifications', label: 'Notifications', Icon: FiBell },
    { id: 'security', label: 'Securite', Icon: FiShield },
  ];

  const handleTabClick = (tabId) => {
    setActiveTab(tabId);
    window.location.hash = tabId;
  };

  // ======== RENDER ========
  return (
    <div className={styles.companyContainer}>
      <CompanyHeader />
      <div className={styles.dashboard}>
        <CompanySidebar />
        <main className={styles.content}>
          {/* Zone A — Header sticky (V1/V7) */}
          <div className={styles.settingsHeader}>
            <div className={styles.headerLeft}>
              <h1>Parametres entreprise</h1>
              <p className={styles.headerSubtitle}>
                Gerez les informations, l'exploitation et la facturation
              </p>
            </div>
            <div className={styles.headerActions}>
              {!isEditing ? (
                <button
                  type="button"
                  className={`${styles.button} ${styles.primary}`}
                  onClick={onClickEdit}
                >
                  <FiEdit2 size={14} /> Modifier
                </button>
              ) : (
                <>
                  <button
                    type="button"
                    className={`${styles.button} ${styles.secondary}`}
                    onClick={onClickCancel}
                    disabled={saving}
                  >
                    <FiX size={14} /> Annuler
                  </button>
                  <button
                    type="button"
                    className={`${styles.button} ${styles.primary}`}
                    onClick={handleSubmit}
                    disabled={saving || hasErrors}
                  >
                    <FiSave size={14} /> {saving ? 'Enregistrement...' : 'Enregistrer'}
                  </button>
                </>
              )}
            </div>
          </div>

          {/* Messages globaux */}
          {loadingCompany && <p>Chargement...</p>}
          {loadError && <div className={styles.error}>{loadError}</div>}
          {message && <div className={styles.success}>{message}</div>}
          {error && !showPasswordModal && <div className={styles.error}>{error}</div>}

          {/* Zone B — Tabs segmentees (V4/V5) */}
          <div className={styles.tabsContainer} role="tablist">
            {tabs.map((tab) => {
              const TabIcon = tab.Icon;
              const active = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  role="tab"
                  aria-selected={active}
                  className={`${styles.tab} ${active ? styles.tabActive : ''}`}
                  onClick={() => handleTabClick(tab.id)}
                >
                  <TabIcon size={14} />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </div>

          {/* Zone C — Contenu de l'onglet actif */}
          {company && (
            <div className={styles.tabContent}>
              {activeTab === 'general' && (
                <GeneralTab
                  company={company}
                  isEditing={isEditing}
                  form={form}
                  fieldErrors={fieldErrors}
                  handleChange={handleChange}
                  handleAddressSelect={handleAddressSelect}
                  handleDomicileAddressSelect={handleDomicileAddressSelect}
                  logoPreview={logoPreview}
                  onClickPickFile={() => fileInputRef.current?.click()}
                  onPickFile={onPickFile}
                  logoUrlEditOpen={logoUrlEditOpen}
                  setLogoUrlEditOpen={setLogoUrlEditOpen}
                  logoUrlInput={logoUrlInput}
                  setLogoUrlInput={setLogoUrlInput}
                  onSaveLogoUrl={onSaveLogoUrl}
                  onRemoveLogo={onRemoveLogo}
                  logoBusy={logoBusy}
                />
              )}

              {activeTab === 'operations' && <OperationsTab ref={operationsRef} isEditing={isEditing} />}
              {activeTab === 'partnerships' && <PartnershipsTab ref={partnershipsRef} isEditing={isEditing} />}
              {activeTab === 'vehicles' && <VehiclesTab ref={vehiclesRef} isEditing={isEditing} />}
              {activeTab === 'billing' && <BillingTab ref={billingRef} companyId={company?.id} isEditing={isEditing} />}
              {activeTab === 'notifications' && <NotificationsTab isEditing={isEditing} />}
              {activeTab === 'security' && <SecurityTab isEditing={isEditing} />}
            </div>
          )}
        </main>
      </div>

      {/* Modal mot de passe (V3) */}
      {showPasswordModal && (
        <div
          className={styles.modalOverlay}
          onClick={() => {
            setShowPasswordModal(false);
            setPasswordInput('');
            setPendingPayload(null);
            setError('');
          }}
        >
          <div className={styles.modalContent} onClick={(e) => e.stopPropagation()}>
            <h2 className={styles.modalTitle}>Verification requise</h2>
            <p className={styles.modalText}>
              Pour des raisons de securite, veuillez entrer votre mot de passe pour confirmer cette modification.
            </p>
            <form onSubmit={handlePasswordSubmit}>
              <div className={styles.formGroup}>
                <label htmlFor="password">Mot de passe</label>
                <input
                  id="password"
                  type="password"
                  value={passwordInput}
                  onChange={(e) => setPasswordInput(e.target.value)}
                  className={styles.input}
                  autoFocus
                  disabled={saving}
                />
              </div>
              {error && <div className={styles.error}>{error}</div>}
              <div className={styles.modalActions}>
                <button
                  type="button"
                  className={`${styles.button} ${styles.secondary}`}
                  onClick={() => {
                    setShowPasswordModal(false);
                    setPasswordInput('');
                    setPendingPayload(null);
                    setError('');
                  }}
                  disabled={saving}
                >
                  Annuler
                </button>
                <button
                  type="submit"
                  className={`${styles.button} ${styles.primary}`}
                  disabled={saving || !passwordInput.trim()}
                >
                  {saving ? 'Verification...' : 'Confirmer'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
