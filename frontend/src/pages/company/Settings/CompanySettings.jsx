// frontend/src/pages/company/Settings/CompanySettings.jsx
import React, { useEffect, useMemo, useRef, useState } from 'react';
import styles from './CompanySettings.module.css';
import CompanyHeader from '../../../components/layout/Header/CompanyHeader';
import CompanySidebar from '../../../components/layout/Sidebar/CompanySidebar/CompanySidebar';
import GeneralTab from './tabs/GeneralTab';
import OperationsTab from './tabs/OperationsTab';
import BillingTab from './tabs/BillingTab';
import NotificationsTab from './tabs/NotificationsTab';
import SecurityTab from './tabs/SecurityTab';

import useCompanyData from '../../../hooks/useCompanyData';
import { updateCompanyInfo, uploadCompanyLogo } from '../../../services/companyService';

// ======== Helpers globaux ========
const API_BASE = (process.env.REACT_APP_API_BASE_URL || '').replace(/\/+$/, '');

const resolveLogoUrl = (val) => {
  if (!val) return null;
  if (/^(https?:|data:)/i.test(val)) return val;

  // ✅ En mode dev (localhost:3000), retourner le chemin relatif pour utiliser le proxy
  const isDevelopment =
    typeof window !== 'undefined' &&
    window.location &&
    /localhost:3000$/i.test(window.location.host);

  if (val.startsWith('/uploads/') || val.startsWith('/')) {
    // En dev, utiliser le chemin relatif (proxy)
    if (isDevelopment) {
      return val;
    }
    // En prod, construire l'URL complète
    const baseUrl = (process.env.REACT_APP_API_BASE_URL || '').replace(/\/api.*$/, '');
    return baseUrl ? `${baseUrl}${val}` : val;
  }

  const path = val.startsWith('/') ? val : `/${val}`;
  return `${API_BASE}${path}`;
};

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
  const { company, error: loadError, reloadCompany } = useCompanyData();

  // Onglet actif (détecte le hash dans l'URL)
  const [activeTab, setActiveTab] = useState(() => {
    const hash = window.location.hash.replace('#', '');
    const validTabs = ['general', 'operations', 'billing', 'notifications', 'security'];
    return validTabs.includes(hash) ? hash : 'general';
  });

  // Écouter les changements de hash
  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash.replace('#', '');
      const validTabs = ['general', 'operations', 'billing', 'notifications', 'security'];
      if (validTabs.includes(hash)) {
        setActiveTab(hash);
      }
    };

    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  const [editMode, setEditMode] = useState(false);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  // -------- Logo --------
  const [logoPreview, setLogoPreview] = useState(null);
  const [logoUrlEditOpen, setLogoUrlEditOpen] = useState(false);
  const [logoUrlInput, setLogoUrlInput] = useState('');
  const [logoBusy, setLogoBusy] = useState(false);
  const fileInputRef = useRef(null);

  useEffect(() => {
    setLogoPreview(resolveLogoUrl(company?.logo_url));
  }, [company?.logo_url]);

  // -------- Form principal (Général) --------
  const [form, setForm] = useState({
    name: '',
    address: '',
    contact_email: '',
    contact_phone: '',
    iban: '',
    uid_ide: '',
    billing_email: '',
    billing_notes: '',
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
      contact_email: company.contact_email || company.email || '',
      contact_phone: company.contact_phone || company.phone || '',
      iban: company.iban ? formatIbanPretty(company.iban) : '',
      uid_ide: company.uid_ide || '',
      billing_email: company.billing_email || '',
      billing_notes: company.billing_notes || '',
      domicile_address_line1: company.domicile_address_line1 || '',
      domicile_address_line2: company.domicile_address_line2 || '',
      domicile_zip: company.domicile_zip || '',
      domicile_city: company.domicile_city || '',
      domicile_country: company.domicile_country || 'CH',
    });
    setLogoUrlInput(company.logo_url || '');
  }, [company]);

  const fieldErrors = useMemo(() => {
    if (!editMode) return {};
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
  }, [form, editMode]);
  const hasErrors = Object.keys(fieldErrors).length > 0;

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({
      ...prev,
      [name]: name === 'iban' ? formatIbanPretty(value) : value,
    }));
  };

  const onClickEdit = () => {
    setMessage('');
    setError('');
    setEditMode(true);
  };

  const onClickCancel = () => {
    if (company) {
      setForm({
        name: company.name || '',
        address: company.address || '',
        contact_email: company.contact_email || company.email || '',
        contact_phone: company.contact_phone || company.phone || '',
        iban: company.iban ? formatIbanPretty(company.iban) : '',
        uid_ide: company.uid_ide || '',
        billing_email: company.billing_email || '',
        billing_notes: company.billing_notes || '',
        domicile_address_line1: company.domicile_address_line1 || '',
        domicile_address_line2: company.domicile_address_line2 || '',
        domicile_zip: company.domicile_zip || '',
        domicile_city: company.domicile_city || '',
        domicile_country: company.domicile_country || 'CH',
      });
    }
    setEditMode(false);
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
    const payload = {
      name: form.name || undefined,
      address: form.address || undefined,
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

    setSaving(true);
    try {
      const updated = await updateCompanyInfo(payload);
      setMessage('Paramètres enregistrés avec succès.');
      setEditMode(false);
      await reloadCompany?.();
      setForm((prev) => ({
        ...prev,
        iban: updated?.iban ? formatIbanPretty(updated.iban) : prev.iban,
        uid_ide: updated?.uid_ide ?? prev.uid_ide,
      }));
    } catch (err) {
      setError(err?.response?.data?.error || err?.message || 'Erreur lors de la sauvegarde.');
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

      if (result?.logo_url) {
        setLogoPreview(resolveLogoUrl(result.logo_url));
      }

      await reloadCompany?.();
      setMessage('Logo mis à jour avec succès.');
      setLogoUrlEditOpen(false);
    } catch (err) {
      setError(err?.response?.data?.error || err?.message || "Échec de l'upload du logo.");
      setLogoPreview(resolveLogoUrl(company?.logo_url));
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
      await reloadCompany?.();
      setMessage('Logo mis à jour via URL.');
      setLogoUrlEditOpen(false);
    } catch (err) {
      setError(
        err?.response?.data?.error || err?.message || "Impossible d'enregistrer l'URL du logo."
      );
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
      await reloadCompany?.();
      setMessage('Logo supprimé.');
      setLogoUrlInput('');
      setLogoPreview(null);
    } catch (err) {
      setError(err?.response?.data?.error || err?.message || 'Impossible de supprimer le logo.');
    } finally {
      setLogoBusy(false);
    }
  };

  // ======== Configuration des onglets ========
  const tabs = [
    { id: 'general', label: 'Général', icon: '🏢' },
    { id: 'operations', label: 'Opérations', icon: '🚗' },
    { id: 'billing', label: 'Facturation', icon: '💰' },
    { id: 'notifications', label: 'Notifications', icon: '📧' },
    { id: 'security', label: 'Sécurité', icon: '🔐' },
  ];

  // ======== RENDER ========
  return (
    <div className={styles.companyContainer}>
      <CompanyHeader />
      <div className={styles.dashboard}>
        <CompanySidebar />
        <main className={styles.content}>
          {/* Header */}
          <div className={styles.settingsHeader}>
            <div className={styles.headerLeft}>
              <h1>⚙️ Paramètres de l'entreprise</h1>
              <p className={styles.headerSubtitle}>Gérez tous les aspects de votre entreprise</p>
            </div>
          </div>

          {/* Messages globaux */}
          {!company && !loadError && <p>Chargement…</p>}
          {loadError && <div className={styles.error}>{loadError}</div>}
          {activeTab === 'general' && message && <div className={styles.success}>{message}</div>}
          {activeTab === 'general' && error && <div className={styles.error}>{error}</div>}

          {/* Navigation par onglets avec bouton Modifier */}
          <div className={styles.tabsContainer}>
            <div className={styles.tabs}>
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  className={`${styles.tab} ${activeTab === tab.id ? styles.active : ''}`}
                  onClick={() => setActiveTab(tab.id)}
                >
                  <span>
                    {tab.icon} {tab.label}
                  </span>
                </button>
              ))}
            </div>
            {activeTab === 'general' && !editMode && (
              <button className={`${styles.submitButton} ${styles.primary}`} onClick={onClickEdit}>
                ✏️ Modifier
              </button>
            )}
          </div>

          {/* Contenu de l'onglet actif */}
          {company && (
            <div className={styles.tabContent}>
              {activeTab === 'general' && (
                <>
                  <GeneralTab
                    company={company}
                    editMode={editMode}
                    form={form}
                    fieldErrors={fieldErrors}
                    handleChange={handleChange}
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

                  {editMode && (
                    <div className={styles.actionsRow}>
                      <button
                        type="button"
                        onClick={onClickCancel}
                        className={`${styles.button} ${styles.secondary}`}
                      >
                        Annuler
                      </button>
                      <button
                        type="button"
                        onClick={handleSubmit}
                        className={`${styles.button} ${styles.primary}`}
                        disabled={saving || hasErrors}
                      >
                        {saving ? '💾 Enregistrement…' : '💾 Enregistrer'}
                      </button>
                    </div>
                  )}
                </>
              )}

              {activeTab === 'operations' && <OperationsTab />}
              {activeTab === 'billing' && <BillingTab />}
              {activeTab === 'notifications' && <NotificationsTab />}
              {activeTab === 'security' && <SecurityTab />}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
