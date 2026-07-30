// frontend/src/pages/company/Settings/CompanySettings.jsx
import React, { useEffect, useMemo, useRef, useState, Suspense, lazy } from 'react';
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
import { useLirieCompany } from '../../../hooks/useLirieCompany';
import { updateCompanyInfo, uploadCompanyLogo } from '../../../services/companyService';
import resolveLogoUrl from '../../../utils/resolveLogoUrl';

// Onglets chargés à la demande (Lot 7 perf) — chunk importé seulement au premier
// affichage réel de chaque onglet, jamais les 7 au chargement initial de la page.
const GeneralTab = lazy(() => import('./tabs/GeneralTab'));
const OperationsTab = lazy(() => import('./tabs/OperationsTab'));
const PartnershipsTab = lazy(() => import('./tabs/PartnershipsTab'));
const BillingTab = lazy(() => import('./tabs/BillingTab'));
const NotificationsTab = lazy(() => import('./tabs/NotificationsTab'));
const SecurityTab = lazy(() => import('./tabs/SecurityTab'));
const VehiclesTab = lazy(() => import('./tabs/VehiclesTab'));

// Validations locales
const emailRx = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const phoneRx = /^\+?[0-9\s\-()]{7,20}$/;
const uidRx = /^(CHE[- ]?\d{3}\.\d{3}\.\d{3}(\s*TVA)?)$|^(CHE[- ]?\d{9}(\s*TVA)?)$/i;

export default function CompanySettings() {
  const { company, companyError: loadError, loadingCompany, reloadCompany } = useLirieCompany();
  const location = useLocation();

  // Onglet actif (détecte le hash dans l'URL)
  const [activeTab, setActiveTab] = useState(() => {
    const hash = location.hash.replace('#', '');
    const validTabs = ['general', 'operations', 'partnerships', 'billing', 'notifications', 'security', 'vehicles'];
    return validTabs.includes(hash) ? hash : 'general';
  });

  // Onglets déjà affichés au moins une fois : restent montés (masqués en CSS) après
  // un changement d'onglet pour préserver les formulaires non enregistrés (Lot 7 perf) —
  // le "Enregistrer" global appelle les refs de tous les onglets édités, pas seulement
  // celui actuellement visible, donc un démontage ferait perdre ces changements.
  const [visitedTabs, setVisitedTabs] = useState(() => new Set([activeTab]));
  useEffect(() => {
    setVisitedTabs((prev) => (prev.has(activeTab) ? prev : new Set(prev).add(activeTab)));
  }, [activeTab]);

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
    if (!form.name?.trim()) errs.name = "Le nom de l'entreprise est requis.";
    return errs;
  }, [form, isEditing]);
  const hasErrors = Object.keys(fieldErrors).length > 0;

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({
      ...prev,
      [name]: value,
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
      const errorMsg = err?.response?.data?.error || err?.message || 'Erreur lors de la sauvegarde.';
      setError(errorMsg);
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
          {error && <div className={styles.error}>{error}</div>}

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

          {/* Zone C — Contenu de l'onglet actif (les onglets déjà visités restent montés,
              masqués en CSS, pour ne pas perdre un formulaire en cours d'édition) */}
          {company && (
            <div className={styles.tabContent}>
              <Suspense fallback={<p>Chargement de l'onglet…</p>}>
                {visitedTabs.has('general') && (
                  <div hidden={activeTab !== 'general'}>
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
                  </div>
                )}

                {visitedTabs.has('operations') && (
                  <div hidden={activeTab !== 'operations'}>
                    <OperationsTab ref={operationsRef} isEditing={isEditing} />
                  </div>
                )}
                {visitedTabs.has('partnerships') && (
                  <div hidden={activeTab !== 'partnerships'}>
                    <PartnershipsTab ref={partnershipsRef} isEditing={isEditing} />
                  </div>
                )}
                {visitedTabs.has('vehicles') && (
                  <div hidden={activeTab !== 'vehicles'}>
                    <VehiclesTab ref={vehiclesRef} isEditing={isEditing} />
                  </div>
                )}
                {visitedTabs.has('billing') && (
                  <div hidden={activeTab !== 'billing'}>
                    <BillingTab ref={billingRef} companyId={company?.id} isEditing={isEditing} />
                  </div>
                )}
                {visitedTabs.has('notifications') && (
                  <div hidden={activeTab !== 'notifications'}>
                    <NotificationsTab isEditing={isEditing} />
                  </div>
                )}
                {visitedTabs.has('security') && (
                  <div hidden={activeTab !== 'security'}>
                    <SecurityTab isEditing={isEditing} />
                  </div>
                )}
              </Suspense>
            </div>
          )}
        </main>
  );
}
