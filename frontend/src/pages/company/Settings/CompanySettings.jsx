// src/pages/company/Settings/CompanySettings.jsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import styles from "./CompanySettings.module.css";
import CompanyHeader from "../../../components/layout/Header/CompanyHeader";
import CompanySidebar from "../../../components/layout/Sidebar/CompanySidebar/CompanySidebar";

import useCompanyData from "../../../hooks/useCompanyData";
import { updateCompanyInfo, uploadCompanyLogo } from "../../../services/companyService";

// ======== Helpers globaux (OK au niveau module) ========

// URL base backend depuis .env CRA
const API_BASE = (process.env.REACT_APP_API_BASE_URL || "").replace(/\/+$/, "");

// Normalise une URL logo (absolue si besoin)
const resolveLogoUrl = (val) => {
  if (!val) return null;
  if (/^(https?:|data:)/i.test(val)) return val;      // déjà absolu ou data:
  const path = val.startsWith("/") ? val : `/${val}`; // ex: "/uploads/…"
  return `${API_BASE}${path}`;
};

// Validations locales
const emailRx = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const phoneRx = /^\+?[0-9\s\-()]{7,20}$/;
const uidRx = /^(CHE[- ]?\d{3}\.\d{3}\.\d{3}(\s*TVA)?)$|^(CHE[- ]?\d{9}(\s*TVA)?)$/i;

function normalizeIban(value = "") {
  return value.replace(/\s+/g, "").toUpperCase();
}
function formatIbanPretty(value = "") {
  const v = normalizeIban(value);
  return v.replace(/(.{4})/g, "$1 ").trim();
}
function ibanChecksumIsValid(iban) {
  const v = normalizeIban(iban);
  if (!v) return true; // champ optionnel
  if (v.length < 15 || v.length > 34) return false;
  if (!/^[A-Z]{2}\d{2}[A-Z0-9]+$/.test(v)) return false;
  const rearranged = v.slice(4) + v.slice(0, 4);
  const expanded = rearranged.replace(/[A-Z]/g, ch => (ch.charCodeAt(0) - 55).toString());
  let remainder = 0;
  for (let i = 0; i < expanded.length; i += 7) {
    remainder = parseInt(String(remainder) + expanded.slice(i, i + 7), 10) % 97;
  }
  return remainder === 1;
}

// Petit composant pour l’affichage lecture seule
function ReadonlyField({ label, value }) {
  return (
    <div className={`${styles.formGroup} ${styles.readonlyField}`}>
      <label style={{ opacity: 0.75 }}>{label}</label>
      <div className={styles.readonlyBox}>
        {value || "—"}
      </div>
    </div>
  );
}

export default function CompanySettings() {
  const { company, error: loadError, reloadCompany } = useCompanyData();

  const [editMode, setEditMode] = useState(false);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  // -------- Logo (hooks DOIVENT être dans le composant) --------
  const [logoPreview, setLogoPreview] = useState(null);
  const [logoUrlEditOpen, setLogoUrlEditOpen] = useState(false);
  const [logoUrlInput, setLogoUrlInput] = useState("");
  const [logoBusy, setLogoBusy] = useState(false);
  const fileInputRef = useRef(null);

  // Met à jour l’aperçu quand logo_url change (supporte les chemins relatifs)
  useEffect(() => {
    setLogoPreview(resolveLogoUrl(company?.logo_url));
  }, [company?.logo_url]);

  // -------- Form principal --------
  const [form, setForm] = useState({
    // Coordonnées
    name: "",
    address: "",
    contact_email: "",
    contact_phone: "",
    // Légal & facturation
    iban: "",
    uid_ide: "",
    billing_email: "",
    billing_notes: "",
    // Domiciliation
    domicile_address_line1: "",
    domicile_address_line2: "",
    domicile_zip: "",
    domicile_city: "",
    domicile_country: "CH",
  });

  // Hydrate le formulaire quand les données arrivent
  useEffect(() => {
    if (!company) return;
    setForm({
      name: company.name || "",
      address: company.address || "",
      contact_email: company.contact_email || company.email || "",
      contact_phone: company.contact_phone || company.phone || "",
      iban: company.iban ? formatIbanPretty(company.iban) : "",
      uid_ide: company.uid_ide || "",
      billing_email: company.billing_email || "",
      billing_notes: company.billing_notes || "",
      domicile_address_line1: company.domicile_address_line1 || "",
      domicile_address_line2: company.domicile_address_line2 || "",
      domicile_zip: company.domicile_zip || "",
      domicile_city: company.domicile_city || "",
      domicile_country: company.domicile_country || "CH",
    });
    setLogoUrlInput(company.logo_url || "");
  }, [company]);

  // Erreurs de champ (édition uniquement)
  const fieldErrors = useMemo(() => {
    if (!editMode) return {};
    const errs = {};
    if (form.contact_email && !emailRx.test(form.contact_email)) errs.contact_email = "Email invalide.";
    if (form.billing_email && !emailRx.test(form.billing_email)) errs.billing_email = "Email de facturation invalide.";
    if (form.contact_phone && !phoneRx.test(form.contact_phone)) errs.contact_phone = "Téléphone invalide.";
    if (form.uid_ide && !uidRx.test(form.uid_ide.trim())) errs.uid_ide = "IDE/UID invalide (ex: CHE-123.456.789).";
    if (form.iban && !ibanChecksumIsValid(form.iban)) errs.iban = "IBAN invalide (checksum).";
    if (!form.name?.trim()) errs.name = "Le nom de l’entreprise est requis.";
    return errs;
  }, [form, editMode]);
  const hasErrors = Object.keys(fieldErrors).length > 0;

  // Handlers formulaire principal
  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm(prev => ({
      ...prev,
      [name]: name === "iban" ? formatIbanPretty(value) : value,
    }));
  };

  const onClickEdit = () => {
    setMessage("");
    setError("");
    setEditMode(true);
  };

  const onClickCancel = () => {
    if (company) {
      setForm({
        name: company.name || "",
        address: company.address || "",
        contact_email: company.contact_email || company.email || "",
        contact_phone: company.contact_phone || company.phone || "",
        iban: company.iban ? formatIbanPretty(company.iban) : "",
        uid_ide: company.uid_ide || "",
        billing_email: company.billing_email || "",
        billing_notes: company.billing_notes || "",
        domicile_address_line1: company.domicile_address_line1 || "",
        domicile_address_line2: company.domicile_address_line2 || "",
        domicile_zip: company.domicile_zip || "",
        domicile_city: company.domicile_city || "",
        domicile_country: company.domicile_country || "CH",
      });
    }
    setEditMode(false);
    setError("");
    setMessage("");
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setMessage("");
    setError("");
    if (hasErrors) {
      setError("Veuillez corriger les erreurs du formulaire.");
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
      setMessage("Paramètres enregistrés avec succès.");
      setEditMode(false);
      await reloadCompany?.();
      setForm(prev => ({
        ...prev,
        iban: updated?.iban ? formatIbanPretty(updated.iban) : prev.iban,
        uid_ide: updated?.uid_ide ?? prev.uid_ide,
      }));
    } catch (err) {
      setError(err?.response?.data?.error || err?.message || "Erreur lors de la sauvegarde.");
    } finally {
      setSaving(false);
    }
  };

  // ======== LOGO: upload fichier ========
  const onClickPickFile = () => fileInputRef.current?.click();

  const onPickFile = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // validations simples
    const allowed = ["image/png", "image/jpeg", "image/jpg", "image/svg+xml"];
    if (!allowed.includes(file.type)) {
      setError("Format de logo non supporté (PNG, JPG, SVG).");
      return;
    }
    if (file.size > 2 * 1024 * 1024) { // 2 Mo
      setError("Le fichier est trop volumineux (max 2 Mo).");
      return;
    }

    // Aperçu immédiat local
    const localUrl = URL.createObjectURL(file);
    setLogoPreview(localUrl);

    setLogoBusy(true);
    setError("");
    setMessage("");
    try {
      await uploadCompanyLogo(file); // POST /companies/me/logo
      await reloadCompany?.();       // mettra à jour company.logo_url -> useEffect remettra logoPreview avec resolveLogoUrl
      setMessage("Logo mis à jour.");
      setLogoUrlEditOpen(false);
    } catch (err) {
      setError(err?.response?.data?.error || err?.message || "Échec de l’upload du logo.");
      // rollback: si erreur, on repasse à l’URL du backend
      setLogoPreview(resolveLogoUrl(company?.logo_url));
    } finally {
      setLogoBusy(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
      // facultatif: URL.revokeObjectURL(localUrl);
    }
  };

  // ======== LOGO: URL directe ========
  const onSaveLogoUrl = async () => {
    if (!logoUrlInput?.trim()) {
      setError("Veuillez saisir une URL valide.");
      return;
    }
    setLogoBusy(true);
    setError("");
    setMessage("");
    try {
      await updateCompanyInfo({ logo_url: logoUrlInput.trim() });
      await reloadCompany?.();
      setMessage("Logo mis à jour via URL.");
      setLogoUrlEditOpen(false);
    } catch (err) {
      setError(err?.response?.data?.error || err?.message || "Impossible d’enregistrer l’URL du logo.");
    } finally {
      setLogoBusy(false);
    }
  };

  const onRemoveLogo = async () => {
    if (!window.confirm("Supprimer le logo ?")) return;
    setLogoBusy(true);
    setError("");
    setMessage("");
    try {
      await updateCompanyInfo({ logo_url: null });
      await reloadCompany?.();
      setMessage("Logo supprimé.");
      setLogoUrlInput("");
      setLogoPreview(null);
    } catch (err) {
      setError(err?.response?.data?.error || err?.message || "Impossible de supprimer le logo.");
    } finally {
      setLogoBusy(false);
    }
  };

  // ======== RENDER ========
  return (
    <div className={styles.companyContainer}>
      <CompanyHeader />
      <div className={styles.dashboard}>
        <CompanySidebar />
        <main className={styles.content}>
          {/* Header + bouton Modifier */}
          <div className={styles.settingsHeader}>
            <h1>Paramètres de l’entreprise</h1>
            {!editMode && (
              <button className={`${styles.submitButton} ${styles.primary}`} onClick={onClickEdit}>
                Modifier
              </button>
            )}
          </div>

          {/* Messages globaux */}
          {!company && !loadError && <p>Chargement…</p>}
          {loadError && <div className={styles.error}>{loadError}</div>}
          {message && <div className={styles.success}>{message}</div>}
          {error && <div className={styles.error}>{error}</div>}

          {/* SECTION LOGO */}
          <section className={`${styles.section} ${styles.logoSection}`}>
            <h2>Identité visuelle</h2>

            <div className={styles.logoRow}>
              <div className={styles.logoBox}>
                {logoPreview ? (
                  <img
                    src={logoPreview}
                    alt="Logo de l’entreprise"
                    className={styles.logoPreview}
                    loading="lazy"
                    onError={(e) => { e.currentTarget.src = ""; setLogoPreview(null); }}
                  />
                ) : (
                  <div className={styles.logoPlaceholder}><span>Pas de logo</span></div>
                )}
              </div>

              <div className={styles.logoActions}>
                <div className={styles.chip}>Formats: PNG / JPG / SVG — Max 2 Mo</div>

                <div className={styles.actionsRow}>
                  <button
                    type="button"
                    className={`${styles.button} ${styles.primary}`}
                    onClick={onClickPickFile}
                    disabled={logoBusy}
                  >
                    {logoBusy ? "Téléversement…" : "Téléverser un fichier"}
                  </button>

                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/png, image/jpeg, image/svg+xml"
                    style={{ display: "none" }}
                    onChange={onPickFile}
                  />

                  <button
                    type="button"
                    className={`${styles.button} ${styles.secondary}`}
                    onClick={() => setLogoUrlEditOpen((v) => !v)}
                    disabled={logoBusy}
                  >
                    {logoUrlEditOpen ? "Annuler URL" : "Utiliser une URL"}
                  </button>

                  {company?.logo_url && (
                    <button
                      type="button"
                      className={`${styles.button} ${styles.danger}`}
                      onClick={onRemoveLogo}
                      disabled={logoBusy}
                    >
                      Supprimer le logo
                    </button>
                  )}
                </div>

                {logoUrlEditOpen && (
                  <div className={styles.urlRow}>
                    <input
                      type="url"
                      placeholder="https://…/mon-logo.png"
                      value={logoUrlInput}
                      onChange={(e) => setLogoUrlInput(e.target.value)}
                      className={styles.input}
                    />
                    <button
                      type="button"
                      className={`${styles.button} ${styles.primary}`}
                      onClick={onSaveLogoUrl}
                      disabled={logoBusy || !logoUrlInput?.trim()}
                    >
                      Enregistrer l’URL
                    </button>
                  </div>
                )}
              </div>
            </div>
          </section>

          {/* LECTURE SEULE */}
          {company && !editMode && (
            <>
              <section className={styles.section}>
                <h2>Coordonnées</h2>
                <ReadonlyField label="Nom de l'entreprise" value={company.name} />
                <ReadonlyField label="Adresse opérationnelle" value={company.address} />
                <ReadonlyField label="Email de contact" value={company.contact_email || company.email} />
                <ReadonlyField label="Téléphone" value={company.contact_phone || company.phone} />
              </section>

              <section className={styles.section}>
                <h2>Légal & facturation</h2>
                <ReadonlyField label="IBAN" value={company.iban ? formatIbanPretty(company.iban) : ""} />
                <ReadonlyField label="IDE / UID" value={company.uid_ide} />
                <ReadonlyField label="Email de facturation" value={company.billing_email} />
                <ReadonlyField label="Notes de facturation" value={company.billing_notes} />
              </section>

              <section className={styles.section}>
                <h2>Adresse de domiciliation</h2>
                <ReadonlyField label="Adresse (ligne 1)" value={company.domicile_address_line1} />
                <ReadonlyField label="Adresse (ligne 2)" value={company.domicile_address_line2} />
                <ReadonlyField label="NPA" value={company.domicile_zip} />
                <ReadonlyField label="Ville" value={company.domicile_city} />
                <ReadonlyField label="Pays (ISO-2)" value={company.domicile_country || "CH"} />
              </section>
            </>
          )}

          {/* EDITION */}
          {company && editMode && (
            <form onSubmit={handleSubmit} className={`${styles.settingsForm} ${styles.section}`}>
              <h2>Coordonnées</h2>
              <div className={styles.formGroup}>
                <label htmlFor="name">Nom de l'entreprise</label>
                <input id="name" name="name" value={form.name} onChange={handleChange} required />
                {fieldErrors.name && <small className={styles.fieldError}>{fieldErrors.name}</small>}
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="address">Adresse opérationnelle</label>
                <input id="address" name="address" value={form.address} onChange={handleChange} />
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="contact_email">Email de contact</label>
                <input type="email" id="contact_email" name="contact_email" value={form.contact_email} onChange={handleChange} />
                {fieldErrors.contact_email && <small className={styles.fieldError}>{fieldErrors.contact_email}</small>}
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="contact_phone">Téléphone</label>
                <input id="contact_phone" name="contact_phone" value={form.contact_phone} onChange={handleChange} />
                {fieldErrors.contact_phone && <small className={styles.fieldError}>{fieldErrors.contact_phone}</small>}
              </div>

              <h2>Légal & facturation</h2>
              <div className={styles.formGroup}>
                <label htmlFor="iban">IBAN</label>
                <input id="iban" name="iban" value={form.iban} onChange={handleChange} placeholder="CH93 0076 2011 6238 5295 7" />
                {fieldErrors.iban && <small className={styles.fieldError}>{fieldErrors.iban}</small>}
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="uid_ide">IDE / UID</label>
                <input id="uid_ide" name="uid_ide" value={form.uid_ide} onChange={handleChange} placeholder="CHE-123.456.789" />
                {fieldErrors.uid_ide && <small className={styles.fieldError}>{fieldErrors.uid_ide}</small>}
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="billing_email">Email de facturation</label>
                <input type="email" id="billing_email" name="billing_email" value={form.billing_email} onChange={handleChange} />
                {fieldErrors.billing_email && <small className={styles.fieldError}>{fieldErrors.billing_email}</small>}
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="billing_notes">Notes de facturation</label>
                <textarea id="billing_notes" name="billing_notes" value={form.billing_notes} onChange={handleChange} rows={3} />
              </div>

              <h2>Adresse de domiciliation</h2>
              <div className={styles.formGroup}>
                <label htmlFor="domicile_address_line1">Adresse (ligne 1)</label>
                <input id="domicile_address_line1" name="domicile_address_line1" value={form.domicile_address_line1} onChange={handleChange} />
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="domicile_address_line2">Adresse (ligne 2)</label>
                <input id="domicile_address_line2" name="domicile_address_line2" value={form.domicile_address_line2} onChange={handleChange} />
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="domicile_zip">NPA</label>
                <input id="domicile_zip" name="domicile_zip" value={form.domicile_zip} onChange={handleChange} />
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="domicile_city">Ville</label>
                <input id="domicile_city" name="domicile_city" value={form.domicile_city} onChange={handleChange} />
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="domicile_country">Pays (ISO-2)</label>
                <input id="domicile_country" name="domicile_country" value={form.domicile_country} onChange={handleChange} maxLength={2} />
              </div>

              <div className={styles.actionsRow}>
                <button type="button" onClick={onClickCancel} className={`${styles.button} ${styles.secondary}`}>
                  Annuler
                </button>
                <button type="submit" className={`${styles.button} ${styles.primary}`} disabled={saving || hasErrors}>
                  {saving ? "Enregistrement…" : "Enregistrer"}
                </button>
              </div>
            </form>
          )}
        </main>
      </div>
    </div>
  );
}
