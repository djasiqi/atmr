// frontend/src/pages/company/Settings/tabs/GeneralTab.jsx
import React, { useRef } from 'react';
import styles from '../CompanySettings.module.css';
import AddressAutocomplete from '../../../../components/common/AddressAutocomplete';

// Helpers
const formatIbanPretty = (value = '') => {
  const v = value.replace(/\s+/g, '').toUpperCase();
  return v.replace(/(.{4})/g, '$1 ').trim();
};

const ReadonlyField = ({ label, value }) => (
  <div className={`${styles.formGroup} ${styles.readonlyField}`}>
    <label style={{ opacity: 0.75 }}>{label}</label>
    <div className={styles.readonlyBox}>{value || '—'}</div>
  </div>
);

const GeneralTab = ({
  company,
  editMode,
  form,
  fieldErrors,
  handleChange,
  handleAddressSelect,
  logoPreview,
  onClickPickFile: _onClickPickFile,
  onPickFile,
  logoUrlEditOpen,
  setLogoUrlEditOpen,
  logoUrlInput,
  setLogoUrlInput,
  onSaveLogoUrl,
  onRemoveLogo,
  logoBusy,
}) => {
  const fileInputRef = useRef(null);

  return (
    <>
      {/* SECTION LOGO */}
      <section className={`${styles.section} ${styles.logoSection}`}>
        <h2>🎨 Identité visuelle</h2>

        <div className={styles.logoRow}>
          <div className={styles.logoBox}>
            {logoPreview ? (
              <img
                src={logoPreview}
                alt="Logo de l'entreprise"
                className={styles.logoPreview}
                loading="lazy"
                onError={(e) => {
                  console.warn('Erreur de chargement du logo dans les paramètres:', logoPreview);
                  e.currentTarget.style.display = 'none';
                  // Afficher le placeholder si l'image échoue
                  const placeholder = e.currentTarget.nextElementSibling;
                  if (placeholder && placeholder.classList.contains(styles.logoPlaceholder)) {
                    placeholder.style.display = 'flex';
                  }
                }}
              />
            ) : null}
            {!logoPreview && (
              <div className={styles.logoPlaceholder}>
                <span>Pas de logo</span>
              </div>
            )}
          </div>

          <div className={styles.logoActions}>
            <div className={styles.chip}>Formats: PNG / JPG / SVG — Max 2 Mo</div>

            <div className={styles.actionsRow}>
              <button
                type="button"
                className={`${styles.button} ${styles.primary}`}
                onClick={() => fileInputRef.current?.click()}
                disabled={logoBusy}
              >
                {logoBusy ? 'Téléversement…' : '📤 Téléverser un fichier'}
              </button>

              <input
                ref={fileInputRef}
                type="file"
                accept="image/png, image/jpeg, image/svg+xml"
                style={{ display: 'none' }}
                onChange={onPickFile}
              />

              <button
                type="button"
                className={`${styles.button} ${styles.secondary}`}
                onClick={() => setLogoUrlEditOpen((v) => !v)}
                disabled={logoBusy}
              >
                {logoUrlEditOpen ? 'Annuler URL' : '🔗 Utiliser une URL'}
              </button>

              {company?.logo_url && (
                <button
                  type="button"
                  className={`${styles.button} ${styles.danger}`}
                  onClick={onRemoveLogo}
                  disabled={logoBusy}
                >
                  🗑️ Supprimer le logo
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
                  💾 Enregistrer l'URL
                </button>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* MODE LECTURE SEULE */}
      {!editMode && (
        <>
          <section className={styles.section}>
            <h2>📍 Coordonnées</h2>
            <ReadonlyField label="Nom de l'entreprise" value={company.name} />
            <ReadonlyField label="Adresse opérationnelle" value={company.address} />
            <ReadonlyField
              label="Email de contact"
              value={company.contact_email || company.email}
            />
            <ReadonlyField label="Téléphone" value={company.contact_phone || company.phone} />
          </section>

          <section className={styles.section}>
            <h2>💼 Légal & facturation</h2>
            <ReadonlyField
              label="IBAN"
              value={company.iban ? formatIbanPretty(company.iban) : ''}
            />
            <ReadonlyField label="IDE / UID" value={company.uid_ide} />
            <ReadonlyField label="Email de facturation" value={company.billing_email} />
            <ReadonlyField label="Notes de facturation" value={company.billing_notes} />
          </section>

          <section className={styles.section}>
            <h2>🏢 Adresse de domiciliation</h2>
            <ReadonlyField label="Adresse (ligne 1)" value={company.domicile_address_line1} />
            <ReadonlyField label="Adresse (ligne 2)" value={company.domicile_address_line2} />
            <ReadonlyField label="NPA" value={company.domicile_zip} />
            <ReadonlyField label="Ville" value={company.domicile_city} />
            <ReadonlyField label="Pays (ISO-2)" value={company.domicile_country || 'CH'} />
          </section>
        </>
      )}

      {/* MODE EDITION */}
      {editMode && (
        <section className={styles.section}>
          <div className={styles.settingsForm}>
            <h2 style={{ gridColumn: '1 / -1' }}>📍 Coordonnées</h2>

            <div className={styles.formGroup}>
              <label htmlFor="name">Nom de l'entreprise</label>
              <input id="name" name="name" value={form.name} onChange={handleChange} required />
              {fieldErrors.name && <small className={styles.fieldError}>{fieldErrors.name}</small>}
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="address">Adresse opérationnelle</label>
              <AddressAutocomplete
                name="address"
                value={form.address}
                onChange={handleChange}
                onSelect={handleAddressSelect}
                placeholder="Saisir l'adresse opérationnelle..."
              />
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="contact_email">Email de contact</label>
              <input
                type="email"
                id="contact_email"
                name="contact_email"
                value={form.contact_email}
                onChange={handleChange}
              />
              {fieldErrors.contact_email && (
                <small className={styles.fieldError}>{fieldErrors.contact_email}</small>
              )}
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="contact_phone">Téléphone</label>
              <input
                id="contact_phone"
                name="contact_phone"
                value={form.contact_phone}
                onChange={handleChange}
              />
              {fieldErrors.contact_phone && (
                <small className={styles.fieldError}>{fieldErrors.contact_phone}</small>
              )}
            </div>

            <h2 style={{ gridColumn: '1 / -1' }}>💼 Légal & facturation</h2>

            <div className={styles.formGroup}>
              <label htmlFor="iban">IBAN</label>
              <input
                id="iban"
                name="iban"
                value={form.iban}
                onChange={handleChange}
                placeholder="CH93 0076 2011 6238 5295 7"
              />
              {fieldErrors.iban && <small className={styles.fieldError}>{fieldErrors.iban}</small>}
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="uid_ide">IDE / UID</label>
              <input
                id="uid_ide"
                name="uid_ide"
                value={form.uid_ide}
                onChange={handleChange}
                placeholder="CHE-123.456.789"
              />
              {fieldErrors.uid_ide && (
                <small className={styles.fieldError}>{fieldErrors.uid_ide}</small>
              )}
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="billing_email">Email de facturation</label>
              <input
                type="email"
                id="billing_email"
                name="billing_email"
                value={form.billing_email}
                onChange={handleChange}
              />
              {fieldErrors.billing_email && (
                <small className={styles.fieldError}>{fieldErrors.billing_email}</small>
              )}
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="billing_notes">Notes de facturation</label>
              <textarea
                id="billing_notes"
                name="billing_notes"
                value={form.billing_notes}
                onChange={handleChange}
                rows={3}
              />
            </div>

            <h2 style={{ gridColumn: '1 / -1' }}>🏢 Adresse de domiciliation</h2>

            <div className={styles.formGroup}>
              <label htmlFor="domicile_address_line1">Adresse (ligne 1)</label>
              <input
                id="domicile_address_line1"
                name="domicile_address_line1"
                value={form.domicile_address_line1}
                onChange={handleChange}
              />
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="domicile_address_line2">Adresse (ligne 2)</label>
              <input
                id="domicile_address_line2"
                name="domicile_address_line2"
                value={form.domicile_address_line2}
                onChange={handleChange}
              />
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="domicile_zip">NPA</label>
              <input
                id="domicile_zip"
                name="domicile_zip"
                value={form.domicile_zip}
                onChange={handleChange}
              />
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="domicile_city">Ville</label>
              <input
                id="domicile_city"
                name="domicile_city"
                value={form.domicile_city}
                onChange={handleChange}
              />
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="domicile_country">Pays (ISO-2)</label>
              <input
                id="domicile_country"
                name="domicile_country"
                value={form.domicile_country}
                onChange={handleChange}
                maxLength={2}
              />
            </div>
          </div>
        </section>
      )}
    </>
  );
};

export default GeneralTab;
