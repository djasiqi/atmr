// frontend/src/pages/company/Settings/tabs/GeneralTab.jsx
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { FiUpload, FiLink, FiTrash2, FiChevronDown, FiChevronUp, FiImage, FiMapPin, FiFileText, FiHome } from 'react-icons/fi';
import styles from '../CompanySettings.module.css';
import AddressAutocomplete from '../../../../components/common/AddressAutocomplete';

const ReadonlyField = ({ label, value }) => (
  <div className={styles.fieldRow}>
    <span className={styles.labelMuted}>{label}</span>
    <span className={`${styles.valueText}${!value ? ` ${styles.valueEmpty}` : ''}`}>
      {value || '\u2014'}
    </span>
  </div>
);

const NotesField = ({ label, value }) => {
  const ref = useRef(null);
  const [overflows, setOverflows] = useState(false);
  const [expanded, setExpanded] = useState(false);

  const checkOverflow = useCallback(() => {
    if (ref.current) {
      setOverflows(ref.current.scrollHeight > ref.current.clientHeight + 2);
    }
  }, []);

  useEffect(() => {
    checkOverflow();
  }, [value, checkOverflow]);

  if (!value) {
    return (
      <div className={`${styles.fieldRow} ${styles.fieldGridFull}`}>
        <span className={styles.labelMuted}>{label}</span>
        <span className={`${styles.valueText} ${styles.valueEmpty}`}>{'\u2014'}</span>
      </div>
    );
  }

  return (
    <div className={`${styles.fieldRow} ${styles.fieldGridFull}`}>
      <span className={styles.labelMuted}>{label}</span>
      <div
        ref={ref}
        className={`${styles.valueMultiline} ${expanded ? styles.valueMultilineExpanded : ''}`}
      >
        {value}
      </div>
      {(overflows || expanded) && (
        <button
          type="button"
          className={styles.expandBtn}
          onClick={() => setExpanded((v) => !v)}
        >
          {expanded ? (
            <><FiChevronUp size={11} /> Reduire</>
          ) : (
            <><FiChevronDown size={11} /> Voir plus</>
          )}
        </button>
      )}
    </div>
  );
};

const GeneralTab = ({
  company,
  isEditing,
  form,
  fieldErrors,
  handleChange,
  handleAddressSelect,
  handleDomicileAddressSelect,
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
      {/* Carte 1 : Identite visuelle */}
      <div className={styles.card}>
        <div className={styles.cardHeader}>
          <div className={styles.cardIcon}><FiImage size={16} /></div>
          <div className={styles.cardHeaderText}>
            <h3 className={styles.cardTitle}>Identite visuelle</h3>
            <p className={styles.cardHint}>Logo affiche sur les documents et factures</p>
          </div>
        </div>

        <div className={styles.logoRow}>
          <div className={styles.logoBox}>
            {logoPreview ? (
              <img
                src={logoPreview}
                alt="Logo de l'entreprise"
                className={styles.logoPreview}
                loading="lazy"
                onError={(e) => {
                  e.currentTarget.style.display = 'none';
                  const placeholder = e.currentTarget.nextElementSibling;
                  if (placeholder && placeholder.classList.contains(styles.logoPlaceholder)) {
                    placeholder.style.display = 'flex';
                  }
                }}
              />
            ) : null}
            {!logoPreview && (
              <div className={styles.logoPlaceholder}>
                <span>Aucun logo</span>
              </div>
            )}
          </div>

          <div className={styles.logoActions}>
            <span className={styles.cardHint}>PNG, JPG ou SVG \u2014 2 Mo max</span>

            <div className={styles.logoActionsRow}>
              {logoPreview ? (
                <>
                  <button
                    type="button"
                    className={`${styles.button} ${styles.secondary}`}
                    onClick={() => fileInputRef.current?.click()}
                    disabled={logoBusy}
                  >
                    <FiUpload size={13} /> {logoBusy ? 'Televersement...' : 'Remplacer'}
                  </button>
                  <button
                    type="button"
                    className={`${styles.button} ${styles.secondary}`}
                    onClick={() => setLogoUrlEditOpen((v) => !v)}
                    disabled={logoBusy}
                  >
                    <FiLink size={13} /> Lien
                  </button>
                  {company?.logo_url && (
                    <button
                      type="button"
                      className={`${styles.button} ${styles.danger}`}
                      onClick={onRemoveLogo}
                      disabled={logoBusy}
                    >
                      <FiTrash2 size={13} /> Supprimer
                    </button>
                  )}
                </>
              ) : (
                <>
                  <button
                    type="button"
                    className={`${styles.button} ${styles.primary}`}
                    onClick={() => fileInputRef.current?.click()}
                    disabled={logoBusy}
                  >
                    <FiUpload size={13} /> {logoBusy ? 'Televersement...' : 'Ajouter'}
                  </button>
                  <button
                    type="button"
                    className={`${styles.button} ${styles.secondary}`}
                    onClick={() => setLogoUrlEditOpen((v) => !v)}
                    disabled={logoBusy}
                  >
                    <FiLink size={13} /> Lien
                  </button>
                </>
              )}
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept="image/png, image/jpeg, image/svg+xml"
              className={styles.hiddenInput}
              onChange={onPickFile}
            />

            {logoUrlEditOpen && (
              <div className={styles.urlRow}>
                <input
                  type="url"
                  placeholder="https://exemple.com/logo.png"
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
                  Enregistrer
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* MODE LECTURE */}
      {!isEditing && (
        <>
          {/* Carte 2 : Coordonnees */}
          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <div className={styles.cardIcon}><FiMapPin size={16} /></div>
              <div className={styles.cardHeaderText}>
                <h3 className={styles.cardTitle}>Coordonnees</h3>
              </div>
            </div>
            <div className={styles.fieldGrid}>
              <ReadonlyField label="Nom" value={company.name} />
              <ReadonlyField
                label="Adresse operationnelle"
                value={company.address}
              />
              <ReadonlyField
                label="Email"
                value={company.contact_email || company.email}
              />
              <ReadonlyField
                label="Telephone"
                value={company.contact_phone || company.phone}
              />
            </div>
          </div>

          {/* Carte 3 : Legal et facturation (V6) */}
          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <div className={styles.cardIcon}><FiFileText size={16} /></div>
              <div className={styles.cardHeaderText}>
                <h3 className={styles.cardTitle}>Legal et facturation</h3>
              </div>
            </div>
            <div className={styles.fieldGrid}>
              <ReadonlyField label="IDE / UID" value={company.uid_ide} />
              <ReadonlyField label="Email de facturation" value={company.billing_email} />
              {company.preferential_rate && (
                <ReadonlyField
                  label="Tarif preferentiel"
                  value={`${company.preferential_rate.toFixed(2)} CHF / trajet`}
                />
              )}
              <NotesField label="Notes de facturation" value={company.billing_notes} />
            </div>
          </div>

          {/* Carte 4 : Domiciliation */}
          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <div className={styles.cardIcon}><FiHome size={16} /></div>
              <div className={styles.cardHeaderText}>
                <h3 className={styles.cardTitle}>Domiciliation</h3>
              </div>
            </div>
            <div className={styles.fieldGrid}>
              <ReadonlyField label="Adresse (ligne 1)" value={company.domicile_address_line1} />
              <ReadonlyField label="Adresse (ligne 2)" value={company.domicile_address_line2} />
              <ReadonlyField label="NPA" value={company.domicile_zip} />
              <ReadonlyField label="Ville" value={company.domicile_city} />
              <ReadonlyField label="Pays" value={company.domicile_country || 'CH'} />
            </div>
          </div>
        </>
      )}

      {/* MODE EDITION */}
      {isEditing && (
        <>
          {/* Carte 2 : Coordonnees (edit) */}
          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <div className={styles.cardIcon}><FiMapPin size={16} /></div>
              <div className={styles.cardHeaderText}>
                <h3 className={styles.cardTitle}>Coordonnees</h3>
              </div>
            </div>
            <div className={styles.settingsForm}>
              <div className={styles.formGroup}>
                <label htmlFor="name">Nom</label>
                <input id="name" name="name" value={form.name} onChange={handleChange} required />
                {fieldErrors.name && <small className={styles.fieldError}>{fieldErrors.name}</small>}
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="address">Adresse operationnelle</label>
                <AddressAutocomplete
                  name="address"
                  value={form.address}
                  onChange={handleChange}
                  onSelect={handleAddressSelect}
                  placeholder="Saisir l'adresse..."
                />
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="contact_email">Email</label>
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
                <label htmlFor="contact_phone">Telephone</label>
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
            </div>
          </div>

          {/* Carte 3 : Legal et facturation (edit) */}
          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <div className={styles.cardIcon}><FiFileText size={16} /></div>
              <div className={styles.cardHeaderText}>
                <h3 className={styles.cardTitle}>Legal et facturation</h3>
              </div>
            </div>
            <div className={styles.settingsForm}>
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
                <label htmlFor="preferential_rate">
                  Tarif preferentiel (CHF / trajet)
                  <small className={styles.labelHint}>Pour les cliniques uniquement</small>
                </label>
                <input
                  type="number"
                  id="preferential_rate"
                  name="preferential_rate"
                  value={form.preferential_rate}
                  onChange={handleChange}
                  step="0.01"
                  min="0"
                  placeholder="40.00"
                />
              </div>

              <div className={`${styles.formGroup} ${styles.formGroupFull}`}>
                <label htmlFor="billing_notes">Notes de facturation</label>
                <textarea
                  id="billing_notes"
                  name="billing_notes"
                  value={form.billing_notes}
                  onChange={handleChange}
                  rows={3}
                />
              </div>
            </div>
          </div>

          {/* Carte 4 : Domiciliation (edit) */}
          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <div className={styles.cardIcon}><FiHome size={16} /></div>
              <div className={styles.cardHeaderText}>
                <h3 className={styles.cardTitle}>Domiciliation</h3>
              </div>
            </div>
            <div className={styles.settingsForm}>
              <div className={styles.formGroup}>
                <label htmlFor="domicile_address_line1">Adresse (ligne 1)</label>
                <AddressAutocomplete
                  name="domicile_address_line1"
                  value={form.domicile_address_line1}
                  onChange={handleChange}
                  onSelect={handleDomicileAddressSelect}
                  placeholder="Saisir l'adresse..."
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
                <label htmlFor="domicile_country">Pays</label>
                <input
                  id="domicile_country"
                  name="domicile_country"
                  value={form.domicile_country}
                  onChange={handleChange}
                  maxLength={2}
                />
              </div>
            </div>
          </div>
        </>
      )}
    </>
  );
};

export default GeneralTab;
