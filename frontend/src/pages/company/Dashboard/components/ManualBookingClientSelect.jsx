import React, { useCallback, useMemo } from 'react';
import AsyncCreatableSelect from 'react-select/async-creatable';
import Label from './ui/Label';
import styles from './ManualBookingForm.module.css';

const MENU_PORTAL_TARGET = typeof window !== 'undefined' ? document.body : null;

const CLIENT_SELECT_STYLES = {
  menuPortal: (base) => ({ ...base, zIndex: 'var(--z-modal-popover)' }),
};

const formatCreateLabel = (input) => `➕ Créer "${input}"`;

const noOptionsMessage = ({ inputValue }) =>
  inputValue
    ? `Aucun client trouvé pour "${inputValue}"`
    : 'Aucun client chargé. Ouvrez la liste ou tapez pour rechercher.';

const loadingMessage = () => '🔍 Recherche en cours...';

function formatOptionLabel(option, { context }) {
  if (option.__isNew__) return option.label;
  const c = option.raw || option;
  const label = option.label || '';
  if (context === 'value') {
    return <span className={styles.clientOptionLabel}>{label}</span>;
  }
  const phone = c?.phone || c?.contact_phone || '';
  const metaParts = [];
  if (c?.is_institution) metaParts.push('🏥 Institution');
  if (phone) metaParts.push(phone);
  const metaText = metaParts.join(' · ');
  return (
    <div className={styles.clientOption}>
      <span className={styles.clientOptionLabel}>{label}</span>
      {metaText && <span className={styles.clientOptionMeta}>{metaText}</span>}
    </div>
  );
}

function ManualBookingClientSelect({
  selectedClient,
  defaultClientOptions,
  loadClientOptions,
  onChange,
  onCreateOption,
  activeStay,
  billToPatient,
  onBillToPatientChange,
  getClinicPickupAddress,
}) {
  const defaultOptions = useMemo(
    () => (defaultClientOptions.length > 0 ? defaultClientOptions : true),
    [defaultClientOptions]
  );

  const handleCreateOption = useCallback(
    (input) => {
      onCreateOption(input);
    },
    [onCreateOption]
  );

  return (
    <div className={styles.formGroup} data-tour-id="booking-client">
      <Label htmlFor="client-select">Client *</Label>
      <AsyncCreatableSelect
        inputId="client-select"
        cacheOptions
        defaultOptions={defaultOptions}
        loadOptions={loadClientOptions}
        onChange={onChange}
        onCreateOption={handleCreateOption}
        value={selectedClient}
        placeholder="Rechercher un client…"
        formatCreateLabel={formatCreateLabel}
        formatOptionLabel={formatOptionLabel}
        noOptionsMessage={noOptionsMessage}
        loadingMessage={loadingMessage}
        menuPortalTarget={MENU_PORTAL_TARGET}
        menuPosition="fixed"
        styles={CLIENT_SELECT_STYLES}
        classNamePrefix="react-select"
        openMenuOnFocus={false}
        blurInputOnSelect
        tabSelectsValue={false}
      />

      {activeStay && activeStay.clinic && (
        <div className={styles.activeStayCard}>
          <div className={styles.activeStayRow}>
            <span className={styles.activeStayIcon}>🏥</span>
            <div className={styles.activeStayContent}>
              <strong className={styles.activeStayTitle}>
                Client hospitalisé à {activeStay.clinic.name}
              </strong>
              <small className={styles.activeStayMeta}>
                Adresse de départ: {getClinicPickupAddress(activeStay.clinic)}
                {activeStay.clinic.preferential_rate && (
                  <span className={styles.activeStayRate}>
                    💰 Tarif préférentiel: {activeStay.clinic.preferential_rate.toFixed(2)} CHF
                  </span>
                )}
              </small>
            </div>
          </div>
          <div className={styles.activeStayOverride}>
            <label className={styles.activeStayLabel}>
              <input
                type="checkbox"
                checked={billToPatient}
                onChange={(e) => onBillToPatientChange(e.target.checked)}
                className={styles.checkbox}
              />
              <span>Facturation patient (override)</span>
            </label>
            {billToPatient && (
              <small className={styles.activeStayWarning}>
                ⚠️ La facturation sera adressée au client (le départ reste la clinique)
              </small>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function areClientSelectPropsEqual(prev, next) {
  return (
    prev.selectedClient === next.selectedClient &&
    prev.defaultClientOptions === next.defaultClientOptions &&
    prev.loadClientOptions === next.loadClientOptions &&
    prev.onChange === next.onChange &&
    prev.onCreateOption === next.onCreateOption &&
    prev.activeStay === next.activeStay &&
    prev.billToPatient === next.billToPatient &&
    prev.onBillToPatientChange === next.onBillToPatientChange &&
    prev.getClinicPickupAddress === next.getClinicPickupAddress
  );
}

export default React.memo(ManualBookingClientSelect, areClientSelectPropsEqual);
