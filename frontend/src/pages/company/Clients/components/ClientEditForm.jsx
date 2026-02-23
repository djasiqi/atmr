// frontend/src/pages/company/Clients/components/ClientEditForm.jsx
import React, { useRef, useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import {
  FiX,
  FiFileText,
  FiMapPin,
  FiCreditCard,
  FiBriefcase,
  FiActivity,
  FiSettings,
  FiChevronDown,
  FiChevronRight,
} from 'react-icons/fi';
import styles from './ClientEditForm.module.css';
import ntStyles from '../../Settings/tabs/NotificationsTab.module.css';
import AddressAutocomplete from '../../../../components/common/AddressAutocomplete';
import InlineDatePicker from '../../../../components/ui/InlineDatePicker';
import { parseAddressWithEstablishment } from '../../../../utils/addressParser';
import {
  normalizePhone,
  getPhoneValidationError,
} from '../../../../utils/phone';
import ClientStaysSection from './ClientStaysSection';
import ClientBillingPartiesSection from './ClientBillingPartiesSection';
import ClinicBillingMappingSection from './ClinicBillingMappingSection';

/**
 * Formulaire d'édition dans le drawer
 * Réutilise la logique de EditClientModal mais adapté pour le drawer avec accordéons
 */
const PLACEHOLDER_VALUES = ['Non spécifié', 'Non renseigné', 'Non renseignée', '—', '\u2014'];
const clean = (val) => {
  if (!val) return '';
  const str = String(val).trim();
  return PLACEHOLDER_VALUES.includes(str) ? '' : str;
};

const capitalizeFirstName = (str) => {
  if (!str) return '';
  return str.replace(/(^|[\s\-'])(\S)/g, (_m, sep, ch) => sep + ch.toUpperCase());
};

const upperLastName = (str) => (str ? str.toUpperCase() : '');

const GENDER_OPTIONS = [
  { value: '', label: 'Sélectionner' },
  { value: 'male', label: 'Monsieur' },
  { value: 'female', label: 'Madame' },
];

function GenderDropdown({ value, onChange, disabled }) {
  const [open, setOpen] = React.useState(false);
  const btnRef = React.useRef(null);
  const menuRef = React.useRef(null);
  const [pos, setPos] = React.useState({ top: 0, left: 0, width: 0 });

  React.useEffect(() => {
    if (!open) return;
    const onClick = (e) => {
      if (btnRef.current?.contains(e.target) || menuRef.current?.contains(e.target)) return;
      setOpen(false);
    };
    const onKey = (e) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', onClick);
    document.addEventListener('keydown', onKey);
    return () => { document.removeEventListener('mousedown', onClick); document.removeEventListener('keydown', onKey); };
  }, [open]);

  const reposition = React.useCallback(() => {
    if (!btnRef.current) return;
    const r = btnRef.current.getBoundingClientRect();
    setPos({ top: r.bottom + 4, left: r.left, width: r.width });
  }, []);

  React.useEffect(() => {
    if (!open) return;
    reposition();
    window.addEventListener('scroll', reposition, true);
    window.addEventListener('resize', reposition);
    return () => { window.removeEventListener('scroll', reposition, true); window.removeEventListener('resize', reposition); };
  }, [open, reposition]);

  const selected = GENDER_OPTIONS.find((o) => o.value === value);

  return (
    <div className={styles.chipDrop}>
      <button
        ref={btnRef}
        type="button"
        className={`${styles.chipBtn} ${value ? styles.chipBtnActive : ''}`}
        onClick={() => !disabled && setOpen((p) => !p)}
        disabled={disabled}
      >
        <span className={styles.chipText}>{selected?.label || 'Sélectionner'}</span>
        <FiChevronDown size={11} className={`${styles.chipArrow} ${open ? styles.chipArrowOpen : ''}`} />
      </button>
      {open && createPortal(
        <div
          ref={menuRef}
          className={styles.chipMenu}
          style={{ position: 'fixed', top: pos.top, left: pos.left, width: pos.width, zIndex: 10000 }}
        >
          {GENDER_OPTIONS.map((o) => (
            <button
              key={o.value}
              type="button"
              className={`${styles.chipOption} ${o.value === value ? styles.chipOptionActive : ''}`}
              onClick={() => { onChange(o.value); setOpen(false); }}
            >
              {o.label}
            </button>
          ))}
        </div>,
        document.body
      )}
    </div>
  );
}

const BILLING_TYPE_OPTIONS = [
  { value: '', label: 'Par défaut' },
  { value: 'patient', label: 'Patient' },
  { value: 'clinic', label: 'Clinique' },
  { value: 'insurance', label: 'Assurance' },
];

function BillingTypeDropdown({ value, onChange, disabled }) {
  const [open, setOpen] = React.useState(false);
  const btnRef = React.useRef(null);
  const menuRef = React.useRef(null);
  const [pos, setPos] = React.useState({ top: 0, left: 0, width: 0 });

  React.useEffect(() => {
    if (!open) return;
    const onClick = (e) => {
      if (btnRef.current?.contains(e.target) || menuRef.current?.contains(e.target)) return;
      setOpen(false);
    };
    const onKey = (e) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', onClick);
    document.addEventListener('keydown', onKey);
    return () => { document.removeEventListener('mousedown', onClick); document.removeEventListener('keydown', onKey); };
  }, [open]);

  const reposition = React.useCallback(() => {
    if (!btnRef.current) return;
    const r = btnRef.current.getBoundingClientRect();
    setPos({ top: r.bottom + 4, left: r.left, width: r.width });
  }, []);

  React.useEffect(() => {
    if (!open) return;
    reposition();
    window.addEventListener('scroll', reposition, true);
    window.addEventListener('resize', reposition);
    return () => { window.removeEventListener('scroll', reposition, true); window.removeEventListener('resize', reposition); };
  }, [open, reposition]);

  const selected = BILLING_TYPE_OPTIONS.find((o) => o.value === value);

  return (
    <div className={styles.chipDrop}>
      <button
        ref={btnRef}
        type="button"
        className={`${styles.chipBtn} ${value ? styles.chipBtnActive : ''}`}
        onClick={() => !disabled && setOpen((p) => !p)}
        disabled={disabled}
      >
        <span className={styles.chipText}>{selected?.label || 'Par défaut'}</span>
        <FiChevronDown size={11} className={`${styles.chipArrow} ${open ? styles.chipArrowOpen : ''}`} />
      </button>
      {open && createPortal(
        <div
          ref={menuRef}
          className={styles.chipMenu}
          style={{ position: 'fixed', top: pos.top, left: pos.left, width: pos.width, zIndex: 10000 }}
        >
          {BILLING_TYPE_OPTIONS.map((o) => (
            <button
              key={o.value}
              type="button"
              className={`${styles.chipOption} ${o.value === value ? styles.chipOptionActive : ''}`}
              onClick={() => { onChange(o.value); setOpen(false); }}
            >
              {o.label}
            </button>
          ))}
        </div>,
        document.body
      )}
    </div>
  );
}

const ClientEditForm = ({
  client,
  onSave,
  onCancel,
  onClose: _onClose,
  loading = false,
  hasUnsavedChanges: externalHasUnsavedChanges = false,
  onUnsavedChangesChange,
  onReloadClient,
}) => {
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(externalHasUnsavedChanges);
  
  // L'institution est-elle liée à une institution officielle ?
  const isLinkedInstitution = !!(client.linked_institution_id || client.linked_institution);

  const [formData, setFormData] = useState({
    is_institution: client.is_institution || false,
    institution_name: client.institution_name || '',
    linked_institution_id: client.linked_institution_id || null,
    first_name: capitalizeFirstName(client.user_first_name || client.first_name || client.user?.first_name || ''),
    last_name: upperLastName(client.user_last_name || client.last_name || client.user?.last_name || ''),
    residence_facility: client.residence_facility || '',
    birth_date: client.user_birth_date || client.user?.birth_date || '',
    gender: (() => {
      const genderValue = client.user_gender || client.user?.gender || '';
      if (!genderValue) return '';
      const genderStr = String(genderValue).toLowerCase();
      if (genderStr === 'homme' || genderStr === 'male') return 'male';
      if (genderStr === 'femme' || genderStr === 'female') return 'female';
      return genderStr;
    })(),
    avs_number: client.avs_number || '',
    phone: clean(client.phone || client.user_phone || client.user?.phone),
    contact_email: clean(client.contact_email),
    contact_phone: clean(client.contact_phone),
    domicile_address: client.domicile_address || client.domicile?.address || '',
    domicile_zip: client.domicile_zip || client.domicile?.zip || '',
    domicile_city: client.domicile_city || client.domicile?.city || '',
    door_code: client.access?.door_code ?? client.door_code ?? '',
    floor: client.access?.floor ?? client.floor ?? '',
    access_notes: client.access?.notes ?? client.access_notes ?? '',
    gp_name: client.gp?.name ?? client.gp_name ?? '',
    gp_phone: client.gp?.phone ?? client.gp_phone ?? '',
    billing_address: client.billing_address || '',
    show_billing_info: !!(client.billing_address && String(client.billing_address).trim()),
    default_billed_to_type: client.default_billing?.billed_to_type ?? client.default_billed_to_type ?? '',
    default_billed_to_contact: client.default_billing?.billed_to_contact ?? client.default_billed_to_contact ?? '',
    preferential_rate: client.preferential_rate || '',
    is_active: client.is_active !== false,
  });

  const [error, setError] = useState(null);
  /** Erreurs par champ (ex: phone, contact_phone, gp_phone) pour validation UI / API */
  const [fieldErrors, setFieldErrors] = useState({ phone: null, contact_phone: null, gp_phone: null });
  const [domicileCoords, setDomicileCoords] = useState({
    lat: client.domicile_lat || client.domicile?.lat || null,
    lon: client.domicile_lon || client.domicile?.lon || null,
  });
  const [billingCoords, setBillingCoords] = useState({
    lat: client.billing_lat ?? null,
    lon: client.billing_lon ?? null,
  });
  const [expandedSections, setExpandedSections] = useState({
    essential: true,
    address: false,
    billing: false,
    clinicMapping: false,
    stays: false,
    billingParties: false,
  });
  const [billingPartiesScrollBottom, setBillingPartiesScrollBottom] = useState(16);
  const billingPartiesScrollRef = useRef(null);
  const billingPartiesSectionRef = useRef(null);
  const [showAdvancedBilling, setShowAdvancedBilling] = useState(
    !!(
      ((client.default_billing?.billed_to_type && client.default_billing.billed_to_type !== 'patient') ||
        (client.default_billing?.billed_to_contact && client.default_billing.billed_to_contact.trim()))
    )
  );

  // Détecter les modifications
  useEffect(() => {
    const orig = {
      is_institution: client.is_institution || false,
      institution_name: client.institution_name || '',
      first_name: capitalizeFirstName(client.user_first_name || client.first_name || client.user?.first_name || ''),
      last_name: upperLastName(client.user_last_name || client.last_name || client.user?.last_name || ''),
      residence_facility: client.residence_facility || '',
      birth_date: client.user_birth_date || client.user?.birth_date || '',
      gender: (() => {
        const g = client.user_gender || client.user?.gender || '';
        if (!g) return '';
        const s = String(g).toLowerCase();
        if (s === 'homme' || s === 'male') return 'male';
        if (s === 'femme' || s === 'female') return 'female';
        return s;
      })(),
      avs_number: client.avs_number || '',
      phone: clean(client.phone || client.user_phone || client.user?.phone),
      contact_email: clean(client.contact_email),
      contact_phone: clean(client.contact_phone),
      domicile_address: client.domicile_address || client.domicile?.address || '',
      domicile_zip: client.domicile_zip || client.domicile?.zip || '',
      domicile_city: client.domicile_city || client.domicile?.city || '',
      door_code: client.access?.door_code ?? client.door_code ?? '',
      floor: client.access?.floor ?? client.floor ?? '',
      access_notes: client.access?.notes ?? client.access_notes ?? '',
      gp_name: client.gp?.name ?? client.gp_name ?? '',
      gp_phone: client.gp?.phone ?? client.gp_phone ?? '',
      billing_address: client.billing_address || '',
      show_billing_info: !!(client.billing_address && String(client.billing_address).trim()),
      default_billed_to_type: client.default_billing?.billed_to_type ?? client.default_billed_to_type ?? '',
      default_billed_to_contact: client.default_billing?.billed_to_contact ?? client.default_billed_to_contact ?? '',
      preferential_rate: client.preferential_rate || '',
      is_active: client.is_active !== false,
    };
    const domOrig = { lat: client.domicile_lat || client.domicile?.lat || null, lon: client.domicile_lon || client.domicile?.lon || null };
    const billOrig = { lat: client.billing_lat ?? null, lon: client.billing_lon ?? null };
    const hasChanges = JSON.stringify(formData) !== JSON.stringify(orig) ||
                       JSON.stringify(domicileCoords) !== JSON.stringify(domOrig) ||
                       JSON.stringify(billingCoords) !== JSON.stringify(billOrig);
    if (hasUnsavedChanges !== hasChanges) {
      setHasUnsavedChanges(hasChanges);
      onUnsavedChangesChange?.(hasChanges);
    }
  }, [formData, domicileCoords, billingCoords, client, hasUnsavedChanges, onUnsavedChangesChange]);

  useEffect(() => {
    if (expandedSections.billingParties && billingPartiesScrollRef.current) {
      const container = billingPartiesScrollRef.current;
      requestAnimationFrame(() => {
        container.scrollTop = 0;
      });
    }
  }, [expandedSections.billingParties]);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    let finalValue = type === 'checkbox' ? checked : value;
    if (name === 'first_name') finalValue = capitalizeFirstName(finalValue);
    if (name === 'last_name') finalValue = upperLastName(finalValue);
    setFormData((prev) => ({
      ...prev,
      [name]: finalValue,
    }));
    if (['phone', 'contact_phone', 'gp_phone'].includes(name) && fieldErrors[name]) {
      setFieldErrors((prev) => ({ ...prev, [name]: null }));
    }
  };

  const handleDomicileAddressSelect = (item) => {
    const label = item.label || '';
    const parsed = parseAddressWithEstablishment(label, item);
    const address =
      parsed.streetNumber && parsed.street
        ? `${parsed.street} ${parsed.streetNumber}`.trim()
        : parsed.street || item.address || '';

    setFormData((prev) => ({
      ...prev,
      residence_facility: parsed.establishment || prev.residence_facility,
      domicile_address: address,
      domicile_zip: parsed.postcode,
      domicile_city: parsed.city,
    }));

    setDomicileCoords({ lat: item.lat ?? null, lon: item.lon ?? null });
  };

  const handleBillingAddressSelect = (item) => {
    const full = item.label || '';
    setFormData((prev) => ({ ...prev, billing_address: full }));
    setBillingCoords({ lat: item.lat ?? null, lon: item.lon ?? null });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (formData.is_institution && !formData.institution_name.trim()) {
      setError("Le nom de l'institution est requis pour les institutions");
      return;
    }

    if (!formData.is_institution && (!formData.first_name.trim() || !formData.last_name.trim())) {
      setError("Le prénom et le nom sont requis pour les clients");
      return;
    }

    if (!formData.is_institution && !formData.gender) {
      setError("Merci de sélectionner une civilité");
      return;
    }

    setError(null);
    setFieldErrors({ phone: null, contact_phone: null, gp_phone: null });

    // Normalisation et validation des numéros de téléphone avant envoi
    const normalizedPhone = normalizePhone(formData.phone);
    const normalizedContactPhone = normalizePhone(formData.contact_phone);
    const normalizedGpPhone = normalizePhone(formData.gp_phone);

    const phoneError = getPhoneValidationError(normalizedPhone);
    const contactPhoneError = getPhoneValidationError(normalizedContactPhone);
    const gpPhoneError = getPhoneValidationError(normalizedGpPhone);

    if (phoneError || contactPhoneError || gpPhoneError) {
      setFieldErrors({
        phone: phoneError || null,
        contact_phone: contactPhoneError || null,
        gp_phone: gpPhoneError || null,
      });
      if (phoneError) setError(phoneError);
      else if (contactPhoneError) setError(contactPhoneError);
      else setError(gpPhoneError);
      return;
    }

    try {
      const hasSeparateBilling = formData.show_billing_info && formData.billing_address?.trim();
      const fullDomicile = [formData.domicile_address, formData.domicile_zip, formData.domicile_city]
        .filter(Boolean)
        .join(', ');
      const billingAddress = hasSeparateBilling
        ? formData.billing_address.trim()
        : fullDomicile || null;

      const payload = {
        is_institution: formData.is_institution,
        institution_name: formData.institution_name || null,
        residence_facility: formData.residence_facility || null,
        avs_number: formData.avs_number?.trim() || null,
        contact_email: formData.contact_email?.trim() || null,
        contact_phone: normalizedContactPhone,
        billing_address: billingAddress,
        billing_lat: hasSeparateBilling ? billingCoords.lat : domicileCoords.lat,
        billing_lon: hasSeparateBilling ? billingCoords.lon : domicileCoords.lon,
        domicile_address: formData.domicile_address?.trim() || null,
        domicile_zip: formData.domicile_zip?.trim() || null,
        domicile_city: formData.domicile_city?.trim() || null,
        preferential_rate: formData.preferential_rate ? parseFloat(formData.preferential_rate) : null,
        is_active: formData.is_active,
        domicile_lat: domicileCoords.lat,
        domicile_lon: domicileCoords.lon,
        door_code: formData.door_code?.trim() || null,
        floor: formData.floor?.trim() || null,
        access_notes: formData.access_notes?.trim() || null,
        gp_name: formData.gp_name?.trim() || null,
        gp_phone: normalizedGpPhone,
        default_billed_to_type: formData.default_billed_to_type || null,
        default_billed_to_contact: formData.default_billed_to_contact?.trim() || null,
      };

      if (!formData.is_institution) {
        payload.first_name = formData.first_name?.trim() || null;
        payload.last_name = formData.last_name?.trim() || null;
        payload.phone = normalizedPhone;
        if (formData.gender?.trim()) payload.gender = formData.gender;
        if (formData.birth_date?.trim()) payload.birth_date = formData.birth_date;
      }

      // Enregistrer d'abord le lien tiers payeur (et numéro SPC) si un tiers payeur est sélectionné,
      // avant onSave, pour que la sauvegarde soit faite avant le passage en mode lecture.
      if (billingPartiesSectionRef.current?.saveBillingPartyLink) {
        try {
          await billingPartiesSectionRef.current.saveBillingPartyLink();
        } catch (linkErr) {
          setError(
            linkErr.response?.data?.error || linkErr.message || 'Erreur lors de l\'enregistrement du tiers payeur'
          );
          return;
        }
      }

      await onSave(payload);
    } catch (err) {
      // companyService.updateClient lance err.response?.data || err → err peut être { error, error_code }
      const errorMessage =
        (typeof err === 'object' && err !== null && err.error) ||
        err.response?.data?.error ||
        err.message ||
        'Erreur lors de la sauvegarde';
      setError(errorMessage);

      // Mapper validation_error backend vers le champ phone si le message concerne le téléphone
      const isValidationError =
        (typeof err === 'object' && err !== null && err.error_code === 'validation_error') ||
        err.response?.data?.error_code === 'validation_error';
      const msg = (typeof err === 'object' && err?.error) || err.response?.data?.error || '';
      const isPhoneRelated =
        typeof msg === 'string' &&
        (msg.toLowerCase().includes('téléphone') || msg.toLowerCase().includes('phone') || msg.toLowerCase().includes('numéro'));
      if (isValidationError && isPhoneRelated) {
        setFieldErrors((prev) => ({ ...prev, phone: errorMessage }));
      }

      // Ne pas relancer l'erreur pour éviter "Uncaught (in promise)"
    }
  };

  const toggleSection = (section) => {
    setExpandedSections((prev) => {
      // Si la section est déjà ouverte, on la ferme
      if (prev[section]) {
        return {
          ...prev,
          [section]: false,
        };
      }
      // Sinon, on ferme toutes les autres et on ouvre celle-ci
      return {
        essential: false,
        address: false,
        billing: false,
        clinicMapping: false,
        stays: false,
        billingParties: false,
        [section]: true,
      };
    });
  };

  const displayName = formData.is_institution
    ? formData.institution_name || `Institution #${client.id}`
    : `${formData.first_name || ''} ${formData.last_name || ''}`.trim() ||
      `${client.user_first_name || ''} ${client.user_last_name || ''}`.trim() ||
      `Client #${client.id}`;

  return (
    <div className={styles.editForm}>
      {/* Header */}
      <header className={styles.header}>
        <div className={styles.headerTop}>
          <button
            onClick={onCancel}
            className={styles.closeButton}
            aria-label="Annuler"
            title="Annuler (ESC)"
          >
            <FiX size={18} />
          </button>
          <div className={styles.headerTitle}>
            <h2 className={styles.clientName}>
              Modifier : {displayName}
            </h2>
          </div>
        </div>
      </header>

      {/* Formulaire */}
      <form onSubmit={handleSubmit} className={styles.form}>
        {error && <div className={styles.error}>{error}</div>}

        {/* Accordéon : Informations client/clinique */}
        <div className={styles.accordion}>
          <button
            type="button"
            onClick={() => toggleSection('essential')}
            className={styles.accordionHeader}
          >
            <span className={styles.accordionTitle}>
              <FiFileText size={14} className={styles.accordionTitleIcon} />
              {formData.is_institution ? 'Informations clinique' : 'Informations client'}
            </span>
            <span className={styles.accordionChevron}>
              {expandedSections.essential ? <FiChevronDown size={16} /> : <FiChevronRight size={16} />}
            </span>
          </button>
          {expandedSections.essential && (
            <div className={styles.accordionContent}>
              {formData.is_institution ? (
                <>
                  <div className={styles.formGroup}>
                    <label htmlFor="institution_name" className={styles.label}>
                      Nom de la clinique *
                    </label>
                    {isLinkedInstitution && (
                      <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px',
                        marginBottom: '6px',
                        padding: '4px 8px',
                        background: 'var(--bg-success-light, #f0fdf4)',
                        border: '1px solid var(--border-success, #86efac)',
                        borderRadius: '6px',
                        fontSize: '12px',
                        color: 'var(--text-success, #16a34a)',
                      }}>
                        <span>Institution officielle</span>
                      </div>
                    )}
                    <input
                      type="text"
                      id="institution_name"
                      name="institution_name"
                      value={formData.institution_name}
                      onChange={handleChange}
                      className={styles.input}
                      required
                      disabled={loading}
                      readOnly={isLinkedInstitution}
                      placeholder="Ex: Clinique des Grangettes"
                      style={isLinkedInstitution ? {
                        backgroundColor: 'var(--bg-disabled, #f5f5f5)',
                        cursor: 'not-allowed',
                      } : undefined}
                    />
                    {isLinkedInstitution && (
                      <small style={{ color: 'var(--text-secondary, #666)' }}>
                        Ce nom est synchronisé avec l'institution officielle et ne peut pas être modifié.
                      </small>
                    )}
                  </div>
                  <div className={styles.formRowTwo}>
                    <div className={styles.formGroup}>
                      <label htmlFor="contact_email" className={styles.label}>
                        Email de contact
                      </label>
                      <input
                        type="email"
                        id="contact_email"
                        name="contact_email"
                        value={formData.contact_email}
                        onChange={handleChange}
                        className={styles.input}
                        placeholder="contact@clinique.ch"
                        disabled={loading}
                      />
                    </div>
                    <div className={styles.formGroup}>
                      <label htmlFor="contact_phone" className={styles.label}>
                        Téléphone
                      </label>
                      <input
                        type="tel"
                        id="contact_phone"
                        name="contact_phone"
                        value={formData.contact_phone}
                        onChange={handleChange}
                        className={styles.input}
                        placeholder="ex: +41791234567"
                        disabled={loading}
                        aria-invalid={!!fieldErrors.contact_phone}
                        aria-describedby={fieldErrors.contact_phone ? 'contact_phone-error' : undefined}
                      />
                      {fieldErrors.contact_phone && (
                        <div id="contact_phone-error" className={styles.fieldError} role="alert">
                          {fieldErrors.contact_phone}
                        </div>
                      )}
                    </div>
                  </div>
                </>
              ) : (
                <>
                  {/* Ligne 1 : Civilité, Prénom, Nom */}
                  <div className={styles.formRow}>
                    <div className={styles.formGroup}>
                      <label htmlFor="gender" className={styles.label}>
                        Civilité
                      </label>
                      <GenderDropdown
                        value={formData.gender}
                        onChange={(v) => handleChange({ target: { name: 'gender', value: v } })}
                        disabled={loading}
                      />
                    </div>
                    <div className={styles.formGroup}>
                      <label htmlFor="first_name" className={styles.label}>
                        Prénom *
                      </label>
                      <input
                        type="text"
                        id="first_name"
                        name="first_name"
                        value={formData.first_name}
                        onChange={handleChange}
                        className={styles.input}
                        placeholder="Prénom"
                        required
                        disabled={loading}
                      />
                    </div>
                    <div className={styles.formGroup}>
                      <label htmlFor="last_name" className={styles.label}>
                        Nom *
                      </label>
                      <input
                        type="text"
                        id="last_name"
                        name="last_name"
                        value={formData.last_name}
                        onChange={handleChange}
                        className={styles.input}
                        placeholder="Nom"
                        required
                        disabled={loading}
                      />
                    </div>
                  </div>
                  {/* Ligne 2 : Date de naissance, Numéro AVS */}
                  <div className={styles.formRowTwo}>
                    <div className={styles.formGroup}>
                      <label className={styles.label}>Date de naissance</label>
                      <InlineDatePicker
                        value={formData.birth_date}
                        onChange={(v) => handleChange({ target: { name: 'birth_date', value: v } })}
                        placeholder="Date de naissance"
                      />
                    </div>
                    <div className={styles.formGroup}>
                      <label htmlFor="avs_number" className={styles.label}>
                        Numéro AVS
                      </label>
                      <input
                        type="text"
                        id="avs_number"
                        name="avs_number"
                        value={formData.avs_number}
                        onChange={handleChange}
                        className={styles.input}
                        placeholder="756.XXXX.XXXX.XX"
                        disabled={loading}
                      />
                    </div>
                  </div>
                  <div className={styles.formGroup}>
                    <label htmlFor="phone" className={styles.label}>Téléphone</label>
                    <input
                      type="tel"
                      id="phone"
                      name="phone"
                      value={formData.phone}
                      onChange={handleChange}
                      className={styles.input}
                      placeholder="ex: +41791234567"
                      disabled={loading}
                      aria-invalid={!!fieldErrors.phone}
                      aria-describedby={fieldErrors.phone ? 'phone-error' : undefined}
                    />
                    {fieldErrors.phone && (
                      <div id="phone-error" className={styles.fieldError} role="alert">
                        {fieldErrors.phone}
                      </div>
                    )}
                  </div>
                  <div className={styles.formRowTwo}>
                    <div className={styles.formGroup}>
                      <label htmlFor="contact_email" className={styles.label}>Email de contact</label>
                      <input
                        type="email"
                        id="contact_email"
                        name="contact_email"
                        value={formData.contact_email}
                        onChange={handleChange}
                        className={styles.input}
                        placeholder="contact@exemple.ch"
                        disabled={loading}
                      />
                    </div>
                    <div className={styles.formGroup}>
                      <label htmlFor="contact_phone" className={styles.label}>Téléphone de contact</label>
                      <input
                        type="tel"
                        id="contact_phone"
                        name="contact_phone"
                        value={formData.contact_phone}
                        onChange={handleChange}
                        className={styles.input}
                        placeholder="ex: +41791234567"
                        disabled={loading}
                        aria-invalid={!!fieldErrors.contact_phone}
                        aria-describedby={fieldErrors.contact_phone ? 'contact_phone-error' : undefined}
                      />
                      {fieldErrors.contact_phone && (
                        <div id="contact_phone-error" className={styles.fieldError} role="alert">
                          {fieldErrors.contact_phone}
                        </div>
                      )}
                    </div>
                  </div>
                </>
              )}
            </div>
          )}
        </div>

        {/* Accordéon : Adresse & résidence */}
        <div className={styles.accordion}>
          <button
            type="button"
            onClick={() => toggleSection('address')}
            className={styles.accordionHeader}
          >
            <span className={styles.accordionTitle}>
              <FiMapPin size={14} className={styles.accordionTitleIcon} />
              {formData.is_institution ? 'Adresse de la clinique' : 'Adresse et residence'}
            </span>
            <span className={styles.accordionChevron}>
              {expandedSections.address ? <FiChevronDown size={16} /> : <FiChevronRight size={16} />}
            </span>
          </button>
          {expandedSections.address && (
            <div className={styles.accordionContent}>
              {!formData.is_institution && (
                <div className={styles.formGroup}>
                  <label htmlFor="residence_facility" className={styles.label}>
                    Établissement de résidence
                  </label>
                  <input
                    type="text"
                    id="residence_facility"
                    name="residence_facility"
                    value={formData.residence_facility}
                    onChange={handleChange}
                    className={styles.input}
                    placeholder="Ex: EMS Maison de Vessy..."
                    disabled={loading}
                  />
                </div>
              )}
              <div className={styles.formGroup}>
                <label htmlFor="domicile_address" className={styles.label}>
                  {formData.is_institution ? 'Adresse de la clinique' : 'Adresse complète'}
                </label>
                <AddressAutocomplete
                  name="domicile_address"
                  value={`${formData.domicile_address}${
                    formData.domicile_zip ? ', ' + formData.domicile_zip : ''
                  }${formData.domicile_city ? ', ' + formData.domicile_city : ''}`}
                  onChange={(_e) => {
                    setDomicileCoords({ lat: null, lon: null });
                  }}
                  onSelect={handleDomicileAddressSelect}
                  placeholder={
                    formData.is_institution
                      ? "Ex: Avenue Ernest-Pictet 9, 1203, Genève"
                      : "Ex: Avenue Ernest-Pictet 9, 1203, Genève"
                  }
                  disabled={loading}
                />
              </div>
              <div className={styles.formRow}>
                <div className={styles.formGroup}>
                  <label htmlFor="domicile_address_street" className={styles.label}>Rue et numéro</label>
                  <input
                    type="text"
                    id="domicile_address_street"
                    value={formData.domicile_address}
                    className={styles.input}
                    readOnly
                    disabled={loading}
                  />
                </div>
                <div className={styles.formGroup}>
                  <label htmlFor="domicile_zip" className={styles.label}>Code postal</label>
                  <input
                    type="text"
                    id="domicile_zip"
                    value={formData.domicile_zip}
                    className={styles.input}
                    readOnly
                    disabled={loading}
                  />
                </div>
                <div className={styles.formGroup}>
                  <label htmlFor="domicile_city" className={styles.label}>Ville</label>
                  <input
                    type="text"
                    id="domicile_city"
                    value={formData.domicile_city}
                    className={styles.input}
                    readOnly
                    disabled={loading}
                  />
                </div>
              </div>
              {!formData.is_institution && (
                <>
                  <div className={styles.formRow}>
                    <div className={styles.formGroup}>
                      <label htmlFor="door_code" className={styles.label}>Code porte</label>
                      <input
                        type="text"
                        id="door_code"
                        name="door_code"
                        value={formData.door_code}
                        onChange={handleChange}
                        className={styles.input}
                        placeholder="Ex: 4521"
                        disabled={loading}
                      />
                    </div>
                    <div className={styles.formGroup}>
                      <label htmlFor="floor" className={styles.label}>Étage</label>
                      <input
                        type="text"
                        id="floor"
                        name="floor"
                        value={formData.floor}
                        onChange={handleChange}
                        className={styles.input}
                        placeholder="Ex: 2e étage"
                        disabled={loading}
                      />
                    </div>
                  </div>
                  <div className={styles.formGroup}>
                    <label htmlFor="access_notes" className={styles.label}>Notes d&apos;accès</label>
                    <textarea
                      id="access_notes"
                      name="access_notes"
                      value={formData.access_notes}
                      onChange={handleChange}
                      className={styles.textarea}
                      placeholder="Ex: appeler avant d'arriver…"
                      disabled={loading}
                    />
                  </div>
                  <div className={styles.formRow}>
                    <div className={styles.formGroup}>
                      <label htmlFor="gp_name" className={styles.label}>Médecin traitant</label>
                      <input
                        type="text"
                        id="gp_name"
                        name="gp_name"
                        value={formData.gp_name}
                        onChange={handleChange}
                        className={styles.input}
                        placeholder="Nom du médecin"
                        disabled={loading}
                      />
                    </div>
                    <div className={styles.formGroup}>
                      <label htmlFor="gp_phone" className={styles.label}>Téléphone du médecin</label>
                      <input
                        type="tel"
                        id="gp_phone"
                        name="gp_phone"
                        value={formData.gp_phone}
                        onChange={handleChange}
                        className={styles.input}
                        placeholder="ex: +41791234567"
                        disabled={loading}
                        aria-invalid={!!fieldErrors.gp_phone}
                        aria-describedby={fieldErrors.gp_phone ? 'gp_phone-error' : undefined}
                      />
                      {fieldErrors.gp_phone && (
                        <div id="gp_phone-error" className={styles.fieldError} role="alert">
                          {fieldErrors.gp_phone}
                        </div>
                      )}
                    </div>
                  </div>
                </>
              )}
            </div>
          )}
        </div>

        {/* Accordéon : Tarif préférentiel (clinique ou patient) */}
        <div className={styles.accordion}>
          <button
            type="button"
            onClick={() => toggleSection('billing')}
            className={styles.accordionHeader}
          >
            <span className={styles.accordionTitle}>
              <FiCreditCard size={14} className={styles.accordionTitleIcon} />
              {formData.is_institution ? 'Tarif preferentiel' : 'Tarif patient'}
            </span>
            <span className={styles.accordionChevron}>
              {expandedSections.billing ? <FiChevronDown size={16} /> : <FiChevronRight size={16} />}
            </span>
          </button>
          {expandedSections.billing && (
            <div className={styles.accordionContent}>
              {!formData.is_institution && (
                <>
                  <label className={`${ntStyles.notifRow} ${styles.toggleRow}`} htmlFor="toggle-billing-addr">
                    <div className={ntStyles.notifInfo}>
                      <span className={ntStyles.notifLabel}>Adresse de facturation différente</span>
                      <span className={ntStyles.notifHint}>Par défaut, la facturation utilise l'adresse de domicile</span>
                    </div>
                    <div className={ntStyles.miniToggle}>
                      <input
                        id="toggle-billing-addr"
                        type="checkbox"
                        name="show_billing_info"
                        checked={formData.show_billing_info}
                        onChange={handleChange}
                        disabled={loading}
                      />
                      <span className={ntStyles.miniSlider} />
                    </div>
                  </label>
                  {formData.show_billing_info && (
                    <div className={styles.formGroup}>
                      <label htmlFor="billing_address" className={styles.label}>Adresse de facturation</label>
                      <AddressAutocomplete
                        name="billing_address"
                        value={formData.billing_address}
                        onChange={(e) => {
                          setBillingCoords({ lat: null, lon: null });
                          setFormData((prev) => ({ ...prev, billing_address: e.target.value }));
                        }}
                        onSelect={handleBillingAddressSelect}
                        placeholder="Ex: Avenue de la Gare 5, 1003, Lausanne"
                        disabled={loading}
                      />
                      <small className={styles.hint}>Si differente de l&apos;adresse de domicile</small>
                    </div>
                  )}
                  <label className={`${ntStyles.notifRow} ${styles.toggleRow}`} htmlFor="toggle-advanced-billing">
                    <div className={ntStyles.notifInfo}>
                      <span className={ntStyles.notifLabel}>Options de facturation avancées</span>
                      <span className={ntStyles.notifHint}>Type de facturation par défaut et contact principal</span>
                    </div>
                    <div className={ntStyles.miniToggle}>
                      <input
                        id="toggle-advanced-billing"
                        type="checkbox"
                        checked={showAdvancedBilling}
                        onChange={(e) => setShowAdvancedBilling(e.target.checked)}
                        disabled={loading}
                      />
                      <span className={ntStyles.miniSlider} />
                    </div>
                  </label>
                  {showAdvancedBilling && (
                    <div className={styles.formRowTwo}>
                      <div className={styles.formGroup}>
                        <label className={styles.label}>Type de facturation</label>
                        <BillingTypeDropdown
                          value={formData.default_billed_to_type}
                          onChange={(v) => handleChange({ target: { name: 'default_billed_to_type', value: v } })}
                          disabled={loading}
                        />
                      </div>
                      <div className={styles.formGroup}>
                        <label htmlFor="default_billed_to_contact" className={styles.label}>
                          Contact facturation
                        </label>
                        <input
                          type="text"
                          id="default_billed_to_contact"
                          name="default_billed_to_contact"
                          value={formData.default_billed_to_contact}
                          onChange={handleChange}
                          className={styles.input}
                          placeholder="Ex: Service facturation"
                          disabled={loading}
                        />
                      </div>
                    </div>
                  )}
                </>
              )}
              <div className={styles.formGroup}>
                <label htmlFor="preferential_rate" className={styles.label}>
                  {formData.is_institution
                    ? 'Tarif préférentiel (CHF)'
                    : 'Tarif patient (CHF)'}
                </label>
                <input
                  type="number"
                  id="preferential_rate"
                  name="preferential_rate"
                  value={formData.preferential_rate}
                  onChange={handleChange}
                  className={styles.input}
                  placeholder="Ex: 40.00"
                  step="0.50"
                  min="0"
                  disabled={loading}
                />
                <small className={styles.hint}>
                  {formData.is_institution
                    ? "Prix d'un trajet simple pour les patients hospitalisés dans cette clinique."
                    : "Prix d'un trajet simple pour ce patient."}
                  {' '}
                  Laisser vide pour utiliser le tarif standard.
                </small>
              </div>
            </div>
          )}
        </div>

        {/* Accordéon : Mapping de facturation (uniquement pour les cliniques) */}
        {formData.is_institution && (
          <div className={styles.accordion}>
            <button
              type="button"
              onClick={() => toggleSection('clinicMapping')}
              className={styles.accordionHeader}
            >
              <span className={styles.accordionTitle}><FiSettings size={14} className={styles.accordionTitleIcon} />Mapping de facturation</span>
              <span className={styles.accordionChevron}>
                {expandedSections.clinicMapping ? <FiChevronDown size={16} /> : <FiChevronRight size={16} />}
              </span>
            </button>
            {expandedSections.clinicMapping && (
              <ClinicBillingMappingSection
                clinicCompanyId={
                  client.default_billing?.billed_to_company?.id ||
                  client.default_billed_to_company_id
                }
                clientId={client.id}
                clinicCompanyName={
                  client.default_billing?.billed_to_company?.name ||
                  client.institution_name
                }
                onCompanyCreated={async (company) => {
                  // Recharger les données du client pour mettre à jour default_billing
                  if (company && company.id && onReloadClient) {
                    console.log('Company creee, rechargement des donnees du client...', company);
                    // Attendre un peu pour laisser le backend finaliser la transaction
                    await new Promise(resolve => setTimeout(resolve, 500));
                    // Forcer un rechargement complet
                    await onReloadClient();
                    // Recharger à nouveau après un délai supplémentaire pour s'assurer que tout est à jour
                    setTimeout(async () => {
                      await onReloadClient();
                    }, 1500);
                  }
                }}
                onClinicCompanyIdChange={(newCompanyId) => {
                  // Mettre à jour localement le client avec la nouvelle Company
                  // Cela permettra au composant de se mettre à jour immédiatement
                  console.log('🔄 Nouvelle clinicCompanyId:', newCompanyId);
                }}
              />
            )}
          </div>
        )}

        {/* Tiers payeur (uniquement pour clients) */}
        {!formData.is_institution && (
          <div className={styles.accordion}>
            <button
              type="button"
              onClick={() => toggleSection('billingParties')}
              className={styles.accordionHeader}
            >
              <span className={styles.accordionTitle}><FiBriefcase size={14} className={styles.accordionTitleIcon} />Tiers payeur / Curateur</span>
              <span className={styles.accordionChevron}>
                {expandedSections.billingParties ? <FiChevronDown size={16} /> : <FiChevronRight size={16} />}
              </span>
            </button>
            {expandedSections.billingParties && (
              <div
                ref={billingPartiesScrollRef}
                className={`${styles.accordionContent} ${styles.accordionContentScrollable}`}
                style={{
                  paddingBottom: `${billingPartiesScrollBottom}px`,
                  scrollPaddingBottom: `${billingPartiesScrollBottom}px`,
                }}
              >
                <ClientBillingPartiesSection
                  ref={billingPartiesSectionRef}
                  clientId={client.id}
                  showTitle={false}
                  autoShowForm={true}
                  integratedSave={true}
                  onScrollBottomGapChange={setBillingPartiesScrollBottom}
                />
              </div>
            )}
          </div>
        )}

        {/* Séjours d'hospitalisation (uniquement pour clients) */}
        {!formData.is_institution && (
          <div className={styles.accordion}>
            <button
              type="button"
              onClick={() => toggleSection('stays')}
              className={styles.accordionHeader}
            >
              <span className={styles.accordionTitle}><FiActivity size={14} className={styles.accordionTitleIcon} />Sejours d&apos;hospitalisation</span>
              <span className={styles.accordionChevron}>
                {expandedSections.stays ? <FiChevronDown size={16} /> : <FiChevronRight size={16} />}
              </span>
            </button>
            {expandedSections.stays && (
              <div className={styles.accordionContent}>
                <ClientStaysSection clientId={client.id} />
              </div>
            )}
          </div>
        )}

        {/* Statut */}
        <label className={`${ntStyles.notifRow} ${styles.toggleRow}`} htmlFor="toggle-is-active">
          <div className={ntStyles.notifInfo}>
            <span className={ntStyles.notifLabel}>
              {formData.is_institution ? 'Clinique active' : 'Client actif'}
            </span>
            <span className={ntStyles.notifHint}>
              {formData.is_institution
                ? 'Les cliniques inactives n\'apparaissent pas dans les sélections'
                : 'Les clients inactifs n\'apparaissent pas dans les sélections'}
            </span>
          </div>
          <div className={ntStyles.miniToggle}>
            <input
              id="toggle-is-active"
              type="checkbox"
              name="is_active"
              checked={formData.is_active}
              onChange={handleChange}
              disabled={loading}
            />
            <span className={ntStyles.miniSlider} />
          </div>
        </label>

        {/* Actions */}
        <div className={styles.actions}>
          <button
            type="button"
            onClick={onCancel}
            className={styles.cancelButton}
            disabled={loading}
          >
            Annuler
          </button>
          <button type="submit" className={styles.saveButton} disabled={loading}>
            {loading ? 'Sauvegarde...' : 'Enregistrer'}
          </button>
        </div>
      </form>
    </div>
  );
};

export default ClientEditForm;
