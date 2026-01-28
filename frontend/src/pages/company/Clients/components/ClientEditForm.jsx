// frontend/src/pages/company/Clients/components/ClientEditForm.jsx
import React, { useRef, useState, useEffect } from 'react';
import styles from './ClientEditForm.module.css';
import AddressAutocomplete from '../../../../components/common/AddressAutocomplete';
import { parseAddressWithEstablishment } from '../../../../utils/addressParser';
import ClientStaysSection from './ClientStaysSection';
import ClientBillingPartiesSection from './ClientBillingPartiesSection';
import ClinicBillingMappingSection from './ClinicBillingMappingSection';

/**
 * Formulaire d'édition dans le drawer
 * Réutilise la logique de EditClientModal mais adapté pour le drawer avec accordéons
 */
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
  
  const [formData, setFormData] = useState({
    is_institution: client.is_institution || false,
    institution_name: client.institution_name || '',
    first_name: client.user_first_name || client.first_name || client.user?.first_name || '',
    last_name: client.user_last_name || client.last_name || client.user?.last_name || '',
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
    phone: client.phone || client.user_phone || client.user?.phone || '',
    contact_email: client.contact_email || '',
    contact_phone: client.contact_phone || '',
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
      first_name: client.user_first_name || client.first_name || client.user?.first_name || '',
      last_name: client.user_last_name || client.last_name || client.user?.last_name || '',
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
      phone: client.phone || client.user_phone || client.user?.phone || '',
      contact_email: client.contact_email || '',
      contact_phone: client.contact_phone || '',
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
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }));
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
        contact_phone: formData.contact_phone?.trim() || null,
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
        gp_phone: formData.gp_phone?.trim() || null,
        default_billed_to_type: formData.default_billed_to_type || null,
        default_billed_to_contact: formData.default_billed_to_contact?.trim() || null,
      };

      if (!formData.is_institution) {
        payload.first_name = formData.first_name?.trim() || null;
        payload.last_name = formData.last_name?.trim() || null;
        payload.phone = formData.phone?.trim() || null;
        if (formData.gender?.trim()) payload.gender = formData.gender;
        if (formData.birth_date?.trim()) payload.birth_date = formData.birth_date;
      }

      await onSave(payload);
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Erreur lors de la sauvegarde');
      throw err;
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
            ✕
          </button>
          <div className={styles.headerTitle}>
            <h2 className={styles.clientName}>
              Modifier : {displayName}
              {hasUnsavedChanges && <span className={styles.unsavedIndicator}> •</span>}
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
              {formData.is_institution ? '📋 Informations clinique' : '📋 Informations client'}
            </span>
            <span className={styles.accordionIcon}>
              {expandedSections.essential ? '▾' : '▸'}
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
                    <input
                      type="text"
                      id="institution_name"
                      name="institution_name"
                      value={formData.institution_name}
                      onChange={handleChange}
                      className={styles.input}
                      required
                      disabled={loading}
                      placeholder="Ex: Clinique des Grangettes"
                    />
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
                        placeholder="+41 22 123 45 67"
                        disabled={loading}
                      />
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
                      <select
                        id="gender"
                        name="gender"
                        value={formData.gender}
                        onChange={handleChange}
                        className={styles.input}
                        disabled={loading}
                      >
                        <option value="">-- Sélectionnez --</option>
                        <option value="male">Monsieur</option>
                        <option value="female">Madame</option>
                      </select>
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
                      <label htmlFor="birth_date" className={styles.label}>
                        Date de naissance
                      </label>
                      <input
                        type="date"
                        id="birth_date"
                        name="birth_date"
                        value={formData.birth_date}
                        onChange={handleChange}
                        className={styles.input}
                        disabled={loading}
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
                      placeholder="+41 22 123 45 67"
                      disabled={loading}
                    />
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
                        placeholder="+41 22 123 45 67"
                        disabled={loading}
                      />
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
              {formData.is_institution ? '🏠 Adresse de la clinique' : '🏠 Adresse & résidence'}
            </span>
            <span className={styles.accordionIcon}>
              {expandedSections.address ? '▾' : '▸'}
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
                        placeholder="+41 22 000 00 00"
                        disabled={loading}
                      />
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
              {formData.is_institution ? '💰 Tarif préférentiel' : '💰 Tarif patient'}
            </span>
            <span className={styles.accordionIcon}>
              {expandedSections.billing ? '▾' : '▸'}
            </span>
          </button>
          {expandedSections.billing && (
            <div className={styles.accordionContent}>
              {!formData.is_institution && (
                <>
                  <div className={styles.checkboxGroup}>
                    <label className={styles.checkboxLabel}>
                      <input
                        type="checkbox"
                        name="show_billing_info"
                        checked={formData.show_billing_info}
                        onChange={handleChange}
                        disabled={loading}
                      />
                      <span className={styles.checkboxText}>
                        <strong>Adresse de facturation différente</strong>
                        <small>Par défaut, la facturation utilise l&apos;adresse de domicile</small>
                      </span>
                    </label>
                  </div>
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
                      <small className={styles.hint}>💡 Si différente de l&apos;adresse de domicile</small>
                    </div>
                  )}
                  <div className={styles.checkboxGroup}>
                    <label className={styles.checkboxLabel}>
                      <input
                        type="checkbox"
                        checked={showAdvancedBilling}
                        onChange={(e) => setShowAdvancedBilling(e.target.checked)}
                        disabled={loading}
                      />
                      <span className={styles.checkboxText}>
                        <strong>Options de facturation avancées</strong>
                        <small>Type de facturation par défaut et contact principal</small>
                      </span>
                    </label>
                  </div>
                  {showAdvancedBilling && (
                    <div className={styles.formRow}>
                      <div className={styles.formGroup}>
                        <label htmlFor="default_billed_to_type" className={styles.label}>
                          Type de facturation par défaut
                        </label>
                        <select
                          id="default_billed_to_type"
                          name="default_billed_to_type"
                          value={formData.default_billed_to_type}
                          onChange={handleChange}
                          className={styles.input}
                          disabled={loading}
                        >
                          <option value="">— Laisser par défaut —</option>
                          <option value="patient">Patient</option>
                          <option value="clinic">Clinique</option>
                          <option value="insurance">Assurance</option>
                        </select>
                      </div>
                      <div className={styles.formGroup}>
                        <label htmlFor="default_billed_to_contact" className={styles.label}>
                          Contact facturation par défaut
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
              <span className={styles.accordionTitle}>📋 Mapping de facturation</span>
              <span className={styles.accordionIcon}>
                {expandedSections.clinicMapping ? '▾' : '▸'}
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
                    console.log('✅ Company créée, rechargement des données du client...', company);
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
              <span className={styles.accordionTitle}>💰 Tiers payeur / Curateur</span>
              <span className={styles.accordionIcon}>
                {expandedSections.billingParties ? '▾' : '▸'}
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
                  clientId={client.id} 
                  showTitle={false}
                  autoShowForm={true}
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
              <span className={styles.accordionTitle}>🏥 Séjours d'hospitalisation</span>
              <span className={styles.accordionIcon}>
                {expandedSections.stays ? '▾' : '▸'}
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
        <div className={styles.formGroup}>
          <label className={styles.checkboxLabel}>
            <input
              type="checkbox"
              name="is_active"
              checked={formData.is_active}
              onChange={handleChange}
              disabled={loading}
              className={styles.checkbox}
            />
            <span className={styles.checkboxText}>
              <strong>{formData.is_institution ? 'Clinique active' : 'Client actif'}</strong>
              <small>
                {formData.is_institution
                  ? 'Les cliniques inactives n\'apparaissent pas dans les sélections'
                  : 'Les clients inactifs n\'apparaissent pas dans les sélections'}
              </small>
            </span>
          </label>
        </div>

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
