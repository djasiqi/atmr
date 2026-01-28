import React, { useEffect, useState } from 'react';
import styles from './ClientFormModal.module.css';
import AddressAutocomplete from '../../../../components/common/AddressAutocomplete';
import { parseAddressWithEstablishment } from '../../../../utils/addressParser';
import { fetchBillingParties, fetchClinicBillingMappings } from '../../../../services/settingsService';

const NewClientModal = ({ onClose, onSave }) => {
  const [formData, setFormData] = useState({
    // ✅ client_type et email supprimés - tous les clients sont PRIVATE
    first_name: '',
    last_name: '',
    phone: '',
    address: '',
    birth_date: '',
    gender: '', // ✅ Civilité obligatoire (male/female)
    avs_number: '', // ✅ Numéro AVS optionnel
    is_institution: false,
    institution_name: '',
    residence_facility: '',
    billing_address: '',
    contact_email: '', // Email de facturation (optionnel)
    contact_phone: '',
    domicile_address: '',
    domicile_zip: '',
    domicile_city: '',
    preferential_rate: '',
    door_code: '',
    floor: '',
    access_notes: '',
    gp_name: '',
    gp_phone: '',
    default_billed_to_type: '',
    default_billed_to_contact: '',
    is_active: true,
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showBillingInfo, setShowBillingInfo] = useState(false);
  const [showAdvancedBilling, setShowAdvancedBilling] = useState(false);
  const [showHospitalization, setShowHospitalization] = useState(false);
  const [showCurator, setShowCurator] = useState(false);
  const [createdClient, setCreatedClient] = useState(null);

  const [expandedSections, setExpandedSections] = useState({
    identity: true,
    contact: true,
    residence: false,
    billing: false,
    hospitalization: false,
    curator: false,
  });

  // Coordonnées GPS pour adresse de domicile
  const [domicileCoords, setDomicileCoords] = useState({
    lat: null,
    lon: null,
  });

  // Coordonnées GPS pour adresse de facturation
  const [billingCoords, setBillingCoords] = useState({ lat: null, lon: null });

  const [stayData, setStayData] = useState({
    company_id: '',
    start_date: '',
    end_date: '',
    notes: '',
  });

  const [billingPartyData, setBillingPartyData] = useState({
    billing_party_id: '',
    role: '',
    is_default: false,
    contact_name: '',
    contact_email: '',
    contact_phone: '',
  });

  const [clinics, setClinics] = useState([]);
  const [billingParties, setBillingParties] = useState([]);
  const [loadingClinics, setLoadingClinics] = useState(false);
  const [loadingBillingParties, setLoadingBillingParties] = useState(false);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }));
  };

  const toggleSection = (section) => {
    setExpandedSections((prev) => ({
      ...prev,
      [section]: !prev[section],
    }));
  };

  useEffect(() => {
    if (!formData.is_institution) return;
    setShowHospitalization(false);
    setShowCurator(false);
    setStayData({
      company_id: '',
      start_date: '',
      end_date: '',
      notes: '',
    });
    setBillingPartyData({
      billing_party_id: '',
      role: '',
      is_default: false,
      contact_name: '',
      contact_email: '',
      contact_phone: '',
    });
  }, [formData.is_institution]);

  useEffect(() => {
    if (!showHospitalization || clinics.length > 0) return;
    const loadClinics = async () => {
      try {
        setLoadingClinics(true);
        const response = await fetchClinicBillingMappings();
        const mappings = response.data || [];
        const uniqueClinics = [];
        const seen = new Set();
        mappings.forEach((mapping) => {
          if (mapping.clinic_company_id && !seen.has(mapping.clinic_company_id)) {
            seen.add(mapping.clinic_company_id);
            uniqueClinics.push({
              id: mapping.clinic_company_id,
              name: mapping.clinic_company_name,
            });
          }
        });
        setClinics(uniqueClinics);
      } catch (err) {
        console.error('Erreur lors du chargement des cliniques:', err);
      } finally {
        setLoadingClinics(false);
      }
    };
    loadClinics();
  }, [showHospitalization, clinics.length]);

  useEffect(() => {
    if (!showCurator || billingParties.length > 0) return;
    const loadBillingParties = async () => {
      try {
        setLoadingBillingParties(true);
        const response = await fetchBillingParties({ active: true });
        setBillingParties(response.data || []);
      } catch (err) {
        console.error('Erreur lors du chargement des tiers payeurs:', err);
      } finally {
        setLoadingBillingParties(false);
      }
    };
    loadBillingParties();
  }, [showCurator, billingParties.length]);

  useEffect(() => {
    if (!showHospitalization) return;
    setExpandedSections((prev) => ({
      ...prev,
      hospitalization: true,
    }));
  }, [showHospitalization]);

  useEffect(() => {
    if (!showCurator) return;
    setExpandedSections((prev) => ({
      ...prev,
      curator: true,
    }));
  }, [showCurator]);

  // Gérer la sélection d'adresse de domicile via autocomplete
  const handleDomicileAddressSelect = (item) => {
    console.log('📍 [Domicile] Adresse sélectionnée:', item);

    // ✅ Utiliser la fonction utilitaire pour parser l'adresse avec détection d'établissement
    const label = item.label || '';
    const parsed = parseAddressWithEstablishment(label, item);

    // Construire l'adresse complète (rue + numéro)
    const address =
      parsed.streetNumber && parsed.street
        ? `${parsed.street} ${parsed.streetNumber}`.trim()
        : parsed.street || item.address || '';

    console.log('📍 [Domicile] Composants extraits:', {
      establishment: parsed.establishment,
      streetNumber: parsed.streetNumber,
      street: parsed.street,
      address,
      postcode: parsed.postcode,
      city: parsed.city,
    });

    setFormData((prev) => ({
      ...prev,
      address: label,
      // ✅ Si un établissement est détecté, le mettre dans residence_facility
      residence_facility: parsed.establishment || prev.residence_facility,
      domicile_address: address,
      domicile_zip: parsed.postcode,
      domicile_city: parsed.city,
      // ✅ NE PAS toucher au champ address global ici
      // Il sera construit dans le payload si nécessaire
    }));

    // Sauvegarder les coordonnées GPS
    setDomicileCoords({
      lat: item.lat ?? null,
      lon: item.lon ?? null,
    });

    console.log(`📍 [Domicile] GPS: ${item.lat}, ${item.lon}`);
  };

  // Gérer la sélection d'adresse de facturation via autocomplete
  const handleBillingAddressSelect = (item) => {
    console.log('📍 [Facturation] Adresse sélectionnée:', item);

    const fullAddress = item.label || '';
    setFormData((prev) => ({
      ...prev,
      billing_address: fullAddress,
    }));

    // Sauvegarder les coordonnées GPS
    setBillingCoords({
      lat: item.lat ?? null,
      lon: item.lon ?? null,
    });

    console.log(`📍 [Facturation] GPS: ${item.lat}, ${item.lon}`);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    setError(null);

    // Validation
    if (!formData.first_name.trim() || !formData.last_name.trim()) {
      setError('Le prénom et le nom sont requis');
      return;
    }

    // ✅ Validation civilité obligatoire
    if (!formData.gender) {
      setError('La civilité (Madame/Monsieur) est obligatoire');
      return;
    }

    // Vérifier que l'adresse de domicile est complète (rue + code postal + ville)
    const hasCompleteAddress =
      formData.domicile_address.trim() &&
      formData.domicile_zip.trim() &&
      formData.domicile_city.trim();

    if (!hasCompleteAddress && !formData.address.trim()) {
      setError("L'adresse de domicile complète est requise (rue, code postal et ville)");
      return;
    }

    if (formData.is_institution && !formData.institution_name.trim()) {
      setError("Le nom de l'institution est requis pour les institutions");
      return;
    }

    if (showHospitalization) {
      if (!stayData.company_id) {
        setError('Merci de sélectionner une clinique pour l’hospitalisation');
        return;
      }
      if (!stayData.start_date) {
        setError('Merci de renseigner la date de début d’hospitalisation');
        return;
      }
    }

    if (showCurator && !billingPartyData.billing_party_id) {
      setError('Merci de sélectionner un tiers payeur pour le curateur');
      return;
    }

    setLoading(true);

    try {
      // Préparer le payload
      // ✅ Si pas d'adresse de facturation spécifique → copier domicile
      const hasSeparateBilling = showBillingInfo && formData.billing_address.trim();

      console.log('📋 [NewClient] Préparation payload:');
      console.log('  - Checkbox facturation active:', showBillingInfo);
      console.log('  - Adresse de facturation remplie:', formData.billing_address.trim() !== '');
      console.log('  - Facturation séparée:', hasSeparateBilling);
      console.log('  - Domicile GPS:', domicileCoords);
      console.log('  - Facturation GPS:', billingCoords);

      const manualAddress = formData.address.trim();
      const domicileAddress = formData.domicile_address.trim() || manualAddress;
      const domicileZip = formData.domicile_zip.trim();
      const domicileCity = formData.domicile_city.trim();
      const fullDomicile = [domicileAddress, domicileZip, domicileCity]
        .filter(Boolean)
        .join(', ');
      const billingAddress = hasSeparateBilling
        ? formData.billing_address.trim()
        : fullDomicile || domicileAddress || manualAddress;

      const payload = {
        // ✅ TOUS les clients créés depuis le Dashboard sont PRIVATE
        // (pas de compte SELF_SERVICE, pas de connexion app mobile)
        client_type: 'PRIVATE',
        first_name: formData.first_name.trim(),
        last_name: formData.last_name.trim(),
        // ✅ Civilité obligatoire
        gender: formData.gender,
        address: fullDomicile || domicileAddress || manualAddress,
        birth_date: formData.birth_date || undefined,
        // ✅ Numéro AVS optionnel
        avs_number: formData.avs_number?.trim() || undefined,
        // ✅ Établissement de résidence (EMS, clinique, foyer, etc.)
        residence_facility: formData.residence_facility?.trim() || undefined,
        // Adresse de domicile (structurée)
        domicile_address: domicileAddress || undefined,
        domicile_zip: domicileZip || undefined,
        domicile_city: domicileCity || undefined,
        // Coordonnées GPS du domicile
        domicile_lat: domicileCoords.lat,
        domicile_lon: domicileCoords.lon,
        // Adresse de facturation (si différente, sinon copie du domicile)
        billing_address: billingAddress || undefined,
        // Coordonnées GPS de facturation (si différentes, sinon copie du domicile)
        billing_lat: hasSeparateBilling ? billingCoords.lat : domicileCoords.lat,
        billing_lon: hasSeparateBilling ? billingCoords.lon : domicileCoords.lon,
        // Tarif préférentiel
        preferential_rate: formData.preferential_rate
          ? parseFloat(formData.preferential_rate)
          : undefined,
        // Institution
        is_institution: formData.is_institution,
        institution_name: formData.is_institution ? formData.institution_name.trim() : undefined,
        // Accès logement
        door_code: formData.door_code?.trim() || undefined,
        floor: formData.floor?.trim() || undefined,
        access_notes: formData.access_notes?.trim() || undefined,
        // Médecin traitant
        gp_name: formData.gp_name?.trim() || undefined,
        gp_phone: formData.gp_phone?.trim() || undefined,
        // Facturation par défaut
        default_billed_to_type: formData.default_billed_to_type || undefined,
        default_billed_to_contact: formData.default_billed_to_contact?.trim() || undefined,
      };

      // ✅ TOUS les clients : générer un email interne unique pour le User
      // Les vrais emails de contact vont dans contact_email (facturation)
      const randomId = Math.random().toString(36).substring(2, 10);
      const timestamp = Date.now().toString(36);

      if (formData.is_institution) {
        payload.email = `institution-${randomId}-${timestamp}@internal.atmr.local`;
      } else {
        payload.email = `client-${randomId}-${timestamp}@internal.atmr.local`;
      }

      // Téléphone et emails de contact (pour facturation)
      payload.phone = formData.phone?.trim() || null;
      payload.contact_email = formData.contact_email?.trim() || null;
      payload.contact_phone = formData.contact_phone?.trim() || null;

      // Nettoyer le payload : supprimer les valeurs null/undefined/vides (sauf is_active)
      Object.keys(payload).forEach((key) => {
        if (key === 'is_active') return;
        if (payload[key] === null || payload[key] === undefined || payload[key] === '') {
          delete payload[key];
        }
      });

      if (showHospitalization) {
        payload.hospitalization = {
          company_id: stayData.company_id,
          start_date: stayData.start_date,
          end_date: stayData.end_date || null,
          notes: stayData.notes || null,
        };
      }

      if (showCurator) {
        payload.billing_party_link = {
          billing_party_id: billingPartyData.billing_party_id,
          role: billingPartyData.role || null,
          is_default: !!billingPartyData.is_default,
          contact_name: billingPartyData.contact_name || null,
          contact_email: billingPartyData.contact_email || null,
          contact_phone: billingPartyData.contact_phone || null,
        };
      }

      console.log('📤 Payload envoyé au backend:', payload);
      await onSave(payload, { existingClient: createdClient });
      setCreatedClient(null);
      setLoading(false);
      onClose();
    } catch (err) {
      const apiError =
        err.response?.data?.error || err.error || err.message || 'Erreur lors de la création du client';
      if (err?.createdClient) {
        setCreatedClient(err.createdClient);
        setError(`Client créé, mais erreur lors des informations complémentaires: ${apiError}`);
      } else {
        setError(apiError);
      }
      setLoading(false);
    }
  };

  const selectedClinic = clinics.find(
    (clinic) => String(clinic.id) === String(stayData.company_id)
  );
  const selectedBillingParty = billingParties.find(
    (party) => String(party.id) === String(billingPartyData.billing_party_id)
  );

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content modal-xl" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2 className="modal-title">
            {formData.is_institution ? 'Nouvelle institution' : 'Nouveau client'}
          </h2>
          <button className="modal-close" onClick={onClose}>
            ✕
          </button>
        </div>

        <form onSubmit={handleSubmit} className={styles.form}>
          {error && <div className={styles.error}>{error}</div>}

          {/* 1) Identité */}
          <div className={styles.accordion}>
            <button
              type="button"
              className={styles.accordionHeader}
              onClick={() => toggleSection('identity')}
            >
              <span className={styles.accordionTitle}>📋 Identité du client</span>
              <span className={styles.accordionIcon}>
                {expandedSections.identity ? '▾' : '▸'}
              </span>
            </button>
            {expandedSections.identity && (
              <div className={styles.accordionContent}>
                <div className={styles.checkboxGroup}>
                  <label className={styles.checkboxLabel}>
                    <input
                      type="checkbox"
                      name="is_institution"
                      checked={formData.is_institution}
                      onChange={handleChange}
                      disabled={loading}
                    />
                    <span className={styles.checkboxText}>
                      <strong>Est une institution</strong>
                      <small>Clinique, hôpital, centre médical, etc.</small>
                    </span>
                  </label>
                </div>

                {formData.is_institution && (
                  <div className={styles.formGroup}>
                    <label htmlFor="institution_name" className={styles.label}>
                      Nom de l'institution *
                    </label>
                    <input
                      type="text"
                      id="institution_name"
                      name="institution_name"
                      value={formData.institution_name}
                      onChange={handleChange}
                      className={styles.input}
                      placeholder="Ex: Clinique du Léman"
                      required={formData.is_institution}
                      disabled={loading}
                    />
                  </div>
                )}

                {formData.is_institution && (
                  <p className={styles.sectionDescription}>
                    <em>
                      Ces informations concernent la personne de contact pour l'institution, pas
                      l'institution elle-même.
                    </em>
                  </p>
                )}

                <div className={styles.formRow}>
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
                      required
                      disabled={loading}
                    />
                  </div>
                </div>

                <div className={styles.formRow}>
                  <div className={styles.formGroup}>
                    <label htmlFor="gender" className={styles.label}>
                      Civilité *
                    </label>
                    <select
                      id="gender"
                      name="gender"
                      value={formData.gender}
                      onChange={handleChange}
                      className={styles.input}
                      required
                      disabled={loading}
                    >
                      <option value="">-- Sélectionnez --</option>
                      <option value="male">Monsieur</option>
                      <option value="female">Madame</option>
                    </select>
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

                {!formData.is_institution && (
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
                )}
              </div>
            )}
          </div>

          {/* 2) Contact & domicile */}
          <div className={styles.accordion}>
            <button
              type="button"
              className={styles.accordionHeader}
              onClick={() => toggleSection('contact')}
            >
              <span className={styles.accordionTitle}>📞 Contact & domicile</span>
              <span className={styles.accordionIcon}>
                {expandedSections.contact ? '▾' : '▸'}
              </span>
            </button>
            {expandedSections.contact && (
              <div className={styles.accordionContent}>
                <div className={styles.formGroup}>
                  <label htmlFor="phone" className={styles.label}>
                    Téléphone
                  </label>
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

                <div className={styles.formGroup}>
                  <label htmlFor="domicile_address" className={styles.label}>
                    {formData.is_institution ? "Adresse de l'institution *" : 'Adresse de domicile *'}
                  </label>
                  <AddressAutocomplete
                    name="domicile_address"
                    value={formData.address}
                    onChange={(e) => {
                      setDomicileCoords({ lat: null, lon: null });
                      setFormData((prev) => ({
                        ...prev,
                        address: e.target.value,
                      }));
                    }}
                    onSelect={handleDomicileAddressSelect}
                    placeholder="Ex: Avenue Ernest-Pictet 9, 1203, Genève"
                    disabled={loading}
                  />
                  <small className={styles.hint}>
                    💡 Tapez pour rechercher une adresse avec autocomplete
                  </small>
                </div>

                <div className={styles.formRow}>
                  <div className={styles.formGroup}>
                    <label htmlFor="domicile_address_street" className={styles.label}>
                      Rue et numéro
                    </label>
                    <input
                      type="text"
                      id="domicile_address_street"
                      name="domicile_address"
                      value={formData.domicile_address}
                      onChange={handleChange}
                      className={styles.input}
                      placeholder="Rempli automatiquement"
                      disabled={loading}
                      readOnly
                    />
                  </div>

                  <div className={styles.formGroup}>
                    <label htmlFor="domicile_zip" className={styles.label}>
                      Code postal
                    </label>
                    <input
                      type="text"
                      id="domicile_zip"
                      name="domicile_zip"
                      value={formData.domicile_zip}
                      onChange={handleChange}
                      className={styles.input}
                      placeholder="Rempli automatiquement"
                      disabled={loading}
                      readOnly
                    />
                  </div>

                  <div className={styles.formGroup}>
                    <label htmlFor="domicile_city" className={styles.label}>
                      Ville
                    </label>
                    <input
                      type="text"
                      id="domicile_city"
                      name="domicile_city"
                      value={formData.domicile_city}
                      onChange={handleChange}
                      className={styles.input}
                      placeholder="Rempli automatiquement"
                      disabled={loading}
                      readOnly
                    />
                  </div>
                </div>

                <div className={styles.formRow}>
                  <div className={styles.formGroup}>
                    <label htmlFor="door_code" className={styles.label}>
                      Code porte
                    </label>
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
                    <label htmlFor="floor" className={styles.label}>
                      Étage
                    </label>
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
                  <label htmlFor="access_notes" className={styles.label}>
                    Notes d'accès
                  </label>
                  <textarea
                    id="access_notes"
                    name="access_notes"
                    value={formData.access_notes}
                    onChange={handleChange}
                    className={styles.textarea}
                    placeholder="Ex: appeler avant d'arriver, sonnette à gauche..."
                    disabled={loading}
                  />
                </div>

                <div className={styles.formRow}>
                  <div className={styles.formGroup}>
                    <label htmlFor="gp_name" className={styles.label}>
                      Médecin traitant
                    </label>
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
                    <label htmlFor="gp_phone" className={styles.label}>
                      Téléphone du médecin
                    </label>
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
              </div>
            )}
          </div>

          {/* 3) Établissement de résidence */}
          <div className={styles.accordion}>
            <button
              type="button"
              className={styles.accordionHeader}
              onClick={() => toggleSection('residence')}
            >
              <span className={styles.accordionTitle}>🏠 Établissement de résidence</span>
              <span className={styles.accordionIcon}>
                {expandedSections.residence ? '▾' : '▸'}
              </span>
            </button>
            {expandedSections.residence && (
              <div className={styles.accordionContent}>
                <div className={styles.formGroup}>
                  <label htmlFor="residence_facility" className={styles.label}>
                    Établissement de résidence (EMS, foyer, etc.)
                  </label>
                  <input
                    type="text"
                    id="residence_facility"
                    name="residence_facility"
                    value={formData.residence_facility}
                    onChange={handleChange}
                    className={styles.input}
                    placeholder="Ex: EMS Maison de Vessy, Foyer Clair Bois..."
                    disabled={loading}
                  />
                  <small className={styles.hint}>
                    💡 Indiquer si le client habite dans un EMS, Foyer, ou autre établissement
                  </small>
                </div>
              </div>
            )}
          </div>

          {/* 4) Facturation */}
          <div className={styles.accordion}>
            <button
              type="button"
              className={styles.accordionHeader}
              onClick={() => toggleSection('billing')}
            >
              <span className={styles.accordionTitle}>💰 Facturation</span>
              <span className={styles.accordionIcon}>
                {expandedSections.billing ? '▾' : '▸'}
              </span>
            </button>
            {expandedSections.billing && (
              <div className={styles.accordionContent}>
                <div className={styles.checkboxGroup}>
                  <label className={styles.checkboxLabel}>
                    <input
                      type="checkbox"
                      id="show_billing_info"
                      checked={showBillingInfo}
                      onChange={(e) => setShowBillingInfo(e.target.checked)}
                      disabled={loading}
                    />
                    <span className={styles.checkboxText}>
                      <strong>Adresse de facturation différente</strong>
                      <small>Par défaut, la facturation utilise l'adresse de domicile</small>
                    </span>
                  </label>
                </div>

                <div className={styles.formRow}>
                  <div className={styles.formGroup}>
                    <label htmlFor="contact_email" className={styles.label}>
                      Email de contact / facturation
                    </label>
                    <input
                      type="email"
                      id="contact_email"
                      name="contact_email"
                      value={formData.contact_email}
                      onChange={handleChange}
                      className={styles.input}
                      placeholder="facturation@institution.ch"
                      disabled={loading}
                    />
                  </div>

                  <div className={styles.formGroup}>
                    <label htmlFor="contact_phone" className={styles.label}>
                      Téléphone de contact
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

                {showBillingInfo && (
                  <div className={styles.formGroup}>
                    <label htmlFor="billing_address" className={styles.label}>
                      Adresse de facturation
                    </label>
                    <AddressAutocomplete
                      name="billing_address"
                      value={formData.billing_address}
                      onChange={(e) => {
                        setBillingCoords({ lat: null, lon: null });
                        setFormData((prev) => ({
                          ...prev,
                          billing_address: e.target.value,
                        }));
                      }}
                      onSelect={handleBillingAddressSelect}
                      placeholder="Ex: Avenue de la Gare 5, 1003, Lausanne"
                      disabled={loading}
                    />
                    <small className={styles.hint}>
                      💡 Si différente de l'adresse de domicile
                    </small>
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
                        <option value="">-- Laisser par défaut --</option>
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

                <div className={styles.formGroup}>
                  <label htmlFor="preferential_rate" className={styles.label}>
                    Tarif préférentiel (CHF)
                  </label>
                  <small className={styles.hint}>
                    Prix d'un trajet simple. Pour un aller-retour, ce tarif est appliqué 2 fois.
                  </small>
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
                </div>
              </div>
            )}
          </div>

          {/* Statut */}
          <div className={styles.section}>
            <div className={styles.checkboxGroup}>
              <label className={styles.checkboxLabel}>
                <input
                  type="checkbox"
                  name="is_active"
                  checked={formData.is_active}
                  onChange={handleChange}
                  disabled={loading}
                />
                <span className={styles.checkboxText}>
                  <strong>Client actif</strong>
                  <small>Les clients inactifs n&apos;apparaissent pas dans les sélections</small>
                </span>
              </label>
            </div>
          </div>

          {/* Options de création */}
          {!formData.is_institution && (
            <div className={styles.section}>
              <div className={styles.checkboxGroup}>
                <label className={styles.checkboxLabel}>
                  <input
                    type="checkbox"
                    checked={showHospitalization}
                    onChange={(e) => {
                      setShowHospitalization(e.target.checked);
                      if (e.target.checked && !stayData.start_date) {
                        setStayData((prev) => ({
                          ...prev,
                          start_date: new Date().toISOString().split('T')[0],
                        }));
                      }
                    }}
                    disabled={loading}
                  />
                  <span className={styles.checkboxText}>
                    <strong>🏥 Ajouter une hospitalisation maintenant</strong>
                    <small>Sélectionner une clinique et des dates de séjour</small>
                  </span>
                </label>
              </div>

              <div className={styles.checkboxGroup}>
                <label className={styles.checkboxLabel}>
                  <input
                    type="checkbox"
                    checked={showCurator}
                    onChange={(e) => setShowCurator(e.target.checked)}
                    disabled={loading}
                  />
                  <span className={styles.checkboxText}>
                    <strong>💼 Ajouter un curateur / tiers payeur</strong>
                    <small>Définir le payeur et le contact curateur</small>
                  </span>
                </label>
              </div>
            </div>
          )}

          {/* 5) Hospitalisation */}
          {!formData.is_institution && showHospitalization && (
            <div className={styles.accordion}>
              <button
                type="button"
                className={styles.accordionHeader}
                onClick={() => toggleSection('hospitalization')}
              >
                <span className={styles.accordionTitle}>🏥 Hospitalisation (optionnel)</span>
                <span className={styles.accordionIcon}>
                  {expandedSections.hospitalization ? '▾' : '▸'}
                </span>
              </button>
              {expandedSections.hospitalization && (
                <div className={styles.accordionContent}>
                  <div className={styles.formGroup}>
                    <label htmlFor="stay_company_id" className={styles.label}>
                      Clinique / Établissement *
                    </label>
                    <select
                      id="stay_company_id"
                      value={stayData.company_id}
                      onChange={(e) =>
                        setStayData((prev) => ({ ...prev, company_id: e.target.value }))
                      }
                      className={styles.input}
                      disabled={loading || loadingClinics}
                      required={showHospitalization}
                    >
                      <option value="">
                        {loadingClinics
                          ? 'Chargement des cliniques...'
                          : '-- Sélectionnez une clinique --'}
                      </option>
                      {clinics.map((clinic) => (
                        <option key={clinic.id} value={clinic.id}>
                          {clinic.name}
                        </option>
                      ))}
                    </select>
                    {!loadingClinics && clinics.length === 0 && (
                      <small className={styles.hint}>
                        ⚠️ Aucune clinique disponible (vérifier les mappings cliniques).
                      </small>
                    )}
                  </div>

                  <div className={styles.formRow}>
                    <div className={styles.formGroup}>
                      <label htmlFor="stay_start_date" className={styles.label}>
                        Date de début *
                      </label>
                      <input
                        type="date"
                        id="stay_start_date"
                        value={stayData.start_date}
                        onChange={(e) =>
                          setStayData((prev) => ({ ...prev, start_date: e.target.value }))
                        }
                        className={styles.input}
                        disabled={loading}
                      />
                    </div>

                    <div className={styles.formGroup}>
                      <label htmlFor="stay_end_date" className={styles.label}>
                        Date de fin (optionnel)
                      </label>
                      <input
                        type="date"
                        id="stay_end_date"
                        value={stayData.end_date}
                        onChange={(e) =>
                          setStayData((prev) => ({ ...prev, end_date: e.target.value }))
                        }
                        className={styles.input}
                        disabled={loading}
                      />
                    </div>
                  </div>

                  <div className={styles.formGroup}>
                    <label htmlFor="stay_notes" className={styles.label}>
                      Notes (optionnel)
                    </label>
                    <textarea
                      id="stay_notes"
                      value={stayData.notes}
                      onChange={(e) =>
                        setStayData((prev) => ({ ...prev, notes: e.target.value }))
                      }
                      className={styles.textarea}
                      placeholder="Informations complémentaires sur le séjour..."
                      disabled={loading}
                    />
                  </div>
                </div>
              )}
            </div>
          )}

          {/* 6) Curateur / Tiers payeur */}
          {!formData.is_institution && showCurator && (
            <div className={styles.accordion}>
              <button
                type="button"
                className={styles.accordionHeader}
                onClick={() => toggleSection('curator')}
              >
                <span className={styles.accordionTitle}>💼 Curateur / tiers payeur</span>
                <span className={styles.accordionIcon}>
                  {expandedSections.curator ? '▾' : '▸'}
                </span>
              </button>
              {expandedSections.curator && (
                <div className={styles.accordionContent}>
                  <div className={styles.formGroup}>
                    <label htmlFor="billing_party_id" className={styles.label}>
                      Tiers payeur *
                    </label>
                    <select
                      id="billing_party_id"
                      value={billingPartyData.billing_party_id}
                      onChange={(e) =>
                        setBillingPartyData((prev) => ({
                          ...prev,
                          billing_party_id: e.target.value,
                        }))
                      }
                      className={styles.input}
                      disabled={loading || loadingBillingParties}
                      required={showCurator}
                    >
                      <option value="">
                        {loadingBillingParties
                          ? 'Chargement des tiers payeurs...'
                          : '-- Sélectionnez un tiers payeur --'}
                      </option>
                      {billingParties
                        .filter((party) => party.type !== 'clinic')
                        .map((party) => (
                          <option key={party.id} value={party.id}>
                            {party.display_name} ({party.type})
                          </option>
                        ))}
                    </select>
                    {!loadingBillingParties && billingParties.length === 0 && (
                      <small className={styles.hint}>
                        ⚠️ Aucun tiers payeur disponible (vérifier les paramètres de facturation).
                      </small>
                    )}
                  </div>

                  <div className={styles.formRow}>
                    <div className={styles.formGroup}>
                      <label htmlFor="billing_party_role" className={styles.label}>
                        Rôle (optionnel)
                      </label>
                      <input
                        type="text"
                        id="billing_party_role"
                        value={billingPartyData.role}
                        onChange={(e) =>
                          setBillingPartyData((prev) => ({ ...prev, role: e.target.value }))
                        }
                        className={styles.input}
                        placeholder="Ex: curateur principal"
                        disabled={loading}
                      />
                    </div>

                    <div className={styles.formGroup}>
                      <label className={styles.checkboxLabel}>
                        <input
                          type="checkbox"
                          checked={billingPartyData.is_default}
                          onChange={(e) =>
                            setBillingPartyData((prev) => ({
                              ...prev,
                              is_default: e.target.checked,
                            }))
                          }
                          disabled={loading}
                        />
                        <span className={styles.checkboxText}>
                          <strong>Définir comme payeur par défaut</strong>
                          <small>Ce payeur sera utilisé automatiquement</small>
                        </span>
                      </label>
                    </div>
                  </div>

                  <div className={styles.formRow}>
                    <div className={styles.formGroup}>
                      <label htmlFor="curator_name" className={styles.label}>
                        Curateur (optionnel)
                      </label>
                      <input
                        type="text"
                        id="curator_name"
                        value={billingPartyData.contact_name}
                        onChange={(e) =>
                          setBillingPartyData((prev) => ({
                            ...prev,
                            contact_name: e.target.value,
                          }))
                        }
                        className={styles.input}
                        placeholder="Ex: Curateur A"
                        disabled={loading}
                      />
                      <small className={styles.hint}>
                        Contact spécifique à ce client (ne modifie pas le tiers payeur)
                      </small>
                    </div>

                    <div className={styles.formGroup}>
                      <label htmlFor="curator_email" className={styles.label}>
                        Email du curateur (optionnel)
                      </label>
                      <input
                        type="email"
                        id="curator_email"
                        value={billingPartyData.contact_email}
                        onChange={(e) =>
                          setBillingPartyData((prev) => ({
                            ...prev,
                            contact_email: e.target.value,
                          }))
                        }
                        className={styles.input}
                        placeholder="curateur@opad.ch"
                        disabled={loading}
                      />
                    </div>
                  </div>

                  <div className={styles.formGroup}>
                    <label htmlFor="curator_phone" className={styles.label}>
                      Téléphone du curateur (optionnel)
                    </label>
                    <input
                      type="tel"
                      id="curator_phone"
                      value={billingPartyData.contact_phone}
                      onChange={(e) =>
                        setBillingPartyData((prev) => ({
                          ...prev,
                          contact_phone: e.target.value,
                        }))
                      }
                      className={styles.input}
                      placeholder="+41 22 000 00 00"
                      disabled={loading}
                    />
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Résumé */}
          <div className={styles.summaryCard}>
            <div className={styles.summaryRow}>
              <span className={styles.summaryLabel}>Client</span>
              <span className={styles.summaryValue}>
                {formData.is_institution
                  ? formData.institution_name || 'Institution'
                  : `${formData.first_name} ${formData.last_name}`.trim() || 'Client'}
              </span>
            </div>
            <div className={styles.summaryRow}>
              <span className={styles.summaryLabel}>Adresse</span>
              <span className={styles.summaryValue}>
                {formData.domicile_address ||
                  formData.address ||
                  'Adresse non renseignée'}
              </span>
            </div>
            <div className={styles.summaryRow}>
              <span className={styles.summaryLabel}>Facturation</span>
              <span className={styles.summaryValue}>
                {showBillingInfo && formData.billing_address
                  ? formData.billing_address
                  : formData.domicile_address || formData.address || 'Adresse non renseignée'}
              </span>
            </div>
            {showHospitalization && (
              <div className={styles.summaryRow}>
                <span className={styles.summaryLabel}>Hospitalisation</span>
                <span className={styles.summaryValue}>
                  {selectedClinic
                    ? `${selectedClinic.name} • ${stayData.start_date || 'date à définir'}`
                    : 'Clinique à définir'}
                </span>
              </div>
            )}
            {showCurator && (
              <div className={styles.summaryRow}>
                <span className={styles.summaryLabel}>Curateur</span>
                <span className={styles.summaryValue}>
                  {selectedBillingParty
                    ? selectedBillingParty.display_name
                    : 'Tiers payeur à définir'}
                </span>
              </div>
            )}
          </div>

          {/* Actions */}
          <div className="modal-footer">
            <button
              type="button"
              onClick={onClose}
              className="btn btn-secondary"
              disabled={loading}
            >
              Annuler
            </button>
            <button type="submit" className="btn btn-primary" disabled={loading}>
              {loading
                ? 'Création...'
                : createdClient
                  ? 'Finaliser'
                  : 'Créer le client'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default NewClientModal;
