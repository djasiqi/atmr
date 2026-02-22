import React, { useState } from 'react';
import styles from './ClientFormModal.module.css';
import AddressAutocomplete from '../../../../components/common/AddressAutocomplete';
import InlineDatePicker from '../../../../components/ui/InlineDatePicker';
import { parseAddressWithEstablishment } from '../../../../utils/addressParser';
import { normalizePhone, getPhoneValidationError } from '../../../../utils/phone';
import ClientStaysSection from './ClientStaysSection';
import ClientBillingPartiesSection from './ClientBillingPartiesSection';

const EditClientModal = ({ client, onClose, onSave }) => {
  // CORRECTION: Les données viennent directement de client, pas de client.domicile
  const [formData, setFormData] = useState({
    is_institution: client.is_institution || false,
    institution_name: client.institution_name || '',
    first_name: client.user_first_name ?? client.first_name ?? client.user?.first_name ?? '',
    last_name: client.user_last_name ?? client.last_name ?? client.user?.last_name ?? '',
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
    billing_address: client.billing_address || '',
    domicile_address: client.domicile_address || client.domicile?.address || '',
    domicile_zip: client.domicile_zip || client.domicile?.zip || '',
    domicile_city: client.domicile_city || client.domicile?.city || '',
    door_code: client.access?.door_code ?? client.door_code ?? '',
    floor: client.access?.floor ?? client.floor ?? '',
    access_notes: client.access?.notes ?? client.access_notes ?? '',
    gp_name: client.gp?.name ?? client.gp_name ?? '',
    gp_phone: client.gp?.phone ?? client.gp_phone ?? '',
    show_billing_info: !!(client.billing_address && String(client.billing_address).trim()),
    default_billed_to_type: client.default_billing?.billed_to_type ?? client.default_billed_to_type ?? '',
    default_billed_to_contact: client.default_billing?.billed_to_contact ?? client.default_billed_to_contact ?? '',
    preferential_rate: client.preferential_rate || '',
    is_active: client.is_active !== false,
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showAdvancedBilling, setShowAdvancedBilling] = useState(
    !!(
      (client.default_billing?.billed_to_type && client.default_billing.billed_to_type !== 'patient') ||
      (client.default_billing?.billed_to_contact && client.default_billing.billed_to_contact.trim())
    )
  );

  // Coordonnées GPS pour adresse de domicile
  // CORRECTION: Lire directement depuis client.domicile_lat, pas client.domicile.lat
  const [domicileCoords, setDomicileCoords] = useState({
    lat: client.domicile_lat || client.domicile?.lat || null,
    lon: client.domicile_lon || client.domicile?.lon || null,
  });

  // Coordonnées GPS pour adresse de facturation
  const [billingCoords, setBillingCoords] = useState({
    lat: client.billing_lat || null,
    lon: client.billing_lon || null,
  });

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }));
  };

  // Gérer la sélection d'adresse de domicile via autocomplete
  const handleDomicileAddressSelect = (item) => {
    console.log('[Domicile] Adresse sélectionnée:', item);

    // Utiliser la fonction utilitaire pour parser l'adresse avec détection d'établissement
    const label = item.label || '';
    const parsed = parseAddressWithEstablishment(label, item);

    // Construire l'adresse complète (rue + numéro)
    const address =
      parsed.streetNumber && parsed.street
        ? `${parsed.street} ${parsed.streetNumber}`.trim()
        : parsed.street || item.address || '';

    console.log('[Domicile] Composants extraits:', {
      establishment: parsed.establishment,
      streetNumber: parsed.streetNumber,
      street: parsed.street,
      address,
      postcode: parsed.postcode,
      city: parsed.city,
    });

    setFormData((prev) => ({
      ...prev,
      // Si un établissement est détecté, le mettre dans residence_facility
      residence_facility: parsed.establishment || prev.residence_facility,
      domicile_address: address,
      domicile_zip: parsed.postcode,
      domicile_city: parsed.city,
    }));

    setDomicileCoords({
      lat: item.lat ?? null,
      lon: item.lon ?? null,
    });

    console.log(`[Domicile] GPS: ${item.lat}, ${item.lon}`);
  };

  // Gérer la sélection d'adresse de facturation via autocomplete
  const handleBillingAddressSelect = (item) => {
    console.log('[Facturation] Adresse sélectionnée:', item);

    const fullAddress = item.label || '';
    setFormData((prev) => ({
      ...prev,
      billing_address: fullAddress,
    }));

    setBillingCoords({
      lat: item.lat ?? null,
      lon: item.lon ?? null,
    });

    console.log(`[Facturation] GPS: ${item.lat}, ${item.lon}`);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (formData.is_institution && !formData.institution_name.trim()) {
      setError("Le nom de l'institution est requis pour les institutions");
      return;
    }

    if (!formData.is_institution) {
      if (!formData.first_name?.trim() || !formData.last_name?.trim()) {
        setError('Le prénom et le nom sont requis');
        return;
      }
      if (!formData.gender) {
        setError("Merci de sélectionner une civilité");
        return;
      }
    }

    setLoading(true);
    setError(null);

    // Normalisation et validation des numéros de téléphone avant envoi
    const normalizedPhone = normalizePhone(formData.phone);
    const normalizedContactPhone = normalizePhone(formData.contact_phone);
    const normalizedGpPhone = normalizePhone(formData.gp_phone);

    const phoneError = getPhoneValidationError(normalizedPhone);
    const contactPhoneError = getPhoneValidationError(normalizedContactPhone);
    const gpPhoneError = getPhoneValidationError(normalizedGpPhone);

    if (phoneError || contactPhoneError || gpPhoneError) {
      setError(phoneError || contactPhoneError || gpPhoneError);
      setLoading(false);
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

      console.log('Payload envoyé:', payload);

      const result = await onSave(payload);
      console.log('Sauvegarde réussie:', result);
      
      // Ne pas fermer ici - laisser handleSaveClient gérer la fermeture après rechargement
      setLoading(false);
      // onClose() sera appelé par handleSaveClient après le rechargement des données
    } catch (err) {
      console.error('Erreur lors de la sauvegarde:', err);
      const errorMessage =
        (typeof err === 'object' && err !== null && err.error) ||
        err.response?.data?.error ||
        err.message ||
        'Erreur lors de la sauvegarde';
      setError(errorMessage);
      setLoading(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content modal-xl" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2 className="modal-title">
            {formData.is_institution
              ? `Éditer l'institution${
                  formData.institution_name ? ` : ${formData.institution_name}` : ''
                }`
              : 'Éditer le client'}
          </h2>
          <button className="modal-close" onClick={onClose}>
            {'\u00D7'}
          </button>
        </div>

        <form onSubmit={handleSubmit} className={styles.form}>
          {error && <div className={styles.error}>{error}</div>}

          {/* Informations client */}
          <div className={styles.section}>
            <h3 className={styles.sectionTitle}>
              {formData.is_institution ? 'Informations institution' : 'Informations client'}
            </h3>

            <div className={formData.is_institution ? styles.infoCard : undefined}>
              {formData.is_institution ? (
                <>
                  <div className={styles.infoRow}>
                    <span className={styles.label}>Institution :</span>
                    <span className={styles.value}>
                      {formData.institution_name || client.institution_name || 'Non defini'}
                    </span>
                  </div>
                  <div className={styles.infoRow}>
                    <span className={styles.label}>Contact interne :</span>
                    <span className={styles.value}>
                      {formData.first_name || formData.last_name
                        ? `${formData.first_name} ${formData.last_name}`.trim()
                        : '—'}
                    </span>
                  </div>
                </>
              ) : (
                <>
                  <div className={styles.formRow}>
                    <div className={styles.formGroup}>
                      <label htmlFor="first_name" className={styles.label}>Prénom *</label>
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
                      <label htmlFor="last_name" className={styles.label}>Nom *</label>
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
                  <div className={styles.formRow}>
                    <div className={styles.formGroup}>
                      <label htmlFor="gender" className={styles.label}>Civilité *</label>
                      <select
                        id="gender"
                        name="gender"
                        value={formData.gender}
                        onChange={handleChange}
                        className={styles.input}
                        required
                        disabled={loading}
                      >
                        <option value="">— Sélectionnez —</option>
                        <option value="male">Monsieur</option>
                        <option value="female">Madame</option>
                      </select>
                    </div>
                    <div className={styles.formGroup}>
                      <label htmlFor="avs_number" className={styles.label}>Numéro AVS</label>
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
                    <label className={styles.label}>Date de naissance</label>
                    <InlineDatePicker
                      value={formData.birth_date}
                      onChange={(v) => handleChange({ target: { name: 'birth_date', value: v } })}
                      placeholder="Date de naissance"
                    />
                  </div>
                </>
              )}
              {(client.user_email || client.user?.email) && (
                <div className={styles.infoRow}>
                  <span className={styles.label}>Email utilisateur :</span>
                  <span className={styles.value}>{client.user_email || client.user?.email}</span>
                </div>
              )}
            </div>
          </div>

          {/* Type de client */}
          <div className={styles.section}>
            <h3 className={styles.sectionTitle}>Type de client</h3>

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
          </div>

          {/* Coordonnées / Facturation */}
          <div className={styles.section}>
            <h3 className={styles.sectionTitle}>Coordonnées de facturation</h3>

            <div className={styles.formRow}>
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
                <small className={styles.hint}>Si différente de l&apos;adresse de domicile</small>
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
          </div>

          {/* Adresse de domicile */}
          <div className={styles.section}>
            <h3 className={styles.sectionTitle}>Adresse de domicile</h3>
            <p className={styles.sectionDescription}>
              Adresse où le client habite (utilisée pour la prise en charge par défaut)
            </p>

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
                placeholder="Ex: EMS Maison de Vessy, Foyer Clair Bois..."
                disabled={loading}
              />
              <small className={styles.hint}>
                Indiquer si le client habite dans un EMS, Foyer, ou autre établissement
              </small>
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="domicile_address" className={styles.label}>
                Adresse complète
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
                placeholder="Ex: Avenue Ernest-Pictet 9, 1203, Genève"
                disabled={loading}
              />
              <small className={styles.hint}>Tapez pour rechercher une nouvelle adresse</small>
            </div>

            <div className={styles.formRow}>
              <div className={styles.formGroup}>
                <label htmlFor="domicile_address_street" className={styles.label}>Rue et numéro</label>
                <input
                  type="text"
                  id="domicile_address_street"
                  name="domicile_address"
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
                  name="domicile_zip"
                  value={formData.domicile_zip}
                  className={styles.input}
                  placeholder="Rempli automatiquement"
                  readOnly
                  disabled={loading}
                />
              </div>
              <div className={styles.formGroup}>
                <label htmlFor="domicile_city" className={styles.label}>Ville</label>
                <input
                  type="text"
                  id="domicile_city"
                  name="domicile_city"
                  value={formData.domicile_city}
                  className={styles.input}
                  placeholder="Rempli automatiquement"
                  readOnly
                  disabled={loading}
                />
              </div>
            </div>

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
                placeholder="Ex: appeler avant d'arriver, sonnette à gauche…"
                disabled={loading}
              />
            </div>

            {!formData.is_institution && (
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
            )}
          </div>

          {/* Tarif préférentiel */}
          <div className={styles.section}>
            <h3 className={styles.sectionTitle}>Tarif preferentiel</h3>

            <div className={styles.formGroup}>
              <label htmlFor="preferential_rate" className={styles.label}>
                Tarif par trajet (CHF)
              </label>
              <small className={styles.hint}>
                Prix d'un trajet simple. Pour un aller-retour, ce tarif sera appliqué 2 fois.
                Laisser vide pour utiliser le tarif standard.
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

          {/* Statut */}
          <div className={styles.section}>
            <h3 className={styles.sectionTitle}>Statut</h3>

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
                  <small>Les clients inactifs n'apparaissent pas dans les sélections</small>
                </span>
              </label>
            </div>
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
              {loading ? 'Sauvegarde...' : 'Enregistrer'}
            </button>
          </div>
        </form>

        {/* Sejours d'hospitalisation - Uniquement pour les clients (pas les institutions) */}
        {/* Placé en dehors du formulaire principal pour éviter les formulaires imbriqués */}
        {!formData.is_institution && (
          <ClientStaysSection clientId={client.id} />
        )}

        {/* Tiers payeur / Curateur - Uniquement pour les clients (pas les institutions) */}
        {/* Placé en dehors du formulaire principal pour éviter les formulaires imbriqués */}
        {!formData.is_institution && (
          <ClientBillingPartiesSection clientId={client.id} />
        )}
      </div>
    </div>
  );
};

export default EditClientModal;
