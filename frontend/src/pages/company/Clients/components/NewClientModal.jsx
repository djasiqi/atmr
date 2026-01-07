import React, { useState } from 'react';
import styles from './ClientFormModal.module.css';
import AddressAutocomplete from '../../../../components/common/AddressAutocomplete';
import { parseAddressWithEstablishment } from '../../../../utils/addressParser';

const NewClientModal = ({ onClose, onSave }) => {
  const [formData, setFormData] = useState({
    // ✅ client_type et email supprimés - tous les clients sont PRIVATE
    first_name: '',
    last_name: '',
    phone: '',
    address: '',
    birth_date: '',
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
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showBillingInfo, setShowBillingInfo] = useState(false);

  // Coordonnées GPS pour adresse de domicile
  const [domicileCoords, setDomicileCoords] = useState({
    lat: null,
    lon: null,
  });

  // Coordonnées GPS pour adresse de facturation
  const [billingCoords, setBillingCoords] = useState({ lat: null, lon: null });

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }));
  };

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

    // Validation
    if (!formData.first_name.trim() || !formData.last_name.trim()) {
      setError('Le prénom et le nom sont requis');
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

    setLoading(true);
    setError(null);

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

      const payload = {
        // ✅ TOUS les clients créés depuis le Dashboard sont PRIVATE
        // (pas de compte SELF_SERVICE, pas de connexion app mobile)
        client_type: 'PRIVATE',
        first_name: formData.first_name.trim(),
        last_name: formData.last_name.trim(),
        address:
          `${formData.domicile_address}, ${formData.domicile_zip}, ${formData.domicile_city}`.trim(),
        birth_date: formData.birth_date || undefined,
        // Adresse de domicile (structurée)
        domicile_address: formData.domicile_address.trim() || undefined,
        domicile_zip: formData.domicile_zip.trim() || undefined,
        domicile_city: formData.domicile_city.trim() || undefined,
        // Coordonnées GPS du domicile
        domicile_lat: domicileCoords.lat,
        domicile_lon: domicileCoords.lon,
        // Adresse de facturation (si différente, sinon copie du domicile)
        billing_address: hasSeparateBilling
          ? formData.billing_address.trim()
          : `${formData.domicile_address}, ${formData.domicile_zip}, ${formData.domicile_city}`.trim(),
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

      // Nettoyer le payload : supprimer les valeurs null/undefined/vides
      Object.keys(payload).forEach((key) => {
        if (payload[key] === null || payload[key] === undefined || payload[key] === '') {
          delete payload[key];
        }
      });

      console.log('📤 Payload envoyé au backend:', payload);
      await onSave(payload);
    } catch (err) {
      setError(err.error || err.message || 'Erreur lors de la création du client');
      setLoading(false);
    }
  };

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

          {/* Informations personnelles */}
          <div className={styles.section}>
            <h3 className={styles.sectionTitle}>
              {formData.is_institution
                ? 'Contact principal (personne de référence)'
                : 'Informations personnelles'}
            </h3>

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

            {/* ✅ Email supprimé de la section principale - uniquement dans facturation */}

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

          {/* Adresse de domicile */}
          <div className={styles.section}>
            <h3 className={styles.sectionTitle}>
              {formData.is_institution ? "📍 Adresse de l'institution" : '🏠 Adresse de domicile'}
            </h3>
            <p className={styles.sectionDescription}>
              {formData.is_institution
                ? "Adresse de l'institution"
                : 'Adresse où le client habite (utilisée pour la prise en charge par défaut)'}
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
                💡 Indiquer si le client habite dans un EMS, Foyer, ou autre établissement
              </small>
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="domicile_address" className={styles.label}>
                Adresse complète *
              </label>
              <AddressAutocomplete
                name="domicile_address"
                value={formData.address}
                onChange={(e) => {
                  // Si l'utilisateur tape manuellement, vider les coordonnées
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
          </div>

          {/* Checkbox pour afficher les coordonnées de facturation */}
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
                <strong>📋 Ajouter des coordonnées de facturation différentes</strong>
              </span>
            </label>
          </div>

          {/* Coordonnées de facturation */}
          {showBillingInfo && (
            <div className={styles.section}>
              <h3 className={styles.sectionTitle}>Coordonnées de facturation</h3>

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
                <small className={styles.hint}>
                  💡 Pour recevoir les factures par email (optionnel)
                </small>
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

              <div className={styles.formGroup}>
                <label htmlFor="billing_address" className={styles.label}>
                  Adresse de facturation
                </label>
                <AddressAutocomplete
                  name="billing_address"
                  value={formData.billing_address}
                  onChange={(e) => {
                    // Si l'utilisateur tape manuellement, vider les coordonnées
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
                <small className={styles.hint}>💡 Si différente de l'adresse de domicile</small>
              </div>
            </div>
          )}

          {/* Tarif préférentiel */}
          {!formData.is_institution && (
            <div className={styles.section}>
              <h3 className={styles.sectionTitle}>💰 Tarif préférentiel</h3>

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
          )}

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
              {loading ? 'Création...' : 'Créer le client'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default NewClientModal;
