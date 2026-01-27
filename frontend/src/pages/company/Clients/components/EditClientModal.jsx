import React, { useState } from 'react';
import styles from './ClientFormModal.module.css';
import AddressAutocomplete from '../../../../components/common/AddressAutocomplete';
import { parseAddressWithEstablishment } from '../../../../utils/addressParser';
import ClientStaysSection from './ClientStaysSection';
import ClientBillingPartiesSection from './ClientBillingPartiesSection';

const EditClientModal = ({ client, onClose, onSave }) => {
  // ✅ CORRECTION: Les données viennent directement de client, pas de client.domicile
  const [formData, setFormData] = useState({
    is_institution: client.is_institution || false,
    institution_name: client.institution_name || '',
    residence_facility: client.residence_facility || '',
    // ✅ CORRECTION: Les données user viennent avec préfixe user_ (user_gender, user_birth_date)
    birth_date: client.user_birth_date || client.user?.birth_date || '',
    // ✅ Convertir gender de "HOMME"/"FEMME" vers "male"/"female" pour le select
    gender: (() => {
      const genderValue = client.user_gender || client.user?.gender || '';
      if (!genderValue) return '';
      const genderStr = String(genderValue).toLowerCase();
      if (genderStr === 'homme' || genderStr === 'male') return 'male';
      if (genderStr === 'femme' || genderStr === 'female') return 'female';
      return genderStr;
    })(),
    avs_number: client.avs_number || '',
    contact_email: client.contact_email || '',
    contact_phone: client.contact_phone || '',
    billing_address: client.billing_address || '',
    // ✅ CORRECTION: Lire directement depuis client.domicile_address, pas client.domicile.address
    domicile_address: client.domicile_address || client.domicile?.address || '',
    domicile_zip: client.domicile_zip || client.domicile?.zip || '',
    domicile_city: client.domicile_city || client.domicile?.city || '',
    preferential_rate: client.preferential_rate || '',
    is_active: client.is_active !== false, // Par défaut actif
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Coordonnées GPS pour adresse de domicile
  // ✅ CORRECTION: Lire directement depuis client.domicile_lat, pas client.domicile.lat
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
    }));

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

    setBillingCoords({
      lat: item.lat ?? null,
      lon: item.lon ?? null,
    });

    console.log(`📍 [Facturation] GPS: ${item.lat}, ${item.lon}`);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    // Validation
    if (formData.is_institution && !formData.institution_name.trim()) {
      setError("Le nom de l'institution est requis pour les institutions");
      return;
    }

    if (!formData.is_institution && !formData.gender) {
      setError("Merci de sélectionner une civilité");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Construire le payload de base sans les champs utilisateur
      const payload = {
        is_institution: formData.is_institution,
        institution_name: formData.institution_name,
        residence_facility: formData.residence_facility,
        avs_number: formData.avs_number,
        contact_email: formData.contact_email,
        contact_phone: formData.contact_phone,
        billing_address: formData.billing_address,
        domicile_address: formData.domicile_address,
        domicile_zip: formData.domicile_zip,
        domicile_city: formData.domicile_city,
        preferential_rate: formData.preferential_rate,
        is_active: formData.is_active,
        domicile_lat: domicileCoords.lat,
        domicile_lon: domicileCoords.lon,
        billing_lat: billingCoords.lat,
        billing_lon: billingCoords.lon,
      };

      // Ajouter les champs utilisateur uniquement si ce n'est pas une institution
      if (!formData.is_institution) {
        // Ne pas envoyer gender si vide ou undefined
        if (formData.gender && formData.gender.trim()) {
          payload.gender = formData.gender;
        }
        if (formData.birth_date && formData.birth_date.trim()) {
          payload.birth_date = formData.birth_date;
        }
      }

      console.log('📤 Payload envoyé:', payload);

      const result = await onSave(payload);
      console.log('✅ Sauvegarde réussie:', result);
      
      // ✅ Ne pas fermer ici - laisser handleSaveClient gérer la fermeture après rechargement
      setLoading(false);
      // onClose() sera appelé par handleSaveClient après le rechargement des données
    } catch (err) {
      console.error('❌ Erreur lors de la sauvegarde:', err);
      setError(err.response?.data?.error || err.message || 'Erreur lors de la sauvegarde');
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
            ✕
          </button>
        </div>

        <form onSubmit={handleSubmit} className={styles.form}>
          {error && <div className={styles.error}>{error}</div>}

          {/* Informations client */}
          <div className={styles.section}>
            <h3 className={styles.sectionTitle}>
              {formData.is_institution ? 'Informations institution' : 'Informations client'}
            </h3>

            <div className={styles.infoCard}>
              {formData.is_institution ? (
                <>
                  <div className={styles.infoRow}>
                    <span className={styles.label}>Institution :</span>
                    <span className={styles.value}>
                      🏥 {formData.institution_name || client.institution_name || 'Non défini'}
                    </span>
                  </div>
                  <div className={styles.infoRow}>
                    <span className={styles.label}>Contact interne :</span>
                    <span className={styles.value}>
                      {client.user_first_name || client.first_name || ''} {client.user_last_name || client.last_name || ''}
                    </span>
                  </div>
                </>
              ) : (
                <>
                  <div className={styles.infoRow}>
                    <span className={styles.label}>Nom :</span>
                    <span className={styles.value}>
                      {client.user_first_name || client.first_name || ''} {client.user_last_name || client.last_name || ''}
                    </span>
                  </div>
                  <div className="form-group mt-sm">
                    <label htmlFor="birth_date" className="form-label">
                      Date de naissance
                    </label>
                    <input
                      type="date"
                      id="birth_date"
                      name="birth_date"
                      value={formData.birth_date}
                      onChange={handleChange}
                      className="form-input"
                      disabled={loading}
                    />
                  </div>

                  {/* ✅ Civilité */}
                  <div className="form-group mt-sm">
                    <label htmlFor="gender" className="form-label">
                      Civilité
                    </label>
                    <select
                      id="gender"
                      name="gender"
                      value={formData.gender}
                      onChange={handleChange}
                      className="form-input"
                      disabled={loading}
                      required={!formData.is_institution}
                    >
                      <option value="">Merci de sélectionner</option>
                      <option value="male">Monsieur</option>
                      <option value="female">Madame</option>
                    </select>
                  </div>

                  {/* ✅ Numéro AVS */}
                  <div className="form-group mt-sm">
                    <label htmlFor="avs_number" className="form-label">
                      Numéro AVS
                    </label>
                    <input
                      type="text"
                      id="avs_number"
                      name="avs_number"
                      value={formData.avs_number}
                      onChange={handleChange}
                      className="form-input"
                      placeholder="756.XXXX.XXXX.XX"
                      disabled={loading}
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

          {/* Coordonnées */}
          <div className={styles.section}>
            <h3 className={styles.sectionTitle}>Coordonnées de facturation</h3>

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
                placeholder="contact@exemple.ch"
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
              <small className={styles.hint}>💡 Si différente de l'adresse de domicile</small>
            </div>
          </div>

          {/* Adresse de domicile */}
          <div className={styles.section}>
            <h3 className={styles.sectionTitle}>🏠 Adresse de domicile</h3>
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
                💡 Indiquer si le client habite dans un EMS, Foyer, ou autre établissement
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
              <small className={styles.hint}>💡 Tapez pour rechercher une nouvelle adresse</small>
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
                  className={styles.input}
                  readOnly
                  disabled={loading}
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
                  className={styles.input}
                  placeholder="Rempli automatiquement"
                  readOnly
                  disabled={loading}
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
                  className={styles.input}
                  placeholder="Rempli automatiquement"
                  readOnly
                  disabled={loading}
                />
              </div>
            </div>
          </div>

          {/* Tarif préférentiel */}
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

        {/* 🏥 Séjours d'hospitalisation - Uniquement pour les clients (pas les institutions) */}
        {/* Placé en dehors du formulaire principal pour éviter les formulaires imbriqués */}
        {!formData.is_institution && (
          <ClientStaysSection clientId={client.id} />
        )}

        {/* 💰 Tiers payeur / Curateur - Uniquement pour les clients (pas les institutions) */}
        {/* Placé en dehors du formulaire principal pour éviter les formulaires imbriqués */}
        {!formData.is_institution && (
          <ClientBillingPartiesSection clientId={client.id} />
        )}
      </div>
    </div>
  );
};

export default EditClientModal;
