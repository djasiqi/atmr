import React, { useState } from "react";
import styles from "./EditClientModal.module.css";
import AddressAutocomplete from "../../../../components/common/AddressAutocomplete";

const EditClientModal = ({ client, onClose, onSave }) => {
  const [formData, setFormData] = useState({
    is_institution: client.is_institution || false,
    institution_name: client.institution_name || "",
    contact_email: client.contact_email || "",
    contact_phone: client.contact_phone || "",
    billing_address: client.billing_address || "",
    domicile_address: client.domicile?.address || "",
    domicile_zip: client.domicile?.zip || "",
    domicile_city: client.domicile?.city || "",
    preferential_rate: client.preferential_rate || "",
    is_active: client.is_active !== false, // Par défaut actif
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Coordonnées GPS pour adresse de domicile
  const [domicileCoords, setDomicileCoords] = useState({
    lat: client.domicile?.lat || null,
    lon: client.domicile?.lon || null,
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
      [name]: type === "checkbox" ? checked : value,
    }));
  };

  // Gérer la sélection d'adresse de domicile via autocomplete
  const handleDomicileAddressSelect = (item) => {
    console.log("📍 [Domicile] Adresse sélectionnée:", item);
    
    const address = item.address || item.label || "";
    const postcode = item.postcode || "";
    const city = item.city || "";
    
    setFormData((prev) => ({
      ...prev,
      domicile_address: address,
      domicile_zip: postcode,
      domicile_city: city,
    }));
    
    setDomicileCoords({
      lat: item.lat ?? null,
      lon: item.lon ?? null,
    });
    
    console.log(`📍 [Domicile] GPS: ${item.lat}, ${item.lon}`);
  };

  // Gérer la sélection d'adresse de facturation via autocomplete
  const handleBillingAddressSelect = (item) => {
    console.log("📍 [Facturation] Adresse sélectionnée:", item);
    
    const fullAddress = item.label || "";
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

    setLoading(true);
    setError(null);

    try {
      // Ajouter les coordonnées GPS au payload
      const payload = {
        ...formData,
        domicile_lat: domicileCoords.lat,
        domicile_lon: domicileCoords.lon,
        billing_lat: billingCoords.lat,
        billing_lon: billingCoords.lon,
      };
      
      await onSave(payload);
    } catch (err) {
      setError(
        err.response?.data?.error ||
          err.message ||
          "Erreur lors de la sauvegarde"
      );
      setLoading(false);
    }
  };

  return (
    <div className={styles.modalOverlay} onClick={onClose}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div className={styles.modalHeader}>
          <h2>
            {formData.is_institution
              ? `Éditer l'institution${
                  formData.institution_name
                    ? ` : ${formData.institution_name}`
                    : ""
                }`
              : "Éditer le client"}
          </h2>
          <button className={styles.closeBtn} onClick={onClose}>
            ✕
          </button>
        </div>

        <form onSubmit={handleSubmit} className={styles.form}>
          {error && <div className={styles.error}>{error}</div>}

          {/* Informations client */}
          <div className={styles.section}>
            <h3 className={styles.sectionTitle}>
              {formData.is_institution
                ? "Informations institution"
                : "Informations client"}
            </h3>

            <div className={styles.infoCard}>
              {formData.is_institution ? (
                <>
                  <div className={styles.infoRow}>
                    <span className={styles.label}>Institution :</span>
                    <span className={styles.value}>
                      🏥{" "}
                      {formData.institution_name ||
                        client.institution_name ||
                        "Non défini"}
                    </span>
                  </div>
                  <div className={styles.infoRow}>
                    <span className={styles.label}>Contact interne :</span>
                    <span className={styles.value}>
                      {client.first_name} {client.last_name}
                    </span>
                  </div>
                </>
              ) : (
                <div className={styles.infoRow}>
                  <span className={styles.label}>Nom :</span>
                  <span className={styles.value}>
                    {client.first_name} {client.last_name}
                  </span>
                </div>
              )}
              {client.user?.email && (
                <div className={styles.infoRow}>
                  <span className={styles.label}>Email utilisateur :</span>
                  <span className={styles.value}>{client.user.email}</span>
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
                    billing_address: e.target.value
                  }));
                }}
                onSelect={handleBillingAddressSelect}
                placeholder="Ex: Avenue de la Gare 5, 1003, Lausanne"
                disabled={loading}
              />
              <small style={{ fontSize: "12px", color: "#666", marginTop: "4px", display: "block" }}>
                💡 Si différente de l'adresse de domicile
              </small>
            </div>
          </div>

          {/* Adresse de domicile */}
          <div className={styles.section}>
            <h3 className={styles.sectionTitle}>🏠 Adresse de domicile</h3>
            <p
              style={{ fontSize: "13px", color: "#666", marginBottom: "12px" }}
            >
              Adresse où le client habite (utilisée pour la prise en charge par
              défaut)
            </p>

            <div className={styles.formGroup}>
              <label htmlFor="domicile_address" className={styles.label}>
                Adresse complète
              </label>
              <AddressAutocomplete
                name="domicile_address"
                value={`${formData.domicile_address}${formData.domicile_zip ? ', ' + formData.domicile_zip : ''}${formData.domicile_city ? ', ' + formData.domicile_city : ''}`}
                onChange={(e) => {
                  setDomicileCoords({ lat: null, lon: null });
                }}
                onSelect={handleDomicileAddressSelect}
                placeholder="Ex: Avenue Ernest-Pictet 9, 1203, Genève"
                disabled={loading}
              />
              <small style={{ fontSize: "12px", color: "#666", marginTop: "4px", display: "block" }}>
                💡 Tapez pour rechercher une nouvelle adresse
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
                  className={styles.input}
                  readOnly
                  style={{ backgroundColor: "#f8f9fa" }}
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
                  style={{ backgroundColor: "#f8f9fa" }}
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
                  style={{ backgroundColor: "#f8f9fa" }}
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
                <small
                  style={{
                    display: "block",
                    fontWeight: "normal",
                    color: "#666",
                    marginTop: "4px",
                  }}
                >
                  Prix d'un trajet simple. Pour un aller-retour, ce tarif sera
                  appliqué 2 fois. Laisser vide pour utiliser le tarif standard.
                </small>
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
                  <small>
                    Les clients inactifs n'apparaissent pas dans les sélections
                  </small>
                </span>
              </label>
            </div>
          </div>

          {/* Actions */}
          <div className={styles.modalActions}>
            <button
              type="button"
              onClick={onClose}
              className={styles.cancelBtn}
              disabled={loading}
            >
              Annuler
            </button>
            <button type="submit" className={styles.saveBtn} disabled={loading}>
              {loading ? "Sauvegarde..." : "Enregistrer"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default EditClientModal;
