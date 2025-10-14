// frontend/src/pages/company/Settings/tabs/BillingTab.jsx
import React, { useState, useEffect } from "react";
import styles from "../CompanySettings.module.css";
import ToggleField from "../../../../components/ui/ToggleField";
import {
  fetchBillingSettings,
  updateBillingSettings,
} from "../../../../services/settingsService";

const BillingTab = () => {
  const [form, setForm] = useState({
    payment_terms_days: 10,
    overdue_fee: 15,
    reminder_schedule_days: {
      1: 10,
      2: 5,
      3: 3,
    },
    reminder1_fee: 5,
    reminder2_fee: 10,
    reminder3_fee: 20,
    auto_reminders_enabled: false,
    email_templates_enabled: false,
    email_sender: "",
    invoice_number_format: "{PREFIX}-{YYYY}-{MM}-{SEQ4}",
    invoice_prefix: "EM",
    invoice_message_template: "",
    reminder1_template: "",
    reminder2_template: "",
    reminder3_template: "",
    legal_footer: "",
    pdf_template_variant: "standard",
    iban: "",
    qr_iban: "",
    esr_ref_base: "",
  });

  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      setLoading(true);
      const data = await fetchBillingSettings();
      if (data) {
        setForm({
          payment_terms_days: data.payment_terms_days || 10,
          overdue_fee: data.overdue_fee || 15,
          reminder_schedule_days: data.reminder_schedule_days || {
            1: 10,
            2: 5,
            3: 3,
          },
          reminder1_fee: data.reminder1_fee || 5,
          reminder2_fee: data.reminder2_fee || 10,
          reminder3_fee: data.reminder3_fee || 20,
          auto_reminders_enabled: data.auto_reminders_enabled || false,
          email_templates_enabled: data.email_templates_enabled || false,
          email_sender: data.email_sender || "",
          invoice_number_format:
            data.invoice_number_format || "{PREFIX}-{YYYY}-{MM}-{SEQ4}",
          invoice_prefix: data.invoice_prefix || "EM",
          invoice_message_template: data.invoice_message_template || "",
          reminder1_template: data.reminder1_template || "",
          reminder2_template: data.reminder2_template || "",
          reminder3_template: data.reminder3_template || "",
          legal_footer: data.legal_footer || "",
          pdf_template_variant: data.pdf_template_variant || "standard",
          iban: data.iban || "",
          qr_iban: data.qr_iban || "",
          esr_ref_base: data.esr_ref_base || "",
        });
      }
    } catch (err) {
      console.error("Erreur lors du chargement des paramètres:", err);
      setError("Erreur lors du chargement des paramètres");
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleToggle = (e) => {
    const { name, checked } = e.target;
    setForm((prev) => ({
      ...prev,
      [name]: checked,
    }));
  };

  const handleReminderScheduleChange = (level, value) => {
    setForm((prev) => ({
      ...prev,
      reminder_schedule_days: {
        ...prev.reminder_schedule_days,
        [level]: parseInt(value) || 0,
      },
    }));
  };

  const generatePreview = () => {
    const format = form.invoice_number_format;
    const prefix = form.invoice_prefix || "EM";
    const today = new Date();
    const year = today.getFullYear();
    const month = String(today.getMonth() + 1).padStart(2, "0");
    const seq = String(1).padStart(4, "0");

    return format
      .replace("{PREFIX}", prefix)
      .replace("{YYYY}", year)
      .replace("{MM}", month)
      .replace("{SEQ4}", seq)
      .replace("{SEQ5}", String(1).padStart(5, "0"))
      .replace("{YYYYMM}", `${year}${month}`)
      .replace("{SEQ3}", String(1).padStart(3, "0"));
  };

  const ibanChecksumIsValid = (iban) => {
    if (!iban || iban.length < 15) return false;

    // Validation basique pour la Suisse
    const swissPattern = /^CH[0-9]{2}[0-9]{5}[0-9A-Z]{12}$/;
    return swissPattern.test(iban.replace(/\s/g, ""));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSaving(true);
    setMessage("");
    setError("");

    try {
      await updateBillingSettings(form);
      setMessage("Paramètres de facturation mis à jour avec succès !");
    } catch (err) {
      console.error("Erreur lors de la sauvegarde:", err);
      setError("Erreur lors de la sauvegarde des paramètres");
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className={styles.loadingContainer}>
        <div className={styles.spinner}></div>
        <p>Chargement des paramètres de facturation...</p>
      </div>
    );
  }

  return (
    <form className={styles.settingsForm} onSubmit={handleSubmit}>
      {message && <div className={styles.success}>{message}</div>}
      {error && <div className={styles.error}>{error}</div>}

      {/* Paramètres de paiement et rappels */}
      <section className={styles.section}>
        <h2>💳 Paramètres de paiement</h2>

        <div className={styles.formGroup}>
          <label htmlFor="payment_terms_days">Délai de paiement</label>
          <div className={styles.inputWithUnit}>
            <input
              type="number"
              id="payment_terms_days"
              name="payment_terms_days"
              value={form.payment_terms_days}
              onChange={handleChange}
              min="1"
              max="90"
            />
            <span className={styles.unit}>jours</span>
          </div>
          <small className={styles.hint}>
            Délai accordé aux clients pour payer (défaut: 10 jours)
          </small>
        </div>

        <div className={styles.formGroup}>
          <label htmlFor="overdue_fee">Frais de retard</label>
          <div className={styles.inputWithUnit}>
            <input
              type="number"
              id="overdue_fee"
              name="overdue_fee"
              value={form.overdue_fee}
              onChange={handleChange}
              step="0.01"
              min="0"
            />
            <span className={styles.unit}>CHF</span>
          </div>
        </div>

        {/* Rappels automatiques */}
        <h2 style={{ marginTop: "24px" }}>📧 Rappels automatiques</h2>

        <ToggleField
          label="Activer les rappels automatiques"
          name="auto_reminders_enabled"
          value={form.auto_reminders_enabled}
          onChange={handleToggle}
          hint="Les rappels seront envoyés automatiquement selon le planning défini"
        />

        {form.auto_reminders_enabled && (
          <>
            {/* 1er rappel */}
            <div className={styles.reminderRow}>
              <h4 className={styles.reminderTitle}>1er rappel</h4>
              <div className={styles.reminderFields}>
                <div className={styles.formGroup}>
                  <label>Délai (jours)</label>
                  <input
                    type="number"
                    value={form.reminder_schedule_days["1"] || 10}
                    onChange={(e) =>
                      handleReminderScheduleChange("1", e.target.value)
                    }
                    min="1"
                    max="90"
                  />
                  <small className={styles.hint}>Après échéance</small>
                </div>
                <div className={styles.formGroup}>
                  <label>Frais (CHF)</label>
                  <input
                    type="number"
                    name="reminder1_fee"
                    value={form.reminder1_fee}
                    onChange={handleChange}
                    step="0.01"
                    min="0"
                  />
                </div>
              </div>
            </div>

            {/* 2e rappel */}
            <div className={styles.reminderRow}>
              <h4 className={styles.reminderTitle}>2e rappel</h4>
              <div className={styles.reminderFields}>
                <div className={styles.formGroup}>
                  <label>Délai (jours)</label>
                  <input
                    type="number"
                    value={form.reminder_schedule_days["2"] || 5}
                    onChange={(e) =>
                      handleReminderScheduleChange("2", e.target.value)
                    }
                    min="1"
                    max="90"
                  />
                  <small className={styles.hint}>Après 1er rappel</small>
                </div>
                <div className={styles.formGroup}>
                  <label>Frais (CHF)</label>
                  <input
                    type="number"
                    name="reminder2_fee"
                    value={form.reminder2_fee}
                    onChange={handleChange}
                    step="0.01"
                    min="0"
                  />
                </div>
              </div>
            </div>

            {/* 3e rappel */}
            <div className={styles.reminderRow}>
              <h4 className={styles.reminderTitle}>
                3e rappel (Mise en demeure)
              </h4>
              <div className={styles.reminderFields}>
                <div className={styles.formGroup}>
                  <label>Délai (jours)</label>
                  <input
                    type="number"
                    value={form.reminder_schedule_days["3"] || 3}
                    onChange={(e) =>
                      handleReminderScheduleChange("3", e.target.value)
                    }
                    min="1"
                    max="90"
                  />
                  <small className={styles.hint}>Après 2e rappel</small>
                </div>
                <div className={styles.formGroup}>
                  <label>Frais (CHF)</label>
                  <input
                    type="number"
                    name="reminder3_fee"
                    value={form.reminder3_fee}
                    onChange={handleChange}
                    step="0.01"
                    min="0"
                  />
                </div>
              </div>
            </div>
          </>
        )}
      </section>

      {/* Format de facturation et pied de page */}
      <section className={styles.section}>
        <h2>🧾 Format de facturation</h2>

        <div className={styles.formGroup}>
          <label htmlFor="invoice_prefix">Préfixe des factures</label>
          <input
            id="invoice_prefix"
            name="invoice_prefix"
            value={form.invoice_prefix}
            onChange={handleChange}
            maxLength={10}
            placeholder="EM"
          />
          <small className={styles.hint}>Ex: EM → {generatePreview()}</small>
        </div>

        <div className={styles.formGroup}>
          <label htmlFor="invoice_number_format">Format de numérotation</label>
          <select
            id="invoice_number_format"
            name="invoice_number_format"
            value={form.invoice_number_format}
            onChange={handleChange}
          >
            <option value="{PREFIX}-{YYYY}-{MM}-{SEQ4}">
              {form.invoice_prefix}-2025-10-0001
            </option>
            <option value="{PREFIX}-{YYYY}-{SEQ5}">
              {form.invoice_prefix}-2025-00001
            </option>
            <option value="{PREFIX}{YYYYMM}{SEQ3}">
              {form.invoice_prefix}202510001
            </option>
          </select>
        </div>

        <div className={styles.previewBadge}>
          <strong>Prévisualisation :</strong> {generatePreview()}
        </div>

        {/* Pied de page légal */}
        <h2 style={{ marginTop: "24px" }}>📄 Pied de page légal</h2>

        <div className={styles.formGroup}>
          <label htmlFor="legal_footer">Texte du pied de page</label>
          <textarea
            id="legal_footer"
            name="legal_footer"
            value={form.legal_footer}
            onChange={handleChange}
            rows={3}
            placeholder="Emmenez Moi Sàrl - CHE-123.456.789 - Genève, Suisse"
          />
          <small className={styles.hint}>
            Affiché sur toutes les factures PDF
          </small>
        </div>

        <div className={styles.formGroup}>
          <label htmlFor="pdf_template_variant">Variante de template PDF</label>
          <select
            id="pdf_template_variant"
            name="pdf_template_variant"
            value={form.pdf_template_variant}
            onChange={handleChange}
          >
            <option value="standard">Standard</option>
            <option value="minimal">Minimal</option>
            <option value="detailed">Détaillé</option>
          </select>
        </div>
      </section>

      {/* Templates d'emails */}
      <section className={styles.section}>
        <h2>✉️ Templates d'emails</h2>

        <ToggleField
          label="Activer les templates d'emails personnalisés"
          name="email_templates_enabled"
          value={form.email_templates_enabled || false}
          onChange={(e) =>
            handleToggle({
              target: {
                name: "email_templates_enabled",
                checked: e.target.checked,
              },
            })
          }
          hint="Personnaliser les messages d'email pour les factures et rappels"
        />

        {form.email_templates_enabled && (
          <>
            <div className={styles.formGroup}>
              <label htmlFor="email_sender">Email expéditeur</label>
              <input
                type="email"
                id="email_sender"
                name="email_sender"
                value={form.email_sender}
                onChange={handleChange}
                placeholder="facturation@emmenezmoi.ch"
              />
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="invoice_message_template">
                Message envoi de facture
              </label>
              <textarea
                id="invoice_message_template"
                name="invoice_message_template"
                value={form.invoice_message_template}
                onChange={handleChange}
                rows={5}
                placeholder="Bonjour {client_name},&#10;&#10;Veuillez trouver ci-joint la facture {invoice_number} d'un montant de {amount} CHF.&#10;&#10;Merci de procéder au paiement avant le {due_date}."
              />
              <small className={styles.hint}>
                Variables: {"{client_name}"}, {"{amount}"}, {"{due_date}"},{" "}
                {"{invoice_number}"}
              </small>
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="reminder1_template">Message 1er rappel</label>
              <textarea
                id="reminder1_template"
                name="reminder1_template"
                value={form.reminder1_template}
                onChange={handleChange}
                rows={4}
                placeholder="Rappel: votre facture {invoice_number} n'a pas encore été réglée."
              />
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="reminder2_template">Message 2e rappel</label>
              <textarea
                id="reminder2_template"
                name="reminder2_template"
                value={form.reminder2_template}
                onChange={handleChange}
                rows={4}
                placeholder="2e rappel: merci de régler la facture {invoice_number} sous 5 jours."
              />
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="reminder3_template">
                Message 3e rappel (Mise en demeure)
              </label>
              <textarea
                id="reminder3_template"
                name="reminder3_template"
                value={form.reminder3_template}
                onChange={handleChange}
                rows={4}
                placeholder="Mise en demeure: dernier rappel avant procédures légales."
              />
            </div>
          </>
        )}
      </section>

      {/* Informations bancaires */}
      <section className={styles.section}>
        <h2>🏦 Informations bancaires</h2>

        <div className={styles.formGroup}>
          <label htmlFor="iban">IBAN</label>
          <input
            id="iban"
            name="iban"
            value={form.iban}
            onChange={handleChange}
            placeholder="CH93 0076 2011 6238 5295 7"
            maxLength={34}
          />
          <small className={styles.hint}>
            {form.iban && (
              <span
                className={
                  ibanChecksumIsValid(form.iban) ? styles.valid : styles.invalid
                }
              >
                {ibanChecksumIsValid(form.iban)
                  ? "✅ IBAN valide"
                  : "❌ IBAN invalide"}
              </span>
            )}
          </small>
        </div>

        <div className={styles.formGroup}>
          <label htmlFor="qr_iban">IBAN pour QR-Code</label>
          <input
            id="qr_iban"
            name="qr_iban"
            value={form.qr_iban}
            onChange={handleChange}
            placeholder="CH93 0076 2011 6238 5295 7"
            maxLength={34}
          />
          <small className={styles.hint}>
            Utilisé pour la génération des QR-codes de paiement
          </small>
        </div>

        <div className={styles.formGroup}>
          <label htmlFor="esr_ref_base">Référence ESR de base</label>
          <input
            id="esr_ref_base"
            name="esr_ref_base"
            value={form.esr_ref_base}
            onChange={handleChange}
            placeholder="00000000000000000000"
            maxLength={27}
          />
          <small className={styles.hint}>
            Référence de base pour les paiements ESR (20 chiffres + 7 chiffres)
          </small>
        </div>
      </section>

      {/* Boutons */}
      <div className={styles.actionsRow}>
        <button
          type="submit"
          className={`${styles.button} ${styles.primary}`}
          disabled={saving}
        >
          {saving ? "💾 Enregistrement…" : "💾 Enregistrer"}
        </button>
      </div>
    </form>
  );
};

export default BillingTab;
