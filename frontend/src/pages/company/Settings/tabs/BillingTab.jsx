// frontend/src/pages/company/Settings/tabs/BillingTab.jsx
import React, { useState, useEffect } from 'react';
import styles from '../CompanySettings.module.css';
import ToggleField from '../../../../components/ui/ToggleField';
import { fetchBillingSettings, updateBillingSettings } from '../../../../services/settingsService';
import EmailConfigSection from './EmailConfigSection';

const BillingTab = ({ companyId }) => {
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
    email_sender: '',
    invoice_number_format: '{PREFIX}-{YYYY}-{MM}-{SEQ4}',
    invoice_prefix: 'EM',
    invoice_message_template: '',
    reminder1_template: '',
    reminder2_template: '',
    reminder3_template: '',
    legal_footer: '',
    pdf_template_variant: 'standard',
    iban: '',
    qr_iban: '',
    esr_ref_base: '',
    // TVA
    vat_applicable: false,
    vat_rate: null,
    vat_label: '',
    vat_number: '',
    // Configuration SMTP
    smtp_enabled: false,
    smtp_server: '',
    smtp_port: 587,
    smtp_use_tls: true,
    smtp_use_ssl: false,
    smtp_username: '',
    smtp_password: '',
    smtp_password_configured: false,
  });

  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      setLoading(true);
      const data = await fetchBillingSettings();
      
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'BillingTab.jsx:loadSettings:after_fetch',message:'Data loaded from API',data:{has_data:!!data,data_keys:data ? Object.keys(data) : [],has_iban:data?.iban !== undefined,iban_value:data?.iban,iban_length:data?.iban?.length,has_qr_iban:data?.qr_iban !== undefined,qr_iban_value:data?.qr_iban},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'D'})}).catch(()=>{});
      // #endregion
      
      if (data) {
        setForm({
          payment_terms_days: data.payment_terms_days ?? 10,
          overdue_fee: data.overdue_fee ?? 15,
          reminder_schedule_days: data.reminder_schedule_days || {
            1: 10,
            2: 5,
            3: 3,
          },
          reminder1_fee: data.reminder1_fee ?? 5,
          reminder2_fee: data.reminder2_fee ?? 10,
          reminder3_fee: data.reminder3_fee ?? 20,
          auto_reminders_enabled: data.auto_reminders_enabled || false,
          email_templates_enabled: data.email_templates_enabled || false,
          email_sender: data.email_sender || '',
          invoice_number_format: data.invoice_number_format || '{PREFIX}-{YYYY}-{MM}-{SEQ4}',
          invoice_prefix: data.invoice_prefix || 'EM',
          invoice_message_template: data.invoice_message_template || '',
          reminder1_template: data.reminder1_template || '',
          reminder2_template: data.reminder2_template || '',
          reminder3_template: data.reminder3_template || '',
          legal_footer: data.legal_footer || '',
          pdf_template_variant: data.pdf_template_variant || 'standard',
          iban: data.iban || '',
          qr_iban: data.qr_iban || '',
          esr_ref_base: data.esr_ref_base || '',
          // TVA
          vat_applicable: data.vat_applicable || false,
          vat_rate: data.vat_rate || null,
          vat_label: data.vat_label || '',
          vat_number: data.vat_number || '',
          // Configuration SMTP
          smtp_enabled: data.smtp_enabled || false,
          smtp_server: data.smtp_server || '',
          smtp_port: data.smtp_port ?? 587,
          smtp_use_tls: data.smtp_use_tls !== undefined ? data.smtp_use_tls : true,
          smtp_use_ssl: data.smtp_use_ssl || false,
          smtp_username: data.smtp_username || '',
          smtp_password: '', // Ne jamais charger le mot de passe
          smtp_password_configured: data.smtp_password_configured || false,
        });
      }
    } catch (err) {
      console.error('Erreur lors du chargement des paramètres:', err);
      setError('Erreur lors du chargement des paramètres');
    } finally {
      setLoading(false);
    }
  };

  // Sauvegarde automatique
  const autoSave = async (updatedForm = null) => {
    setMessage('');
    setError('');

    try {
      const formData = updatedForm || form;

      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'BillingTab.jsx:autoSave:start',message:'Starting autoSave',data:{has_iban:!!formData.iban,iban_value:formData.iban,iban_length:formData.iban?.length,has_qr_iban:!!formData.qr_iban,qr_iban_value:formData.qr_iban},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
      // #endregion

      // Fonction pour normaliser l'IBAN (enlever les espaces)
      const normalizeIban = (iban) => {
        if (!iban) return null;
        return iban.replace(/\s+/g, '').toUpperCase().trim() || null;
      };

      // Nettoyer les données avant envoi
      const cleanedData = {
        ...formData,
        // Normaliser les IBAN (enlever les espaces avant envoi)
        iban: normalizeIban(formData.iban),
        qr_iban: normalizeIban(formData.qr_iban),
        // S'assurer que reminder_schedule_days a les bonnes clés (strings)
        reminder_schedule_days: formData.reminder_schedule_days
          ? {
              1: parseInt(formData.reminder_schedule_days['1']) || 0,
              2: parseInt(formData.reminder_schedule_days['2']) || 0,
              3: parseInt(formData.reminder_schedule_days['3']) || 0,
            }
          : { 1: 10, 2: 5, 3: 3 },
        // Convertir les valeurs null en undefined pour les champs optionnels
        // Pour vat_rate, s'assurer que c'est un nombre valide ou null
        vat_rate: (() => {
          if (
            formData.vat_rate === null ||
            formData.vat_rate === '' ||
            formData.vat_rate === undefined
          ) {
            return null;
          }
          const parsed = parseFloat(formData.vat_rate);
          return isNaN(parsed) || parsed <= 0 ? null : parsed;
        })(),
        vat_label: formData.vat_label || null,
        vat_number: formData.vat_number || null,
        // Convertir les frais en nombres
        reminder1_fee: parseFloat(formData.reminder1_fee) || 0,
        reminder2_fee: parseFloat(formData.reminder2_fee) || 0,
        reminder3_fee: parseFloat(formData.reminder3_fee) || 0,
        overdue_fee: parseFloat(formData.overdue_fee) || 0,
        payment_terms_days: parseInt(formData.payment_terms_days, 10) || (formData.payment_terms_days === '0' ? 0 : 10),
      };

      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'BillingTab.jsx:autoSave:before_send',message:'Data before sending',data:{has_iban:!!cleanedData.iban,iban_value:cleanedData.iban,iban_length:cleanedData.iban?.length,has_qr_iban:!!cleanedData.qr_iban,qr_iban_value:cleanedData.qr_iban},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'B'})}).catch(()=>{});
      // #endregion

      console.log('[BillingTab] Sending data:', cleanedData);
      const response = await updateBillingSettings(cleanedData);
      
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'BillingTab.jsx:autoSave:after_send',message:'Data after sending',data:{has_response:!!response,response_keys:response ? Object.keys(response) : [],has_data:!!response?.data,data_keys:response?.data ? Object.keys(response.data) : [],has_iban_in_response:response?.data?.iban !== undefined,iban_in_response:response?.data?.iban,iban_length_in_response:response?.data?.iban?.length},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'C'})}).catch(()=>{});
      // #endregion
      
      // Mettre à jour le formulaire avec la réponse du serveur
      if (response?.data) {
        const formatIbanPretty = (iban) => {
          if (!iban) return '';
          const v = iban.replace(/\s+/g, '').toUpperCase();
          return v.replace(/(.{4})/g, '$1 ').trim();
        };
        
        setForm((prev) => ({
          ...prev,
          iban: response.data.iban ? formatIbanPretty(response.data.iban) : prev.iban,
          qr_iban: response.data.qr_iban ? formatIbanPretty(response.data.qr_iban) : prev.qr_iban,
        }));
      }
      
      setMessage('✅ Sauvegardé automatiquement');
      setTimeout(() => setMessage(''), 2000);
    } catch (err) {
      console.error('Auto-save failed:', err);
      const errorMessage =
        err?.response?.data?.error || err?.message || 'Erreur lors de la sauvegarde';
      setError(`❌ ${errorMessage}`);
      setTimeout(() => setError(''), 5000);
    }
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleBlur = () => {
    autoSave();
  };

  const handleToggle = (e) => {
    const { name, checked } = e.target;
    const updatedForm = {
      ...form,
      [name]: checked,
    };
    setForm(updatedForm);
    // Sauvegarder immédiatement après changement de toggle
    autoSave(updatedForm);
  };

  const handleReminderScheduleChange = (level, value) => {
    const updatedForm = {
      ...form,
      reminder_schedule_days: {
        ...form.reminder_schedule_days,
        [level]: parseInt(value) || 0,
      },
    };
    setForm(updatedForm);
  };

  const handleReminderScheduleBlur = () => {
    autoSave();
  };

  const generatePreview = () => {
    const format = form.invoice_number_format;
    const prefix = form.invoice_prefix || 'EM';
    const today = new Date();
    const year = today.getFullYear();
    const month = String(today.getMonth() + 1).padStart(2, '0');
    const seq = String(1).padStart(4, '0');

    return format
      .replace('{PREFIX}', prefix)
      .replace('{YYYY}', year)
      .replace('{MM}', month)
      .replace('{SEQ4}', seq)
      .replace('{SEQ5}', String(1).padStart(5, '0'))
      .replace('{YYYYMM}', `${year}${month}`)
      .replace('{SEQ3}', String(1).padStart(3, '0'));
  };

  const ibanChecksumIsValid = (iban) => {
    if (!iban || iban.length < 15) return false;

    // Validation basique pour la Suisse
    const swissPattern = /^CH[0-9]{2}[0-9]{5}[0-9A-Z]{12}$/;
    return swissPattern.test(iban.replace(/\s/g, ''));
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
    <div className={styles.settingsForm} style={{ display: 'block' }}>
      {message && <div className={styles.success}>{message}</div>}
      {error && <div className={styles.error}>{error}</div>}

      {/* Layout en 2 colonnes */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: 'var(--spacing-lg)',
          alignItems: 'start',
          width: '100%',
        }}
      >
        {/* COLONNE GAUCHE */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--spacing-md)' }}>
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
                  onBlur={handleBlur}
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
                  onBlur={handleBlur}
                  step="0.01"
                  min="0"
                />
                <span className={styles.unit}>CHF</span>
              </div>
              <small className={styles.hint}>
                Montant facturé automatiquement lorsque le paiement est en retard après l'échéance
              </small>
            </div>
          </section>

          {/* Section Rappels */}
          <section className={styles.section}>
            <h2>📧 Rappels de paiement</h2>
            <small className={styles.hint} style={{ display: 'block', marginBottom: '16px' }}>
              Configurez les frais et délais pour chaque niveau de rappel. Les frais sont toujours
              facturés lors de l'émission du rappel, même si l'envoi automatique est désactivé.
            </small>

            {/* 1er rappel - Délai et Frais ensemble */}
            <div className={styles.reminderRow}>
              <h4 className={styles.reminderTitle}>1er rappel</h4>
              <div className={styles.reminderFields}>
                <div className={styles.formGroup}>
                  <label>Frais (CHF)</label>
                  <div className={styles.inputWithUnit}>
                    <input
                      type="number"
                      name="reminder1_fee"
                      value={form.reminder1_fee}
                      onChange={handleChange}
                      onBlur={handleBlur}
                      step="0.01"
                      min="0"
                    />
                    <span className={styles.unit}>CHF</span>
                  </div>
                  <small className={styles.hint}>Montant facturé lors de l'émission</small>
                </div>
                {form.auto_reminders_enabled && (
                  <div className={styles.formGroup}>
                    <label>Délai d'envoi (jours)</label>
                    <input
                      type="number"
                      value={form.reminder_schedule_days['1'] || 10}
                      onChange={(e) => handleReminderScheduleChange('1', e.target.value)}
                      onBlur={handleReminderScheduleBlur}
                      min="1"
                      max="90"
                    />
                    <small className={styles.hint}>Jours après l'échéance</small>
                  </div>
                )}
              </div>
            </div>

            {/* 2e rappel - Délai et Frais ensemble */}
            <div className={styles.reminderRow}>
              <h4 className={styles.reminderTitle}>2e rappel</h4>
              <div className={styles.reminderFields}>
                <div className={styles.formGroup}>
                  <label>Frais (CHF)</label>
                  <div className={styles.inputWithUnit}>
                    <input
                      type="number"
                      name="reminder2_fee"
                      value={form.reminder2_fee}
                      onChange={handleChange}
                      onBlur={handleBlur}
                      step="0.01"
                      min="0"
                    />
                    <span className={styles.unit}>CHF</span>
                  </div>
                  <small className={styles.hint}>Montant facturé lors de l'émission</small>
                </div>
                {form.auto_reminders_enabled && (
                  <div className={styles.formGroup}>
                    <label>Délai d'envoi (jours)</label>
                    <input
                      type="number"
                      value={form.reminder_schedule_days['2'] || 5}
                      onChange={(e) => handleReminderScheduleChange('2', e.target.value)}
                      onBlur={handleReminderScheduleBlur}
                      min="1"
                      max="90"
                    />
                    <small className={styles.hint}>Jours après le 1er rappel</small>
                  </div>
                )}
              </div>
            </div>

            {/* 3e rappel - Délai et Frais ensemble */}
            <div className={styles.reminderRow}>
              <h4 className={styles.reminderTitle}>3e rappel (Mise en demeure)</h4>
              <div className={styles.reminderFields}>
                <div className={styles.formGroup}>
                  <label>Frais (CHF)</label>
                  <div className={styles.inputWithUnit}>
                    <input
                      type="number"
                      name="reminder3_fee"
                      value={form.reminder3_fee}
                      onChange={handleChange}
                      onBlur={handleBlur}
                      step="0.01"
                      min="0"
                    />
                    <span className={styles.unit}>CHF</span>
                  </div>
                  <small className={styles.hint}>Montant facturé lors de l'émission</small>
                </div>
                {form.auto_reminders_enabled && (
                  <div className={styles.formGroup}>
                    <label>Délai d'envoi (jours)</label>
                    <input
                      type="number"
                      value={form.reminder_schedule_days['3'] || 3}
                      onChange={(e) => handleReminderScheduleChange('3', e.target.value)}
                      onBlur={handleReminderScheduleBlur}
                      min="1"
                      max="90"
                    />
                    <small className={styles.hint}>Jours après le 2e rappel</small>
                  </div>
                )}
              </div>
            </div>

            {/* Activation des rappels automatiques */}
            <div
              style={{
                marginTop: '24px',
                paddingTop: '24px',
                borderTop: '1px solid var(--border-primary)',
              }}
            >
              <ToggleField
                label="Activer l'envoi automatique des rappels"
                name="auto_reminders_enabled"
                value={form.auto_reminders_enabled}
                onChange={handleToggle}
                hint="Si activé, les rappels seront envoyés automatiquement selon les délais configurés ci-dessus. Les frais seront toujours facturés même si l'envoi est manuel."
              />
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
                    name: 'email_templates_enabled',
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
                    onBlur={handleBlur}
                    placeholder="facturation@emmenezmoi.ch"
                  />
                </div>

                <div className={styles.formGroup}>
                  <label htmlFor="invoice_message_template">Message envoi de facture</label>
                  <textarea
                    id="invoice_message_template"
                    name="invoice_message_template"
                    value={form.invoice_message_template}
                    onChange={handleChange}
                    onBlur={handleBlur}
                    rows={5}
                    placeholder="Bonjour {client_name},&#10;&#10;Veuillez trouver ci-joint la facture {invoice_number} d'un montant de {amount} CHF.&#10;&#10;Merci de procéder au paiement avant le {due_date}."
                  />
                  <small className={styles.hint}>
                    Variables: {'{client_name}'}, {'{amount}'}, {'{due_date}'}, {'{invoice_number}'}
                  </small>
                </div>

                <div className={styles.formGroup}>
                  <label htmlFor="reminder1_template">Message 1er rappel</label>
                  <textarea
                    id="reminder1_template"
                    name="reminder1_template"
                    value={form.reminder1_template}
                    onChange={handleChange}
                    onBlur={handleBlur}
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
                    onBlur={handleBlur}
                    rows={4}
                    placeholder="2e rappel: merci de régler la facture {invoice_number} sous 5 jours."
                  />
                </div>

                <div className={styles.formGroup}>
                  <label htmlFor="reminder3_template">Message 3e rappel (Mise en demeure)</label>
                  <textarea
                    id="reminder3_template"
                    name="reminder3_template"
                    value={form.reminder3_template}
                    onChange={handleChange}
                    onBlur={handleBlur}
                    rows={4}
                    placeholder="Mise en demeure: dernier rappel avant procédures légales."
                  />
                </div>
              </>
            )}
          </section>
        </div>

        {/* COLONNE DROITE */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--spacing-md)' }}>
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
                onBlur={handleBlur}
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
                onChange={(e) => {
                  const newForm = { ...form, invoice_number_format: e.target.value };
                  setForm(newForm);
                  autoSave(newForm);
                }}
              >
                <option value="{PREFIX}-{YYYY}-{MM}-{SEQ4}">
                  {form.invoice_prefix}-2025-10-0001
                </option>
                <option value="{PREFIX}-{YYYY}-{SEQ5}">{form.invoice_prefix}-2025-00001</option>
                <option value="{PREFIX}{YYYYMM}{SEQ3}">{form.invoice_prefix}202510001</option>
              </select>
            </div>

            <div className={styles.previewBadge}>
              <strong>Prévisualisation :</strong> {generatePreview()}
            </div>

            {/* Pied de page légal */}
            <h2 style={{ marginTop: '24px' }}>📄 Pied de page légal</h2>

            <div className={styles.formGroup}>
              <label htmlFor="legal_footer">Texte du pied de page</label>
              <textarea
                id="legal_footer"
                name="legal_footer"
                value={form.legal_footer}
                onChange={handleChange}
                onBlur={handleBlur}
                rows={3}
                placeholder="Emmenez Moi Sàrl - CHE-123.456.789 - Genève, Suisse"
              />
              <small className={styles.hint}>Affiché sur toutes les factures PDF</small>
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="pdf_template_variant">Variante de template PDF</label>
              <select
                id="pdf_template_variant"
                name="pdf_template_variant"
                value={form.pdf_template_variant}
                onChange={(e) => {
                  const newForm = { ...form, pdf_template_variant: e.target.value };
                  setForm(newForm);
                  autoSave(newForm);
                }}
              >
                <option value="standard">Standard</option>
                <option value="minimal">Minimal</option>
                <option value="detailed">Détaillé</option>
              </select>
            </div>
          </section>

          {/* TVA */}
          <section className={styles.section}>
            <h2>💰 TVA (Taxe sur la valeur ajoutée)</h2>

            <ToggleField
              label="TVA applicable"
              name="vat_applicable"
              value={form.vat_applicable}
              onChange={handleToggle}
              hint="Activez la TVA si votre entreprise est assujettie à la TVA"
            />

            {form.vat_applicable && (
              <>
                <div className={styles.formGroup}>
                  <label htmlFor="vat_rate">Taux de TVA (%)</label>
                  <div className={styles.inputWithUnit}>
                    <input
                      type="number"
                      id="vat_rate"
                      name="vat_rate"
                      value={form.vat_rate || ''}
                      onChange={handleChange}
                      onBlur={handleBlur}
                      step="0.01"
                      min="0"
                      max="100"
                      placeholder="7.7"
                    />
                    <span className={styles.unit}>%</span>
                  </div>
                  <small className={styles.hint}>
                    Taux de TVA standard en Suisse: 7.7% (réduit: 2.5%, réduit spécial: 3.7%)
                  </small>
                </div>

                <div className={styles.formGroup}>
                  <label htmlFor="vat_label">Libellé TVA</label>
                  <input
                    type="text"
                    id="vat_label"
                    name="vat_label"
                    value={form.vat_label}
                    onChange={handleChange}
                    onBlur={handleBlur}
                    placeholder="TVA"
                    maxLength={50}
                  />
                  <small className={styles.hint}>
                    Libellé affiché sur les factures (ex: "TVA", "TVA 7.7%", "TVA incluse")
                  </small>
                </div>

                <div className={styles.formGroup}>
                  <label htmlFor="vat_number">Numéro de TVA</label>
                  <input
                    type="text"
                    id="vat_number"
                    name="vat_number"
                    value={form.vat_number}
                    onChange={handleChange}
                    onBlur={handleBlur}
                    placeholder="CHE-123.456.789 TVA"
                    maxLength={50}
                  />
                  <small className={styles.hint}>
                    Numéro d'identification TVA de l'entreprise (optionnel)
                  </small>
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
                onBlur={handleBlur}
                placeholder="CH93 0076 2011 6238 5295 7"
                maxLength={34}
              />
              <small className={styles.hint}>
                {form.iban && (
                  <span className={ibanChecksumIsValid(form.iban) ? styles.valid : styles.invalid}>
                    {ibanChecksumIsValid(form.iban) ? '✅ IBAN valide' : '❌ IBAN invalide'}
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
                onBlur={handleBlur}
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
                onBlur={handleBlur}
                placeholder="00000000000000000000"
                maxLength={27}
              />
              <small className={styles.hint}>
                Référence de base pour les paiements ESR (20 chiffres + 7 chiffres)
              </small>
            </div>
          </section>

          {/* Configuration Email Transactionnel (Brevo) */}
          {companyId && <EmailConfigSection companyId={companyId} />}
        </div>
      </div>
    </div>
  );
};

export default BillingTab;
