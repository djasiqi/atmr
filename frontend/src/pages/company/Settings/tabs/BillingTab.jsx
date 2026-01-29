// frontend/src/pages/company/Settings/tabs/BillingTab.jsx
import React, { useState, useEffect } from 'react';
import styles from '../CompanySettings.module.css';
import ToggleField from '../../../../components/ui/ToggleField';
import { fetchBillingSettings, updateBillingSettings } from '../../../../services/settingsService';
import EmailConfigSection from './EmailConfigSection';

// Helper pour générer le HTML de la signature (prévisualisation côté frontend)
const generateSignaturePreviewHtml = (formData) => {
  const escapeHtml = (text) => {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  };

  // Validation et normalisation (simplifiée côté frontend)
  const truncate = (value, maxLength) => {
    if (!value) return '';
    const trimmed = value.trim();
    return trimmed.length > maxLength ? trimmed.substring(0, maxLength) : trimmed;
  };

  const normalizeEmail = (email) => {
    if (!email) return '';
    const trimmed = email.trim();
    return trimmed.includes('@') ? truncate(trimmed, 200) : '';
  };

  const normalizeWebsite = (website) => {
    if (!website) return '';
    return truncate(website.trim(), 200);
  };

  // Normaliser les champs
  const name = truncate(formData.signature_name || '', 200);
  const title = truncate(formData.signature_title || '', 200);
  const company = truncate(formData.signature_company || '', 200);
  const phoneMain = truncate(formData.signature_phone_main || '', 50);
  const phoneMobile = truncate(formData.signature_phone_mobile || '', 50);
  const email = normalizeEmail(formData.signature_email || '');
  const website = normalizeWebsite(formData.signature_website || '');
  const addressLine = truncate(formData.signature_address_line || '', 200);
  const zip = truncate(formData.signature_zip || '', 10);
  const city = truncate(formData.signature_city || '', 100);
  // Logo: utilise automatiquement company.logo_url (pas de champ séparé)
  const logoUrl = null; // Le logo sera récupéré depuis company.logo_url côté backend

  // Construire la colonne gauche
  const leftColParts = [];
  if (name) leftColParts.push(`<strong style="font-size: 12px;">${escapeHtml(name)}</strong>`);
  if (title) leftColParts.push(escapeHtml(title));
  if (company) leftColParts.push(escapeHtml(company));
  const leftColContent = leftColParts.length > 0 ? leftColParts.join('<br>') : '&nbsp;';

  // Construire la colonne droite
  const rightColParts = [];
  const phones = [];
  if (phoneMain) phones.push(escapeHtml(phoneMain));
  if (phoneMobile) phones.push(escapeHtml(phoneMobile));
  if (phones.length > 0) rightColParts.push(phones.join(' | '));

  if (email) {
    rightColParts.push(`<a href="mailto:${escapeHtml(email)}" style="color: #1b4b7a; text-decoration: none;">${escapeHtml(email)}</a>`);
  }

  if (website) {
    let websiteClean = website;
    if (!websiteClean.startsWith('http://') && !websiteClean.startsWith('https://')) {
      websiteClean = `https://${websiteClean}`;
    }
    const websiteDisplay = website.replace(/^https?:\/\//, '');
    rightColParts.push(`<a href="${escapeHtml(websiteClean)}" style="color: #1b4b7a; text-decoration: none;">${escapeHtml(websiteDisplay)}</a>`);
  }

  const addressParts = [];
  if (addressLine) addressParts.push(escapeHtml(addressLine));
  if (zip && city) {
    addressParts.push(`${escapeHtml(zip)} ${escapeHtml(city)}`);
  } else if (city) {
    addressParts.push(escapeHtml(city));
  }
  if (addressParts.length > 0) rightColParts.push(addressParts.join('<br>'));
  const rightColContent = rightColParts.length > 0 ? rightColParts.join('<br>') : '&nbsp;';

  // Construire le HTML
  let html = `
    <table cellpadding="0" cellspacing="0" border="0" style="font-family: Arial, sans-serif; font-size: 11px; color: #333; margin-top: 12px; width: 100%;">
      <tr>
        <td style="vertical-align: top; padding-right: 12px; width: 50%;">
          ${leftColContent}
        </td>
        <td width="1" style="border-left: 2px solid #1b4b7a; padding-left: 12px; vertical-align: top; width: 50%;">
          ${rightColContent}
        </td>
      </tr>
    </table>
  `;

  // Ligne horizontale + logo
  if (logoUrl) {
    html += `
      <table width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-top: 12px;">
        <tr><td style="border-top: 1px solid #1b4b7a; line-height: 1px; font-size: 1px;">&nbsp;</td></tr>
      </table>
      <div style="padding-top: 8px;">
        <img src="${escapeHtml(logoUrl)}" height="26" alt="Logo" style="display: block; border: 0; outline: none; text-decoration: none;" />
      </div>
    `;
  } else {
    html += `
      <table width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-top: 12px;">
        <tr><td style="border-top: 1px solid #1b4b7a; line-height: 1px; font-size: 1px;">&nbsp;</td></tr>
      </table>
    `;
  }

  return html;
};

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
    email_signature_mode: 'form',
    email_signature_text: '',
    signature_name: '',
    signature_title: '',
    signature_company: '',
    signature_phone_main: '',
    signature_phone_mobile: '',
    signature_email: '',
    signature_website: '',
    signature_address_line: '',
    signature_zip: '',
    signature_city: '',
    email_signature_html_template: '',
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
  const [showSignaturePreview, setShowSignaturePreview] = useState(false);
  const [expandedSections, setExpandedSections] = useState({
    payment: true,
    reminders: false,
    templates: false,
    format: true,
    vat: false,
    banking: false,
    emailConfig: false,
  });

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      setLoading(true);
      const data = await fetchBillingSettings();
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
          email_sender: data.email_sender ?? '',
          invoice_number_format: data.invoice_number_format || '{PREFIX}-{YYYY}-{MM}-{SEQ4}',
          invoice_prefix: data.invoice_prefix || 'EM',
          invoice_message_template: data.invoice_message_template ?? '',
          reminder1_template: data.reminder1_template ?? '',
          reminder2_template: data.reminder2_template ?? '',
          reminder3_template: data.reminder3_template ?? '',
          email_signature_mode: data.email_signature_mode || 'form',
          email_signature_text: data.email_signature_text ?? '',
          signature_name: data.signature_name ?? '',
          signature_title: data.signature_title ?? '',
          signature_company: data.signature_company ?? '',
          signature_phone_main: data.signature_phone_main ?? '',
          signature_phone_mobile: data.signature_phone_mobile ?? '',
          signature_email: data.signature_email ?? '',
          signature_website: data.signature_website ?? '',
          signature_address_line: data.signature_address_line ?? '',
          signature_zip: data.signature_zip ?? '',
          signature_city: data.signature_city ?? '',
          // Note: signature_logo_url supprimé - on utilise maintenant company.logo_url automatiquement
          email_signature_html_template: data.email_signature_html_template ?? '',
          legal_footer: data.legal_footer ?? '',
          pdf_template_variant: data.pdf_template_variant || 'standard',
          iban: data.iban ?? '',
          qr_iban: data.qr_iban ?? '',
          esr_ref_base: data.esr_ref_base ?? '',
          // TVA
          vat_applicable: data.vat_applicable || false,
          vat_rate: data.vat_rate || null,
          vat_label: data.vat_label ?? '',
          vat_number: data.vat_number ?? '',
          // Configuration SMTP
          smtp_enabled: data.smtp_enabled || false,
          smtp_server: data.smtp_server ?? '',
          smtp_port: data.smtp_port ?? 587,
          smtp_use_tls: data.smtp_use_tls !== undefined ? data.smtp_use_tls : true,
          smtp_use_ssl: data.smtp_use_ssl || false,
          smtp_username: data.smtp_username ?? '',
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
      
      // ✅ OPTIMISATION: Utiliser la réponse du PUT si elle contient les données complètes
      // Sinon, recharger depuis le serveur (fallback)
      if (response?.data && response.data.iban !== undefined) {
        // La réponse contient les données, hydrater directement le state
        const formatIbanPretty = (iban) => {
          if (!iban) return '';
          const v = iban.replace(/\s+/g, '').toUpperCase();
          return v.replace(/(.{4})/g, '$1 ').trim();
        };
        
        setForm((prev) => ({
          ...prev,
          iban: response.data.iban ? formatIbanPretty(response.data.iban) : prev.iban,
          qr_iban: response.data.qr_iban ? formatIbanPretty(response.data.qr_iban) : prev.qr_iban,
          esr_ref_base: response.data.esr_ref_base || prev.esr_ref_base,
          // Mettre à jour aussi les autres champs de la réponse
          ...Object.fromEntries(
            Object.entries(response.data).filter(([key]) => 
              ['payment_terms_days', 'overdue_fee', 'reminder1_fee', 'reminder2_fee', 
               'reminder3_fee', 'reminder_schedule_days', 'auto_reminders_enabled',
               'email_sender', 'invoice_number_format', 'invoice_prefix',
               'invoice_message_template', 'reminder1_template', 'reminder2_template',
               'reminder3_template', 'legal_footer', 'pdf_template_variant',
               'vat_applicable', 'vat_rate', 'vat_label', 'vat_number'].includes(key)
            )
          ),
        }));
      } else {
        // Fallback: recharger toutes les données depuis le serveur
        await loadSettings();
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

  const toggleSection = (sectionKey) => {
    setExpandedSections((prev) => ({
      ...prev,
      [sectionKey]: !prev[sectionKey],
    }));
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
          gap: 'var(--spacing-md)',
          alignItems: 'start',
          width: '100%',
        }}
      >
        {/* COLONNE GAUCHE */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--spacing-sm)' }}>
          {/* Paramètres de paiement et rappels */}
          <section className={`${styles.section} ${styles.accordion}`}>
            <button
              type="button"
              className={styles.accordionHeader}
              onClick={() => toggleSection('payment')}
              aria-expanded={expandedSections.payment}
              aria-controls="billing-payment-section"
            >
              <span className={styles.accordionTitle}>💳 Paramètres de paiement</span>
              <span
                className={`${styles.accordionIcon} ${
                  expandedSections.payment ? styles.accordionIconOpen : ''
                }`}
                aria-hidden="true"
              >
                ▾
              </span>
            </button>
            {expandedSections.payment && (
              <div id="billing-payment-section" className={styles.accordionContent}>
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
                    Montant facturé automatiquement lorsque le paiement est en retard après
                    l'échéance
                  </small>
                </div>
              </div>
            )}
          </section>

          {/* Section Rappels */}
          <section className={`${styles.section} ${styles.accordion}`}>
            <button
              type="button"
              className={styles.accordionHeader}
              onClick={() => toggleSection('reminders')}
              aria-expanded={expandedSections.reminders}
              aria-controls="billing-reminders-section"
            >
              <span className={styles.accordionTitle}>📧 Rappels de paiement</span>
              <span
                className={`${styles.accordionIcon} ${
                  expandedSections.reminders ? styles.accordionIconOpen : ''
                }`}
                aria-hidden="true"
              >
                ▾
              </span>
            </button>
            {expandedSections.reminders && (
              <div id="billing-reminders-section" className={styles.accordionContent}>
                <small className={styles.hint} style={{ display: 'block', marginBottom: '16px' }}>
                  Configurez les frais et délais pour chaque niveau de rappel. Les frais sont
                  toujours facturés lors de l'émission du rappel, même si l'envoi automatique est
                  désactivé.
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
                    marginTop: '16px',
                    paddingTop: '16px',
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
              </div>
            )}
          </section>

          {/* Templates d'emails */}
          <section className={`${styles.section} ${styles.accordion}`}>
            <button
              type="button"
              className={styles.accordionHeader}
              onClick={() => toggleSection('templates')}
              aria-expanded={expandedSections.templates}
              aria-controls="billing-templates-section"
            >
              <span className={styles.accordionTitle}>✉️ Templates d'emails</span>
              <span
                className={`${styles.accordionIcon} ${
                  expandedSections.templates ? styles.accordionIconOpen : ''
                }`}
                aria-hidden="true"
              >
                ▾
              </span>
            </button>
            {expandedSections.templates && (
              <div id="billing-templates-section" className={styles.accordionContent}>
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
                        Variables: {'{client_name}'}, {'{amount}'}, {'{due_date}'},{' '}
                        {'{invoice_number}'}
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
                      <label htmlFor="reminder3_template">
                        Message 3e rappel (Mise en demeure)
                      </label>
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

                    <div className={styles.formGroup}>
                      <label htmlFor="email_signature_mode">Mode signature</label>
                      <select
                        id="email_signature_mode"
                        name="email_signature_mode"
                        value={form.email_signature_mode}
                        onChange={handleChange}
                        onBlur={handleBlur}
                      >
                        <option value="form">Formulaire</option>
                        <option value="text">Texte simple</option>
                        <option value="html">HTML (expert)</option>
                      </select>
                      <small className={styles.hint}>
                        Mode formulaire: champs normalisés (génération auto du HTML). Mode texte: signature simple multi-lignes. Mode HTML: template personnalisé avec variables.
                      </small>
                    </div>

                    {form.email_signature_mode === 'form' ? (
                      <>
                        <div className={styles.formGroup}>
                          <label htmlFor="signature_name">Nom complet *</label>
                          <input
                            type="text"
                            id="signature_name"
                            name="signature_name"
                            value={form.signature_name}
                            onChange={handleChange}
                            onBlur={handleBlur}
                            placeholder="Khalid ALAOUI"
                          />
                          <small className={styles.hint}>
                            Nom complet qui apparaîtra en gras dans la colonne gauche.
                          </small>
                        </div>

                        <div className={styles.formGroup}>
                          <label htmlFor="signature_title">Titre (optionnel)</label>
                          <input
                            type="text"
                            id="signature_title"
                            name="signature_title"
                            value={form.signature_title}
                            onChange={handleChange}
                            onBlur={handleBlur}
                            placeholder="Associé gérant"
                          />
                        </div>

                        <div className={styles.formGroup}>
                          <label htmlFor="signature_company">Société (optionnel)</label>
                          <input
                            type="text"
                            id="signature_company"
                            name="signature_company"
                            value={form.signature_company}
                            onChange={handleChange}
                            onBlur={handleBlur}
                            placeholder="Emmenez-moi Sàrl"
                          />
                        </div>

                        <div className={styles.formGroup}>
                          <label htmlFor="signature_phone_main">Téléphone principal (optionnel)</label>
                          <input
                            type="text"
                            id="signature_phone_main"
                            name="signature_phone_main"
                            value={form.signature_phone_main}
                            onChange={handleChange}
                            onBlur={handleBlur}
                            placeholder="022 512 02 03"
                          />
                        </div>

                        <div className={styles.formGroup}>
                          <label htmlFor="signature_phone_mobile">Téléphone mobile (optionnel)</label>
                          <input
                            type="text"
                            id="signature_phone_mobile"
                            name="signature_phone_mobile"
                            value={form.signature_phone_mobile}
                            onChange={handleChange}
                            onBlur={handleBlur}
                            placeholder="079 291 50 37"
                          />
                          <small className={styles.hint}>
                            Les téléphones seront affichés dans la colonne droite, séparés par " | ".
                          </small>
                        </div>

                        <div className={styles.formGroup}>
                          <label htmlFor="signature_email">Email (optionnel)</label>
                          <input
                            type="email"
                            id="signature_email"
                            name="signature_email"
                            value={form.signature_email}
                            onChange={handleChange}
                            onBlur={handleBlur}
                            placeholder="info@casa-famiglia.ch"
                          />
                          <small className={styles.hint}>
                            L'email sera automatiquement transformé en lien mailto: dans la colonne droite.
                          </small>
                        </div>

                        <div className={styles.formGroup}>
                          <label htmlFor="signature_website">Site web (optionnel)</label>
                          <input
                            type="url"
                            id="signature_website"
                            name="signature_website"
                            value={form.signature_website}
                            onChange={handleChange}
                            onBlur={handleBlur}
                            placeholder="www.transport-emmenez-moi.ch"
                          />
                          <small className={styles.hint}>
                            Le site web sera automatiquement transformé en lien cliquable dans la colonne droite.
                          </small>
                        </div>

                        <div className={styles.formGroup}>
                          <label htmlFor="signature_address_line">Ligne adresse (optionnel)</label>
                          <input
                            type="text"
                            id="signature_address_line"
                            name="signature_address_line"
                            value={form.signature_address_line}
                            onChange={handleChange}
                            onBlur={handleBlur}
                            placeholder="Route de Chevrens 145"
                          />
                        </div>

                        <div style={{ display: 'flex', gap: 'var(--spacing-sm)' }}>
                          <div className={styles.formGroup} style={{ flex: 1 }}>
                            <label htmlFor="signature_zip">Code postal (optionnel)</label>
                            <input
                              type="text"
                              id="signature_zip"
                              name="signature_zip"
                              value={form.signature_zip}
                              onChange={handleChange}
                              onBlur={handleBlur}
                              placeholder="1247"
                            />
                          </div>

                          <div className={styles.formGroup} style={{ flex: 2 }}>
                            <label htmlFor="signature_city">Ville (optionnel)</label>
                            <input
                              type="text"
                              id="signature_city"
                              name="signature_city"
                              value={form.signature_city}
                              onChange={handleChange}
                              onBlur={handleBlur}
                              placeholder="Anières"
                            />
                          </div>
                        </div>

                        <div className={styles.formGroup}>
                          <button
                            type="button"
                            onClick={() => setShowSignaturePreview(true)}
                            style={{
                              padding: '8px 16px',
                              backgroundColor: '#1b4b7a',
                              color: 'white',
                              border: 'none',
                              borderRadius: '4px',
                              cursor: 'pointer',
                              fontSize: '14px',
                            }}
                          >
                            Prévisualiser la signature
                          </button>
                          <small className={styles.hint}>
                            Aperçu de la signature telle qu'elle apparaîtra dans les emails.
                          </small>
                        </div>
                      </>
                    ) : form.email_signature_mode === 'text' ? (
                      <>
                        <div className={styles.formGroup}>
                          <label htmlFor="email_signature_text">Signature email (texte)</label>
                          <textarea
                            id="email_signature_text"
                            name="email_signature_text"
                            value={form.email_signature_text}
                            onChange={handleChange}
                            onBlur={handleBlur}
                            rows={6}
                            placeholder="Khalid ALAOUI
Associé gérant – Emmenez-moi Sàrl
022 512 02 03 | 079 291 50 37
info@casa-famiglia.ch
www.transport-emmenez-moi.ch
Route de Chevrens 145, 1247 Anières"
                          />
                          <small className={styles.hint}>
                            Cette signature sera automatiquement ajoutée à la fin de tous les emails
                            de facturation et rappels (même si les templates personnalisés sont désactivés).
                          </small>
                        </div>

                        <div className={styles.formGroup}>
                          <button
                            type="button"
                            onClick={() => setShowSignaturePreview(true)}
                            style={{
                              padding: '8px 16px',
                              backgroundColor: '#1b4b7a',
                              color: 'white',
                              border: 'none',
                              borderRadius: '4px',
                              cursor: 'pointer',
                              fontSize: '14px',
                            }}
                          >
                            Prévisualiser la signature
                          </button>
                          <small className={styles.hint}>
                            Aperçu de la signature telle qu'elle apparaîtra dans les emails.
                          </small>
                        </div>
                      </>
                    ) : (
                      <>
                        <div className={styles.formGroup}>
                          <label htmlFor="email_signature_html_template">Signature email (HTML)</label>
                          <textarea
                            id="email_signature_html_template"
                            name="email_signature_html_template"
                            value={form.email_signature_html_template}
                            onChange={handleChange}
                            onBlur={handleBlur}
                            rows={12}
                            placeholder={`<table cellpadding="0" cellspacing="0" border="0" style="font-family: Arial, sans-serif; font-size: 11px; color: #333;">
  <tr>
    <td style="vertical-align: top; padding-right: 12px;">
      <strong>{{ name }}</strong><br>
      {{ phone }}<br>
      {{ email }}<br>
      {{ address }}
    </td>
    <td width="1" style="border-left: 2px solid #1b4b7a; padding-left: 12px; vertical-align: top;">
      <!-- Colonne droite optionnelle -->
    </td>
  </tr>
</table>
<div style="border-top: 1px solid #1b4b7a; margin-top: 12px; padding-top: 8px;">
  {% if logo_url %}
    <img src="{{ logo_url }}" height="26" alt="Logo" style="display: block;" />
  {% endif %}
</div>`}
                          />
                          <small className={styles.hint}>
                            Template HTML avec variables Jinja2. Variables disponibles: <code>&#123;&#123; name &#125;&#125;</code>, <code>&#123;&#123; phone &#125;&#125;</code>, <code>&#123;&#123; email &#125;&#125;</code>, <code>&#123;&#123; address &#125;&#125;</code>, <code>&#123;&#123; logo_url &#125;&#125;</code>.
                            Les balises &lt;script&gt; et événements onclick sont automatiquement supprimés pour la sécurité.
                          </small>
                        </div>

                      </>
                    )}
                  </>
                )}
              </div>
            )}
          </section>
        </div>

        {/* COLONNE DROITE */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--spacing-sm)' }}>
          {/* Format de facturation et pied de page */}
          <section className={`${styles.section} ${styles.accordion}`}>
            <button
              type="button"
              className={styles.accordionHeader}
              onClick={() => toggleSection('format')}
              aria-expanded={expandedSections.format}
              aria-controls="billing-format-section"
            >
              <span className={styles.accordionTitle}>🧾 Format de facturation</span>
              <span
                className={`${styles.accordionIcon} ${
                  expandedSections.format ? styles.accordionIconOpen : ''
                }`}
                aria-hidden="true"
              >
                ▾
              </span>
            </button>
            {expandedSections.format && (
              <div id="billing-format-section" className={styles.accordionContent}>
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
                <h2 style={{ marginTop: '16px' }}>📄 Pied de page légal</h2>

                <div className={styles.formGroup}>
                  <label htmlFor="legal_footer">Texte du pied de page</label>
                  <textarea
                    id="legal_footer"
                    name="legal_footer"
                    value={form.legal_footer}
                    onChange={handleChange}
                    onBlur={handleBlur}
                    rows={3}
                    placeholder="En votre aimable règlement net sous {payment_terms_days} {jours} avec nos remerciements anticipés. En cas de retard de paiement, des frais de rappel d'un montant de CHF {overdue_fee} vous seront facturés, conformément à nos conditions générales."
                  />
                  <small className={styles.hint}>
                    Affiché sur toutes les factures PDF. Placeholders : {'{payment_terms_days}'}, {'{overdue_fee}'}, {'{jours}'} (valeurs depuis Paramètres de paiement)
                  </small>
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
              </div>
            )}
          </section>

          {/* TVA */}
          <section className={`${styles.section} ${styles.accordion}`}>
            <button
              type="button"
              className={styles.accordionHeader}
              onClick={() => toggleSection('vat')}
              aria-expanded={expandedSections.vat}
              aria-controls="billing-vat-section"
            >
              <span className={styles.accordionTitle}>💰 TVA (Taxe sur la valeur ajoutée)</span>
              <span
                className={`${styles.accordionIcon} ${
                  expandedSections.vat ? styles.accordionIconOpen : ''
                }`}
                aria-hidden="true"
              >
                ▾
              </span>
            </button>
            {expandedSections.vat && (
              <div id="billing-vat-section" className={styles.accordionContent}>
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
              </div>
            )}
          </section>

          {/* Informations bancaires */}
          <section className={`${styles.section} ${styles.accordion}`}>
            <button
              type="button"
              className={styles.accordionHeader}
              onClick={() => toggleSection('banking')}
              aria-expanded={expandedSections.banking}
              aria-controls="billing-banking-section"
            >
              <span className={styles.accordionTitle}>🏦 Informations bancaires</span>
              <span
                className={`${styles.accordionIcon} ${
                  expandedSections.banking ? styles.accordionIconOpen : ''
                }`}
                aria-hidden="true"
              >
                ▾
              </span>
            </button>
            {expandedSections.banking && (
              <div id="billing-banking-section" className={styles.accordionContent}>
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
                      <span
                        className={ibanChecksumIsValid(form.iban) ? styles.valid : styles.invalid}
                      >
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
              </div>
            )}
          </section>

          {/* Configuration Email Transactionnel (Brevo) */}
          {companyId && (
            <section className={`${styles.section} ${styles.accordion}`}>
              <button
                type="button"
                className={styles.accordionHeader}
                onClick={() => toggleSection('emailConfig')}
                aria-expanded={expandedSections.emailConfig}
                aria-controls="billing-email-config-section"
              >
                <span className={styles.accordionTitle}>📧 Configuration Email Transactionnel</span>
                <span
                  className={`${styles.accordionIcon} ${
                    expandedSections.emailConfig ? styles.accordionIconOpen : ''
                  }`}
                  aria-hidden="true"
                >
                  ▾
                </span>
              </button>
              {expandedSections.emailConfig && (
                <div id="billing-email-config-section" className={styles.accordionContent}>
                  <EmailConfigSection companyId={companyId} showHeader={false} compact />
                </div>
              )}
            </section>
          )}


        </div>
      </div>

      {/* Modal de prévisualisation de la signature */}
      {showSignaturePreview && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
          }}
          onClick={() => setShowSignaturePreview(false)}
        >
          <div
            style={{
              backgroundColor: 'white',
              padding: '24px',
              borderRadius: '8px',
              maxWidth: '600px',
              maxHeight: '80vh',
              overflow: 'auto',
              boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <h3 style={{ margin: 0, fontSize: '18px', fontWeight: '600' }}>Prévisualisation de la signature</h3>
              <button
                onClick={() => setShowSignaturePreview(false)}
                style={{
                  background: 'none',
                  border: 'none',
                  fontSize: '24px',
                  cursor: 'pointer',
                  color: '#666',
                  padding: '0',
                  width: '30px',
                  height: '30px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                ×
              </button>
            </div>
            <div
              style={{
                border: '1px solid #ddd',
                padding: '16px',
                backgroundColor: '#f9f9f9',
                borderRadius: '4px',
              }}
              dangerouslySetInnerHTML={{
                __html: form.email_signature_mode === 'form'
                  ? generateSignaturePreviewHtml(form)
                  : form.email_signature_mode === 'text'
                  ? form.email_signature_text
                      ? (() => {
                          const escapeHtml = (text) => {
                            if (!text) return '';
                            const div = document.createElement('div');
                            div.textContent = text;
                            return div.innerHTML;
                          };
                          return form.email_signature_text
                            .split('\n')
                            .map((line) => escapeHtml(line))
                            .join('<br>');
                        })()
                      : '<em>Aucune signature configurée</em>'
                  : '<em>Mode HTML : prévisualisation non disponible</em>',
              }}
            />
            <div style={{ marginTop: '16px', fontSize: '12px', color: '#666' }}>
              <p style={{ margin: 0 }}>
                Cette prévisualisation montre comment la signature apparaîtra dans les emails envoyés.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default BillingTab;
