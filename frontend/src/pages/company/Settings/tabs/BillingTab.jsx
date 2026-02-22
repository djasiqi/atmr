// frontend/src/pages/company/Settings/tabs/BillingTab.jsx
import React, { useState, useEffect, useRef, forwardRef, useImperativeHandle, useCallback } from 'react';
import {
  FiCreditCard,
  FiSliders,
  FiBell,
  FiMail,
  FiFileText,
  FiPercent,
  FiCheck,
  FiX,
  FiXCircle,
  FiSend,
  FiChevronDown,
} from 'react-icons/fi';
import styles from '../CompanySettings.module.css';
import n from './NotificationsTab.module.css';
import { fetchBillingSettings, updateBillingSettings } from '../../../../services/settingsService';
import EmailConfigSection from './EmailConfigSection';
import CancellationPolicyEditor from './components/CancellationPolicyEditor';

const generateSignaturePreviewHtml = (formData) => {
  const escapeHtml = (text) => {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  };

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
  const logoUrl = null;

  const leftColParts = [];
  if (name) leftColParts.push(`<strong style="font-size: 12px;">${escapeHtml(name)}</strong>`);
  if (title) leftColParts.push(escapeHtml(title));
  if (company) leftColParts.push(escapeHtml(company));
  const leftColContent = leftColParts.length > 0 ? leftColParts.join('<br>') : '&nbsp;';

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

const ReadonlyField = ({ label, value, suffix }) => {
  const display = value != null && value !== '' ? `${value}${suffix ? ` ${suffix}` : ''}` : null;
  return (
    <div className={styles.fieldRow}>
      <span className={styles.labelMuted}>{label}</span>
      <span className={`${styles.valueText}${!display ? ` ${styles.valueEmpty}` : ''}`}>
        {display || '\u2014'}
      </span>
    </div>
  );
};

const BILLING_DEFAULTS = {
  payment_terms_days: 10,
  overdue_fee: 15,
  reminder_schedule_days: { 1: 10, 2: 5, 3: 5 },
  reminder1_fee: 0,
  reminder2_fee: 40,
  reminder3_fee: 0,
  material_delivery_price_fixed: null,
  auto_reminders_enabled: true,
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
  pdf_template_variant: 'default',
  iban: '',
  qr_iban: '',
  esr_ref_base: '',
  vat_applicable: true,
  vat_rate: 7.7,
  vat_label: '',
  vat_number: '',
  smtp_enabled: false,
  smtp_server: '',
  smtp_port: 587,
  smtp_use_tls: true,
  smtp_use_ssl: false,
  smtp_username: '',
  smtp_password: '',
  smtp_password_configured: false,
  cancellation_policy: null,
};

const SectionCard = ({ icon: Icon, title, hint, expanded, onToggle, children }) => (
  <div
    className={styles.card}
    style={expanded ? undefined : { padding: 0, marginBottom: 8 }}
  >
    <button
      type="button"
      className={styles.cardHeader}
      onClick={onToggle}
      style={{
        cursor: 'pointer',
        width: '100%',
        background: 'none',
        border: 'none',
        borderBottom: expanded ? '1px solid var(--border-primary, #e5e7eb)' : 'none',
        textAlign: 'left',
        ...(expanded ? {} : {
          marginBottom: 0,
          padding: '12px 16px',
        }),
      }}
    >
      <div className={styles.cardIcon}><Icon size={16} /></div>
      <div className={styles.cardHeaderText} style={{ flex: 1 }}>
        <h3 className={styles.cardTitle}>{title}</h3>
        {hint && <p className={styles.cardHint}>{hint}</p>}
      </div>
      <FiChevronDown
        size={16}
        style={{
          color: 'var(--text-tertiary, #6b7280)',
          transition: 'transform 0.2s',
          transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)',
          flexShrink: 0,
        }}
      />
    </button>
    {expanded && <div style={{ padding: '4px 20px 16px' }}>{children}</div>}
  </div>
);

const BillingTab = forwardRef(({ companyId, isEditing }, ref) => {
  const [form, setForm] = useState({ ...BILLING_DEFAULTS });

  const [loading, setLoading] = useState(true);
  const [isHydrated, setIsHydrated] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [showSignaturePreview, setShowSignaturePreview] = useState(false);
  const [expandedSections, setExpandedSections] = useState({
    payment: true,
    format: true,
    reminders: false,
    vat: false,
    banking: false,
    templates: false,
    cancellation: false,
    emailConfig: false,
  });

  const serverFormRef = useRef(null);

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      setLoading(true);
      const data = await fetchBillingSettings();
      if (data) {
        const D = BILLING_DEFAULTS;
        const loaded = {
          payment_terms_days: data.payment_terms_days ?? D.payment_terms_days,
          overdue_fee: data.overdue_fee ?? D.overdue_fee,
          reminder_schedule_days: data.reminder_schedule_days || D.reminder_schedule_days,
          reminder1_fee: data.reminder1_fee ?? D.reminder1_fee,
          reminder2_fee: data.reminder2_fee ?? D.reminder2_fee,
          reminder3_fee: data.reminder3_fee ?? D.reminder3_fee,
          material_delivery_price_fixed: data.material_delivery_price_fixed ?? D.material_delivery_price_fixed,
          auto_reminders_enabled: data.auto_reminders_enabled ?? D.auto_reminders_enabled,
          email_templates_enabled: data.email_templates_enabled ?? D.email_templates_enabled,
          email_sender: data.email_sender ?? D.email_sender,
          invoice_number_format: data.invoice_number_format || D.invoice_number_format,
          invoice_prefix: data.invoice_prefix || D.invoice_prefix,
          invoice_message_template: data.invoice_message_template ?? D.invoice_message_template,
          reminder1_template: data.reminder1_template ?? D.reminder1_template,
          reminder2_template: data.reminder2_template ?? D.reminder2_template,
          reminder3_template: data.reminder3_template ?? D.reminder3_template,
          email_signature_mode: data.email_signature_mode || D.email_signature_mode,
          email_signature_text: data.email_signature_text ?? D.email_signature_text,
          signature_name: data.signature_name ?? D.signature_name,
          signature_title: data.signature_title ?? D.signature_title,
          signature_company: data.signature_company ?? D.signature_company,
          signature_phone_main: data.signature_phone_main ?? D.signature_phone_main,
          signature_phone_mobile: data.signature_phone_mobile ?? D.signature_phone_mobile,
          signature_email: data.signature_email ?? D.signature_email,
          signature_website: data.signature_website ?? D.signature_website,
          signature_address_line: data.signature_address_line ?? D.signature_address_line,
          signature_zip: data.signature_zip ?? D.signature_zip,
          signature_city: data.signature_city ?? D.signature_city,
          email_signature_html_template: data.email_signature_html_template ?? D.email_signature_html_template,
          legal_footer: data.legal_footer ?? D.legal_footer,
          pdf_template_variant: data.pdf_template_variant || D.pdf_template_variant,
          iban: data.iban ?? D.iban,
          qr_iban: data.qr_iban ?? D.qr_iban,
          esr_ref_base: data.esr_ref_base ?? D.esr_ref_base,
          vat_applicable: data.vat_applicable ?? D.vat_applicable,
          vat_rate: data.vat_rate ?? D.vat_rate,
          vat_label: data.vat_label ?? D.vat_label,
          vat_number: data.vat_number ?? D.vat_number,
          smtp_enabled: data.smtp_enabled ?? D.smtp_enabled,
          smtp_server: data.smtp_server ?? D.smtp_server,
          smtp_port: data.smtp_port ?? D.smtp_port,
          smtp_use_tls: data.smtp_use_tls ?? D.smtp_use_tls,
          smtp_use_ssl: data.smtp_use_ssl ?? D.smtp_use_ssl,
          smtp_username: data.smtp_username ?? D.smtp_username,
          smtp_password: '',
          smtp_password_configured: data.smtp_password_configured ?? false,
          cancellation_policy: data.cancellation_policy ?? D.cancellation_policy,
        };
        setForm(loaded);
        serverFormRef.current = loaded;
      }
    } catch (err) {
      console.error('Erreur lors du chargement des paramètres:', err);
      setError('Erreur lors du chargement des paramètres');
    } finally {
      setLoading(false);
      setIsHydrated(true);
    }
  };

  const normalizeIban = (iban) => {
    if (!iban) return null;
    return iban.replace(/\s+/g, '').toUpperCase().trim() || null;
  };

  const buildCleanedData = useCallback((formData) => {
    const cleaned = {
      ...formData,
      iban: normalizeIban(formData.iban),
      qr_iban: normalizeIban(formData.qr_iban),
      reminder_schedule_days: formData.reminder_schedule_days
        ? {
            1: parseInt(formData.reminder_schedule_days['1']) || 0,
            2: parseInt(formData.reminder_schedule_days['2']) || 0,
            3: parseInt(formData.reminder_schedule_days['3']) || 0,
          }
        : BILLING_DEFAULTS.reminder_schedule_days,
      vat_rate: (() => {
        if (formData.vat_rate === null || formData.vat_rate === '' || formData.vat_rate === undefined) return null;
        const parsed = parseFloat(formData.vat_rate);
        return isNaN(parsed) || parsed <= 0 ? null : parsed;
      })(),
      vat_label: formData.vat_label || null,
      vat_number: formData.vat_number || null,
      reminder1_fee: parseFloat(formData.reminder1_fee) || 0,
      reminder2_fee: parseFloat(formData.reminder2_fee) || 0,
      reminder3_fee: parseFloat(formData.reminder3_fee) || 0,
      material_delivery_price_fixed: formData.material_delivery_price_fixed != null && formData.material_delivery_price_fixed !== ''
        ? parseFloat(formData.material_delivery_price_fixed)
        : null,
      overdue_fee: parseFloat(formData.overdue_fee) || 0,
      payment_terms_days: parseInt(formData.payment_terms_days, 10) || (formData.payment_terms_days === '0' ? 0 : 10),
    };

    delete cleaned.smtp_password_configured;
    if (!cleaned.smtp_password || !cleaned.smtp_password.trim()) {
      delete cleaned.smtp_password;
    }

    return cleaned;
  }, []);

  const saveBilling = useCallback(async () => {
    const cleanedData = buildCleanedData(form);
    await updateBillingSettings(cleanedData);
    await loadSettings();
  }, [form, buildCleanedData]);

  const resetBilling = useCallback(() => {
    if (serverFormRef.current) {
      setForm(serverFormRef.current);
    }
    setMessage('');
    setError('');
  }, []);

  useImperativeHandle(ref, () => ({
    save: saveBilling,
    reset: resetBilling,
    isReady: () => isHydrated,
  }), [saveBilling, resetBilling, isHydrated]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  };

  const handleToggle = (e) => {
    const { name, checked } = e.target;
    setForm((prev) => ({ ...prev, [name]: checked }));
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

  const toggleSection = (key) => {
    setExpandedSections((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const ibanChecksumIsValid = (iban) => {
    if (!iban || iban.length < 15) return false;
    const swissPattern = /^CH[0-9]{2}[0-9]{5}[0-9A-Z]{12}$/;
    return swissPattern.test(iban.replace(/\s/g, ''));
  };

  const getSectionHint = (key) => {
    switch (key) {
      case 'payment': return `${form.payment_terms_days} jours / ${form.overdue_fee} CHF`;
      case 'reminders': return form.auto_reminders_enabled ? 'Envoi automatique actif' : 'Envoi manuel';
      case 'templates': return form.email_templates_enabled ? 'Personnalises' : 'Par defaut';
      case 'format': return generatePreview();
      case 'vat': return form.vat_applicable ? `${form.vat_rate || 0}%` : 'Inactive';
      case 'banking': return form.iban ? 'IBAN renseigne' : 'IBAN manquant';
      case 'cancellation': return form.cancellation_policy?.enabled ? 'Actif' : 'Inactif';
      case 'emailConfig': return form.smtp_enabled ? 'Configure' : 'Non configure';
      default: return '';
    }
  };

  if (loading) {
    return (
      <div className={styles.loadingContainer}>
        <div className={styles.spinner}></div>
        <p>Chargement des parametres de facturation...</p>
      </div>
    );
  }

  return (
    <div className={`${styles.settingsForm} ${styles.billingFormBlock}`}>
      {message && <div className={styles.success}>{message}</div>}
      {error && <div className={styles.error}>{error}</div>}

      <div className={styles.billingGrid}>
        {/* ═══ Colonne gauche: Finances ═══ */}
        <div className={styles.billingCol}>

          {/* Parametres de paiement */}
          <SectionCard
            icon={FiSliders}
            title="Parametres de paiement"
            hint={getSectionHint('payment')}
            expanded={expandedSections.payment}
            onToggle={() => toggleSection('payment')}
          >
            {isEditing ? (
              <>
                <div className={styles.formGroup}>
                  <label htmlFor="payment_terms_days">Delai de paiement</label>
                  <div className={styles.inputWithUnit} style={{ maxWidth: 120 }}>
                    <input type="number" id="payment_terms_days" name="payment_terms_days" value={form.payment_terms_days} onChange={handleChange} min="1" max="90" />
                    <span className={styles.unit}>jours</span>
                  </div>
                  <small className={styles.hint}>Delai accorde aux clients pour payer (1-90 jours)</small>
                </div>
                <div className={styles.formGroup}>
                  <label htmlFor="overdue_fee">Frais de retard</label>
                  <div className={styles.inputWithUnit}>
                    <input type="number" id="overdue_fee" name="overdue_fee" value={form.overdue_fee} onChange={handleChange} step="0.01" min="0" />
                    <span className={styles.unit}>CHF</span>
                  </div>
                  <small className={styles.hint}>Montant facture en cas de retard de paiement</small>
                </div>
                <div className={styles.formGroup}>
                  <label htmlFor="material_delivery_price_fixed">Prix livraison fixe</label>
                  <div className={styles.inputWithUnit}>
                    <input type="number" id="material_delivery_price_fixed" name="material_delivery_price_fixed" value={form.material_delivery_price_fixed ?? ''} onChange={handleChange} step="0.01" min="0" placeholder="Ex: 35.00" />
                    <span className={styles.unit}>CHF</span>
                  </div>
                  <small className={styles.hint}>Prix fixe pour les livraisons materiel</small>
                </div>
              </>
            ) : (
              <div className={styles.fieldGrid}>
                <ReadonlyField label="Delai de paiement" value={form.payment_terms_days} suffix="jours" />
                <ReadonlyField label="Frais de retard" value={form.overdue_fee} suffix="CHF" />
                <ReadonlyField label="Prix livraison fixe" value={form.material_delivery_price_fixed} suffix="CHF" />
              </div>
            )}
          </SectionCard>

          {/* TVA */}
          <SectionCard
            icon={FiPercent}
            title="TVA"
            hint={getSectionHint('vat')}
            expanded={expandedSections.vat}
            onToggle={() => toggleSection('vat')}
          >
            <label className={n.notifRow} htmlFor="vat_applicable" style={{ padding: '8px 0' }}>
              <div className={n.notifInfo}>
                <span className={n.notifLabel}>TVA applicable</span>
                <span className={n.notifHint}>Activez si votre entreprise est assujettie</span>
              </div>
              <div className={n.miniToggle}>
                <input id="vat_applicable" type="checkbox" name="vat_applicable" checked={form.vat_applicable} onChange={handleToggle} />
                <span className={n.miniSlider} />
              </div>
            </label>
            {form.vat_applicable && (
              isEditing ? (
                <>
                  <div className={styles.formGroup}>
                    <label htmlFor="vat_rate">Taux de TVA</label>
                    <div className={styles.inputWithUnit}>
                      <input type="number" id="vat_rate" name="vat_rate" value={form.vat_rate || ''} onChange={handleChange} step="0.01" min="0" max="100" placeholder="7.7" />
                      <span className={styles.unit}>%</span>
                    </div>
                    <small className={styles.hint}>Standard: 7.7% / Reduit: 2.5% / Special: 3.7%</small>
                  </div>
                  <div className={styles.formGroup}>
                    <label htmlFor="vat_label">Libelle TVA</label>
                    <input type="text" id="vat_label" name="vat_label" value={form.vat_label} onChange={handleChange} placeholder="TVA" maxLength={50} />
                  </div>
                  <div className={styles.formGroup}>
                    <label htmlFor="vat_number">Numero de TVA</label>
                    <input type="text" id="vat_number" name="vat_number" value={form.vat_number} onChange={handleChange} placeholder="CHE-123.456.789 TVA" maxLength={50} />
                  </div>
                </>
              ) : (
                <div className={styles.fieldGrid}>
                  <ReadonlyField label="Taux" value={form.vat_rate} suffix="%" />
                  <ReadonlyField label="Libelle" value={form.vat_label} />
                  <ReadonlyField label="Numero TVA" value={form.vat_number} />
                </div>
              )
            )}
          </SectionCard>

          {/* Informations bancaires */}
          <SectionCard
            icon={FiCreditCard}
            title="Informations bancaires"
            hint={getSectionHint('banking')}
            expanded={expandedSections.banking}
            onToggle={() => toggleSection('banking')}
          >
            {isEditing ? (
              <>
                <div className={styles.formGroup}>
                  <label htmlFor="iban">IBAN</label>
                  <input id="iban" name="iban" value={form.iban} onChange={handleChange} placeholder="CH93 0076 2011 6238 5295 7" maxLength={34} />
                  <small className={styles.hint}>
                    {form.iban && (
                      <span className={`${styles.ibanStatus} ${ibanChecksumIsValid(form.iban) ? styles.valid : styles.invalid}`} role="status">
                        {ibanChecksumIsValid(form.iban) ? <><FiCheck aria-hidden /> IBAN valide</> : <><FiX aria-hidden /> IBAN invalide</>}
                      </span>
                    )}
                  </small>
                </div>
                <div className={styles.formGroup}>
                  <label htmlFor="qr_iban">IBAN pour QR-Code</label>
                  <input id="qr_iban" name="qr_iban" value={form.qr_iban} onChange={handleChange} placeholder="CH93 0076 2011 6238 5295 7" maxLength={34} />
                </div>
                <div className={styles.formGroup}>
                  <label htmlFor="esr_ref_base">Reference ESR</label>
                  <input id="esr_ref_base" name="esr_ref_base" value={form.esr_ref_base} onChange={handleChange} placeholder="00000000000000000000" maxLength={27} />
                </div>
              </>
            ) : (
              <div className={styles.fieldGrid}>
                <ReadonlyField label="IBAN" value={form.iban} />
                <ReadonlyField label="IBAN QR-Code" value={form.qr_iban} />
                <ReadonlyField label="Reference ESR" value={form.esr_ref_base} />
              </div>
            )}
          </SectionCard>

          {/* Frais d'annulation */}
          <SectionCard
            icon={FiXCircle}
            title="Frais d'annulation"
            hint={getSectionHint('cancellation')}
            expanded={expandedSections.cancellation}
            onToggle={() => toggleSection('cancellation')}
          >
            {isEditing ? (
              <CancellationPolicyEditor
                policy={form.cancellation_policy}
                onChange={(nextPolicy) => {
                  setForm((prev) => ({ ...prev, cancellation_policy: nextPolicy }));
                }}
              />
            ) : (
              <div className={styles.fieldGrid}>
                <ReadonlyField label="Frais d'annulation" value={form.cancellation_policy?.enabled ? 'Actif' : 'Inactif'} />
                {form.cancellation_policy?.enabled && (
                  <>
                    <ReadonlyField label="Base" value={form.cancellation_policy?.basis === 'booking_amount' ? 'Montant course' : form.cancellation_policy?.basis} />
                    <ReadonlyField label="Paliers" value={`${form.cancellation_policy?.tiers?.length || 0} configure(s)`} />
                  </>
                )}
              </div>
            )}
          </SectionCard>

        </div>

        {/* ═══ Colonne droite: Documents & Communication ═══ */}
        <div className={styles.billingCol}>

          {/* Format de facturation */}
          <SectionCard
            icon={FiFileText}
            title="Format de facturation"
            hint={getSectionHint('format')}
            expanded={expandedSections.format}
            onToggle={() => toggleSection('format')}
          >
            {isEditing ? (
              <>
                <div className={styles.formGroup}>
                  <label htmlFor="invoice_prefix">Prefixe des factures</label>
                  <input id="invoice_prefix" name="invoice_prefix" value={form.invoice_prefix} onChange={handleChange} maxLength={10} placeholder="EM" />
                  <span className={styles.previewInline}>{generatePreview()}</span>
                </div>
                <div className={styles.formGroup}>
                  <label htmlFor="invoice_number_format">Format de numerotation</label>
                  <select id="invoice_number_format" name="invoice_number_format" value={form.invoice_number_format} onChange={handleChange}>
                    <option value="{PREFIX}-{YYYY}-{MM}-{SEQ4}">{form.invoice_prefix}-2025-10-0001</option>
                    <option value="{PREFIX}-{YYYY}-{SEQ5}">{form.invoice_prefix}-2025-00001</option>
                    <option value="{PREFIX}{YYYYMM}{SEQ3}">{form.invoice_prefix}202510001</option>
                  </select>
                </div>
                <div style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid var(--border-primary, #e5e7eb)' }}>
                  <div className={styles.formGroup}>
                    <label htmlFor="legal_footer">Pied de page</label>
                    <textarea id="legal_footer" name="legal_footer" value={form.legal_footer ?? ''} onChange={handleChange} rows={3} placeholder="En votre aimable reglement net sous {payment_terms_days} {jours}..." />
                    <small className={styles.hint}>Placeholders : {'{payment_terms_days}'}, {'{overdue_fee}'}, {'{jours}'}</small>
                  </div>
                </div>
                <div className={styles.formGroup}>
                  <label htmlFor="pdf_template_variant">Variante PDF</label>
                  <select id="pdf_template_variant" name="pdf_template_variant" value={form.pdf_template_variant} onChange={handleChange}>
                    <option value="standard">Standard</option>
                    <option value="minimal">Minimal</option>
                    <option value="detailed">Detaille</option>
                  </select>
                </div>
              </>
            ) : (
              <div className={styles.fieldGrid}>
                <ReadonlyField label="Prefixe" value={form.invoice_prefix} />
                <ReadonlyField label="Numero" value={generatePreview()} />
                <ReadonlyField label="Variante PDF" value={form.pdf_template_variant === 'detailed' ? 'Detaille' : form.pdf_template_variant === 'minimal' ? 'Minimal' : 'Standard'} />
                <div className={styles.fieldGridFull}>
                  <ReadonlyField label="Pied de page" value={form.legal_footer} />
                </div>
              </div>
            )}
          </SectionCard>

          {/* Rappels de paiement */}
          <SectionCard
            icon={FiBell}
            title="Rappels de paiement"
            hint={getSectionHint('reminders')}
            expanded={expandedSections.reminders}
            onToggle={() => toggleSection('reminders')}
          >
            {isEditing ? (
              <>
                <small className={`${styles.hint} ${styles.hintBlock}`}>
                  Configurez les frais et delais pour chaque niveau de rappel.
                </small>
                {[
                  { key: '1', label: '1er rappel', field: 'reminder1_fee', defaultDelay: 10, delayHint: "Jours apres l'echeance" },
                  { key: '2', label: '2e rappel', field: 'reminder2_fee', defaultDelay: 5, delayHint: 'Jours apres le 1er rappel' },
                  { key: '3', label: '3e rappel (Mise en demeure)', field: 'reminder3_fee', defaultDelay: 3, delayHint: 'Jours apres le 2e rappel' },
                ].map((r) => (
                  <div key={r.key} className={styles.reminderRow}>
                    <h4 className={styles.reminderTitle}>{r.label}</h4>
                    <div className={styles.reminderFields}>
                      <div className={styles.formGroup}>
                        <label>Frais</label>
                        <div className={styles.inputWithUnit}>
                          <input type="number" name={r.field} value={form[r.field]} onChange={handleChange} step="0.01" min="0" />
                          <span className={styles.unit}>CHF</span>
                        </div>
                      </div>
                      {form.auto_reminders_enabled && (
                        <div className={styles.formGroup}>
                          <label>Delai d'envoi</label>
                          <input type="number" value={form.reminder_schedule_days[r.key] || r.defaultDelay} onChange={(e) => handleReminderScheduleChange(r.key, e.target.value)} min="1" max="90" />
                          <small className={styles.hint}>{r.delayHint}</small>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </>
            ) : (
              <div className={styles.fieldGrid}>
                <ReadonlyField label="1er rappel" value={form.reminder1_fee} suffix="CHF" />
                <ReadonlyField label="2e rappel" value={form.reminder2_fee} suffix="CHF" />
                <ReadonlyField label="3e rappel" value={form.reminder3_fee} suffix="CHF" />
              </div>
            )}
            <div className={styles.reminderBorderTop}>
              <label className={n.notifRow} htmlFor="auto_reminders_enabled" style={{ padding: '8px 0' }}>
                <div className={n.notifInfo}>
                  <span className={n.notifLabel}>Envoi automatique</span>
                  <span className={n.notifHint}>Frais facturés même si envoi manuel</span>
                </div>
                <div className={n.miniToggle}>
                  <input id="auto_reminders_enabled" type="checkbox" name="auto_reminders_enabled" checked={form.auto_reminders_enabled} onChange={handleToggle} />
                  <span className={n.miniSlider} />
                </div>
              </label>
            </div>
          </SectionCard>

          {/* Templates d'emails */}
          <SectionCard
        icon={FiMail}
        title="Templates d'emails"
        hint={getSectionHint('templates')}
        expanded={expandedSections.templates}
        onToggle={() => toggleSection('templates')}
      >
        <label className={n.notifRow} htmlFor="email_templates_enabled" style={{ padding: '8px 0' }}>
          <div className={n.notifInfo}>
            <span className={n.notifLabel}>Templates personnalisés</span>
            <span className={n.notifHint}>Messages d'email pour factures et rappels</span>
          </div>
          <div className={n.miniToggle}>
            <input id="email_templates_enabled" type="checkbox" name="email_templates_enabled" checked={form.email_templates_enabled || false} onChange={(e) => handleToggle({ target: { name: 'email_templates_enabled', checked: e.target.checked } })} />
            <span className={n.miniSlider} />
          </div>
        </label>
        {form.email_templates_enabled && (
          isEditing ? (
            <>
              <div className={styles.formGroup}>
                <label htmlFor="email_sender">Email expediteur</label>
                <input type="email" id="email_sender" name="email_sender" value={form.email_sender} onChange={handleChange} placeholder="facturation@emmenezmoi.ch" />
              </div>
              <div className={styles.formGroup}>
                <label htmlFor="invoice_message_template">Message envoi de facture</label>
                <textarea id="invoice_message_template" name="invoice_message_template" value={form.invoice_message_template ?? ''} onChange={handleChange} rows={5} placeholder="Bonjour {client_name},&#10;&#10;Veuillez trouver ci-joint la facture {invoice_number}." />
                <small className={styles.hint}>Variables: {'{client_name}'}, {'{amount}'}, {'{due_date}'}, {'{invoice_number}'}</small>
              </div>
              <div className={styles.formGroup}>
                <label htmlFor="reminder1_template">Message 1er rappel</label>
                <textarea id="reminder1_template" name="reminder1_template" value={form.reminder1_template ?? ''} onChange={handleChange} rows={4} placeholder="Rappel: votre facture {invoice_number} n'a pas encore ete reglee." />
              </div>
              <div className={styles.formGroup}>
                <label htmlFor="reminder2_template">Message 2e rappel</label>
                <textarea id="reminder2_template" name="reminder2_template" value={form.reminder2_template ?? ''} onChange={handleChange} rows={4} placeholder="2e rappel: merci de regler la facture {invoice_number} sous 5 jours." />
              </div>
              <div className={styles.formGroup}>
                <label htmlFor="reminder3_template">Message 3e rappel (Mise en demeure)</label>
                <textarea id="reminder3_template" name="reminder3_template" value={form.reminder3_template ?? ''} onChange={handleChange} rows={4} placeholder="Mise en demeure: dernier rappel avant procedures legales." />
              </div>
              <div className={styles.formGroup}>
                <label htmlFor="email_signature_mode">Mode signature</label>
                <select id="email_signature_mode" name="email_signature_mode" value={form.email_signature_mode} onChange={handleChange}>
                  <option value="form">Formulaire</option>
                  <option value="text">Texte simple</option>
                  <option value="html">HTML (expert)</option>
                </select>
              </div>
              {form.email_signature_mode === 'form' && (
                <>
                  <div className={styles.formGroup}>
                    <label htmlFor="signature_name">Nom complet *</label>
                    <input type="text" id="signature_name" name="signature_name" value={form.signature_name} onChange={handleChange} placeholder="Khalid ALAOUI" />
                  </div>
                  <div className={styles.formGroup}>
                    <label htmlFor="signature_title">Titre</label>
                    <input type="text" id="signature_title" name="signature_title" value={form.signature_title} onChange={handleChange} placeholder="Associe gerant" />
                  </div>
                  <div className={styles.formGroup}>
                    <label htmlFor="signature_company">Societe</label>
                    <input type="text" id="signature_company" name="signature_company" value={form.signature_company} onChange={handleChange} placeholder="Emmenez-moi Sarl" />
                  </div>
                  <div className={styles.formGroup}>
                    <label htmlFor="signature_phone_main">Telephone principal</label>
                    <input type="text" id="signature_phone_main" name="signature_phone_main" value={form.signature_phone_main} onChange={handleChange} placeholder="022 512 02 03" />
                  </div>
                  <div className={styles.formGroup}>
                    <label htmlFor="signature_phone_mobile">Telephone mobile</label>
                    <input type="text" id="signature_phone_mobile" name="signature_phone_mobile" value={form.signature_phone_mobile} onChange={handleChange} placeholder="079 291 50 37" />
                  </div>
                  <div className={styles.formGroup}>
                    <label htmlFor="signature_email">Email</label>
                    <input type="email" id="signature_email" name="signature_email" value={form.signature_email} onChange={handleChange} placeholder="info@casa-famiglia.ch" />
                  </div>
                  <div className={styles.formGroup}>
                    <label htmlFor="signature_website">Site web</label>
                    <input type="url" id="signature_website" name="signature_website" value={form.signature_website} onChange={handleChange} placeholder="www.transport-emmenez-moi.ch" />
                  </div>
                  <div className={styles.formGroup}>
                    <label htmlFor="signature_address_line">Adresse</label>
                    <input type="text" id="signature_address_line" name="signature_address_line" value={form.signature_address_line} onChange={handleChange} placeholder="Route de Chevrens 145" />
                  </div>
                  <div className={styles.billingFlexRow}>
                    <div className={`${styles.formGroup} ${styles.billingFlex1}`}>
                      <label htmlFor="signature_zip">NPA</label>
                      <input type="text" id="signature_zip" name="signature_zip" value={form.signature_zip} onChange={handleChange} placeholder="1247" />
                    </div>
                    <div className={`${styles.formGroup} ${styles.billingFlex2}`}>
                      <label htmlFor="signature_city">Ville</label>
                      <input type="text" id="signature_city" name="signature_city" value={form.signature_city} onChange={handleChange} placeholder="Anieres" />
                    </div>
                  </div>
                </>
              )}
              {form.email_signature_mode === 'text' && (
                <div className={styles.formGroup}>
                  <label htmlFor="email_signature_text">Signature (texte)</label>
                  <textarea id="email_signature_text" name="email_signature_text" value={form.email_signature_text ?? ''} onChange={handleChange} rows={6} placeholder={"Khalid ALAOUI\nAssocie gerant\n022 512 02 03"} />
                </div>
              )}
              {form.email_signature_mode === 'html' && (
                <div className={styles.formGroup}>
                  <label htmlFor="email_signature_html_template">Signature (HTML)</label>
                  <textarea id="email_signature_html_template" name="email_signature_html_template" value={form.email_signature_html_template ?? ''} onChange={handleChange} rows={12} placeholder={'<table>...</table>'} />
                  <small className={styles.hint}>Variables Jinja2: {'{{ name }}'}, {'{{ phone }}'}, {'{{ email }}'}, {'{{ address }}'}, {'{{ logo_url }}'}</small>
                </div>
              )}
              <button type="button" className={`${styles.button} ${styles.secondary}`} onClick={() => setShowSignaturePreview(true)}>
                Previsualiser la signature
              </button>
            </>
          ) : (
            <div className={styles.fieldGrid}>
              <ReadonlyField label="Email expediteur" value={form.email_sender} />
              <ReadonlyField label="Mode signature" value={
                form.email_signature_mode === 'form' ? 'Formulaire' :
                form.email_signature_mode === 'text' ? 'Texte simple' : 'HTML'
              } />
              {form.signature_name && <ReadonlyField label="Signataire" value={form.signature_name} />}
            </div>
          )
        )}
      </SectionCard>

      {/* Configuration Email Transactionnel */}
      {companyId && (
        <SectionCard
          icon={FiSend}
          title="Configuration email transactionnel"
          hint={getSectionHint('emailConfig')}
          expanded={expandedSections.emailConfig}
          onToggle={() => toggleSection('emailConfig')}
        >
          {isEditing ? (
            <EmailConfigSection companyId={companyId} showHeader={false} compact />
          ) : (
            <div className={styles.fieldGrid}>
              <ReadonlyField label="Statut" value={form.smtp_enabled ? 'Configure' : 'Non configure'} />
              {form.smtp_enabled && <ReadonlyField label="Serveur SMTP" value={form.smtp_server} />}
            </div>
          )}
        </SectionCard>
      )}

        </div>
      </div>

      {showSignaturePreview && (
        <div className={styles.modalOverlay} onClick={() => setShowSignaturePreview(false)} role="dialog" aria-modal="true" aria-labelledby="signature-preview-title">
          <div className={styles.modalContentLarge} onClick={(e) => e.stopPropagation()}>
            <div className={styles.signatureModalHeader}>
              <h3 id="signature-preview-title" className={styles.modalTitle}>Previsualisation de la signature</h3>
              <button type="button" className={styles.signatureModalClose} onClick={() => setShowSignaturePreview(false)} aria-label="Fermer">
                <FiX size={18} />
              </button>
            </div>
            <div
              className={styles.signatureModalPreview}
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
                          return form.email_signature_text.split('\n').map((line) => escapeHtml(line)).join('<br>');
                        })()
                      : '<em>Aucune signature configuree</em>'
                  : '<em>Mode HTML : previsualisation non disponible</em>',
              }}
            />
            <p className={styles.signatureModalFooter}>Previsualisation telle qu'elle apparaitra dans les emails envoyes.</p>
          </div>
        </div>
      )}
    </div>
  );
});

BillingTab.displayName = 'BillingTab';
export default BillingTab;
