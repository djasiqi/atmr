import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  fetchPlatformBillingCreditor,
  putPlatformBillingCreditor,
} from '../../../services/adminService';
import styles from './AdminSettings.module.css';

const emptyForm = {
  legal_name: '',
  street_name: '',
  building_number: '',
  postal_code: '',
  city: '',
  country_code: 'CH',
  uid_ide: '',
  vat_number: '',
  legal_form: 'sole_proprietorship',
  signatory_name: '',
  signatory_title: 'Exploitant',
  default_tax_rate: '0',
  iban: '',
  qr_iban: '',
  payment_reference_mode: 'QRR',
  payment_terms_days_default: '30',
  is_active: true,
};

const LEGAL_FORM_LABELS = {
  sole_proprietorship: 'Indépendant',
  sarl: 'Sàrl',
  sa: 'SA',
  association: 'Association',
  foundation: 'Fondation',
  other: 'Autre',
};

const maskIban = (raw) => {
  const compact = String(raw || '').replace(/\s+/g, '').toUpperCase();
  if (!compact) return 'Non renseigné';
  if (compact.length < 8) return compact;
  return `${compact.slice(0, 4)} ···· ···· ${compact.slice(-4)}`;
};

const formatIbanDisplay = (raw) => {
  const compact = String(raw || '').replace(/\s+/g, '').toUpperCase();
  return compact.replace(/(.{4})/g, '$1 ').trim();
};

const AdminPlatformCreditorSettings = () => {
  const [form, setForm] = useState(emptyForm);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [revealIban, setRevealIban] = useState(false);
  const [revealQrIban, setRevealQrIban] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const res = await fetchPlatformBillingCreditor();
      const c = res?.creditor;
      if (c) {
        setForm({
          legal_name: c.legal_name || '',
          street_name: c.street_name || '',
          building_number: c.building_number || '',
          postal_code: c.postal_code || '',
          city: c.city || '',
          country_code: c.country_code || 'CH',
          uid_ide: c.uid_ide || '',
          vat_number: c.vat_number || '',
          legal_form: c.legal_form || 'sole_proprietorship',
          signatory_name: c.signatory_name || '',
          signatory_title: c.signatory_title || 'Exploitant',
          default_tax_rate:
            c.default_tax_rate != null ? String(c.default_tax_rate) : '0',
          iban: c.iban || '',
          qr_iban: c.qr_iban || '',
          payment_reference_mode: c.payment_reference_mode || 'QRR',
          payment_terms_days_default:
            c.payment_terms_days_default != null
              ? String(c.payment_terms_days_default)
              : '30',
          is_active: c.is_active !== false,
        });
      } else {
        setForm(emptyForm);
      }
    } catch (e) {
      setError(e?.response?.data?.message || e?.message || 'Erreur chargement créancier');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const setField = (key, value) => {
    setForm((f) => ({ ...f, [key]: value }));
  };

  const readiness = useMemo(() => {
    const hasIdentity = Boolean(form.legal_name.trim());
    const hasAddress = Boolean(
      form.street_name.trim() && form.postal_code.trim() && form.city.trim()
    );
    const hasIban = Boolean(form.iban.trim() || form.qr_iban.trim());
    const tax = Number(String(form.default_tax_rate).replace(',', '.'));
    return {
      ready: hasIdentity && hasAddress && hasIban,
      hasIdentity,
      hasAddress,
      hasIban,
      taxZero: Number.isFinite(tax) && tax === 0,
      taxLabel: Number.isFinite(tax) && tax === 0 ? 'Franchise / non assujetti' : `${form.default_tax_rate} %`,
      addressLine: [form.street_name, form.building_number].filter(Boolean).join(' '),
      cityLine: [form.postal_code, form.city].filter(Boolean).join(' '),
    };
  }, [form]);

  const onSave = async (e) => {
    e.preventDefault();
    setSaving(true);
    setError('');
    setSuccess('');
    try {
      if (!form.legal_name.trim()) {
        throw new Error('Raison sociale obligatoire');
      }
      if (!form.street_name.trim() || !form.postal_code.trim() || !form.city.trim()) {
        throw new Error('Adresse de domicile incomplète (rue, NPA, localité)');
      }
      if (!form.iban.trim() && !form.qr_iban.trim()) {
        throw new Error('IBAN ou QR-IBAN obligatoire pour la QR-facture');
      }
      await putPlatformBillingCreditor({
        ...form,
        legal_name: form.legal_name.trim(),
        street_name: form.street_name.trim(),
        building_number: form.building_number.trim() || null,
        postal_code: form.postal_code.trim(),
        city: form.city.trim(),
        country_code: (form.country_code || 'CH').trim().toUpperCase(),
        uid_ide: form.uid_ide.trim() || null,
        vat_number: form.vat_number.trim() || null,
        legal_form: form.legal_form || null,
        signatory_name: form.signatory_name.trim() || null,
        signatory_title: form.signatory_title.trim() || null,
        iban: form.iban.replace(/\s+/g, '').toUpperCase() || null,
        qr_iban: form.qr_iban.replace(/\s+/g, '').toUpperCase() || null,
        default_tax_rate: form.default_tax_rate,
        payment_terms_days_default: Number(form.payment_terms_days_default) || 30,
        is_active: true,
      });
      setSuccess('Créancier enregistré.');
      setRevealIban(false);
      setRevealQrIban(false);
      await load();
    } catch (err) {
      setError(err?.response?.data?.message || err?.message || 'Erreur enregistrement');
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className={styles.panelBody}>
        <p className={styles.helperText}>Chargement du créancier…</p>
      </div>
    );
  }

  return (
    <div className={styles.panelBody}>
      <div className={styles.panelIntro}>
        <h2 className={styles.panelTitle}>Créancier LIRIE</h2>
        <p className={styles.panelLead}>
          Identité figée sur les QR-factures et les contrats partenaires. En indépendant,
          utilisez le nom de la personne physique.
        </p>
      </div>

      <aside className={styles.summaryStrip} aria-label="Résumé créancier">
        <div className={styles.summaryMain}>
          <strong>{form.legal_name || 'Non configuré'}</strong>
          <span>
            {LEGAL_FORM_LABELS[form.legal_form] || form.legal_form}
            {form.signatory_name ? ` · ${form.signatory_name}` : ''}
          </span>
          <span>
            {[readiness.addressLine, readiness.cityLine, form.country_code]
              .filter(Boolean)
              .join(' · ') || 'Adresse manquante'}
          </span>
        </div>
        <div className={styles.summaryMeta}>
          <span
            className={`${styles.statusChip} ${
              readiness.ready ? styles.statusOk : styles.statusWarn
            }`}
          >
            {readiness.ready ? 'QR prêt' : 'Incomplet'}
          </span>
          <span className={`${styles.statusChip} ${styles.statusMuted}`}>
            {readiness.taxLabel}
          </span>
          <span className={styles.ibanPreview}>{maskIban(form.iban || form.qr_iban)}</span>
        </div>
      </aside>

      {error ? <div className={styles.error}>{error}</div> : null}
      {success ? <div className={styles.success}>{success}</div> : null}

      <form onSubmit={onSave} className={styles.splitForm}>
        <div className={styles.splitCol}>
          <h3 className={styles.colTitle}>Identité</h3>
          <label className={styles.formGroup}>
            {form.legal_form === 'sole_proprietorship'
              ? 'Nom (personne physique)'
              : 'Raison sociale'}
            <input
              value={form.legal_name}
              onChange={(e) => setField('legal_name', e.target.value)}
              required
              autoComplete="organization"
              placeholder={
                form.legal_form === 'sole_proprietorship'
                  ? 'Drin Jasiqi'
                  : 'Raison sociale au registre'
              }
            />
          </label>
          <div className={styles.formRow2}>
            <label className={styles.formGroup}>
              Forme juridique
              <select
                value={form.legal_form}
                onChange={(e) => setField('legal_form', e.target.value)}
              >
                {Object.entries(LEGAL_FORM_LABELS).map(([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </select>
            </label>
            <label className={styles.formGroup}>
              IDE / UID
              <input
                value={form.uid_ide}
                onChange={(e) => setField('uid_ide', e.target.value)}
                placeholder="CHE-…"
              />
            </label>
          </div>
          <div className={styles.formRow2}>
            <label className={styles.formGroup}>
              Signataire
              <input
                value={form.signatory_name}
                onChange={(e) => setField('signatory_name', e.target.value)}
              />
            </label>
            <label className={styles.formGroup}>
              Titre
              <input
                value={form.signatory_title}
                onChange={(e) => setField('signatory_title', e.target.value)}
                placeholder="Exploitant"
              />
            </label>
          </div>

          <h3 className={styles.colTitle}>Adresse</h3>
          <div className={styles.formRowStreet}>
            <label className={styles.formGroup}>
              Rue
              <input
                value={form.street_name}
                onChange={(e) => setField('street_name', e.target.value)}
                required
                autoComplete="address-line1"
              />
            </label>
            <label className={styles.formGroup}>
              N°
              <input
                value={form.building_number}
                onChange={(e) => setField('building_number', e.target.value)}
                autoComplete="address-line2"
              />
            </label>
          </div>
          <div className={styles.formRow3}>
            <label className={styles.formGroup}>
              NPA
              <input
                value={form.postal_code}
                onChange={(e) => setField('postal_code', e.target.value)}
                required
                autoComplete="postal-code"
              />
            </label>
            <label className={styles.formGroup}>
              Localité
              <input
                value={form.city}
                onChange={(e) => setField('city', e.target.value)}
                required
                autoComplete="address-level2"
              />
            </label>
            <label className={styles.formGroup}>
              Pays
              <input
                value={form.country_code}
                onChange={(e) => setField('country_code', e.target.value)}
                maxLength={2}
                required
              />
            </label>
          </div>
        </div>

        <div className={styles.splitCol}>
          <h3 className={styles.colTitle}>Paiement QR-Bill</h3>
          <label className={styles.formGroup}>
            IBAN
            <div className={styles.sensitiveField}>
              <input
                value={
                  revealIban ? formatIbanDisplay(form.iban) : maskIban(form.iban)
                }
                readOnly={!revealIban}
                onChange={(e) => setField('iban', e.target.value)}
                onFocus={() => setRevealIban(true)}
                placeholder="CH93 0076 2011 6238 5295 7"
                autoComplete="off"
                spellCheck={false}
              />
              <button
                type="button"
                className={styles.revealButton}
                onClick={() => setRevealIban((v) => !v)}
              >
                {revealIban ? 'Masquer' : 'Afficher'}
              </button>
            </div>
          </label>
          <label className={styles.formGroup}>
            QR-IBAN (si différent)
            <div className={styles.sensitiveField}>
              <input
                value={
                  revealQrIban
                    ? formatIbanDisplay(form.qr_iban)
                    : form.qr_iban
                      ? maskIban(form.qr_iban)
                      : ''
                }
                readOnly={!revealQrIban}
                onChange={(e) => setField('qr_iban', e.target.value)}
                onFocus={() => setRevealQrIban(true)}
                placeholder="Optionnel"
                autoComplete="off"
                spellCheck={false}
              />
              <button
                type="button"
                className={styles.revealButton}
                onClick={() => setRevealQrIban((v) => !v)}
              >
                {revealQrIban ? 'Masquer' : 'Afficher'}
              </button>
            </div>
          </label>
          <label className={styles.formGroup}>
            Mode de référence
            <select
              value={form.payment_reference_mode}
              onChange={(e) => setField('payment_reference_mode', e.target.value)}
            >
              <option value="QRR">QRR (QR-IBAN)</option>
              <option value="SCOR">SCOR</option>
              <option value="NON">NON (sans référence)</option>
            </select>
            <span className={styles.fieldHint}>
              IBAN classique → NON appliqué automatiquement.
            </span>
          </label>
          <div className={styles.formRow2}>
            <label className={styles.formGroup}>
              TVA (%)
              <input
                value={form.default_tax_rate}
                onChange={(e) => setField('default_tax_rate', e.target.value)}
                inputMode="decimal"
              />
              <span className={styles.fieldHint}>
                0 = franchise. 8.1 si assujetti.
              </span>
            </label>
            <label className={styles.formGroup}>
              Délai (jours)
              <input
                value={form.payment_terms_days_default}
                onChange={(e) => setField('payment_terms_days_default', e.target.value)}
                inputMode="numeric"
              />
            </label>
          </div>
        </div>

        <div className={styles.formFooter}>
          <button type="submit" className={styles.primaryButton} disabled={saving}>
            {saving ? 'Enregistrement…' : 'Enregistrer'}
          </button>
        </div>
      </form>
    </div>
  );
};

export default AdminPlatformCreditorSettings;
