import React, { useCallback, useEffect, useState } from 'react';
import {
  fetchPlatformBillingCreditor,
  putPlatformBillingCreditor,
} from '../../../services/adminService';
import styles from './AdminSettings.module.css';

const emptyForm = {
  legal_name: 'LIRIE',
  street_name: '',
  building_number: '',
  postal_code: '',
  city: '',
  country_code: 'CH',
  uid_ide: '',
  vat_number: '',
  legal_form: 'sole_proprietorship',
  signatory_name: '',
  signatory_title: '',
  default_tax_rate: '0',
  iban: '',
  qr_iban: '',
  payment_reference_mode: 'QRR',
  payment_terms_days_default: '30',
  is_active: true,
};

const AdminPlatformCreditorSettings = () => {
  const [form, setForm] = useState(emptyForm);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const res = await fetchPlatformBillingCreditor();
      const c = res?.creditor;
      if (c) {
        setForm({
          legal_name: c.legal_name || 'LIRIE',
          street_name: c.street_name || '',
          building_number: c.building_number || '',
          postal_code: c.postal_code || '',
          city: c.city || '',
          country_code: c.country_code || 'CH',
          uid_ide: c.uid_ide || '',
          vat_number: c.vat_number || '',
          legal_form: c.legal_form || 'sole_proprietorship',
          signatory_name: c.signatory_name || '',
          signatory_title: c.signatory_title || '',
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
      setSuccess('Créancier LIRIE enregistré — prêt pour les QR-factures plateforme.');
      await load();
    } catch (err) {
      setError(err?.response?.data?.message || err?.message || 'Erreur enregistrement');
    } finally {
      setSaving(false);
    }
  };

  return (
    <section className={styles.cardWide} aria-labelledby="creditor-settings-title">
      <h2 id="creditor-settings-title">Facturation plateforme LIRIE (créancier)</h2>
      <p className={styles.helperText}>
        Adresse de domicile et IBAN utilisés sur les <strong>QR-factures</strong> émises par LIRIE
        vers les transporteurs. Sans ces informations, l’émission PDF/QR reste bloquée.
      </p>

      {error ? <div className={styles.error}>{error}</div> : null}
      {success ? <div className={styles.success}>{success}</div> : null}

      {loading ? (
        <p className={styles.helperText}>Chargement…</p>
      ) : (
        <form onSubmit={onSave}>
          <div className={styles.formRow}>
            <label className={styles.formGroup}>
              Raison sociale
              <input
                value={form.legal_name}
                onChange={(e) => setField('legal_name', e.target.value)}
                required
                autoComplete="organization"
              />
            </label>
            <label className={styles.formGroup}>
              IDE / UID (optionnel)
              <input
                value={form.uid_ide}
                onChange={(e) => setField('uid_ide', e.target.value)}
                placeholder="CHE-123.456.789"
              />
            </label>
          </div>
          <div className={styles.formRow}>
            <label className={styles.formGroup}>
              Forme juridique
              <select
                value={form.legal_form}
                onChange={(e) => setField('legal_form', e.target.value)}
              >
                <option value="sole_proprietorship">Indépendant</option>
                <option value="sarl">Sàrl</option>
                <option value="sa">SA</option>
                <option value="association">Association</option>
                <option value="foundation">Fondation</option>
                <option value="other">Autre</option>
              </select>
            </label>
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

          <h3 className={styles.subSectionTitle}>Adresse de domicile</h3>
          <div className={styles.formRow}>
            <label className={`${styles.formGroup} ${styles.formGroupGrow}`}>
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
          <div className={styles.formRow}>
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

          <h3 className={styles.subSectionTitle}>Paiement QR-Bill</h3>
          <div className={styles.formRow}>
            <label className={styles.formGroup}>
              IBAN
              <input
                value={form.iban}
                onChange={(e) => setField('iban', e.target.value)}
                placeholder="CH93 0076 2011 6238 5295 7"
                autoComplete="off"
              />
            </label>
            <label className={styles.formGroup}>
              QR-IBAN (si différent)
              <input
                value={form.qr_iban}
                onChange={(e) => setField('qr_iban', e.target.value)}
                placeholder="Optionnel si IBAN renseigné"
                autoComplete="off"
              />
            </label>
          </div>
          <div className={styles.formRow}>
            <label className={styles.formGroup}>
              Mode de référence
              <select
                value={form.payment_reference_mode}
                onChange={(e) => setField('payment_reference_mode', e.target.value)}
              >
                <option value="QRR">QRR (nécessite un QR-IBAN)</option>
                <option value="SCOR">SCOR (référence créancier)</option>
                <option value="NON">NON (sans référence)</option>
              </select>
              <small className={styles.helperText}>
                Avec un IBAN classique (pas QR-IBAN), le système utilise automatiquement NON.
              </small>
            </label>
            <label className={styles.formGroup}>
              Taux TVA par défaut (%)
              <input
                value={form.default_tax_rate}
                onChange={(e) => setField('default_tax_rate', e.target.value)}
                inputMode="decimal"
              />
              <small className={styles.helperText}>
                0 = non assujetti / franchise suisse (&lt; 100&apos;000 CHF de chiffre
                d&apos;affaires). Mettre 8.1 lorsque LIRIE sera assujettie à la TVA.
              </small>
            </label>
            <label className={styles.formGroup}>
              Délai de paiement (jours)
              <input
                value={form.payment_terms_days_default}
                onChange={(e) => setField('payment_terms_days_default', e.target.value)}
                inputMode="numeric"
              />
            </label>
          </div>

          <div className={styles.advancedActions}>
            <button type="submit" className={styles.primaryButton} disabled={saving}>
              {saving ? 'Enregistrement…' : 'Enregistrer le créancier LIRIE'}
            </button>
          </div>
        </form>
      )}
    </section>
  );
};

export default AdminPlatformCreditorSettings;
