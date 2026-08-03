import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { fetchAdminIndicativeFare, updateAdminIndicativeFare } from '../../services/settingsService';
import { computeIndicativeFromConfigChf } from '../../utils/indicativeFarePreview';
import styles from '../../pages/admin/Settings/AdminSettings.module.css';

const defaultSim = { distance_km: '13.5', duration_min: '20' };

const PARAMS = [
  { key: 'min_fare_chf', label: 'Minimum', suffix: 'CHF' },
  { key: 'base_chf', label: 'Base', suffix: 'CHF' },
  { key: 'per_minute_chf', label: 'Par minute', suffix: 'CHF' },
  { key: 'ref_km', label: 'Réf. distance', suffix: 'km' },
  { key: 'ref_min', label: 'Réf. durée', suffix: 'min' },
];

function IndicativeFareAdminSection() {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const [form, setForm] = useState({
    is_enabled: true,
    min_fare_chf: 45,
    base_chf: 18,
    per_minute_chf: 0.35,
    ref_km: 13.5,
    ref_min: 20,
    calibration_note: '',
  });
  const [audit, setAudit] = useState({
    config_version: null,
    updated_at: null,
    updated_by_user_id: null,
  });
  const [sim, setSim] = useState(defaultSim);

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const data = await fetchAdminIndicativeFare();
      setForm((prev) => ({
        ...prev,
        is_enabled: Boolean(data.is_enabled),
        min_fare_chf: Number(data.min_fare_chf),
        base_chf: Number(data.base_chf),
        per_minute_chf: Number(data.per_minute_chf),
        ref_km: Number(data.ref_km),
        ref_min: Number(data.ref_min),
        calibration_note: data.calibration_note || '',
      }));
      setAudit({
        config_version: data.config_version,
        updated_at: data.updated_at,
        updated_by_user_id: data.updated_by_user_id,
      });
    } catch (e) {
      setError(
        e?.response?.data?.message ||
          e?.response?.data?.error ||
          e?.message ||
          'Chargement impossible.'
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const derivedPerKm = useMemo(() => {
    const refK = Number(form.ref_km);
    if (!Number.isFinite(refK) || refK <= 0) return null;
    return (
      (Number(form.min_fare_chf) -
        Number(form.base_chf) -
        Number(form.ref_min) * Number(form.per_minute_chf)) /
      refK
    );
  }, [form.min_fare_chf, form.base_chf, form.per_minute_chf, form.ref_km, form.ref_min]);

  const previewAmount = useMemo(() => {
    const dk = parseFloat(String(sim.distance_km).replace(',', '.'));
    const dm = Number.isFinite(dk) && dk > 0 ? dk * 1000 : 0;
    const mins = parseFloat(String(sim.duration_min).replace(',', '.'));
    const durationS = Number.isFinite(mins) ? Math.max(0, Math.round(mins * 60)) : 0;
    return computeIndicativeFromConfigChf(
      {
        min_fare_chf: form.min_fare_chf,
        base_chf: form.base_chf,
        per_minute_chf: form.per_minute_chf,
        ref_km: form.ref_km,
        ref_min: form.ref_min,
      },
      dm,
      durationS
    );
  }, [form, sim.distance_km, sim.duration_min]);

  const updatedAtLabel = useMemo(() => {
    if (!audit.updated_at) return null;
    try {
      return new Date(audit.updated_at).toLocaleString('fr-CH', {
        dateStyle: 'medium',
        timeStyle: 'short',
      });
    } catch {
      return audit.updated_at;
    }
  }, [audit.updated_at]);

  const onSave = async () => {
    setSaving(true);
    setError('');
    setSuccess('');
    try {
      const out = await updateAdminIndicativeFare({
        is_enabled: form.is_enabled,
        min_fare_chf: form.min_fare_chf,
        base_chf: form.base_chf,
        per_minute_chf: form.per_minute_chf,
        ref_km: form.ref_km,
        ref_min: form.ref_min,
        calibration_note: form.calibration_note,
      });
      setAudit({
        config_version: out.config_version,
        updated_at: out.updated_at,
        updated_by_user_id: out.updated_by_user_id,
      });
      setSuccess('Configuration enregistrée.');
    } catch (e) {
      const msg = e?.response?.data?.message || e?.response?.data?.error || e?.message;
      setError(String(msg || 'Enregistrement refusé.'));
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className={styles.panelBody}>
        <p className={styles.helperText}>Chargement de l’indicatif…</p>
      </div>
    );
  }

  return (
    <div className={styles.panelBody}>
      <div className={styles.panelIntroRow}>
        <div className={styles.panelIntro}>
          <h2 className={styles.panelTitle}>Indicatif portail client</h2>
          <p className={styles.panelLead}>
            Estimation affichée aux clients — distincte de <code>compute_price</code>.
          </p>
        </div>
        <div className={styles.statusRow}>
          <span
            className={`${styles.statusChip} ${
              form.is_enabled ? styles.statusOk : styles.statusWarn
            }`}
          >
            {form.is_enabled ? 'Activé' : 'Désactivé'}
          </span>
          <span className={`${styles.statusChip} ${styles.statusMuted}`}>
            v{audit.config_version != null ? audit.config_version : '—'}
          </span>
        </div>
      </div>

      {error ? <div className={styles.error}>{error}</div> : null}
      {success ? <div className={styles.success}>{success}</div> : null}

      <div className={styles.splitForm}>
        <div className={styles.splitCol}>
          <label className={styles.toggleRow}>
            <input
              type="checkbox"
              checked={form.is_enabled}
              onChange={(e) => setForm((p) => ({ ...p, is_enabled: e.target.checked }))}
            />
            <span>Indicatif actif (sinon estimate → 412)</span>
          </label>

          <div className={styles.paramGrid}>
            {PARAMS.map(({ key, label, suffix }) => (
              <label key={key} className={styles.formGroup}>
                {label}
                <div className={styles.inputWithSuffix}>
                  <input
                    type="number"
                    step="0.0001"
                    value={form[key]}
                    onChange={(e) => {
                      const v = e.target.value;
                      setForm((p) => ({ ...p, [key]: v === '' ? '' : Number(v) }));
                    }}
                  />
                  <span className={styles.inputSuffix}>{suffix}</span>
                </div>
              </label>
            ))}
          </div>

          <div className={styles.derivedValue}>
            <span>Tarif au km (dérivé)</span>
            <strong>
              {derivedPerKm == null || Number.isNaN(derivedPerKm)
                ? '—'
                : `${derivedPerKm.toFixed(4)} CHF/km`}
            </strong>
          </div>

          <label className={styles.formGroup}>
            Note interne
            <textarea
              rows={3}
              value={form.calibration_note}
              onChange={(e) => setForm((p) => ({ ...p, calibration_note: e.target.value }))}
              placeholder="Contexte de calibration…"
            />
          </label>

          {updatedAtLabel ? (
            <p className={styles.metaLine}>
              Dernière mise à jour : {updatedAtLabel}
              {audit.updated_by_user_id != null
                ? ` · #${audit.updated_by_user_id}`
                : ''}
            </p>
          ) : null}
        </div>

        <div className={`${styles.splitCol} ${styles.simCard}`}>
          <h3 className={styles.colTitle}>Simulation</h3>
          <p className={styles.helperText}>
            Trajet simple théorique (sans A/R ni récurrence).
          </p>
          <div className={styles.formRow2}>
            <label className={styles.formGroup}>
              Distance
              <div className={styles.inputWithSuffix}>
                <input
                  value={sim.distance_km}
                  onChange={(e) => setSim((s) => ({ ...s, distance_km: e.target.value }))}
                />
                <span className={styles.inputSuffix}>km</span>
              </div>
            </label>
            <label className={styles.formGroup}>
              Durée
              <div className={styles.inputWithSuffix}>
                <input
                  value={sim.duration_min}
                  onChange={(e) => setSim((s) => ({ ...s, duration_min: e.target.value }))}
                />
                <span className={styles.inputSuffix}>min</span>
              </div>
            </label>
          </div>
          <div className={styles.simAmount}>
            <span>Indicatif</span>
            <strong>
              {previewAmount == null ? '—' : `${previewAmount.toFixed(2)} CHF`}
            </strong>
          </div>
        </div>

        <div className={styles.formFooter}>
          <button
            type="button"
            className={styles.primaryButton}
            onClick={onSave}
            disabled={saving}
          >
            {saving ? 'Enregistrement…' : 'Enregistrer'}
          </button>
        </div>
      </div>
    </div>
  );
}

export default IndicativeFareAdminSection;
