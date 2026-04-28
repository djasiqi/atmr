import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { fetchAdminIndicativeFare, updateAdminIndicativeFare } from '../../services/settingsService';
import { computeIndicativeFromConfigChf } from '../../utils/indicativeFarePreview';
import styles from '../../pages/admin/Settings/AdminSettings.module.css';

const defaultSim = { distance_km: '13.5', duration_min: '20' };

/**
 * Bloc autonome : GET/PUT /admin/client-indicative-fare, aperçu per_km, simulation distance/durée.
 */
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
  const [audit, setAudit] = useState({ config_version: null, updated_at: null, updated_by_user_id: null });
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
    return (Number(form.min_fare_chf) - Number(form.base_chf) - Number(form.ref_min) * Number(form.per_minute_chf)) / refK;
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
      setSuccess('Configuration enregistrée. La version a été incrémentée.');
    } catch (e) {
      const msg = e?.response?.data?.message || e?.response?.data?.error || e?.message;
      setError(String(msg || 'Enregistrement refusé.'));
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <section className={styles.card} style={{ marginBottom: '1.5rem' }}>
        <h2>Indicatif portail client</h2>
        <p>Chargement…</p>
      </section>
    );
  }

  return (
    <section className={styles.card} style={{ marginBottom: '1.5rem' }}>
      <h2>Indicatif portail client (calibration)</h2>
      <p className={styles.helperText}>
        Règle serveur unique (non confondue avec la prévisualisation de réservation / compute_price). Le
        <code> per_km</code> affiché est dérivé pour respecter l’ancrage à ref_km / ref_min.
      </p>
      {error ? <div className={styles.error}>{error}</div> : null}
      {success ? <div className={styles.success}>{success}</div> : null}
      <div className={styles.previewBox}>
        <p>
          <strong>Version config</strong> : {audit.config_version != null ? audit.config_version : '—'}
        </p>
        {audit.updated_at ? (
          <p>
            <strong>Dernière MAJ</strong> : {audit.updated_at} — éditeur (user_id){' '}
            {audit.updated_by_user_id != null ? audit.updated_by_user_id : '—'}
          </p>
        ) : null}
      </div>
      <label className={styles.formRow} style={{ marginTop: '0.5rem', display: 'flex', alignItems: 'center', gap: 8 }}>
        <input
          type="checkbox"
          checked={form.is_enabled}
          onChange={(e) => setForm((p) => ({ ...p, is_enabled: e.target.checked }))}
        />
        <span>Indicatif activé (sinon 412 côté API estimate)</span>
      </label>
      <div className={styles.formRow} style={{ display: 'grid', gap: 8, marginTop: 8 }}>
        {['min_fare_chf', 'base_chf', 'per_minute_chf', 'ref_km', 'ref_min'].map((k) => (
          <label key={k} style={{ display: 'flex', flexDirection: 'column', maxWidth: 360 }}>
            <span>{k}</span>
            <input
              type="number"
              step="0.0001"
              value={form[k]}
              onChange={(e) => {
                const v = e.target.value;
                setForm((p) => ({ ...p, [k]: v === '' ? '' : Number(v) }));
              }}
            />
          </label>
        ))}
        <p>
          <strong>per_km dérivé (lecture seule)</strong> : {derivedPerKm == null || Number.isNaN(derivedPerKm) ? '—' : derivedPerKm.toFixed(4)} CHF/km
        </p>
        <label style={{ display: 'flex', flexDirection: 'column', maxWidth: 480 }}>
          <span>Note de calibration (interne)</span>
          <textarea
            rows={3}
            value={form.calibration_note}
            onChange={(e) => setForm((p) => ({ ...p, calibration_note: e.target.value }))}
          />
        </label>
      </div>
      <h3 style={{ marginTop: 16 }}>Trajet de référence simulé (aperçu)</h3>
      <p className={styles.helperText}>
        Saisis une distance (km) et une durée (min) : montant indicatif théorique (un trajet simple, sans
        A/R ni récurrence).
      </p>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
        <label>
          Distance (km)
          <input
            value={sim.distance_km}
            onChange={(e) => setSim((s) => ({ ...s, distance_km: e.target.value }))}
            style={{ marginLeft: 8, width: 100 }}
          />
        </label>
        <label>
          Durée (min)
          <input
            value={sim.duration_min}
            onChange={(e) => setSim((s) => ({ ...s, duration_min: e.target.value }))}
            style={{ marginLeft: 8, width: 80 }}
          />
        </label>
      </div>
      <p>
        <strong>Indicatif simulé</strong> : {previewAmount == null ? '—' : `${previewAmount.toFixed(2)} CHF`}
      </p>
      <div style={{ marginTop: 12 }}>
        <button type="button" className={styles.primaryButton} onClick={onSave} disabled={saving}>
          {saving ? 'Enregistrement…' : 'Enregistrer'}
        </button>
      </div>
    </section>
  );
}

export default IndicativeFareAdminSection;
