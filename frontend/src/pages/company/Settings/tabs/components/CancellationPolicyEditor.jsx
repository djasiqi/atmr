import React, { useState, useCallback, useEffect } from 'react';
import { FiPlus, FiTrash2, FiClock, FiTruck } from 'react-icons/fi';
import styles from './CancellationPolicyEditor.module.css';

const DEFAULT_POLICY = {
  enabled: false,
  basis: 'booking_amount',
  apply_when_driver_assigned_only: true,
  tiers: [],
  min_fee_chf: 0,
  max_fee_chf: null,
  reason_overrides: {},
};

const REASON_OPTIONS = [
  { value: 'LAST_MINUTE', label: 'Annulation dernière minute' },
  { value: 'NO_SHOW', label: 'Client ne s\'est pas présenté' },
  { value: 'CLIENT_REQUEST', label: 'Client a demandé l\'annulation' },
  { value: 'COMPANY_ISSUE', label: 'Problème entreprise' },
  { value: 'MAJOR_DELAY', label: 'Retard important' },
  { value: 'VEHICLE_ISSUE', label: 'Problème véhicule' },
  { value: 'OTHER', label: 'Autre raison' },
];

const nextTierId = () => `t${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;

export default function CancellationPolicyEditor({ policy, onChange }) {
  const [data, setData] = useState(() => ({ ...DEFAULT_POLICY, ...(policy || {}) }));
  const [initialized, setInitialized] = useState(false);

  useEffect(() => {
    if (!initialized && policy) {
      setData({ ...DEFAULT_POLICY, ...policy });
      setInitialized(true);
    }
  }, [policy, initialized]);

  const commit = useCallback(
    (next) => {
      setData(next);
      onChange(next);
    },
    [onChange],
  );

  const update = (key, value) => commit({ ...data, [key]: value });

  const addTimeTier = () => {
    const tiers = [
      ...data.tiers,
      { id: nextTierId(), type: 'time', hours_before: 24, percent: 20, label: '' },
    ];
    commit({ ...data, tiers });
  };

  const addStatusTier = () => {
    const hasEnRoute = data.tiers.some((t) => t.type === 'status' && t.status === 'EN_ROUTE');
    if (hasEnRoute) return;
    const tiers = [
      ...data.tiers,
      { id: nextTierId(), type: 'status', status: 'EN_ROUTE', percent: 70, label: 'Chauffeur en route' },
    ];
    commit({ ...data, tiers });
  };

  const updateTier = (idx, field, value) => {
    const tiers = data.tiers.map((t, i) => (i === idx ? { ...t, [field]: value } : t));
    commit({ ...data, tiers });
  };

  const removeTier = (idx) => {
    commit({ ...data, tiers: data.tiers.filter((_, i) => i !== idx) });
  };

  const toggleOverride = (code, billable) => {
    const overrides = { ...data.reason_overrides };
    if (overrides[code] && overrides[code].billable === billable) {
      delete overrides[code];
    } else {
      overrides[code] = { billable };
    }
    commit({ ...data, reason_overrides: overrides });
  };

  const previewFee = (amount, percent) => {
    if (!amount || !percent) return '—';
    let fee = (amount * percent) / 100;
    if (data.min_fee_chf && fee < data.min_fee_chf) fee = data.min_fee_chf;
    if (data.max_fee_chf && fee > data.max_fee_chf) fee = data.max_fee_chf;
    return `${fee.toFixed(2)} CHF`;
  };

  const timeTiers = data.tiers.filter((t) => t.type === 'time');
  const statusTiers = data.tiers.filter((t) => t.type === 'status');

  return (
    <div className={styles.container}>
      <div className={styles.toggleRow}>
        <label className={styles.toggle}>
          <input
            type="checkbox"
            checked={data.enabled}
            onChange={(e) => update('enabled', e.target.checked)}
          />
          <span>Activer les frais d'annulation paramétrables</span>
        </label>
      </div>

      {data.enabled && (
        <>
          <div className={styles.optionRow}>
            <label className={styles.toggle}>
              <input
                type="checkbox"
                checked={data.apply_when_driver_assigned_only}
                onChange={(e) => update('apply_when_driver_assigned_only', e.target.checked)}
              />
              <span>Appliquer uniquement si un chauffeur est assigné</span>
            </label>
          </div>

          <div className={styles.minMaxRow}>
            <div className={styles.field}>
              <label>Min. frais (CHF)</label>
              <input
                type="number"
                min="0"
                step="0.05"
                value={data.min_fee_chf ?? ''}
                onChange={(e) => update('min_fee_chf', e.target.value ? parseFloat(e.target.value) : 0)}
                onBlur={() => onChange(data)}
              />
            </div>
            <div className={styles.field}>
              <label>Max. frais (CHF)</label>
              <input
                type="number"
                min="0"
                step="0.05"
                value={data.max_fee_chf ?? ''}
                placeholder="Illimité"
                onChange={(e) => update('max_fee_chf', e.target.value ? parseFloat(e.target.value) : null)}
                onBlur={() => onChange(data)}
              />
            </div>
          </div>

          {/* Paliers temps */}
          <div className={styles.tiersSection}>
            <h4 className={styles.tierTitle}>
              <FiClock size={14} /> Paliers temps
            </h4>
            {timeTiers.length === 0 && (
              <p className={styles.emptyHint}>Aucun palier temps configuré</p>
            )}
            {timeTiers.map((tier) => {
              const idx = data.tiers.indexOf(tier);
              return (
                <div key={tier.id} className={styles.tierRow}>
                  <div className={styles.tierField}>
                    <label>Seuil (heures)</label>
                    <input
                      type="number"
                      min="1"
                      step="1"
                      value={tier.hours_before}
                      onChange={(e) => updateTier(idx, 'hours_before', parseInt(e.target.value, 10) || 1)}
                    />
                  </div>
                  <div className={styles.tierField}>
                    <label>Pourcentage (%)</label>
                    <input
                      type="number"
                      min="0"
                      max="100"
                      value={tier.percent}
                      onChange={(e) => updateTier(idx, 'percent', parseInt(e.target.value, 10) || 0)}
                    />
                  </div>
                  <div className={styles.tierField}>
                    <label>Label</label>
                    <input
                      type="text"
                      placeholder={`< ${tier.hours_before}h`}
                      value={tier.label || ''}
                      onChange={(e) => updateTier(idx, 'label', e.target.value)}
                    />
                  </div>
                  <div className={styles.tierPreview}>
                    Ex: {previewFee(100, tier.percent)} sur 100 CHF
                  </div>
                  <button
                    type="button"
                    className={styles.removeBtn}
                    onClick={() => removeTier(idx)}
                    title="Supprimer"
                  >
                    <FiTrash2 size={14} />
                  </button>
                </div>
              );
            })}
            <button type="button" className={styles.addBtn} onClick={addTimeTier}>
              <FiPlus size={14} /> Ajouter un palier temps
            </button>
          </div>

          {/* Paliers statut */}
          <div className={styles.tiersSection}>
            <h4 className={styles.tierTitle}>
              <FiTruck size={14} /> Paliers statut
            </h4>
            {statusTiers.length === 0 && (
              <p className={styles.emptyHint}>Aucun palier statut configuré</p>
            )}
            {statusTiers.map((tier) => {
              const idx = data.tiers.indexOf(tier);
              return (
                <div key={tier.id} className={styles.tierRow}>
                  <div className={styles.tierField}>
                    <label>Statut</label>
                    <select
                      value={tier.status || 'EN_ROUTE'}
                      onChange={(e) => updateTier(idx, 'status', e.target.value)}
                    >
                      <option value="EN_ROUTE">Chauffeur en route</option>
                    </select>
                  </div>
                  <div className={styles.tierField}>
                    <label>Pourcentage (%)</label>
                    <input
                      type="number"
                      min="0"
                      max="100"
                      value={tier.percent}
                      onChange={(e) => updateTier(idx, 'percent', parseInt(e.target.value, 10) || 0)}
                    />
                  </div>
                  <div className={styles.tierPreview}>
                    Ex: {previewFee(100, tier.percent)} sur 100 CHF
                  </div>
                  <button
                    type="button"
                    className={styles.removeBtn}
                    onClick={() => removeTier(idx)}
                    title="Supprimer"
                  >
                    <FiTrash2 size={14} />
                  </button>
                </div>
              );
            })}
            {!statusTiers.some((t) => t.status === 'EN_ROUTE') && (
              <button type="button" className={styles.addBtn} onClick={addStatusTier}>
                <FiPlus size={14} /> Ajouter palier "Chauffeur en route"
              </button>
            )}
          </div>

          {/* Exceptions par motif */}
          <div className={styles.tiersSection}>
            <h4 className={styles.tierTitle}>Exceptions par motif</h4>
            <p className={styles.emptyHint}>
              Forcer un motif comme facturable ou non-facturable, indépendamment des paliers.
            </p>
            <div className={styles.overridesGrid}>
              {REASON_OPTIONS.map((r) => {
                const override = data.reason_overrides[r.value];
                return (
                  <div key={r.value} className={styles.overrideRow}>
                    <span className={styles.overrideLabel}>{r.label}</span>
                    <div className={styles.overrideBtns}>
                      <button
                        type="button"
                        className={`${styles.overrideBtn} ${override?.billable === false ? styles.activeNo : ''}`}
                        onClick={() => toggleOverride(r.value, false)}
                        title="Forcer non-facturable"
                      >
                        Non fact.
                      </button>
                      <button
                        type="button"
                        className={`${styles.overrideBtn} ${override?.billable === true ? styles.activeYes : ''}`}
                        onClick={() => toggleOverride(r.value, true)}
                        title="Forcer facturable"
                      >
                        Facturable
                      </button>
                      {!override && <span className={styles.overrideDefault}>Par défaut</span>}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
