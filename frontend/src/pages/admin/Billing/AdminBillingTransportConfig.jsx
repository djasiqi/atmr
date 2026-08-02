import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  fetchPlatformBillingCompaniesConfig,
  fetchPlatformBillingFeatureFlags,
  fetchPlatformSubscriptionPricing,
  putPlatformBillingCompanyConfig,
} from '../../../services/adminService';
import AdminBillingDualProductConfig from './AdminBillingDualProductConfig';
import styles from './AdminBillingTransportConfig.module.css';

const DISPATCH_OPTIONS = [
  { value: '', label: '(défaut moteur)' },
  { value: 'manual', label: 'manual' },
  { value: 'semi_auto', label: 'semi_auto' },
  { value: 'fully_auto', label: 'fully_auto' },
];

const fmtMoney = (n) => {
  if (n == null || Number.isNaN(Number(n))) return '—';
  return `${Number(n).toLocaleString('fr-CH', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} CHF`;
};

const toDatetimeLocal = (iso) => {
  if (!iso) return '';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '';
  const pad = (x) => String(x).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
};

const fromDatetimeLocal = (v) => {
  if (!v || !v.trim()) return null;
  const d = new Date(v);
  return Number.isNaN(d.getTime()) ? null : d.toISOString();
};

const emptyForm = {
  is_billing_enabled: false,
  dispatch_mode_override: '',
  commission_rate: '',
  support_hourly_rate_default: '',
  effective_from: '',
  effective_to: '',
  is_active: true,
  notes: '',
};

/** UI legacy V1 — hooks toujours appelés dans ce composant uniquement. */
const AdminBillingTransportConfigLegacy = () => {
  const [items, setItems] = useState([]);
  const [pricing, setPricing] = useState([]);
  const [search, setSearch] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [modalCompany, setModalCompany] = useState(null);
  const [form, setForm] = useState(emptyForm);
  const [saving, setSaving] = useState(false);
  const [modalError, setModalError] = useState(null);

  useEffect(() => {
    const t = setTimeout(() => setDebouncedSearch(search.trim()), 400);
    return () => clearTimeout(t);
  }, [search]);

  const loadCompanies = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const listRes = await fetchPlatformBillingCompaniesConfig({
        q: debouncedSearch || undefined,
      });
      setItems(listRes?.items || []);
    } catch (e) {
      setError(e?.response?.data?.message || e?.message || 'Erreur chargement');
      setItems([]);
    } finally {
      setLoading(false);
    }
  }, [debouncedSearch]);

  useEffect(() => {
    loadCompanies();
  }, [loadCompanies]);

  const loadPricing = useCallback(async () => {
    try {
      const priceRes = await fetchPlatformSubscriptionPricing();
      setPricing(priceRes?.items || []);
    } catch {
      setPricing([]);
    }
  }, []);

  useEffect(() => {
    loadPricing();
  }, [loadPricing]);

  const openEdit = (row) => {
    const c = row.config;
    setModalCompany(row);
    setModalError(null);
    if (!c) {
      setForm({
        ...emptyForm,
        is_billing_enabled: false,
      });
      return;
    }
    setForm({
      is_billing_enabled: !!c.is_billing_enabled,
      dispatch_mode_override: c.dispatch_mode_override || '',
      commission_rate: c.commission_rate != null ? String(c.commission_rate) : '',
      support_hourly_rate_default:
        c.support_hourly_rate_default != null ? String(c.support_hourly_rate_default) : '',
      effective_from: toDatetimeLocal(c.effective_from),
      effective_to: toDatetimeLocal(c.effective_to),
      is_active: c.is_active !== false,
      notes: c.notes || '',
    });
  };

  const onSave = async () => {
    if (!modalCompany) return;
    setSaving(true);
    setModalError(null);
    try {
      const payload = {
        is_billing_enabled: form.is_billing_enabled,
        is_active: form.is_active,
        dispatch_mode_override: form.dispatch_mode_override || null,
        commission_rate:
          form.commission_rate === '' ? null : Number(form.commission_rate.replace(',', '.')),
        support_hourly_rate_default:
          form.support_hourly_rate_default === ''
            ? null
            : Number(form.support_hourly_rate_default.replace(',', '.')),
        effective_from: fromDatetimeLocal(form.effective_from),
        effective_to: fromDatetimeLocal(form.effective_to),
        notes: form.notes || null,
      };
      await putPlatformBillingCompanyConfig(modalCompany.company_id, payload);
      setModalCompany(null);
      await loadCompanies();
    } catch (e) {
      setModalError(e?.response?.data?.message || e?.message || 'Enregistrement impossible');
    } finally {
      setSaving(false);
    }
  };

  const pricingByMode = useMemo(() => {
    const m = {};
    pricing.forEach((r) => {
      if (!m[r.dispatch_mode]) m[r.dispatch_mode] = [];
      m[r.dispatch_mode].push(r);
    });
    return m;
  }, [pricing]);

  return (
    <div className={`${styles.layout} ${styles.layoutWithAside}`}>
      <div>
        <p className={styles.lead}>
          Activez la facturation plateforme par transporteur et renseignez commission, override de mode
          dispatch, tarif support et fenêtre de validité. Sans{' '}
          <strong>facturation activée</strong>, aucun relevé n’est généré pour l’entreprise.
        </p>
        <div className={styles.hint}>
          La grille d’abonnement (paliers par volume et mode dispatch) est globale — récapitulée à
          droite. Les montants d’abonnement sur les relevés sont calculés à partir du mode effectif et
          du volume de la période.
        </div>

        {error ? <div className={styles.errorBanner}>{error}</div> : null}

        <div className={styles.toolbar}>
          <input
            type="search"
            className={styles.search}
            placeholder="Filtrer par nom d’entreprise…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            aria-label="Filtrer"
          />
          <button type="button" className={styles.btn} onClick={() => loadCompanies()} disabled={loading}>
            Actualiser
          </button>
        </div>

        {loading && items.length === 0 ? (
          <p className={styles.muted}>Chargement…</p>
        ) : (
          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>Entreprise</th>
                  <th>Facturation</th>
                  <th>Commission</th>
                  <th>Dispatch</th>
                  <th>Support / h</th>
                  <th>Valide du → au</th>
                  <th>Actif</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {items.map((row) => {
                  const c = row.config;
                  return (
                    <tr key={row.company_id}>
                      <td>
                        <strong>{row.company_name}</strong>
                        <div className={styles.mono}>id {row.company_id}</div>
                      </td>
                      <td>
                        {c?.is_billing_enabled ? (
                          <span className={styles.badgeOn}>Activée</span>
                        ) : (
                          <span className={styles.badgeOff}>Désactivée</span>
                        )}
                      </td>
                      <td className={styles.mono}>
                        {c?.commission_rate != null ? `${Number(c.commission_rate) * 100} %` : '—'}
                      </td>
                      <td className={styles.mono}>{c?.dispatch_mode_override || '—'}</td>
                      <td>{c?.support_hourly_rate_default != null ? fmtMoney(c.support_hourly_rate_default) : '—'}</td>
                      <td className={styles.muted}>
                        {c?.effective_from
                          ? new Date(c.effective_from).toLocaleString('fr-CH')
                          : '—'}
                        <br />
                        {c?.effective_to ? new Date(c.effective_to).toLocaleString('fr-CH') : '—'}
                      </td>
                      <td>{c ? (c.is_active ? 'oui' : 'non') : '—'}</td>
                      <td>
                        <button type="button" className={styles.btn} onClick={() => openEdit(row)}>
                          {c ? 'Modifier' : 'Configurer'}
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <aside className={styles.aside}>
        <h2 className={styles.asideTitle}>Grille abonnement (globale)</h2>
        <p className={styles.asideLead}>Paliers par mode dispatch — lecture seule.</p>
        {['manual', 'semi_auto', 'fully_auto'].map((mode) => (
          <div key={mode} style={{ marginBottom: '1rem' }}>
            <div className={styles.mono} style={{ fontWeight: 600, marginBottom: '0.35rem' }}>
              {mode}
            </div>
            <table className={styles.pricingTable}>
              <thead>
                <tr>
                  <th>Vol.</th>
                  <th>Prix / mois</th>
                </tr>
              </thead>
              <tbody>
                {(pricingByMode[mode] || []).map((r) => (
                  <tr key={r.id}>
                    <td>
                      {r.volume_min}
                      {r.volume_max != null ? `–${r.volume_max}` : '+'}
                    </td>
                    <td>{fmtMoney(r.price_monthly)}</td>
                  </tr>
                ))}
                {!(pricingByMode[mode] || []).length ? (
                  <tr>
                    <td colSpan={2} className={styles.muted}>
                      Aucune ligne
                    </td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          </div>
        ))}
      </aside>

      {modalCompany ? (
        <div
          className={styles.modalOverlay}
          role="presentation"
          onClick={() => !saving && setModalCompany(null)}
        >
          <div
            className={styles.modal}
            role="dialog"
            aria-modal="true"
            aria-labelledby="cfg-modal-title"
            onClick={(e) => e.stopPropagation()}
          >
            <h2 id="cfg-modal-title">
              Paramètres — {modalCompany.company_name}{' '}
              <span className={styles.muted}>(#{modalCompany.company_id})</span>
            </h2>
            {modalError ? <div className={styles.errorBanner}>{modalError}</div> : null}
            <div className={styles.formGrid}>
              <label>
                <input
                  type="checkbox"
                  checked={form.is_billing_enabled}
                  onChange={(e) => setForm((f) => ({ ...f, is_billing_enabled: e.target.checked }))}
                />{' '}
                Facturation plateforme activée
              </label>
              <label>
                Commission (décimal, ex. 0.05 pour 5 %)
                <input
                  type="text"
                  inputMode="decimal"
                  value={form.commission_rate}
                  onChange={(e) => setForm((f) => ({ ...f, commission_rate: e.target.value }))}
                  placeholder="ex. 0.05"
                />
              </label>
              <label>
                Override mode dispatch
                <select
                  value={form.dispatch_mode_override}
                  onChange={(e) => setForm((f) => ({ ...f, dispatch_mode_override: e.target.value }))}
                >
                  {DISPATCH_OPTIONS.map((o) => (
                    <option key={o.value || 'empty'} value={o.value}>
                      {o.label}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Tarif support horaire par défaut (CHF)
                <input
                  type="text"
                  inputMode="decimal"
                  value={form.support_hourly_rate_default}
                  onChange={(e) =>
                    setForm((f) => ({ ...f, support_hourly_rate_default: e.target.value }))
                  }
                  placeholder="ex. 180"
                />
              </label>
              <label>
                Valide du (optionnel)
                <input
                  type="datetime-local"
                  value={form.effective_from}
                  onChange={(e) => setForm((f) => ({ ...f, effective_from: e.target.value }))}
                />
              </label>
              <label>
                Valide au (optionnel)
                <input
                  type="datetime-local"
                  value={form.effective_to}
                  onChange={(e) => setForm((f) => ({ ...f, effective_to: e.target.value }))}
                />
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={form.is_active}
                  onChange={(e) => setForm((f) => ({ ...f, is_active: e.target.checked }))}
                />{' '}
                Ligne de config active
              </label>
              <label>
                Notes internes
                <textarea
                  value={form.notes}
                  onChange={(e) => setForm((f) => ({ ...f, notes: e.target.value }))}
                />
              </label>
            </div>
            <div className={styles.modalActions}>
              <button
                type="button"
                className={styles.btn}
                disabled={saving}
                onClick={() => setModalCompany(null)}
              >
                Annuler
              </button>
              <button type="button" className={`${styles.btn} ${styles.btnPrimary}`} disabled={saving} onClick={onSave}>
                {saving ? '…' : 'Enregistrer'}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
};

/**
 * Routeur config : feature flag avant tout rendu dual/legacy.
 * Les hooks de chaque UI restent dans leur composant (pas de return anticipé
 * au milieu d'une liste de hooks).
 */
const AdminBillingTransportConfig = () => {
  // Nouvelle UI dual-produit par défaut ; legacy uniquement si flag explicitement false.
  const [dualUi, setDualUi] = useState(true);
  const [flagLoaded, setFlagLoaded] = useState(false);

  useEffect(() => {
    fetchPlatformBillingFeatureFlags()
      .then((f) => {
        const raw = f?.PLATFORM_BILLING_DUAL_PRODUCT_CONFIG_UI;
        setDualUi(raw === undefined ? true : Boolean(raw));
      })
      .catch(() => setDualUi(true))
      .finally(() => setFlagLoaded(true));
  }, []);

  if (!flagLoaded) {
    return <p className={styles.muted}>Chargement…</p>;
  }
  if (dualUi) {
    return <AdminBillingDualProductConfig />;
  }
  return <AdminBillingTransportConfigLegacy />;
};

export default AdminBillingTransportConfig;
