import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import {
  createPlatformBillingPeriod,
  fetchPlatformBillingCompaniesConfig,
  fetchPlatformBillingPeriodInvoices,
  fetchPlatformBillingPeriods,
  recalculatePlatformBillingPeriod,
} from '../../../services/adminService';
import styles from './AdminBillingOverview.module.css';
import { adminPaths } from '../routing/adminRoutePaths';

const MONTHS_FR = [
  'janvier',
  'février',
  'mars',
  'avril',
  'mai',
  'juin',
  'juillet',
  'août',
  'septembre',
  'octobre',
  'novembre',
  'décembre',
];

const fmtMoney = (n) => {
  if (n == null || n === '') return '—';
  return `${String(n)} CHF`;
};

const statementStateFr = (status, hasConfig, incomplete) => {
  if (incomplete) return 'Config. incomplète';
  if (!hasConfig) return 'Non générée';
  const s = String(status || '').toUpperCase();
  if (s === 'NEEDS_REVIEW') return 'À contrôler';
  if (s === 'VALIDATED' || s === 'LOCKED') return 'Prête';
  if (s === 'CALCULATED') return 'Calculée';
  if (s === 'DRAFT') return 'Brouillon';
  return status || 'Non générée';
};

const badgeClassForState = (state) => {
  if (state === 'Prête') return styles.badgeReady;
  if (state === 'Calculée') return styles.badgeCalc;
  if (state === 'À contrôler' || state === 'Config. incomplète') return styles.badgeReview;
  if (state === 'Brouillon') return styles.badgeMuted;
  return styles.badgeMuted;
};

const isLikelyTestCompany = (name) => {
  const n = (name || '').trim().toLowerCase();
  if (!n) return true;
  return (
    n.startsWith('test ') ||
    n.startsWith('test company') ||
    n.startsWith('transport ') ||
    n.includes('test co') ||
    n.includes('footer test') ||
    n.includes('header gate') ||
    n.includes('lines gate') ||
    /^test company [0-9a-f]{6,}/i.test(name) ||
    /^transport [0-9a-f]{6,}/i.test(name)
  );
};

const AdminBillingOverview = () => {
  const { public_id: adminId } = useParams();
  const base = adminPaths.finance(adminId);

  const now = new Date();
  const [year, setYear] = useState(now.getFullYear());
  const [month, setMonth] = useState(now.getMonth() + 1);
  const [showTests, setShowTests] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [info, setInfo] = useState(null);
  const [period, setPeriod] = useState(null);
  const [invoices, setInvoices] = useState([]);
  const [companies, setCompanies] = useState([]);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [periodsRes, configRes] = await Promise.all([
        fetchPlatformBillingPeriods(),
        fetchPlatformBillingCompaniesConfig({}),
      ]);
      const periods = periodsRes?.periods || [];
      const p =
        periods.find(
          (x) => x.billing_year === Number(year) && x.billing_month === Number(month)
        ) || null;
      setPeriod(p);
      setCompanies(configRes?.items || []);

      if (p?.id) {
        const invRes = await fetchPlatformBillingPeriodInvoices(p.id);
        setInvoices(invRes?.invoices || []);
      } else {
        setInvoices([]);
      }
    } catch (e) {
      setError(e?.response?.data?.message || e?.message || 'Erreur chargement');
    } finally {
      setLoading(false);
    }
  }, [year, month]);

  useEffect(() => {
    load();
  }, [load]);

  const invByCompany = useMemo(() => {
    const map = new Map();
    for (let i = 0; i < invoices.length; i += 1) {
      map.set(invoices[i].company_id, invoices[i]);
    }
    return map;
  }, [invoices]);

  const rows = useMemo(() => {
    const out = [];
    for (let i = 0; i < companies.length; i += 1) {
      const c = companies[i];
      if (!showTests && isLikelyTestCompany(c.company_name)) continue;
      const inv = invByCompany.get(c.company_id);
      const cfg = c.config;
      if (!cfg?.is_billing_enabled && !inv) continue;
      const incomplete =
        Boolean(cfg?.is_billing_enabled) &&
        !(cfg.own_portfolio_billing_enabled || cfg.lirie_commission_enabled);
      const amount = inv?.total_amount ?? null;
      out.push({
        company_id: c.company_id,
        company_name: c.company_name,
        invoice: inv,
        portfolio: inv?.own_portfolio_count ?? null,
        lirie: inv?.lirie_transport_count ?? null,
        amount,
        amountNum: Number(String(amount || '0').replace(',', '.')) || 0,
        state: statementStateFr(
          inv?.statement_status,
          Boolean(cfg?.is_billing_enabled),
          incomplete
        ),
      });
    }
    out.sort((a, b) => b.amountNum - a.amountNum);
    return out;
  }, [companies, invByCompany, showTests]);

  const kpis = useMemo(() => {
    let ready = 0;
    let toReview = 0;
    let forecast = 0;
    for (let i = 0; i < rows.length; i += 1) {
      const r = rows[i];
      if (r.state === 'Prête' || r.state === 'Calculée') ready += 1;
      if (r.state === 'À contrôler' || r.state === 'Config. incomplète') toReview += 1;
      forecast += r.amountNum;
    }
    return {
      companies: rows.length,
      ready,
      toReview,
      forecast: forecast.toFixed(2),
    };
  }, [rows]);

  const periodLabel = `${MONTHS_FR[Number(month) - 1] || month} ${year}`;

  const periodChip = () => {
    if (!period) {
      return <span className={`${styles.chip} ${styles.chipMissing}`}>Non créée</span>;
    }
    if (period.status === 'locked') {
      return <span className={`${styles.chip} ${styles.chipLocked}`}>Verrouillée</span>;
    }
    return <span className={`${styles.chip} ${styles.chipOpen}`}>Ouverte</span>;
  };

  const ensurePeriodAndRecalculate = async () => {
    setBusy(true);
    setInfo(null);
    setError(null);
    try {
      let p = period;
      if (!p?.id) {
        const created = await createPlatformBillingPeriod(Number(year), Number(month));
        p = created?.period || created;
        setPeriod(p || null);
      }
      if (!p?.id) throw new Error('Période introuvable');
      const res = await recalculatePlatformBillingPeriod(p.id);
      setInfo(`Relevés calculés — ${res?.invoices_generated ?? '—'} entreprise(s).`);
      await load();
    } catch (e) {
      setError(e?.response?.data?.message || e?.message || 'Calcul impossible');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className={styles.page}>
      <div className={styles.toolbar}>
        <div className={styles.toolbarLeft}>
          <label className={styles.field}>
            Année
            <input
              type="number"
              min="2020"
              max="2100"
              value={year}
              onChange={(e) => setYear(Number(e.target.value))}
            />
          </label>
          <label className={styles.field}>
            Mois
            <select value={month} onChange={(e) => setMonth(Number(e.target.value))}>
              {MONTHS_FR.map((label, i) => (
                <option key={label} value={i + 1}>
                  {label}
                </option>
              ))}
            </select>
          </label>
          <div className={styles.periodMeta}>
            <span>{periodLabel}</span>
            {periodChip()}
          </div>
          <label className={styles.checkLabel}>
            <input
              type="checkbox"
              checked={showTests}
              onChange={(e) => setShowTests(e.target.checked)}
            />
            Inclure tests
          </label>
        </div>
        <div className={styles.toolbarRight}>
          <button
            type="button"
            className={styles.btn}
            disabled={loading || busy}
            onClick={() => load()}
          >
            Actualiser
          </button>
          <button
            type="button"
            className={`${styles.btn} ${styles.btnPrimary}`}
            disabled={busy || period?.status === 'locked'}
            onClick={ensurePeriodAndRecalculate}
          >
            {busy ? 'Calcul…' : 'Calculer les relevés'}
          </button>
        </div>
      </div>

      {error ? (
        <div className={`${styles.banner} ${styles.bannerError}`} role="alert">
          {error}
        </div>
      ) : null}
      {info ? (
        <div className={`${styles.banner} ${styles.bannerOk}`} role="status">
          {info}
        </div>
      ) : null}

      <section className={styles.kpiRow} aria-label="Synthèse mensuelle">
        <div className={styles.kpi}>
          <span className={styles.kpiLabel}>Entreprises</span>
          <span className={styles.kpiValue}>{loading ? '—' : kpis.companies}</span>
        </div>
        <div className={styles.kpi}>
          <span className={styles.kpiLabel}>Relevés prêts</span>
          <span className={styles.kpiValue}>{loading ? '—' : kpis.ready}</span>
        </div>
        <div className={`${styles.kpi} ${kpis.toReview ? styles.kpiWarn : ''}`}>
          <span className={styles.kpiLabel}>À traiter</span>
          <span className={styles.kpiValue}>{loading ? '—' : kpis.toReview}</span>
        </div>
        <div className={`${styles.kpi} ${styles.kpiAccent}`}>
          <span className={styles.kpiLabel}>Total prévisionnel</span>
          <span className={styles.kpiValue}>
            {loading ? '—' : fmtMoney(kpis.forecast)}
          </span>
        </div>
      </section>

      <section className={styles.panel}>
        <div className={styles.panelHead}>
          <h2 className={styles.panelTitle}>Entreprises</h2>
          <span className={styles.panelMeta}>
            {loading ? 'Chargement…' : `${rows.length} ligne${rows.length > 1 ? 's' : ''}`}
            {' · '}
            <Link to={`${base}/releves`} className={styles.rowAction}>
              Voir les relevés
            </Link>
          </span>
        </div>
        {loading ? (
          <p className={styles.loading}>Chargement de la période…</p>
        ) : rows.length === 0 ? (
          <div className={styles.empty}>
            Aucune entreprise facturable pour {periodLabel}.
            <br />
            Activez un contrat dans{' '}
            <Link to={`${base}/config`} className={styles.rowAction}>
              Entreprises
            </Link>
            , puis calculez les relevés.
          </div>
        ) : (
          <div className={styles.tableWrap}>
            <table className={`${styles.table} ${styles.tableEntreprises}`}>
              <colgroup>
                <col className={styles.colCompany} />
                <col className={styles.colCount} />
                <col className={styles.colCount} />
                <col className={styles.colAmount} />
                <col className={styles.colState} />
                <col className={styles.colAction} />
              </colgroup>
              <thead>
                <tr>
                  <th scope="col">Entreprise</th>
                  <th scope="col" className={styles.colHead}>
                    <span className={styles.thMain}>Portefeuille</span>
                    <span className={styles.thSub}>nb courses (abo)</span>
                  </th>
                  <th scope="col" className={styles.colHead}>
                    <span className={styles.thMain}>Marketplace</span>
                    <span className={styles.thSub}>nb transports LIRIE</span>
                  </th>
                  <th scope="col" className={styles.colHead}>
                    <span className={styles.thMain}>Montant</span>
                    <span className={styles.thSub}>total TTC</span>
                  </th>
                  <th scope="col" className={styles.colHeadCenter}>
                    État
                  </th>
                  <th scope="col" className={styles.colHeadAction}>
                    <span className={styles.srOnly}>Action</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.company_id}>
                    <td className={styles.companyName}>{r.company_name}</td>
                    <td className={styles.cellCount}>
                      <span className={styles.countValue}>
                        {r.portfolio != null ? r.portfolio : '—'}
                      </span>
                      <span className={styles.countUnit}>courses</span>
                    </td>
                    <td className={styles.cellCount}>
                      <span className={styles.countValue}>
                        {r.lirie != null ? r.lirie : '—'}
                      </span>
                      <span className={styles.countUnit}>transports</span>
                    </td>
                    <td className={styles.cellAmount}>{fmtMoney(r.amount)}</td>
                    <td className={styles.cellState}>
                      <span className={`${styles.badge} ${badgeClassForState(r.state)}`}>
                        {r.state}
                      </span>
                    </td>
                    <td className={styles.cellAction}>
                      {r.invoice ? (
                        <Link
                          className={styles.rowAction}
                          to={`${base}/releves`}
                          state={{ focusInvoiceId: r.invoice.id, periodId: period?.id }}
                        >
                          Ouvrir
                        </Link>
                      ) : (
                        <Link className={styles.rowAction} to={`${base}/config`}>
                          Configurer
                        </Link>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
};

export default AdminBillingOverview;
