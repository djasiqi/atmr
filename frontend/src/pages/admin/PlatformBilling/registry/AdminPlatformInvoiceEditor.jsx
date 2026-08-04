import React, { useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import { FiPlus, FiRefreshCw, FiTrash2, FiX } from 'react-icons/fi';
import {
  fetchPlatformIssuedInvoiceEditor,
  previewPlatformIssuedInvoiceEditor,
  replacePlatformIssuedInvoice,
} from '../../../../services/adminService';
import AdminActionDialog from '../../components/AdminActionDialog';
import styles from './AdminPlatformInvoiceEditor.module.css';

const apiErrorMessage = (e) =>
  e?.response?.data?.message ||
  e?.response?.data?.error ||
  e?.message ||
  'Erreur';

const emptyLine = () => ({
  calculation_mode: 'FIXED_AMOUNT',
  label: '',
  quantity: '',
  unit_amount: '',
  amount: '0.00',
  line_type: 'ADJUSTMENT',
});

const emptyDiscount = () => ({
  calculation_mode: 'FIXED_AMOUNT',
  label: 'Remise',
  quantity: '',
  unit_amount: '',
  amount: '0.00',
  line_type: 'DISCOUNT',
  discount_mode: 'AMOUNT', // AMOUNT (CHF) | PERCENT
  discount_value: '0',
});

const round2 = (n) => {
  const x = Number(n);
  if (!Number.isFinite(x)) return 0;
  return Math.round(x * 100) / 100;
};

const parseNum = (v) => {
  const n = Number(String(v ?? '').replace(',', '.'));
  return Number.isFinite(n) ? n : 0;
};

/** Montant d’une ligne hors remise (toujours positif ou signé selon saisie). */
const regularLineAmount = (line) => {
  if (line.calculation_mode === 'UNIT_PRICE') {
    return round2(parseNum(line.quantity) * parseNum(line.unit_amount));
  }
  return round2(parseNum(line.amount));
};

const baseBeforeDiscounts = (lines) =>
  round2(
    lines
      .filter((ln) => ln.line_type !== 'DISCOUNT')
      .reduce((s, ln) => s + regularLineAmount(ln), 0)
  );

/** Montant signé d’une ligne (remises → négatif). */
const lineAmount = (line, allLines = []) => {
  if (line.line_type === 'DISCOUNT') {
    const val = Math.abs(parseNum(line.discount_value));
    if (line.discount_mode === 'PERCENT') {
      return -round2((baseBeforeDiscounts(allLines) * val) / 100);
    }
    return -round2(val);
  }
  return regularLineAmount(line);
};

const mapBootstrapLine = (ln) => {
  const base = {
    calculation_mode: ln.calculation_mode || 'FIXED_AMOUNT',
    label: ln.label || '',
    quantity: ln.quantity ?? '',
    unit_amount: ln.unit_amount ?? '',
    amount: ln.amount ?? '0',
    line_type: ln.line_type || 'ADJUSTMENT',
  };
  if (base.line_type === 'DISCOUNT') {
    return {
      ...base,
      calculation_mode: 'FIXED_AMOUNT',
      discount_mode: 'AMOUNT',
      discount_value: String(Math.abs(parseNum(ln.amount))),
    };
  }
  return syncDerivedLabel(base);
};

const updateLine = (rows, idx, patch) =>
  rows.map((r, i) => (i === idx ? { ...r, ...patch } : r));

/** Formate un nombre pour libellé (1, 1.5, 2.25…). */
const fmtLabelNum = (v) => {
  const n = parseNum(v);
  if (!Number.isFinite(n)) return '0';
  return String(round2(n)).replace(/\.?0+$/, '') || '0';
};

/** Resynchronise libellés auto (support X h) quand qté / tarif changent. */
const syncDerivedLabel = (line) => {
  const label = (line.label || '').trim();
  const lt = (line.line_type || '').toLowerCase();
  const isSupport = lt.includes('support') || /^support/i.test(label);
  if (
    isSupport &&
    line.calculation_mode === 'UNIT_PRICE' &&
    line.quantity !== '' &&
    line.quantity != null
  ) {
    const hours = fmtLabelNum(line.quantity);
    if (line.unit_amount !== '' && line.unit_amount != null) {
      const rate = fmtLabelNum(line.unit_amount);
      return {
        ...line,
        label: `Support plateforme — ${hours} h à ${rate} CHF/h`,
      };
    }
    return { ...line, label: `Support plateforme — ${hours} h` };
  }
  return line;
};

const patchLine = (rows, idx, patch) =>
  rows.map((r, i) => (i === idx ? syncDerivedLabel({ ...r, ...patch }) : r));

const AdminPlatformInvoiceEditor = ({ issuedId, onClose, onReplaced }) => {
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [bootstrap, setBootstrap] = useState(null);
  const [debtor, setDebtor] = useState({});
  const [lines, setLines] = useState([]);
  const [taxRate, setTaxRate] = useState('0');
  const [dueAt, setDueAt] = useState('');
  const [commercialRef, setCommercialRef] = useState('');
  const [confirmDialog, setConfirmDialog] = useState(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await fetchPlatformIssuedInvoiceEditor(issuedId);
        if (cancelled) return;
        setBootstrap(data);
        setDebtor({ ...(data.debtor_snapshot || {}) });
        setLines((data.lines || []).map(mapBootstrapLine));
        setTaxRate(String(data.tax_rate ?? '0'));
        setDueAt(data.due_at ? String(data.due_at).slice(0, 10) : '');
        setCommercialRef(data.commercial_reference || '');
      } catch (e) {
        if (!cancelled) setError(apiErrorMessage(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [issuedId]);

  const totals = useMemo(() => {
    const subtotal = round2(lines.reduce((s, ln) => s + lineAmount(ln, lines), 0));
    const rate = parseNum(taxRate);
    const tax = round2((subtotal * rate) / 100);
    return { subtotal, tax, total: round2(subtotal + tax) };
  }, [lines, taxRate]);

  const mode = bootstrap?.mode;
  const title = mode === 'correct' ? 'Corriger la facture' : 'Éditer la facture';
  const hasStatementLines = (bootstrap?.statement_lines || []).length > 0;

  const buildPayload = () => ({
    idempotency_key:
      typeof crypto !== 'undefined' && crypto.randomUUID
        ? crypto.randomUUID()
        : `replace-${issuedId}-${Date.now()}`,
    source_updated_at: bootstrap?.source_updated_at,
    debtor_snapshot: debtor,
    lines: lines.map((ln) => {
      if (ln.line_type === 'DISCOUNT') {
        const amt = lineAmount(ln, lines);
        const val = Math.abs(parseNum(ln.discount_value));
        let label = (ln.label || 'Remise').trim();
        if (ln.discount_mode === 'PERCENT') {
          const bare = /^remise\s*$/i.test(label) || /^remise\s+\d+([.,]\d+)?\s*%?$/i.test(label);
          if (bare) label = `Remise ${val} %`;
        }
        return {
          calculation_mode: 'FIXED_AMOUNT',
          label,
          amount: String(amt),
          line_type: 'DISCOUNT',
        };
      }
      if (ln.calculation_mode === 'UNIT_PRICE') {
        return {
          calculation_mode: 'UNIT_PRICE',
          label: ln.label,
          quantity: String(ln.quantity),
          unit_amount: String(ln.unit_amount),
          line_type: ln.line_type || 'ADJUSTMENT',
        };
      }
      return {
        calculation_mode: 'FIXED_AMOUNT',
        label: ln.label,
        amount: String(lineAmount(ln, lines)),
        line_type: ln.line_type || 'ADJUSTMENT',
      };
    }),
    tax_rate: String(taxRate),
    due_at: dueAt || null,
    commercial_reference: commercialRef || null,
  });

  const resetFromStatement = () => {
    setLines((bootstrap?.statement_lines || []).map(mapBootstrapLine));
  };

  const setDebtorField = (key) => (e) =>
    setDebtor((d) => ({ ...d, [key]: e.target.value }));

  const handlePreview = async () => {
    setBusy(true);
    setError(null);
    try {
      await previewPlatformIssuedInvoiceEditor(issuedId, buildPayload());
    } catch (e) {
      setError(apiErrorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  const doSave = async (reason) => {
    setBusy(true);
    setError(null);
    try {
      const payload = buildPayload();
      if (reason) payload.reason = reason;
      const res = await replacePlatformIssuedInvoice(issuedId, payload);
      onReplaced?.(res?.issued_id || res?.issued_invoice?.id);
    } catch (e) {
      setError(apiErrorMessage(e));
      setConfirmDialog(null);
    } finally {
      setBusy(false);
    }
  };

  const handleSave = () => {
    if (mode === 'correct') {
      setConfirmDialog({
        title: 'Corriger la facture',
        description:
          'Cette facture a déjà été envoyée. La correction générera automatiquement un avoir pour la facture actuelle et une nouvelle facture avec les modifications.',
        reason: { required: true, label: 'Motif de correction', minLength: 3 },
        confirmationLabel: 'Enregistrer et réémettre',
        onConfirm: async ({ reason }) => {
          await doSave(reason);
          setConfirmDialog(null);
        },
      });
      return;
    }
    doSave();
  };

  if (mode === 'payments_block') {
    return createPortal(
      <div className={styles.overlay} role="presentation" onClick={onClose}>
        <aside
          className={styles.drawer}
          role="dialog"
          aria-modal="true"
          onClick={(e) => e.stopPropagation()}
        >
          <header className={styles.header}>
            <h2 className={styles.title}>Correction impossible</h2>
            <button type="button" className={styles.closeBtn} onClick={onClose} aria-label="Fermer">
              <FiX size={18} />
            </button>
          </header>
          <div className={styles.body}>
            <p>
              Correction impossible tant que des paiements sont enregistrés. Contre-passez
              d’abord les paiements depuis l’onglet Paiements.
            </p>
          </div>
          <div className={styles.actionBar}>
            <button type="button" className={styles.btn} onClick={onClose}>
              Fermer
            </button>
          </div>
        </aside>
      </div>,
      document.body
    );
  }

  return createPortal(
    <div className={styles.overlay} role="presentation" onClick={onClose}>
      <aside
        className={styles.drawer}
        role="dialog"
        aria-modal="true"
        aria-labelledby="invoice-editor-title"
        onClick={(e) => e.stopPropagation()}
      >
        <header className={styles.header}>
          <div>
            <p className={styles.eyebrow}>Facture</p>
            <h2 id="invoice-editor-title" className={styles.title}>
              {title}
            </h2>
            <p className={styles.meta}>
              {bootstrap?.invoice_number || `Facture #${issuedId}`}
            </p>
          </div>
          <button type="button" className={styles.closeBtn} onClick={onClose} aria-label="Fermer">
            <FiX size={18} />
          </button>
        </header>

        {error ? (
          <div className={styles.bannerError} role="alert">
            {error}
          </div>
        ) : null}

        <div className={styles.body}>
          {loading ? <p className={styles.loading}>Chargement…</p> : null}

          {!loading && bootstrap && (
            <>
              <section className={styles.section}>
                <h3 className={styles.sectionTitle}>Destinataire</h3>
                <div className={styles.debtorGrid}>
                  <label className={`${styles.field} ${styles.spanFull}`}>
                    Raison sociale
                    <input
                      value={debtor.legal_name || ''}
                      onChange={setDebtorField('legal_name')}
                      autoComplete="organization"
                    />
                  </label>
                  <label className={`${styles.field} ${styles.spanStreet}`}>
                    Rue
                    <input
                      value={debtor.street_name || ''}
                      onChange={setDebtorField('street_name')}
                      autoComplete="address-line1"
                    />
                  </label>
                  <label className={`${styles.field} ${styles.spanNum}`}>
                    N°
                    <input
                      value={debtor.building_number || ''}
                      onChange={setDebtorField('building_number')}
                    />
                  </label>
                  <label className={`${styles.field} ${styles.spanNpa}`}>
                    NPA
                    <input
                      value={debtor.postal_code || ''}
                      onChange={setDebtorField('postal_code')}
                      autoComplete="postal-code"
                    />
                  </label>
                  <label className={`${styles.field} ${styles.spanCity}`}>
                    Ville
                    <input
                      value={debtor.city || ''}
                      onChange={setDebtorField('city')}
                      autoComplete="address-level2"
                    />
                  </label>
                  <label className={`${styles.field} ${styles.spanCountry}`}>
                    Pays
                    <input
                      value={debtor.country_code || ''}
                      onChange={setDebtorField('country_code')}
                      maxLength={2}
                      autoComplete="country"
                    />
                  </label>
                  <label className={`${styles.field} ${styles.spanFull}`}>
                    E-mail de facturation
                    <input
                      type="email"
                      value={debtor.billing_email || ''}
                      onChange={setDebtorField('billing_email')}
                      autoComplete="email"
                    />
                  </label>
                  <label className={`${styles.field} ${styles.spanHalf}`}>
                    UID / IDE
                    <input
                      value={debtor.uid_ide || ''}
                      onChange={setDebtorField('uid_ide')}
                      placeholder="CHE-XXX.XXX.XXX"
                    />
                  </label>
                  <label className={`${styles.field} ${styles.spanHalf}`}>
                    N° TVA
                    <input
                      value={debtor.vat_number || ''}
                      onChange={setDebtorField('vat_number')}
                      placeholder="Optionnel"
                    />
                  </label>
                </div>
              </section>

              <section className={styles.section}>
                <h3 className={styles.sectionTitle}>Facturation</h3>
                <div className={styles.metaGrid}>
                  <label className={styles.field}>
                    Échéance
                    <input
                      type="date"
                      value={dueAt}
                      onChange={(e) => setDueAt(e.target.value)}
                    />
                  </label>
                  <label className={`${styles.field} ${styles.spanGrow}`}>
                    Référence commerciale
                    <input
                      value={commercialRef}
                      onChange={(e) => setCommercialRef(e.target.value)}
                      placeholder="Bon de commande, contrat…"
                    />
                  </label>
                  <label className={`${styles.field} ${styles.spanTax}`}>
                    TVA (%)
                    <input
                      inputMode="decimal"
                      value={taxRate}
                      onChange={(e) => setTaxRate(e.target.value)}
                    />
                  </label>
                </div>
              </section>

              <section className={styles.section}>
                <div className={styles.sectionHead}>
                  <h3 className={styles.sectionTitle}>Lignes</h3>
                  <div className={styles.inlineActions}>
                    <button
                      type="button"
                      className={styles.btnSm}
                      onClick={() => setLines((l) => [...l, emptyLine()])}
                    >
                      <FiPlus size={14} aria-hidden />
                      Ligne
                    </button>
                    <button
                      type="button"
                      className={styles.btnSm}
                      onClick={() => setLines((l) => [...l, emptyDiscount()])}
                    >
                      <FiPlus size={14} aria-hidden />
                      Remise
                    </button>
                    {hasStatementLines ? (
                      <button
                        type="button"
                        className={styles.btnSmMuted}
                        onClick={resetFromStatement}
                        title="Remplacer les lignes par celles du relevé"
                      >
                        <FiRefreshCw size={14} aria-hidden />
                        Depuis le relevé
                      </button>
                    ) : null}
                  </div>
                </div>

                {lines.length === 0 ? (
                  <p className={styles.emptyHint}>
                    Aucune ligne. Ajoutez une ligne ou une remise.
                  </p>
                ) : (
                  <div className={styles.linesList}>
                    <div className={styles.linesHeader} aria-hidden>
                      <span>Description</span>
                      <span>Calcul</span>
                      <span>Montant</span>
                      <span />
                    </div>
                    {lines.map((ln, idx) => {
                      const isDiscount = ln.line_type === 'DISCOUNT';
                      const isUnit = !isDiscount && ln.calculation_mode === 'UNIT_PRICE';
                      const amount = lineAmount(ln, lines);
                      const discountIsPercent = ln.discount_mode === 'PERCENT';
                      return (
                        <div
                          key={idx}
                          className={`${styles.lineCard} ${
                            isDiscount ? styles.lineDiscount : ''
                          }`}
                        >
                          <label className={styles.lineLabel}>
                            <span className={styles.srOnly}>Description</span>
                            <input
                              className={styles.lineDesc}
                              value={ln.label}
                              onChange={(e) =>
                                setLines((rows) =>
                                  updateLine(rows, idx, { label: e.target.value })
                                )
                              }
                              placeholder={
                                isDiscount ? 'Libellé de la remise' : 'Libellé de la ligne'
                              }
                            />
                          </label>

                          <div className={styles.lineCalc}>
                            {isDiscount ? (
                              <>
                                <div
                                  className={styles.modeToggle}
                                  role="group"
                                  aria-label="Type de remise"
                                >
                                  <button
                                    type="button"
                                    className={
                                      !discountIsPercent
                                        ? styles.modeActive
                                        : styles.modeBtn
                                    }
                                    onClick={() =>
                                      setLines((rows) =>
                                        updateLine(rows, idx, {
                                          discount_mode: 'AMOUNT',
                                        })
                                      )
                                    }
                                  >
                                    Montant CHF
                                  </button>
                                  <button
                                    type="button"
                                    className={
                                      discountIsPercent
                                        ? styles.modeActive
                                        : styles.modeBtn
                                    }
                                    onClick={() =>
                                      setLines((rows) =>
                                        updateLine(rows, idx, {
                                          discount_mode: 'PERCENT',
                                        })
                                      )
                                    }
                                  >
                                    Pourcentage
                                  </button>
                                </div>
                                <label className={styles.miniField}>
                                  {discountIsPercent ? 'Remise (%)' : 'Remise (CHF)'}
                                  <div className={styles.discountInputWrap}>
                                    <input
                                      inputMode="decimal"
                                      value={ln.discount_value ?? ''}
                                      onChange={(e) =>
                                        setLines((rows) =>
                                          updateLine(rows, idx, {
                                            discount_value: e.target.value,
                                          })
                                        )
                                      }
                                      placeholder={discountIsPercent ? '10' : '10.00'}
                                    />
                                    <span className={styles.discountSuffix} aria-hidden>
                                      {discountIsPercent ? '%' : 'CHF'}
                                    </span>
                                  </div>
                                </label>
                                {discountIsPercent ? (
                                  <p className={styles.discountHint}>
                                    Sur sous-total {baseBeforeDiscounts(lines).toFixed(2)} CHF
                                  </p>
                                ) : null}
                              </>
                            ) : (
                              <>
                                <div
                                  className={styles.modeToggle}
                                  role="group"
                                  aria-label="Mode de calcul"
                                >
                                  <button
                                    type="button"
                                    className={
                                      !isUnit ? styles.modeActive : styles.modeBtn
                                    }
                                    onClick={() =>
                                      setLines((rows) =>
                                        updateLine(rows, idx, {
                                          calculation_mode: 'FIXED_AMOUNT',
                                          amount:
                                            rows[idx].amount ||
                                            String(lineAmount(rows[idx], rows)),
                                        })
                                      )
                                    }
                                  >
                                    Fixe
                                  </button>
                                  <button
                                    type="button"
                                    className={
                                      isUnit ? styles.modeActive : styles.modeBtn
                                    }
                                    onClick={() =>
                                      setLines((rows) =>
                                        patchLine(rows, idx, {
                                          calculation_mode: 'UNIT_PRICE',
                                          quantity: rows[idx].quantity || '1',
                                          unit_amount:
                                            rows[idx].unit_amount ||
                                            rows[idx].amount ||
                                            '0',
                                        })
                                      )
                                    }
                                  >
                                    Qté × prix
                                  </button>
                                </div>

                                {isUnit ? (
                                  <div className={styles.unitInputs}>
                                    <label className={styles.miniField}>
                                      Qté
                                      <input
                                        inputMode="decimal"
                                        value={ln.quantity}
                                        onChange={(e) =>
                                          setLines((rows) =>
                                            patchLine(rows, idx, {
                                              quantity: e.target.value,
                                            })
                                          )
                                        }
                                      />
                                    </label>
                                    <span className={styles.unitTimes} aria-hidden>
                                      ×
                                    </span>
                                    <label className={styles.miniField}>
                                      Prix
                                      <input
                                        inputMode="decimal"
                                        value={ln.unit_amount}
                                        onChange={(e) =>
                                          setLines((rows) =>
                                            patchLine(rows, idx, {
                                              unit_amount: e.target.value,
                                            })
                                          )
                                        }
                                      />
                                    </label>
                                  </div>
                                ) : (
                                  <label className={styles.miniField}>
                                    Montant
                                    <input
                                      inputMode="decimal"
                                      value={ln.amount}
                                      onChange={(e) =>
                                        setLines((rows) =>
                                          updateLine(rows, idx, {
                                            amount: e.target.value,
                                          })
                                        )
                                      }
                                    />
                                  </label>
                                )}
                              </>
                            )}
                          </div>

                          <div className={styles.lineTotal}>
                            <span
                              className={`${styles.lineTotalValue} ${
                                amount < 0 ? styles.lineTotalNeg : ''
                              }`}
                            >
                              {amount.toFixed(2)}
                            </span>
                            <span className={styles.lineTotalCur}>CHF</span>
                          </div>

                          <button
                            type="button"
                            className={styles.lineRemove}
                            onClick={() =>
                              setLines((rows) => rows.filter((_, i) => i !== idx))
                            }
                            aria-label="Supprimer la ligne"
                            title="Supprimer"
                          >
                            <FiTrash2 size={15} />
                          </button>
                        </div>
                      );
                    })}
                  </div>
                )}

                <dl className={styles.totals}>
                  <div>
                    <dt>Sous-total HT</dt>
                    <dd>{totals.subtotal.toFixed(2)} CHF</dd>
                  </div>
                  <div>
                    <dt>TVA ({taxRate || '0'} %)</dt>
                    <dd>{totals.tax.toFixed(2)} CHF</dd>
                  </div>
                  <div className={styles.totalsGrand}>
                    <dt>Total TTC</dt>
                    <dd>{totals.total.toFixed(2)} CHF</dd>
                  </div>
                </dl>
              </section>
            </>
          )}
        </div>

        <div className={styles.actionBar}>
          <button type="button" className={styles.btn} onClick={onClose} disabled={busy}>
            Annuler
          </button>
          <button
            type="button"
            className={styles.btn}
            onClick={handlePreview}
            disabled={busy || loading}
          >
            Prévisualiser le PDF
          </button>
          <button
            type="button"
            className={styles.btnPrimary}
            onClick={handleSave}
            disabled={busy || loading || mode === 'readonly'}
          >
            Enregistrer et réémettre
          </button>
        </div>

        {confirmDialog && (
          <AdminActionDialog
            open
            title={confirmDialog.title}
            description={confirmDialog.description}
            confirmationLabel={confirmDialog.confirmationLabel || 'Confirmer'}
            reason={confirmDialog.reason}
            onConfirm={confirmDialog.onConfirm}
            onClose={() => setConfirmDialog(null)}
            loading={busy}
          />
        )}
      </aside>
    </div>,
    document.body
  );
};

export default AdminPlatformInvoiceEditor;
