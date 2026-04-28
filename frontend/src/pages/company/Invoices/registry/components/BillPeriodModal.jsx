import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { FiX, FiEye, FiZap, FiList, FiLoader, FiUser, FiHome, FiUsers } from 'react-icons/fi';
import { invoiceService, formatCurrencyCHF, generateInvoice } from '../../../../../services/invoiceService';
import { getApiErrorMessage } from '../../../../../utils/apiErrorMessage';
import styles from './BillPeriodModal.module.css';

const unwrapApi = (res) => {
  if (res && typeof res === 'object' && res.data && typeof res.data === 'object' && 'transports_count' in res.data) {
    return res.data;
  }
  if (res && typeof res === 'object' && 'transports_count' in res) {
    return res;
  }
  return res?.data ?? res;
};

const now = new Date();
const defaultYear = now.getFullYear();
const defaultMonth = now.getMonth() + 1;

const BillPeriodModal = ({
  open,
  onClose,
  companyId,
  onSuccess,
  onOpenLegacy,
}) => {
  const [payerType, setPayerType] = useState('patient');
  const [periodYear, setPeriodYear] = useState(defaultYear);
  const [periodMonth, setPeriodMonth] = useState(defaultMonth);
  const [clients, setClients] = useState([]);
  const [institutions, setInstitutions] = useState([]);
  const [clientId, setClientId] = useState('');
  const [clinicKey, setClinicKey] = useState(''); // institution id as string
  const [partnershipId, setPartnershipId] = useState('');
  const [billablePartners, setBillablePartners] = useState([]);
  const [loadingLists, setLoadingLists] = useState(false);
  const [preview, setPreview] = useState(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [generateLoading, setGenerateLoading] = useState(false);
  const [error, setError] = useState('');

  const loadLists = useCallback(async () => {
    if (!companyId || !open) return;
    setLoadingLists(true);
    setError('');
    try {
      const [elig, inst, bpRaw] = await Promise.all([
        invoiceService.fetchEligibleClients(companyId, {
          year: periodYear,
          month: periodMonth,
          limit: 500,
        }),
        invoiceService.fetchInstitutions(companyId),
        invoiceService.fetchBillablePartners(companyId, {
          year: periodYear,
          month: periodMonth,
        }),
      ]);
      const ec = elig?.data?.clients ?? elig?.clients ?? [];
      setClients(Array.isArray(ec) ? ec : []);
      setInstitutions(inst?.institutions || inst?.data?.institutions || []);
      const bpList = bpRaw?.data ?? bpRaw;
      setBillablePartners(Array.isArray(bpList) ? bpList : []);
    } catch (e) {
      setError("Impossible de charger les listes. Réessayez.");
    } finally {
      setLoadingLists(false);
    }
  }, [companyId, open, periodYear, periodMonth]);

  useEffect(() => {
    if (open) {
      void loadLists();
    }
  }, [open, loadLists]);

  useEffect(() => {
    if (!open) {
      setPartnershipId('');
    }
  }, [open]);

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape' && open && !generateLoading && !previewLoading) onClose();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [open, onClose, generateLoading, previewLoading]);

  const selectedClinic = institutions.find((i) => String(i.id) === clinicKey);
  const clinicCompanyId = selectedClinic?.clinic_company_id ?? null;

  const canPreview = useCallback(() => {
    if (payerType === 'patient') return Boolean(clientId);
    if (payerType === 'clinic') return Boolean(clinicKey && clinicCompanyId);
    if (payerType === 'partner') return Boolean(partnershipId);
    return false;
  }, [payerType, clientId, clinicKey, clinicCompanyId, partnershipId]);

  const footerHint = useMemo(() => {
    if (loadingLists) return null;
    if (payerType === 'patient' && !clientId) {
      return 'Sélectionnez un patient pour prévisualiser.';
    }
    if (payerType === 'clinic') {
      if (!clinicKey) return 'Sélectionnez une clinique pour prévisualiser.';
      if (!clinicCompanyId) {
        return 'Cette institution n’a pas d’entreprise S2 associée — impossible de facturer en mode clinique.';
      }
    }
    if (payerType === 'partner' && !partnershipId) {
      return 'Sélectionnez un partenaire à facturer pour cette période.';
    }
    if (preview && preview.transports_count === 0) {
      return payerType === 'partner'
        ? 'Aucun transfert à facturer sur cette période pour ce partenaire.'
        : 'Aucun transport à facturer sur cette période pour ce payeur.';
    }
    if (canPreview() && !preview && !previewLoading && !generateLoading) {
      return 'Utilisez « Prévisualiser » pour estimer le montant avant de générer le brouillon.';
    }
    return null;
  }, [
    loadingLists,
    payerType,
    clientId,
    clinicKey,
    clinicCompanyId,
    preview,
    previewLoading,
    generateLoading,
    partnershipId,
    canPreview,
  ]);

  const runPreview = async () => {
    if (!canPreview()) {
      setError('Sélectionnez un payeur.');
      return;
    }
    setError('');
    setPreview(null);

    if (payerType === 'partner') {
      const row = billablePartners.find((p) => String(p.partnership_id) === partnershipId);
      if (!row) {
        setError('Partenariat introuvable pour cette période.');
        return;
      }
      const validated = Number(row.validated_unbilled_transfers_count ?? 0);
      const total = Number(row.total_amount ?? 0);
      const unbilled = Number(row.unbilled_transfers_count ?? 0);
      const warnings = [];
      if (validated === 0 && unbilled > 0) {
        warnings.push(
          'Certains transferts ne sont pas encore validés — le montant estimé ne les inclut pas.'
        );
      }
      if (validated === 0 && unbilled === 0) {
        warnings.push('Aucun transfert facturable sur cette période pour ce partenaire.');
      }
      setPreview({
        mode: 'partner_monthly',
        transports_count: validated,
        estimated_total: total,
        warnings,
      });
      return;
    }

    setPreviewLoading(true);
    try {
      const res = await invoiceService.fetchPeriodPreview(companyId, {
        year: periodYear,
        month: periodMonth,
        clientId: payerType === 'patient' ? parseInt(clientId, 10) : undefined,
        clinicCompanyId: payerType === 'clinic' ? clinicCompanyId : undefined,
      });
      setPreview(unwrapApi(res));
    } catch (err) {
      setError(getApiErrorMessage(err, 'Prévisualisation impossible'));
    } finally {
      setPreviewLoading(false);
    }
  };

  const runGenerate = async () => {
    if (!canPreview()) {
      setError('Sélectionnez un payeur.');
      return;
    }
    if (!preview) {
      setError('Prévisualisez d’abord l’aperçu.');
      return;
    }
    if (preview.transports_count === 0) {
      setError(
        payerType === 'partner'
          ? 'Aucun transfert à facturer sur cette période. Vérifiez le partenaire, le mois, ou des transferts déjà facturés.'
          : 'Aucun transport à facturer sur cette période. Vérifiez le payeur, le mois, ou des courses déjà facturées.'
      );
      return;
    }
    setError('');
    setGenerateLoading(true);
    try {
      if (payerType === 'partner') {
        const result = await invoiceService.generatePartnerInvoice(companyId, {
          partnership_id: parseInt(partnershipId, 10),
          period_year: periodYear,
          period_month: periodMonth,
        });
        const inv = result?.data ?? result;
        if (inv?.id) {
          onClose();
          onSuccess?.(inv);
        } else {
          setError('Réponse inattendue du serveur.');
        }
        return;
      }

      let payload;
      if (payerType === 'patient') {
        payload = {
          client_id: parseInt(clientId, 10),
          period_year: periodYear,
          period_month: periodMonth,
        };
      } else {
        payload = {
          mode: 'clinic_monthly',
          clinic_company_id: clinicCompanyId,
          period_year: periodYear,
          period_month: periodMonth,
        };
      }
      const result = await generateInvoice(companyId, payload);
      const inv = result?.data ?? result;
      if (inv?.id) {
        onClose();
        onSuccess?.(inv);
      } else {
        setError('Réponse inattendue du serveur.');
      }
    } catch (err) {
      setError(getApiErrorMessage(err, 'Échec de génération'));
    } finally {
      setGenerateLoading(false);
    }
  };

  if (!open) return null;

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.panel} onClick={(e) => e.stopPropagation()}>
        <div className={styles.head}>
          <div className={styles.headText}>
            <h2 className={styles.title}>Facturer une période</h2>
            <p className={styles.subtitle}>
              Choisir le payeur et la période, prévisualiser puis générer le brouillon.
            </p>
          </div>
          <button type="button" className={styles.close} onClick={onClose} aria-label="Fermer">
            <FiX size={18} />
          </button>
        </div>

        <div className={styles.body}>
          <div className={styles.section}>
            <div className={styles.fieldGroup}>
              <span className={styles.fieldLabel}>Type de payeur</span>
              <div className={styles.payerSegment} role="radiogroup" aria-label="Type de payeur">
                <label
                  className={`${styles.payerChoice} ${payerType === 'patient' ? styles.payerChoiceActive : ''}`}
                >
                  <input
                    type="radio"
                    name="payerType"
                    value="patient"
                    className={styles.payerRadio}
                    checked={payerType === 'patient'}
                    onChange={() => {
                      setPayerType('patient');
                      setPartnershipId('');
                      setPreview(null);
                    }}
                  />
                  <span className={styles.payerChoiceIcon} aria-hidden="true">
                    <FiUser strokeWidth={2} />
                  </span>
                  <span className={styles.payerChoiceText}>
                    <span className={styles.payerChoiceTitle}>Patient</span>
                    <span className={styles.payerChoiceHint}>Facture directe</span>
                  </span>
                </label>
                <label
                  className={`${styles.payerChoice} ${payerType === 'clinic' ? styles.payerChoiceActive : ''}`}
                >
                  <input
                    type="radio"
                    name="payerType"
                    value="clinic"
                    className={styles.payerRadio}
                    checked={payerType === 'clinic'}
                    onChange={() => {
                      setPayerType('clinic');
                      setPartnershipId('');
                      setPreview(null);
                    }}
                  />
                  <span className={styles.payerChoiceIcon} aria-hidden="true">
                    <FiHome strokeWidth={2} />
                  </span>
                  <span className={styles.payerChoiceText}>
                    <span className={styles.payerChoiceTitle}>Clinique</span>
                    <span className={styles.payerChoiceHint}>Mensuelle S2</span>
                  </span>
                </label>
                <label
                  className={`${styles.payerChoice} ${payerType === 'partner' ? styles.payerChoiceActive : ''}`}
                >
                  <input
                    type="radio"
                    name="payerType"
                    value="partner"
                    className={styles.payerRadio}
                    checked={payerType === 'partner'}
                    onChange={() => {
                      setPayerType('partner');
                      setPreview(null);
                    }}
                  />
                  <span className={styles.payerChoiceIcon} aria-hidden="true">
                    <FiUsers strokeWidth={2} />
                  </span>
                  <span className={styles.payerChoiceText}>
                    <span className={styles.payerChoiceTitle}>Partenaires</span>
                    <span className={styles.payerChoiceHint}>Inter-entreprises</span>
                  </span>
                </label>
              </div>
            </div>

            <fieldset className={styles.periodFieldset}>
              <legend className={styles.periodLegend}>Période facturée</legend>
              <div className={styles.monthRow}>
                <div className={`${styles.row} ${styles.monthCol}`}>
                  <label htmlFor="bill-period-month">Mois</label>
                  <input
                    id="bill-period-month"
                    className={styles.inputMonth}
                    type="number"
                    min="1"
                    max="12"
                    value={periodMonth}
                    onChange={(e) => {
                      setPeriodMonth(parseInt(e.target.value, 10) || 1);
                      setPreview(null);
                    }}
                  />
                </div>
                <div className={`${styles.row} ${styles.monthCol}`}>
                  <label htmlFor="bill-period-year">Année</label>
                  <input
                    id="bill-period-year"
                    className={styles.inputMonth}
                    type="number"
                    min="2000"
                    max="2100"
                    value={periodYear}
                    onChange={(e) => {
                      setPeriodYear(parseInt(e.target.value, 10) || defaultYear);
                      setPreview(null);
                    }}
                  />
                </div>
              </div>
            </fieldset>

            {payerType === 'patient' && (
              <div className={styles.row}>
                <label htmlFor="bill-period-client">Patient</label>
                <select
                  id="bill-period-client"
                  className={styles.select}
                  value={clientId}
                  onChange={(e) => {
                    setClientId(e.target.value);
                    setPreview(null);
                  }}
                  disabled={loadingLists}
                >
                  <option value="">— Choisir —</option>
                  {clients.map((c) => (
                    <option key={c.id} value={c.id}>
                      {(c.unbilled_count > 0 ? '● ' : '')}
                      {c.first_name} {c.last_name}
                      {c.unbilled_total_amount ? ` (${c.unbilled_total_amount} CHF non fact.)` : ''}
                    </option>
                  ))}
                </select>
              </div>
            )}

            {payerType === 'clinic' && (
              <div className={styles.row}>
                <label htmlFor="bill-period-clinic">Clinique (institution)</label>
                <select
                  id="bill-period-clinic"
                  className={styles.select}
                  value={clinicKey}
                  onChange={(e) => {
                    setClinicKey(e.target.value);
                    setPreview(null);
                  }}
                  disabled={loadingLists}
                >
                  <option value="">— Choisir —</option>
                  {institutions.map((i) => (
                    <option key={i.id} value={String(i.id)}>
                      {i.institution_name}
                      {i.clinic_company_id ? ` (S2 #${i.clinic_company_id})` : ''}
                    </option>
                  ))}
                </select>
              </div>
            )}

            {payerType === 'partner' && (
              <div className={styles.row}>
                <label htmlFor="bill-period-partner">Entreprise partenaire</label>
                <select
                  id="bill-period-partner"
                  className={styles.select}
                  value={partnershipId}
                  onChange={(e) => {
                    setPartnershipId(e.target.value);
                    setPreview(null);
                  }}
                  disabled={loadingLists}
                >
                  <option value="">— Choisir —</option>
                  {billablePartners.map((p) => (
                    <option key={p.partnership_id} value={String(p.partnership_id)}>
                      {p.partner_company_name}
                      {typeof p.validated_unbilled_transfers_count === 'number'
                        ? ` · ${p.validated_unbilled_transfers_count} validé(s)`
                        : ''}
                    </option>
                  ))}
                </select>
              </div>
            )}

            {loadingLists && (
              <p className={styles.formHint} role="status">
                Chargement des listes (patients, institutions, partenaires)…
              </p>
            )}
          </div>

          {error && <div className={styles.err}>{error}</div>}

          {preview && !error && (
            <div className={styles.previewBox}>
              <div className={styles.previewHead}>
                <h3>Aperçu</h3>
                <span className={styles.previewModeBadge}>
                  {preview.mode === 'clinic_monthly'
                    ? 'S2 clinique'
                    : preview.mode === 'partner_monthly'
                      ? 'Facturation partenaire'
                      : 'Direct patient'}
                </span>
              </div>
              <div className={styles.previewStats}>
                <div className={styles.previewStat}>
                  <span className={styles.previewStatValue}>{preview.transports_count ?? 0}</span>
                  <span className={styles.previewStatLabel}>
                    {preview.mode === 'partner_monthly'
                      ? `transfert${preview.transports_count !== 1 ? 's' : ''} validé${preview.transports_count !== 1 ? 's' : ''}`
                      : `transport${preview.transports_count !== 1 ? 's' : ''} éligible${preview.transports_count !== 1 ? 's' : ''}`}
                  </span>
                </div>
                <div className={styles.previewStatHighlight}>
                  <span className={styles.previewStatLabel}>Total estimé</span>
                  <span className={styles.previewStatMoney}>
                    {formatCurrencyCHF(preview.estimated_total ?? 0).replace(' CHF', '')}{' '}
                    <span className={styles.previewStatCurrency}>CHF</span>
                  </span>
                </div>
              </div>
              {Array.isArray(preview.warnings) && preview.warnings.length > 0 && (
                <ul className={styles.warnings}>
                  {preview.warnings.map((w) => (
                    <li key={w}>{w}</li>
                  ))}
                </ul>
              )}
            </div>
          )}

          <div className={styles.footer}>
            {footerHint && (
              <p className={styles.footerHint} id="bill-period-footer-hint">
                {footerHint}
              </p>
            )}
            <div
              className={styles.footerGroup}
              aria-describedby={footerHint ? 'bill-period-footer-hint' : undefined}
            >
              <button
                type="button"
                className={styles.btn}
                onClick={runPreview}
                disabled={!canPreview() || previewLoading || generateLoading}
              >
                {previewLoading ? (
                  <FiLoader className={styles.btnIconSpin} size={14} aria-hidden />
                ) : (
                  <FiEye size={14} aria-hidden />
                )}
                {previewLoading ? 'Prévisualisation…' : 'Prévisualiser'}
              </button>
              <button
                type="button"
                className={`${styles.btn} ${styles.btnPrimary}`}
                onClick={runGenerate}
                disabled={
                  !canPreview() ||
                  !preview ||
                  generateLoading ||
                  previewLoading ||
                  preview.transports_count === 0
                }
              >
                {generateLoading ? (
                  <FiLoader className={styles.btnIconSpin} size={14} aria-hidden />
                ) : (
                  <FiZap size={14} aria-hidden />
                )}
                {generateLoading ? 'Génération…' : 'Générer brouillon'}
              </button>
            </div>
          </div>

          <div className={styles.legacy}>
            <button type="button" className={styles.btnMuted} onClick={onOpenLegacy}>
              <FiList className={styles.btnLinkIcon} aria-hidden />
              Ouvrir l’assistant facture avancé (mode précédent)
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BillPeriodModal;
