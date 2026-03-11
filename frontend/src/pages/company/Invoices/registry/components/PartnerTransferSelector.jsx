import React, { useCallback, useEffect, useRef, useState } from 'react';
import { toast } from 'sonner';
import { invoiceService } from '../../../../../services/invoiceService';
import styles from './PartnerTransferSelector.module.css';

const PartnerTransferSelector = ({
  companyId,
  partnershipId,
  period,
  overrides = {},
  onOverrideChange,
  onSelectionChange,
}) => {
  const [transfers, setTransfers] = useState([]);
  const [selectedIds, setSelectedIds] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showAdjustments, setShowAdjustments] = useState({});
  const [localInputValues, setLocalInputValues] = useState({});
  const amountInputRefs = useRef({});
  const focusedAmountInputIdRef = useRef(null);

  useEffect(() => {
    const loadTransfers = async () => {
      if (!companyId || !partnershipId || !period?.year || !period?.month) return;

      try {
        setLoading(true);
        setError(null);
        const data = await invoiceService.fetchPartnerTransfers(
          companyId,
          partnershipId,
          { year: period.year, month: period.month }
        );
        const list = Array.isArray(data?.transfers) ? data.transfers : [];
        setTransfers(list);
        setSelectedIds(list.map((t) => t.id));
      } catch (err) {
        console.error('Erreur chargement transferts partenaire:', err);
        setError('Erreur lors du chargement des transferts');
        setTransfers([]);
      } finally {
        setLoading(false);
      }
    };

    loadTransfers();
  }, [companyId, partnershipId, period?.year, period?.month]);

  useEffect(() => {
    if (!onSelectionChange) return;
    const selected = selectedIds
      .map((id) => transfers.find((t) => t.id === id))
      .filter(Boolean);
    const currentKey = JSON.stringify({
      ids: selectedIds.slice().sort((a, b) => a - b),
      overrideVersions: selected.map((t) => {
        const ov = overrides?.[String(t.id)] || overrides?.[t.id] || {};
        return `${t.id}:${ov.amount ?? 'null'}:${ov.note ?? 'null'}`;
      }).sort(),
    });
    const prevKey = amountInputRefs.current._prevKey;
    if (prevKey === currentKey) return;
    amountInputRefs.current._prevKey = currentKey;
    onSelectionChange(selected);
  }, [selectedIds, transfers, overrides, onSelectionChange]);

  const handleToggle = (transferId) => {
    setSelectedIds((prev) =>
      prev.includes(transferId)
        ? prev.filter((id) => id !== transferId)
        : [...prev, transferId]
    );
  };

  const handleToggleAdjustments = (transferId) => {
    setShowAdjustments((prev) => ({
      ...prev,
      [transferId]: !prev[transferId],
    }));
  };

  const normalizeAmount = useCallback((value) => {
    if (!value || value === '' || value === null || value === undefined) {
      return { normalized: null, formatted: null, isValid: true };
    }
    const normalized = String(value).replace(/,/g, '.').trim().replace(/\s/g, '');
    const numeric = parseFloat(normalized);
    if (Number.isNaN(numeric) || numeric < 0) {
      return { normalized: null, formatted: null, isValid: false };
    }
    const rounded = Math.round(numeric * 100) / 100;
    return { normalized: rounded, formatted: rounded.toFixed(2), isValid: true };
  }, []);

  const handleAmountChange = useCallback((transferId, value) => {
    setLocalInputValues((prev) => ({ ...prev, [transferId]: value }));
  }, []);

  const handleAmountBlur = useCallback(
    (transferId, value, originalAmount) => {
      if (!onOverrideChange) return;
      const result = normalizeAmount(value);
      if (!result.isValid) {
        toast.error("Montant invalide. Réinitialisation à la valeur d'origine.");
        onOverrideChange(transferId, {
          amount: originalAmount !== undefined ? originalAmount : null,
        });
        setLocalInputValues((prev) => {
          const next = { ...prev };
          delete next[transferId];
          return next;
        });
      } else if (result.normalized === null) {
        onOverrideChange(transferId, {
          amount: originalAmount !== undefined ? originalAmount : null,
        });
        setLocalInputValues((prev) => {
          const next = { ...prev };
          delete next[transferId];
          return next;
        });
      } else {
        onOverrideChange(transferId, { amount: result.normalized });
        setLocalInputValues((prev) => ({
          ...prev,
          [transferId]: result.formatted,
        }));
      }
    },
    [onOverrideChange, normalizeAmount]
  );

  const handleNoteChange = (transferId, value) => {
    if (!onOverrideChange) return;
    onOverrideChange(transferId, { note: value?.trim?.() ? value : null });
  };

  const formatDate = (dateString) => {
    if (!dateString) return '-';
    try {
      return new Date(dateString).toLocaleDateString('fr-FR', {
        day: '2-digit',
        month: '2-digit',
        year: 'numeric',
      });
    } catch {
      return '-';
    }
  };

  const formatCurrency = (value) =>
    `${Number(value || 0).toFixed(2)} CHF`;

  const computeAmount = useCallback(
    (transfer) => {
      const ov = overrides?.[String(transfer.id)] || overrides?.[transfer.id] || {};
      return Number(ov.amount ?? transfer.partner_cost ?? 0);
    },
    [overrides]
  );

  const handleSelectAll = () => setSelectedIds(transfers.map((t) => t.id));
  const handleDeselectAll = () => setSelectedIds([]);

  if (loading) {
    return <div className={styles.loading}>Chargement des transferts...</div>;
  }
  if (error) {
    return <div className={styles.error}>{error}</div>;
  }
  if (transfers.length === 0) {
    return (
      <div className={styles.empty}>
        <div className={styles.emptyIcon}>🚗</div>
        <p>Aucun transfert non facturé pour cette période</p>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      <div className={styles.headerRow}>
        <span className={styles.headerLabel}>
          {transfers.length} transfert{transfers.length > 1 ? 's' : ''}
        </span>
        {transfers.length > 1 && (
          <div className={styles.actionsInline}>
            <button
              type="button"
              onClick={handleSelectAll}
              className={styles.actionLink}
            >
              Tout sélectionner
            </button>
            <button
              type="button"
              onClick={handleDeselectAll}
              className={styles.actionLink}
            >
              Tout désélectionner
            </button>
          </div>
        )}
      </div>

      <div className={styles.transfersList}>
        {transfers.map((transfer) => {
          const isSelected = selectedIds.includes(transfer.id);
          const showAdjust = showAdjustments[transfer.id];
          const amount = computeAmount(transfer);
          const ov = overrides?.[String(transfer.id)] || overrides?.[transfer.id] || {};
          const routeLabel = `${transfer.pickup_location || '—'} → ${transfer.dropoff_location || '—'}`;

          return (
            <label
              key={transfer.id}
              className={`${styles.transferItem} ${isSelected ? styles.selected : ''}`}
            >
              <input
                type="checkbox"
                checked={isSelected}
                onChange={() => handleToggle(transfer.id)}
                className={styles.checkbox}
              />
              <div className={styles.transferContent}>
                <div className={styles.transferSingleLine}>
                  <span className={styles.date}>{formatDate(transfer.date)}</span>
                  <span className={styles.lineSep}>•</span>
                  <span className={styles.route} title={routeLabel}>
                    {routeLabel}
                  </span>
                  <span className={styles.client}>{transfer.client_name}</span>
                  <span className={styles.amount}>{formatCurrency(amount)}</span>
                  {!showAdjust && (
                    <button
                      type="button"
                      className={styles.adjustLink}
                      onClick={(e) => {
                        e.preventDefault();
                        handleToggleAdjustments(transfer.id);
                      }}
                      title="Ajuster le montant"
                      aria-expanded={showAdjust}
                    >
                      ✏️
                    </button>
                  )}
                </div>

                {showAdjust && (
                  <div className={styles.adjustInline}>
                    <div className={styles.adjustRow}>
                      <span className={styles.adjustLabel}>Montant HT</span>
                      <input
                        ref={(el) => {
                          if (el) amountInputRefs.current[transfer.id] = el;
                          else delete amountInputRefs.current[transfer.id];
                        }}
                        type="number"
                        step="0.05"
                        min="0"
                        className={styles.adjustInput}
                        value={
                          localInputValues[transfer.id] !== undefined
                            ? localInputValues[transfer.id]
                            : ov.amount !== undefined
                              ? ov.amount
                              : ''
                        }
                        placeholder={transfer.partner_cost?.toFixed(2) || '0.00'}
                        onChange={(e) =>
                          handleAmountChange(transfer.id, e.target.value)
                        }
                        onFocus={() => {
                          focusedAmountInputIdRef.current = transfer.id;
                        }}
                        onBlur={(e) => {
                          focusedAmountInputIdRef.current = null;
                          handleAmountBlur(
                            transfer.id,
                            e.target.value,
                            transfer.partner_cost
                          );
                        }}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') e.currentTarget.blur();
                        }}
                      />
                      <span className={styles.adjustSuffix}>CHF</span>
                    </div>
                    <div className={styles.adjustRowNote}>
                      <span className={styles.adjustLabel}>Note (optionnelle)</span>
                      <input
                        type="text"
                        className={styles.adjustNoteInput}
                        value={ov.note ?? ''}
                        placeholder="Ex. Ajustement temps d'attente"
                        onChange={(e) =>
                          handleNoteChange(transfer.id, e.target.value)
                        }
                      />
                    </div>
                    <div className={styles.adjustActions}>
                      <button
                        type="button"
                        className={styles.adjustResetBtn}
                        onClick={(e) => {
                          e.preventDefault();
                          if (onOverrideChange) {
                            onOverrideChange(transfer.id, {
                              amount: null,
                              note: null,
                            });
                          }
                        }}
                      >
                        Réinitialiser
                      </button>
                      <button
                        type="button"
                        className={styles.adjustLink}
                        onClick={(e) => {
                          e.preventDefault();
                          handleToggleAdjustments(transfer.id);
                        }}
                      >
                        Fermer
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </label>
          );
        })}
      </div>

      <div className={styles.summaryMinimal}>
        {selectedIds.length} sélectionné{selectedIds.length > 1 ? 's' : ''} • Total{' '}
        {formatCurrency(
          transfers
            .filter((t) => selectedIds.includes(t.id))
            .reduce((sum, t) => sum + computeAmount(t), 0)
        )}
      </div>
    </div>
  );
};

export default PartnerTransferSelector;
