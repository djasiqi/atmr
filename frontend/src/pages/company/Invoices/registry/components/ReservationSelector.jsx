import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { invoiceService } from '../../../../../services/invoiceService';
import styles from './ReservationSelector.module.css';

const ReservationSelector = ({
  companyId,
  clientId,
  clientName,
  period,
  billToType,
  vatConfig,
  overrides = {},
  onOverrideChange,
  onSelectionChange,
  preselectedIds = [],
}) => {
  const [reservations, setReservations] = useState([]);
  const [selectedIds, setSelectedIds] = useState([]);
  const [filter, setFilter] = useState('all');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hasAutoSelected, setHasAutoSelected] = useState(false);
  useEffect(() => {
    if (!reservations.length) return;
    if (!Array.isArray(preselectedIds) || preselectedIds.length === 0) return;
    const normalized = preselectedIds
      .map((id) => Number(id))
      .filter((id) => !Number.isNaN(id) && reservations.some((res) => res.id === id));
    if (normalized.length === 0) return;

    const currentKey = [...selectedIds].sort((a, b) => a - b).join(',');
    const nextKey = [...normalized].sort((a, b) => a - b).join(',');

    if (currentKey !== nextKey) {
      setSelectedIds(normalized);
      setHasAutoSelected(true);
    }
  }, [preselectedIds, reservations, selectedIds]);

  const vatApplicable = Boolean(vatConfig?.applicable);
  const defaultVatRate = vatApplicable
    ? Number.isFinite(Number(vatConfig?.defaultRate))
      ? Number(vatConfig.defaultRate)
      : 0
    : 0;

  useEffect(() => {
    const loadReservations = async () => {
      if (!companyId || !clientId || !period.year || !period.month) return;

      try {
        setLoading(true);
        setError(null);

        const data = await invoiceService.fetchUnbilledReservations(companyId, clientId, {
          year: period.year,
          month: period.month,
          billed_to_type: filter !== 'all' ? filter : undefined,
        });

        const list = Array.isArray(data?.reservations) ? data.reservations : [];
        setReservations(list);

        // Retirer les sélections qui n'existent plus
        setSelectedIds((prev) => prev.filter((id) => list.some((res) => res.id === id)));
      } catch (err) {
        console.error('Erreur chargement réservations:', err);
        setError('Erreur lors du chargement des transports');
        setReservations([]);
      } finally {
        setLoading(false);
      }
    };

    loadReservations();
  }, [companyId, clientId, period, filter]);

  useEffect(() => {
    if (billToType && reservations.length > 0 && !hasAutoSelected) {
      const matching = reservations.filter((r) => r.billed_to_type === billToType).map((r) => r.id);
      if (matching.length > 0) {
        setSelectedIds(matching);
        setHasAutoSelected(true);
      }
    }
  }, [reservations, billToType, hasAutoSelected]);

  useEffect(() => {
    if (!onSelectionChange) return;
    const selected = reservations.filter((r) => selectedIds.includes(r.id));
    onSelectionChange(selected);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedIds, reservations]);

  const handleToggle = (reservationId) => {
    setSelectedIds((prev) =>
      prev.includes(reservationId)
        ? prev.filter((id) => id !== reservationId)
        : [...prev, reservationId]
    );
  };

  const handleSelectAll = () => {
    const allIds = reservations.map((r) => r.id);
    setSelectedIds(allIds);
  };

  const handleDeselectAll = () => {
    setSelectedIds([]);
  };

  const handleAmountChange = (reservationId, value) => {
    const numeric = parseFloat(value);
    if (!onOverrideChange) return;
    if (value === '') {
      onOverrideChange(reservationId, { amount: null });
    } else if (!Number.isNaN(numeric)) {
      onOverrideChange(reservationId, { amount: numeric });
    }
  };

  const handleVatChange = (reservationId, value) => {
    if (!onOverrideChange) return;
    const numeric = parseFloat(value);
    if (value === '') {
      onOverrideChange(reservationId, { vat_rate: null });
    } else if (!Number.isNaN(numeric)) {
      onOverrideChange(reservationId, { vat_rate: numeric });
    }
  };

  const handleNoteChange = (reservationId, value) => {
    if (!onOverrideChange) return;
    onOverrideChange(reservationId, { note: value?.trim?.() ? value : null });
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

  const getBillingTypeLabel = (type) => {
    const labels = {
      patient: '👤 Patient',
      clinic: '🏥 Clinique',
      insurance: '🏢 Assurance',
    };
    return labels[type] || type;
  };

  const getBillingTypeClass = (type) => type || 'patient';

  const computeAmounts = useCallback(
    (reservation) => {
      const override = overrides?.[reservation.id] || {};
      const amount = Number(
        override.amount ?? reservation.amount ?? reservation.estimated_amount ?? 0
      );
      const vatRate = vatApplicable
        ? Number(
            override.vat_rate ??
              reservation.vat_rate ??
              reservation.default_vat_rate ??
              defaultVatRate
          )
        : 0;
      const sanitizedAmount = Number.isNaN(amount) ? 0 : amount;
      const sanitizedVatRate = Number.isNaN(vatRate) ? 0 : vatRate;
      const vatValue = vatApplicable
        ? Number(((sanitizedAmount * sanitizedVatRate) / 100).toFixed(2))
        : 0;
      const total = Number((sanitizedAmount + vatValue).toFixed(2));

      return {
        amount: sanitizedAmount,
        vatRate: sanitizedVatRate,
        vatValue,
        total,
        note: override.note || '',
      };
    },
    [overrides, vatApplicable, defaultVatRate]
  );

  const selectedReservations = useMemo(
    () => reservations.filter((r) => selectedIds.includes(r.id)),
    [reservations, selectedIds]
  );

  const summaryTotals = useMemo(() => {
    return selectedReservations.reduce(
      (acc, reservation) => {
        const figures = computeAmounts(reservation);
        acc.base += figures.amount;
        acc.vat += figures.vatValue;
        acc.total += figures.total;
        return acc;
      },
      { base: 0, vat: 0, total: 0 }
    );
  }, [selectedReservations, computeAmounts]);

  const formatCurrency = (value) => `${Number(value || 0).toFixed(2)} CHF`;

  if (loading) {
    return <div className={styles.loading}>Chargement des transports...</div>;
  }

  if (error) {
    return <div className={styles.error}>{error}</div>;
  }

  if (reservations.length === 0) {
    return (
      <div className={styles.empty}>
        <div className={styles.emptyIcon}>🚗</div>
        <p>Aucun transport non facturé pour cette période</p>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h4 className={styles.clientName}>{clientName}</h4>
        <div className={styles.filterButtons}>
          <button
            type="button"
            className={`${styles.filterBtn} ${filter === 'all' ? styles.active : ''}`}
            onClick={() => setFilter('all')}
          >
            Tous ({reservations.length})
          </button>
          <button
            type="button"
            className={`${styles.filterBtn} ${filter === 'clinic' ? styles.active : ''}`}
            onClick={() => setFilter('clinic')}
          >
            🏥 Clinique
          </button>
          <button
            type="button"
            className={`${styles.filterBtn} ${filter === 'patient' ? styles.active : ''}`}
            onClick={() => setFilter('patient')}
          >
            👤 Patient
          </button>
        </div>
      </div>

      <div className={styles.actions}>
        <button type="button" onClick={handleSelectAll} className={styles.actionBtn}>
          Tout sélectionner
        </button>
        <button type="button" onClick={handleDeselectAll} className={styles.actionBtn}>
          Tout désélectionner
        </button>
      </div>

      <div className={styles.reservationsList}>
        {reservations.map((reservation) => {
          const isSelected = selectedIds.includes(reservation.id);
          const figures = computeAmounts(reservation);

          return (
            <label
              key={reservation.id}
              className={`${styles.reservationItem} ${isSelected ? styles.selected : ''}`}
            >
              <input
                type="checkbox"
                checked={isSelected}
                onChange={() => handleToggle(reservation.id)}
                className={styles.checkbox}
              />

              <div className={styles.reservationContent}>
                <div className={styles.reservationHeader}>
                  <span className={styles.date}>{formatDate(reservation.date)}</span>
                  <div className={styles.amountStack}>
                    <span className={styles.amount}>{formatCurrency(figures.amount)}</span>
                    {vatApplicable && (
                      <span className={styles.amountVat}>
                        TVA {figures.vatRate.toFixed(2)}% · {formatCurrency(figures.vatValue)}
                      </span>
                    )}
                    <span className={styles.amountTotal}>
                      Total {formatCurrency(figures.total)}
                    </span>
                  </div>
                </div>

                <div className={styles.route}>
                  <span className={styles.location}>{reservation.pickup_location}</span>
                  <span className={styles.arrow}>→</span>
                  <span className={styles.location}>{reservation.dropoff_location}</span>
                </div>

                <div className={styles.reservationFooter}>
                  <span
                    className={`${styles.badge} ${
                      styles[getBillingTypeClass(reservation.billed_to_type)]
                    }`}
                  >
                    {getBillingTypeLabel(reservation.billed_to_type)}
                  </span>
                  {reservation.is_return && <span className={styles.returnBadge}>↩ Retour</span>}
                  {reservation.is_urgent && <span className={styles.urgentBadge}>⚡ Urgent</span>}
                  {reservation.medical_facility && (
                    <span className={styles.medicalBadge}>🏥 {reservation.medical_facility}</span>
                  )}
                </div>

                {isSelected && (
                  <div className={styles.adjustments}>
                    <div className={styles.adjustGrid}>
                      <label className={styles.field}>
                        <span>Montant HT</span>
                        <input
                          type="number"
                          step="0.05"
                          min="0"
                          className={styles.input}
                          value={
                            overrides?.[reservation.id]?.amount !== undefined
                              ? overrides[reservation.id].amount
                              : ''
                          }
                          placeholder={figures.amount.toFixed(2)}
                          onChange={(e) => handleAmountChange(reservation.id, e.target.value)}
                        />
                      </label>

                      {vatApplicable && (
                        <label className={styles.field}>
                          <span>TVA %</span>
                          <input
                            type="number"
                            step="0.1"
                            min="0"
                            className={styles.input}
                            value={
                              overrides?.[reservation.id]?.vat_rate !== undefined
                                ? overrides[reservation.id].vat_rate
                                : ''
                            }
                            placeholder={figures.vatRate.toFixed(2)}
                            onChange={(e) => handleVatChange(reservation.id, e.target.value)}
                          />
                        </label>
                      )}
                    </div>

                    <label className={styles.field}>
                      <span>Note d’ajustement (facultatif)</span>
                      <textarea
                        rows={2}
                        className={styles.noteInput}
                        value={overrides?.[reservation.id]?.note ?? ''}
                        placeholder="Ex. Ajustement temps d’attente"
                        onChange={(e) => handleNoteChange(reservation.id, e.target.value)}
                      />
                    </label>

                    <div className={styles.adjustSummary}>
                      <span>
                        HT <strong>{formatCurrency(figures.amount)}</strong>
                      </span>
                      {vatApplicable && (
                        <span>
                          TVA <strong>{formatCurrency(figures.vatValue)}</strong>
                        </span>
                      )}
                      <span>
                        TTC <strong>{formatCurrency(figures.total)}</strong>
                      </span>
                    </div>
                  </div>
                )}
              </div>
            </label>
          );
        })}
      </div>

      <div className={styles.summary}>
        <div className={styles.summaryItem}>
          <div className={styles.summaryLabel}>
            {selectedIds.length} transport(s) sélectionné(s)
          </div>
          <div className={styles.summaryAmounts}>
            <span>
              HT : <strong>{formatCurrency(summaryTotals.base)}</strong>
            </span>
            {vatApplicable && (
              <span>
                TVA : <strong>{formatCurrency(summaryTotals.vat)}</strong>
              </span>
            )}
            <span className={styles.summaryTotal}>
              TTC : <strong>{formatCurrency(summaryTotals.total)}</strong>
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ReservationSelector;
