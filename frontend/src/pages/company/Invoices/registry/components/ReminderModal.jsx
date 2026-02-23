import React, { useState, useEffect } from 'react';
import { FiX, FiAlertTriangle, FiClock, FiDollarSign, FiCheckCircle, FiLock } from 'react-icons/fi';
import styles from './ReminderModal.module.css';
import { getNextReminderLevel } from '../../../../../services/invoiceService';
import { fetchBillingSettings } from '../../../../../services/settingsService';

const LEVEL_BADGES = {
  1: { label: '1er', className: styles.levelBadge1 },
  2: { label: '2e', className: styles.levelBadge2 },
  3: { label: 'Dernier', className: styles.levelBadge3 },
};

const ReminderModal = ({ open, invoice, onClose, onReminder }) => {
  const [level, setLevel] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [billingSettings, setBillingSettings] = useState(null);
  const [settingsLoadFailed, setSettingsLoadFailed] = useState(false);
  const [settingsUsingDefaults, setSettingsUsingDefaults] = useState(false);

  useEffect(() => {
    if (!open) return;

    let cancelled = false;

    const loadBillingSettings = async () => {
      try {
        setSettingsLoadFailed(false);
        setSettingsUsingDefaults(false);
        const data = await fetchBillingSettings();

        if (cancelled) return;

        if (data) {
          setBillingSettings(data);
        } else {
          if (!cancelled) {
            setSettingsUsingDefaults(true);
            setBillingSettings({
              reminder1_fee: 0,
              reminder2_fee: 0,
              reminder3_fee: 0,
              reminder_schedule_days: { '1': 10, '2': 5, '3': 5 },
            });
          }
        }
      } catch (err) {
        console.error('[ReminderModal] Erreur lors du chargement des settings:', err);
        if (!cancelled) {
          setSettingsLoadFailed(true);
          setBillingSettings({
            reminder1_fee: 0,
            reminder2_fee: 0,
            reminder3_fee: 0,
            reminder_schedule_days: { '1': 10, '2': 5, '3': 5 },
          });
        }
      }
    };

    loadBillingSettings();

    return () => {
      cancelled = true;
    };
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const handleKey = (e) => {
      if (e.key === 'Escape') handleClose();
    };
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  const getReminderLevels = () => {
    if (!billingSettings) {
      return [
        { value: 1, label: '1er rappel', description: 'Rappel aimable sans frais', fee: 0, delayDays: 10 },
        { value: 2, label: '2e rappel', description: 'Rappel avec frais supplémentaires', fee: 0, delayDays: 5 },
        { value: 3, label: 'Dernier rappel', description: 'Dernier rappel avant mise en demeure', fee: 0, delayDays: 5 },
      ];
    }

    const fees = {
      1: Number(billingSettings.reminder1_fee ?? 0),
      2: Number(billingSettings.reminder2_fee ?? 0),
      3: Number(billingSettings.reminder3_fee ?? 0),
    };

    const delaysRaw = billingSettings.reminder_schedule_days ?? {};
    const delays = {
      1: Number(delaysRaw[1] ?? delaysRaw['1'] ?? 10),
      2: Number(delaysRaw[2] ?? delaysRaw['2'] ?? 5),
      3: Number(delaysRaw[3] ?? delaysRaw['3'] ?? 5),
    };

    return [
      {
        value: 1,
        label: '1er rappel',
        description: fees[1] === 0
          ? 'Rappel aimable sans frais'
          : `Rappel avec frais de ${fees[1].toFixed(2)} CHF`,
        fee: fees[1],
        delayDays: delays[1],
        delayDescription: `${delays[1]} jours après l'échéance`,
      },
      {
        value: 2,
        label: '2e rappel',
        description: fees[2] === 0
          ? 'Rappel sans frais supplémentaires'
          : `Rappel avec frais de ${fees[2].toFixed(2)} CHF`,
        fee: fees[2],
        delayDays: delays[2],
        delayDescription: `${delays[2]} jours après le 1er rappel`,
      },
      {
        value: 3,
        label: 'Dernier rappel',
        description: fees[3] === 0
          ? 'Dernier rappel sans frais supplémentaires'
          : `Dernier rappel avec frais de ${fees[3].toFixed(2)} CHF`,
        fee: fees[3],
        delayDays: delays[3],
        delayDescription: `${delays[3]} jours après le 2e rappel`,
      },
    ];
  };

  const reminderLevels = getReminderLevels();

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      setLoading(true);
      setError(null);
      await onReminder(invoice.id, level);
    } catch (err) {
      setError(err.message || 'Erreur lors de la génération du rappel');
    } finally {
      setLoading(false);
    }
  };

  const handleClose = () => {
    setLevel(1);
    setError(null);
    onClose();
  };

  useEffect(() => {
    if (open && invoice) {
      const next = getNextReminderLevel(invoice);
      setLevel(next);
    }
  }, [open, invoice]);

  if (!open || !invoice) return null;

  const nextLevel = getNextReminderLevel(invoice);
  const selectedLevel = reminderLevels.find((rl) => rl.value === level);
  const fee = selectedLevel?.fee ?? 0;

  const getClientName = () => {
    if (!invoice) return '';
    if (invoice.billed_to_company_id && invoice.billed_to_company) {
      return invoice.billed_to_company.name || 'Clinique';
    }
    if (invoice.client) {
      return (
        invoice.client.institution_name ||
        `${invoice.client.first_name || ''} ${invoice.client.last_name || ''}`.trim() ||
        invoice.client.username ||
        ''
      );
    }
    return 'Client inconnu';
  };

  return (
    <div className={styles.overlay} onClick={handleClose}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className={styles.header}>
          <div className={styles.headerTitle}>
            <FiAlertTriangle size={18} className={styles.headerIcon} />
            <h2>Générer un rappel</h2>
          </div>
          <button className={styles.closeBtn} onClick={handleClose} aria-label="Fermer">
            <FiX size={18} />
          </button>
        </div>

        {/* Invoice summary card */}
        <div className={styles.invoiceCard}>
          <div className={styles.invoiceCardHeader}>
            <span className={styles.invoiceNumber}>{invoice.invoice_number}</span>
            <span className={styles.clientName}>{getClientName()}</span>
          </div>
          <div className={styles.invoiceDetails}>
            <div className={styles.detailRow}>
              <span className={styles.detailLabel}>Niveau actuel</span>
              <span className={styles.detailValue}>
                {invoice.reminder_level === 0 ? 'Aucun rappel' : `Rappel ${invoice.reminder_level}`}
              </span>
            </div>
            <div className={styles.detailRow}>
              <span className={styles.detailLabel}>Dernier rappel</span>
              <span className={styles.detailValue}>
                {invoice.last_reminder_at
                  ? new Date(invoice.last_reminder_at).toLocaleDateString('fr-CH')
                  : 'Jamais'}
              </span>
            </div>
            <hr className={styles.detailSeparator} />
            <div className={styles.detailRow}>
              <span className={styles.detailLabel}>Solde dû</span>
              <span className={styles.detailValueBold}>{invoice.balance_due.toFixed(2)} CHF</span>
            </div>
          </div>
        </div>

        <form onSubmit={handleSubmit}>
          <div className={styles.body}>
            {/* Level selector */}
            <div className={styles.levelSectionLabel}>Niveau du rappel</div>

            {nextLevel > 3 ? (
              <div className={styles.emptyState}>
                Tous les niveaux de rappel ont été utilisés pour cette facture.
              </div>
            ) : (
              <div className={styles.levelOptions}>
                {reminderLevels.map((rl) => {
                  const badge = LEVEL_BADGES[rl.value];
                  const isNext = rl.value === nextLevel;
                  const isDone = rl.value < nextLevel;
                  const isLocked = rl.value > nextLevel;
                  const isActive = level === rl.value;
                  return (
                    <label
                      key={rl.value}
                      className={[
                        styles.levelCard,
                        isActive ? styles.levelCardActive : '',
                        isDone ? styles.levelCardDone : '',
                        isLocked ? styles.levelCardLocked : '',
                      ].filter(Boolean).join(' ')}
                    >
                      {isNext && (
                        <input
                          type="radio"
                          name="level"
                          value={rl.value}
                          checked={isActive}
                          onChange={() => setLevel(rl.value)}
                          className={styles.levelRadio}
                        />
                      )}
                      {isDone && (
                        <div className={styles.levelDoneIcon}>
                          <FiCheckCircle size={16} />
                        </div>
                      )}
                      {isLocked && (
                        <div className={styles.levelLockedIcon}>
                          <FiLock size={14} />
                        </div>
                      )}
                      <div className={styles.levelContent}>
                        <div className={styles.levelHeader}>
                          <span className={styles.levelTitle}>{rl.label}</span>
                          {badge && (
                            <span className={`${styles.levelBadge} ${badge.className}`}>
                              {isDone ? 'Fait' : badge.label}
                            </span>
                          )}
                        </div>
                        <div className={styles.levelDesc}>
                          {isDone
                            ? 'Rappel déjà envoyé'
                            : isLocked
                              ? `Disponible après le rappel niveau ${rl.value - 1}`
                              : rl.description}
                        </div>
                        {isNext && rl.delayDescription && (
                          <div className={styles.levelDelay}>
                            <FiClock size={11} />
                            {rl.delayDescription}
                          </div>
                        )}
                      </div>
                    </label>
                  );
                })}
              </div>
            )}

            {/* Fee summary */}
            {selectedLevel && (
              <div className={styles.feeSummary}>
                <div className={styles.feeSummaryIcon}>
                  <FiDollarSign size={18} />
                </div>
                <div className={styles.feeSummaryContent}>
                  <p className={styles.feeSummaryTitle}>Frais de rappel</p>
                  <p className={`${styles.feeSummaryAmount} ${fee === 0 ? styles.feeSummaryAmountZero : ''}`}>
                    {fee === 0 ? 'Aucun frais' : `${fee.toFixed(2)} CHF`}
                  </p>
                  {selectedLevel.delayDescription && (
                    <p className={styles.feeSummaryDelay}>
                      {selectedLevel.delayDescription}
                    </p>
                  )}
                </div>
              </div>
            )}

            {/* Warnings */}
            {settingsLoadFailed && (
              <div className={styles.settingsWarning}>
                <FiAlertTriangle size={14} />
                Impossible de charger les paramètres de rappels — valeurs par défaut appliquées
              </div>
            )}
            {settingsUsingDefaults && !settingsLoadFailed && (
              <div className={styles.settingsWarning}>
                <FiAlertTriangle size={14} />
                Paramètres de rappel non configurés — valeurs par défaut appliquées
              </div>
            )}

            {error && <div className={styles.error}>{error}</div>}
          </div>

          {/* Footer */}
          <div className={styles.footer}>
            <button
              type="button"
              className={styles.cancelBtn}
              onClick={handleClose}
              disabled={loading}
            >
              Annuler
            </button>
            <button
              type="submit"
              className={styles.submitBtn}
              disabled={loading || nextLevel > 3}
            >
              {loading ? 'Génération...' : `Générer le rappel niveau ${level}`}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default ReminderModal;
