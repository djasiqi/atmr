import React, { useState, useEffect } from 'react';
import styles from './ReminderModal.module.css';
import { getNextReminderLevel } from '../../../../../services/invoiceService';
import { fetchBillingSettings } from '../../../../../services/settingsService';

const ReminderModal = ({ open, invoice, onClose, onReminder }) => {
  const [level, setLevel] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [billingSettings, setBillingSettings] = useState(null);
  const [settingsLoadFailed, setSettingsLoadFailed] = useState(false);
  const [settingsUsingDefaults, setSettingsUsingDefaults] = useState(false);

  // ✅ Charger les settings de rappels au montage du modal
  useEffect(() => {
    if (!open) return;

    let cancelled = false;

    const loadBillingSettings = async () => {
      try {
        setSettingsLoadFailed(false);
        setSettingsUsingDefaults(false);
        const data = await fetchBillingSettings();
        
        if (cancelled) return; // ✅ Guard: éviter setState après fermeture

        if (data) {
          setBillingSettings(data);
          const REMINDER_DEBUG = process.env.REACT_APP_REMINDER_DEBUG === '1';
          if (REMINDER_DEBUG) {
            console.log('[REMINDER_DEBUG] Billing settings loaded:', {
              reminder1_fee: data.reminder1_fee,
              reminder2_fee: data.reminder2_fee,
              reminder3_fee: data.reminder3_fee,
              reminder_schedule_days: data.reminder_schedule_days,
            });
          }
        } else {
          // Pas de données → utiliser defaults
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
          // Fallback: utiliser des valeurs par défaut
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
      cancelled = true; // ✅ Cleanup: annuler si modal fermé
    };
  }, [open]);

  // ✅ Générer dynamiquement les descriptions selon les settings réels
  const getReminderLevels = () => {
    if (!billingSettings) {
      // Fallback si settings non chargés
      return [
        { value: 1, label: '1er rappel', description: 'Rappel aimable sans frais', fee: 0, delayDays: 10 },
        { value: 2, label: '2e rappel', description: 'Rappel avec frais supplémentaires', fee: 0, delayDays: 5 },
        { value: 3, label: 'Dernier rappel', description: 'Dernier rappel avant mise en demeure', fee: 0, delayDays: 5 },
      ];
    }

    // ✅ Sécurisation: convertir en Number pour éviter crash toFixed sur string/Decimal
    const fees = {
      1: Number(billingSettings.reminder1_fee ?? 0),
      2: Number(billingSettings.reminder2_fee ?? 0),
      3: Number(billingSettings.reminder3_fee ?? 0),
    };

    // ✅ Normalisation: gérer clés string ('1') et int (1) pour reminder_schedule_days
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
        delayDescription: `Envoi ${delays[1]} jours après l'échéance`,
      },
      {
        value: 2,
        label: '2e rappel',
        description: fees[2] === 0
          ? 'Rappel sans frais supplémentaires'
          : `Rappel avec frais de ${fees[2].toFixed(2)} CHF`,
        fee: fees[2],
        delayDays: delays[2],
        delayDescription: `Envoi ${delays[2]} jours après le 1er rappel`,
      },
      {
        value: 3,
        label: 'Dernier rappel',
        description: fees[3] === 0
          ? 'Dernier rappel sans frais supplémentaires'
          : `Dernier rappel avec frais de ${fees[3].toFixed(2)} CHF`,
        fee: fees[3],
        delayDays: delays[3],
        delayDescription: `Envoi ${delays[3]} jours après le 2e rappel`,
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

  if (!open || !invoice) return null;

  const nextLevel = getNextReminderLevel(invoice);
  const availableLevels = reminderLevels.filter((rl) => rl.value >= nextLevel);

  return (
    <div className="modal-overlay">
      <div className="modal-content modal-md">
        <div className="modal-header">
          <h2 className="modal-title">Générer un rappel</h2>
          <button className="modal-close" onClick={handleClose}>
            ✕
          </button>
        </div>

        <div className="modal-body">
          <div className={styles.invoiceInfo}>
            <h3>Facture {invoice.invoice_number}</h3>
            <p>
              Client:{' '}
              {invoice.client
                ? invoice.client.institution_name ||
                  `${invoice.client.first_name || ''} ${invoice.client.last_name || ''}`.trim() ||
                  invoice.client.username
                : 'Client inconnu'}
            </p>
            <p>
              Solde dû: <strong>{invoice.balance_due.toFixed(2)} CHF</strong>
            </p>
            <p>
              Niveau actuel:{' '}
              {invoice.reminder_level === 0 ? 'Aucun' : `Rappel ${invoice.reminder_level}`}
            </p>
            <p>
              Dernier rappel:{' '}
              {invoice.last_reminder_at
                ? new Date(invoice.last_reminder_at).toLocaleDateString('fr-FR')
                : 'Jamais'}
            </p>
          </div>

          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label className="form-label required">Niveau du rappel</label>
              <div className={styles.levelOptions}>
                {availableLevels.map((reminderLevel) => (
                  <div key={reminderLevel.value} className={styles.levelOption}>
                    <input
                      type="radio"
                      id={`level-${reminderLevel.value}`}
                      name="level"
                      value={reminderLevel.value}
                      checked={level === reminderLevel.value}
                      onChange={(e) => setLevel(parseInt(e.target.value))}
                      className={styles.radio}
                    />
                    <label htmlFor={`level-${reminderLevel.value}`} className={styles.levelLabel}>
                      <div className={styles.levelTitle}>{reminderLevel.label}</div>
                      <div className={styles.levelDescription}>{reminderLevel.description}</div>
                      {reminderLevel.delayDescription && (
                        <div className={styles.delayDescription} style={{ fontSize: '0.75rem', color: '#888', marginTop: '0.25rem' }}>
                          {reminderLevel.delayDescription}
                        </div>
                      )}
                    </label>
                  </div>
                ))}
              </div>
            </div>

            {level > 0 && (() => {
              const selectedLevel = reminderLevels.find((rl) => rl.value === level);
              const fee = selectedLevel?.fee ?? 0;
              const delayDescription = selectedLevel?.delayDescription ?? '';

              return (
                <div className={styles.feeInfo}>
                  <h4>Frais associés</h4>
                  <p>
                    {fee === 0 ? (
                      `Aucun frais pour le ${selectedLevel?.label || 'rappel'}`
                    ) : (
                      `Frais: ${fee.toFixed(2)} CHF`
                    )}
                  </p>
                  {delayDescription && (
                    <p className={styles.delayInfo} style={{ fontSize: '0.875rem', color: '#666', marginTop: '0.5rem' }}>
                      {delayDescription}
                    </p>
                  )}
                  {settingsLoadFailed && (
                    <p className={styles.settingsWarning} style={{ fontSize: '0.875rem', color: '#856404', marginTop: '0.5rem', fontStyle: 'italic' }}>
                      ⚠️ Impossible de charger les paramètres de rappels — valeurs par défaut appliquées
                    </p>
                  )}
                  {settingsUsingDefaults && !settingsLoadFailed && (
                    <p className={styles.settingsWarning} style={{ fontSize: '0.875rem', color: '#856404', marginTop: '0.5rem', fontStyle: 'italic' }}>
                      ⚠️ Paramètres de rappel non configurés — valeurs par défaut appliquées
                    </p>
                  )}
                </div>
              );
            })()}

            {error && <div className="alert alert-error mb-md">{error}</div>}

            <div className="modal-footer">
              <button
                type="button"
                className="btn btn-secondary"
                onClick={handleClose}
                disabled={loading}
              >
                Annuler
              </button>
              <button
                type="submit"
                className="btn btn-primary"
                disabled={loading || availableLevels.length === 0}
              >
                {loading ? 'Génération...' : `Générer le rappel niveau ${level}`}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default ReminderModal;
