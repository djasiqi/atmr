// frontend/src/pages/company/Settings/tabs/components/StatementGenerationForm.jsx
import React, { useState } from 'react';
import styles from '../../CompanySettings.module.css';

const StatementGenerationForm = ({ isConsolidated, partnership, onGenerate, onCancel }) => {
  const [periodType, setPeriodType] = useState('monthly');
  const [year, setYear] = useState(new Date().getFullYear());
  const [month, setMonth] = useState(new Date().getMonth() + 1);
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();

    // Validation
    if (periodType === 'periodic' && (!startDate || !endDate)) {
      alert('Veuillez sélectionner les dates de début et de fin');
      return;
    }

    if (periodType === 'periodic' && new Date(startDate) > new Date(endDate)) {
      alert('La date de début doit être antérieure à la date de fin');
      return;
    }

    // Convertir les dates en ISO si périodique
    let startDateISO = null;
    let endDateISO = null;
    if (periodType === 'periodic') {
      startDateISO = new Date(startDate).toISOString();
      endDateISO = new Date(endDate).toISOString();
    }

    onGenerate(
      isConsolidated,
      periodType,
      periodType === 'annual' || periodType === 'monthly' ? year : null,
      periodType === 'monthly' ? month : null,
      startDateISO,
      endDateISO
    );
  };

  // Générer les options d'années (5 dernières années)
  const currentYear = new Date().getFullYear();
  const years = Array.from({ length: 5 }, (_, i) => currentYear - i);

  const months = [
    { value: 1, label: 'Janvier' },
    { value: 2, label: 'Février' },
    { value: 3, label: 'Mars' },
    { value: 4, label: 'Avril' },
    { value: 5, label: 'Mai' },
    { value: 6, label: 'Juin' },
    { value: 7, label: 'Juillet' },
    { value: 8, label: 'Août' },
    { value: 9, label: 'Septembre' },
    { value: 10, label: 'Octobre' },
    { value: 11, label: 'Novembre' },
    { value: 12, label: 'Décembre' },
  ];

  return (
    <form onSubmit={handleSubmit}>
      {!isConsolidated && partnership && (
        <div className={styles.formGroup}>
          <label className={styles.label}>Partenaire</label>
          <div
            style={{
              padding: 'var(--spacing-sm)',
              background: 'var(--bg-secondary)',
              borderRadius: 'var(--radius-md)',
              border: '1px solid var(--border-primary)',
            }}
          >
            {partnership.partner_company_name_display || partnership.partner_company_name || 'Partenaire inconnu'}
          </div>
        </div>
      )}

      <div className={styles.formGroup}>
        <label className={styles.label}>Type de période</label>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
            <input
              type="radio"
              name="periodType"
              value="annual"
              checked={periodType === 'annual'}
              onChange={(e) => setPeriodType(e.target.value)}
            />
            <span>Annuel (pour comptabilité)</span>
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
            <input
              type="radio"
              name="periodType"
              value="monthly"
              checked={periodType === 'monthly'}
              onChange={(e) => setPeriodType(e.target.value)}
            />
            <span>Mensuel</span>
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
            <input
              type="radio"
              name="periodType"
              value="periodic"
              checked={periodType === 'periodic'}
              onChange={(e) => setPeriodType(e.target.value)}
            />
            <span>Périodique (personnalisé)</span>
          </label>
        </div>
      </div>

      {periodType === 'annual' && (
        <div className={styles.formGroup}>
          <label htmlFor="year" className={styles.label}>
            Année
          </label>
          <select
            id="year"
            value={year}
            onChange={(e) => setYear(parseInt(e.target.value))}
            className={styles.input}
          >
            {years.map((y) => (
              <option key={y} value={y}>
                {y}
              </option>
            ))}
          </select>
        </div>
      )}

      {periodType === 'monthly' && (
        <>
          <div className={styles.formGroup}>
            <label htmlFor="year_monthly" className={styles.label}>
              Année
            </label>
            <select
              id="year_monthly"
              value={year}
              onChange={(e) => setYear(parseInt(e.target.value))}
              className={styles.input}
            >
              {years.map((y) => (
                <option key={y} value={y}>
                  {y}
                </option>
              ))}
            </select>
          </div>
          <div className={styles.formGroup}>
            <label htmlFor="month" className={styles.label}>
              Mois
            </label>
            <select
              id="month"
              value={month}
              onChange={(e) => setMonth(parseInt(e.target.value))}
              className={styles.input}
            >
              {months.map((m) => (
                <option key={m.value} value={m.value}>
                  {m.label}
                </option>
              ))}
            </select>
          </div>
        </>
      )}

      {periodType === 'periodic' && (
        <>
          <div className={styles.formGroup}>
            <label htmlFor="startDate" className={styles.label}>
              Date de début
            </label>
            <input
              id="startDate"
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              className={styles.input}
              required
            />
          </div>
          <div className={styles.formGroup}>
            <label htmlFor="endDate" className={styles.label}>
              Date de fin
            </label>
            <input
              id="endDate"
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              className={styles.input}
              required
            />
          </div>
        </>
      )}

      <div className="modal-footer">
        <button type="button" className="btn btn-secondary" onClick={onCancel}>
          Annuler
        </button>
        <button type="submit" className="btn btn-primary">
          Générer le décompte
        </button>
      </div>
    </form>
  );
};

export default StatementGenerationForm;

