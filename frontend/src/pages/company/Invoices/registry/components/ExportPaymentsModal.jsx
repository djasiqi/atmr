import React, { useState, useEffect } from 'react';
import { exportPaymentsCSV } from '../../../../../services/invoiceService';
import styles from './ExportPaymentsModal.module.css';

const ExportPaymentsModal = ({ open, onClose, companyId, companyName, initialYear, initialMonth }) => {
  const currentDate = new Date();
  const currentYear = currentDate.getFullYear();
  const currentMonth = currentDate.getMonth() + 1; // 1-12

  // Valeurs initiales : utiliser les filtres si disponibles, sinon mois/année courants
  const [year, setYear] = useState(initialYear || currentYear);
  const [month, setMonth] = useState(initialMonth || currentMonth);
  const [decimalFormat, setDecimalFormat] = useState('comma'); // comma par défaut (Crésus)
  const [withMeta, setWithMeta] = useState(false); // Métadonnées OFF par défaut
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Mettre à jour les valeurs si initialYear/initialMonth changent
  useEffect(() => {
    if (initialYear) setYear(initialYear);
    if (initialMonth) setMonth(initialMonth);
  }, [initialYear, initialMonth]);

  // Générer les options d'années (5 dernières années + année courante)
  const years = [];
  for (let i = currentYear; i >= currentYear - 5; i--) {
    years.push(i);
  }

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

  const handleSubmit = async (e) => {
    e.preventDefault();

    // Empêcher les doubles clics
    if (loading) return;

    if (!year || !month) {
      setError('Veuillez sélectionner une année et un mois');
      return;
    }

    try {
      setLoading(true);
      setError(null);

      await exportPaymentsCSV(companyId, {
        year,
        month,
        decimal: decimalFormat,
        with_meta: withMeta ? 1 : 0,
        companyName,
      });

      // Fermer le modal après téléchargement réussi
      onClose();
    } catch (err) {
      // Gestion des erreurs spécifiques
      const status = err?.response?.status;
      if (status === 401 || status === 403) {
        setError('Accès refusé');
      } else if (status === 404 || err?.response?.data?.size === 0) {
        setError('Aucun paiement enregistré sur cette période');
      } else {
        setError(err.message || "Erreur lors de l'export du CSV");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleClose = () => {
    setError(null);
    onClose();
  };

  if (!open) return null;

  return (
    <div className="modal-overlay">
      <div className="modal-content modal-md">
        <div className="modal-header">
          <h2 className="modal-title">⬇️ Export compta (paiements)</h2>
          <button className="modal-close" onClick={handleClose}>
            ✕
          </button>
        </div>

        <form onSubmit={handleSubmit} className={styles.form}>
          {error && <div className="alert alert-error mb-md">{error}</div>}

          <div className={styles.formGroup}>
            <label className={styles.label}>
              Année <span className={styles.required}>*</span>
            </label>
            <select
              className={styles.select}
              value={year}
              onChange={(e) => setYear(parseInt(e.target.value, 10))}
              required
              disabled={loading}
            >
              {years.map((y) => (
                <option key={y} value={y}>
                  {y}
                </option>
              ))}
            </select>
          </div>

          <div className={styles.formGroup}>
            <label className={styles.label}>
              Mois <span className={styles.required}>*</span>
            </label>
            <select
              className={styles.select}
              value={month}
              onChange={(e) => setMonth(parseInt(e.target.value, 10))}
              required
              disabled={loading}
            >
              {months.map((m) => (
                <option key={m.value} value={m.value}>
                  {m.label}
                </option>
              ))}
            </select>
            <p className={styles.helperText}>L'export est mensuel (un mois précis).</p>
          </div>

          <div className={styles.formGroup}>
            <label className={styles.label}>Format décimal</label>
            <div className={styles.radioGroup}>
              <label className={styles.radioLabel}>
                <input
                  type="radio"
                  name="decimalFormat"
                  value="comma"
                  checked={decimalFormat === 'comma'}
                  onChange={(e) => setDecimalFormat(e.target.value)}
                  disabled={loading}
                />
                <span>Virgule (Crésus) — 150,50</span>
              </label>
              <label className={styles.radioLabel}>
                <input
                  type="radio"
                  name="decimalFormat"
                  value="dot"
                  checked={decimalFormat === 'dot'}
                  onChange={(e) => setDecimalFormat(e.target.value)}
                  disabled={loading}
                />
                <span>Point — 150.50</span>
              </label>
            </div>
          </div>

          <div className={styles.formGroup}>
            <label className={styles.checkboxLabel}>
              <input
                type="checkbox"
                checked={withMeta}
                onChange={(e) => setWithMeta(e.target.checked)}
                disabled={loading}
              />
              <span>Inclure les métadonnées (période, entreprise, date d'export)</span>
            </label>
            <p className={styles.hint}>
              Les métadonnées peuvent causer des problèmes lors de l'import dans certains logiciels
              de comptabilité. Désactivé par défaut.
            </p>
          </div>

          <div className="modal-footer">
            <button
              type="button"
              className="btn btn-secondary"
              onClick={handleClose}
              disabled={loading}
            >
              Annuler
            </button>
            <button type="submit" className="btn btn-primary" disabled={loading}>
              {loading ? (
                <>
                  <span className={styles.spinner}></span>
                  Téléchargement...
                </>
              ) : (
                '📥 Télécharger CSV'
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default ExportPaymentsModal;
