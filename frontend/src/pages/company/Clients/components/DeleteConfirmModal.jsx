import React, { useState } from 'react';
import styles from './DeleteConfirmModal.module.css';

const DeleteConfirmModal = ({ client, onClose, onConfirm }) => {
  const [deleteType, setDeleteType] = useState('soft');
  const [loading, setLoading] = useState(false);

  const handleConfirm = async () => {
    setLoading(true);
    try {
      await onConfirm(deleteType === 'hard');
    } finally {
      setLoading(false);
    }
  };

  if (!client) return null;

  const isInstitution = client.is_institution;
  const displayName = isInstitution
    ? client.institution_name || 'cette institution'
    : `${client.first_name || ''} ${client.last_name || ''}`.trim() || 'ce client';

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        <h3 className={styles.title}>
          Supprimer {isInstitution ? "l'institution" : 'le client'}
        </h3>
        <p className={styles.message}>
          Voulez-vous vraiment supprimer <strong>{displayName}</strong> ?
        </p>

        <div className={styles.options}>
          <label
            className={`${styles.option} ${deleteType === 'soft' ? styles.optionActive : ''}`}
          >
            <input
              type="radio"
              value="soft"
              checked={deleteType === 'soft'}
              onChange={(e) => setDeleteType(e.target.value)}
              disabled={loading}
              className={styles.radio}
            />
            <div className={styles.optionContent}>
              <span className={styles.optionTitle}>Désactiver (recommandé)</span>
              <span className={styles.optionHint}>Le client sera masqué mais les données seront conservées</span>
            </div>
          </label>

          <label
            className={`${styles.option} ${deleteType === 'hard' ? styles.optionActiveDanger : ''}`}
          >
            <input
              type="radio"
              value="hard"
              checked={deleteType === 'hard'}
              onChange={(e) => setDeleteType(e.target.value)}
              disabled={loading}
              className={styles.radio}
            />
            <div className={styles.optionContent}>
              <span className={styles.optionTitle}>Supprimer définitivement</span>
              <span className={styles.optionHintDanger}>Action irréversible — toutes les données seront perdues</span>
            </div>
          </label>
        </div>

        <div className={styles.actions}>
          <button
            type="button"
            className={styles.cancelBtn}
            onClick={onClose}
            disabled={loading}
          >
            Annuler
          </button>
          <button
            type="button"
            className={`${styles.confirmBtn} ${deleteType === 'hard' ? styles.confirmBtnDanger : styles.confirmBtnDefault}`}
            onClick={handleConfirm}
            disabled={loading}
          >
            {loading
              ? 'Traitement...'
              : deleteType === 'soft'
                ? 'Désactiver'
                : 'Supprimer définitivement'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default DeleteConfirmModal;
