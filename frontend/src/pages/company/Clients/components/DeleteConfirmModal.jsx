import React, { useState } from 'react';
import styles from './DeleteConfirmModal.module.css';

const DeleteConfirmModal = ({ client, onClose, onConfirm }) => {
  const [deleteType, setDeleteType] = useState('soft'); // 'soft' ou 'hard'
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
    <div className={styles.modalOverlay} onClick={onClose}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div className={styles.modalHeader}>
          <h2>Supprimer {isInstitution ? 'l\'institution' : 'le client'}</h2>
          <button className={styles.closeBtn} onClick={onClose}>
            ✕
          </button>
        </div>

        <div className={styles.modalContent}>
          <div className={styles.warningIcon}>⚠️</div>
          
          <p className={styles.message}>
            Voulez-vous vraiment supprimer <strong>{displayName}</strong> ?
          </p>

          <div className={styles.optionsGroup}>
            <label className={styles.optionLabel}>
              <input
                type="radio"
                value="soft"
                checked={deleteType === 'soft'}
                onChange={(e) => setDeleteType(e.target.value)}
                disabled={loading}
              />
              <div className={styles.optionContent}>
                <strong>Désactiver (recommandé)</strong>
                <small>Le client sera masqué mais les données seront conservées</small>
              </div>
            </label>

            <label className={styles.optionLabel}>
              <input
                type="radio"
                value="hard"
                checked={deleteType === 'hard'}
                onChange={(e) => setDeleteType(e.target.value)}
                disabled={loading}
              />
              <div className={styles.optionContent}>
                <strong>Supprimer définitivement</strong>
                <small className={styles.danger}>
                  ⚠️ Action irréversible - Toutes les données seront perdues
                </small>
              </div>
            </label>
          </div>

          <div className={styles.modalActions}>
            <button
              type="button"
              onClick={onClose}
              className={styles.cancelBtn}
              disabled={loading}
            >
              Annuler
            </button>
            <button
              type="button"
              onClick={handleConfirm}
              className={styles.confirmBtn}
              disabled={loading}
            >
              {loading 
                ? 'Suppression...' 
                : deleteType === 'soft' 
                  ? 'Désactiver' 
                  : 'Supprimer définitivement'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DeleteConfirmModal;

