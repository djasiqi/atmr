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
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content modal-md" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2 className="modal-title">Supprimer {isInstitution ? "l'institution" : 'le client'}</h2>
          <button className="modal-close" onClick={onClose}>
            ✕
          </button>
        </div>

        <div className="modal-body">
          <div className="text-center text-6xl mb-md">⚠️</div>

          <p className="text-center text-secondary mb-lg">
            Voulez-vous vraiment supprimer <strong className="text-primary">{displayName}</strong> ?
          </p>

          <div className={styles.optionsGroup}>
            <label className={styles.optionLabel}>
              <input
                type="radio"
                className="form-radio"
                value="soft"
                checked={deleteType === 'soft'}
                onChange={(e) => setDeleteType(e.target.value)}
                disabled={loading}
              />
              <div className="flex-col gap-xs">
                <strong className="text-sm text-primary">Désactiver (recommandé)</strong>
                <small className="text-xs text-tertiary">
                  Le client sera masqué mais les données seront conservées
                </small>
              </div>
            </label>

            <label className={styles.optionLabel}>
              <input
                type="radio"
                className="form-radio"
                value="hard"
                checked={deleteType === 'hard'}
                onChange={(e) => setDeleteType(e.target.value)}
                disabled={loading}
              />
              <div className="flex-col gap-xs">
                <strong className="text-sm text-primary">Supprimer définitivement</strong>
                <small className="text-xs text-error font-medium">
                  ⚠️ Action irréversible - Toutes les données seront perdues
                </small>
              </div>
            </label>
          </div>

          <div className="modal-footer">
            <button
              type="button"
              onClick={onClose}
              className="btn btn-secondary"
              disabled={loading}
            >
              Annuler
            </button>
            <button
              type="button"
              onClick={handleConfirm}
              className="btn btn-danger"
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
