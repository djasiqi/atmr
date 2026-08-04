import React, { useEffect, useId, useState } from 'react';
import styles from './AdminTempPasswordDialog.module.css';

/**
 * Affiche un mot de passe temporaire une seule fois — jamais via alert/toast/console.
 * @param {{
 *   open: boolean,
 *   accountLabel: string,
 *   temporaryPassword: string,
 *   onClose: () => void,
 * }} props
 */
export default function AdminTempPasswordDialog({
  open,
  accountLabel,
  temporaryPassword,
  onClose,
}) {
  const titleId = useId();
  const [visible, setVisible] = useState(false);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!open) return;
    setVisible(false);
    setCopied(false);
  }, [open, temporaryPassword]);

  if (!open) return null;

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(temporaryPassword);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  };

  return (
    <div className={styles.overlay} role="presentation">
      <div
        className={styles.dialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
      >
        <h2 id={titleId} className={styles.title}>
          Mot de passe temporaire généré
        </h2>
        <p className={styles.lead}>
          Compte : <strong>{accountLabel}</strong>
        </p>
        <p className={styles.warn}>
          Ce mot de passe ne sera affiché qu’une seule fois. Communiquez-le de
          façon sécurisée à l’utilisateur, puis fermez cette fenêtre.
        </p>
        <p className={styles.audit}>
          Cette opération est enregistrée côté serveur (réinitialisation
          administrateur).
        </p>

        <div className={styles.secretBox}>
          <code className={styles.secret}>
            {visible ? temporaryPassword : '••••••••••••••••'}
          </code>
          <div className={styles.secretActions}>
            <button type="button" className={styles.btnGhost} onClick={() => setVisible((v) => !v)}>
              {visible ? 'Masquer' : 'Afficher'}
            </button>
            <button type="button" className={styles.btnPrimary} onClick={handleCopy}>
              {copied ? 'Copié' : 'Copier le mot de passe'}
            </button>
          </div>
        </div>

        <div className={styles.footer}>
          <button type="button" className={styles.btnPrimary} onClick={onClose}>
            J’ai noté le mot de passe
          </button>
        </div>
      </div>
    </div>
  );
}
