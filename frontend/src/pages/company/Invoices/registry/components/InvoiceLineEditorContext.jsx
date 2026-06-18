import React from 'react';
import {
  lineEditorContextArTag,
  lineEditorContextSubline,
} from '../../../../../utils/invoiceLineRoundTrip';

/**
 * Ligne contexte compacte : client · date · A/R · liens d’exclusion de jambe (inline).
 */
export default function InvoiceLineEditorContext({ line, styles, legActions }) {
  const base = lineEditorContextSubline(line);
  const arTag = lineEditorContextArTag(line);
  const hasLegActions = Boolean(legActions?.enabled);

  if (!base && !arTag && !hasLegActions) return null;

  const arTagClass =
    arTag === 'Retour' ? styles.lineArTagInlineReturn : styles.lineArTagInline;

  return (
    <div className={styles.lineEditorContext}>
      {base ? <span className={styles.lineEditorContextPart}>{base}</span> : null}
      {arTag ? (
        <>
          {base ? <span className={styles.lineEditorContextSep} aria-hidden="true">·</span> : null}
          <span className={arTagClass}>{arTag}</span>
        </>
      ) : null}
      {hasLegActions ? (
        <>
          {base || arTag ? (
            <span className={styles.lineEditorContextSep} aria-hidden="true">
              ·
            </span>
          ) : null}
          <button
            type="button"
            className={styles.lineContextAction}
            disabled={legActions.disabled}
            title={
              legActions.returnTitle ??
              'Conserver l’aller, retirer le retour de la facture'
            }
            onClick={() => legActions.onExcludeLeg('return')}
          >
            sans retour
          </button>
          <span className={styles.lineEditorContextSep} aria-hidden="true">
            ·
          </span>
          <button
            type="button"
            className={styles.lineContextAction}
            disabled={legActions.disabled}
            title={
              legActions.outboundTitle ??
              'Conserver le retour, retirer l’aller de la facture'
            }
            onClick={() => legActions.onExcludeLeg('outbound')}
          >
            sans aller
          </button>
        </>
      ) : null}
    </div>
  );
}
