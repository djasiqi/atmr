/**
 * VirtualizedReservationTable.jsx
 *
 * Composant virtualisé pour ReservationTable utilisant react-window.
 *
 * Ce composant virtualise le rendu du tableau de réservations pour améliorer
 * les performances avec de grandes listes.
 *
 * @module components/virtualized/VirtualizedReservationTable
 */

import React, { useCallback, useMemo, useRef, useLayoutEffect, useState } from 'react';
import PropTypes from 'prop-types';
import { List } from 'react-window';
import { FiCheckCircle, FiXCircle } from 'react-icons/fi';
import { renderBookingDateTime } from '../../utils/formatDate';
import ReservationActions from '../reservations/ReservationActions';
import styles from './VirtualizedReservationTable.module.css';


/**
 * Composant VirtualizedReservationTable
 *
 * Version virtualisée de ReservationTable qui n'affiche que les lignes visibles
 * dans le viewport.
 */
const VirtualizedReservationTable = ({
  reservations: reservationsProp,
  onRowClick,
  onAccept,
  onReject,
  onAssign,
  onEdit,
  onDelete,
  onSchedule,
  onDispatchNow,
  hideAssign = false,
  hideSchedule = false,
  hideUrgent = false,
  hideEdit = false,
  hideDelete = false,
}) => {
  // #region agent log
  fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'VirtualizedReservationTable.jsx:43',message:'Props reçues',data:{reservationsProp:reservationsProp?reservationsProp.length:0,isArray:Array.isArray(reservationsProp),type:typeof reservationsProp},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
  // #endregion
  
  // ✅ Sécurité: Garantir que reservations est toujours un array (jamais undefined/null)
  // Cela évite les erreurs dans useMemo et dans react-window
  const reservations = useMemo(
    () => {
      const result = Array.isArray(reservationsProp) ? reservationsProp : [];
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'VirtualizedReservationTable.jsx:49',message:'Reservations après useMemo',data:{length:result.length,firstItem:result[0]?Object.keys(result[0]):null},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
      // #endregion
      return result;
    },
    [reservationsProp]
  );
  const listRef = useRef(null);
  const containerRef = useRef(null);
  const headerRef = useRef(null);
  const rowHeightsRef = useRef({});
  const [columnWidths, setColumnWidths] = useState([]);

  // Hauteur estimée par défaut (variable selon le contenu)
  const DEFAULT_ROW_HEIGHT = 80;

  // ✅ Valeur par défaut stable dans useRef (toujours disponible, même avant useMemo)
  const defaultItemDataRef = useRef({
    reservations: [],
    onRowClick: () => {},
    onAccept: () => {},
    onReject: () => {},
    onAssign: () => {},
    onEdit: () => {},
    onDelete: () => {},
    onSchedule: () => {},
    onDispatchNow: () => {},
    hideAssign: false,
    hideSchedule: false,
    hideUrgent: false,
    hideEdit: false,
    hideDelete: false,
  });

  // ✅ FIX PROPRE: itemData calculé via useMemo avec valeur par défaut stable
  // CRITIQUE: react-window's useMemoizedObject appelle Object.values() sur itemData
  // Dans React 18 (StrictMode/concurrent), useMemo peut être évalué après l'accès initial
  // Solution: useMemo retourne TOUJOURS un objet valide, jamais null/undefined
  const itemDataMemo = useMemo(
    () => {
      // ✅ Garantir que l'objet retourné est TOUJOURS valide (non-null, non-undefined, non-array)
      const data = {
        reservations: Array.isArray(reservations) ? reservations : [],
        onRowClick: typeof onRowClick === 'function' ? onRowClick : () => {},
        onAccept: typeof onAccept === 'function' ? onAccept : () => {},
        onReject: typeof onReject === 'function' ? onReject : () => {},
        onAssign: typeof onAssign === 'function' ? onAssign : () => {},
        onEdit: typeof onEdit === 'function' ? onEdit : () => {},
        onDelete: typeof onDelete === 'function' ? onDelete : () => {},
        onSchedule: typeof onSchedule === 'function' ? onSchedule : () => {},
        onDispatchNow: typeof onDispatchNow === 'function' ? onDispatchNow : () => {},
        hideAssign: Boolean(hideAssign),
        hideSchedule: Boolean(hideSchedule),
        hideUrgent: Boolean(hideUrgent),
        hideEdit: Boolean(hideEdit),
        hideDelete: Boolean(hideDelete),
      };

      // ✅ Mettre à jour la ref avec la nouvelle valeur (pour fallback)
      defaultItemDataRef.current = data;

      return data;
    },
    [
      reservations,
      onRowClick,
      onAccept,
      onReject,
      onAssign,
      onEdit,
      onDelete,
      onSchedule,
      onDispatchNow,
      hideAssign,
      hideSchedule,
      hideUrgent,
      hideEdit,
      hideDelete,
    ]
  );

  // ✅ Synchroniser itemDataMemo avec defaultItemDataRef de manière synchrone
  // useLayoutEffect s'exécute de manière synchrone avant le rendu du DOM
  useLayoutEffect(() => {
    if (
      itemDataMemo &&
      typeof itemDataMemo === 'object' &&
      !Array.isArray(itemDataMemo) &&
      itemDataMemo !== null
    ) {
      defaultItemDataRef.current = itemDataMemo;
    }
  }, [itemDataMemo]);

  // ✅ Utiliser directement itemDataMemo avec fallback vers defaultItemDataRef
  // Pas de useState pour éviter les re-renders qui pourraient causer des états transitoires
  // defaultItemDataRef.current est TOUJOURS valide, même si useMemo n'est pas encore évalué
  const itemData = 
    itemDataMemo &&
    typeof itemDataMemo === 'object' &&
    !Array.isArray(itemDataMemo) &&
    itemDataMemo !== null
      ? itemDataMemo
      : defaultItemDataRef.current;

  const getItemSize = useCallback(
    (index) => {
      // Retourner la hauteur stockée ou la hauteur par défaut
      return rowHeightsRef.current[index] || DEFAULT_ROW_HEIGHT;
    },
    [DEFAULT_ROW_HEIGHT]
  );

  // ✅ Protection: garantir que getItemSize est toujours une fonction valide
  const safeGetItemSize = useCallback(
    (index) => {
      if (typeof getItemSize === 'function') {
        return getItemSize(index);
      }
      return DEFAULT_ROW_HEIGHT;
    },
    [getItemSize, DEFAULT_ROW_HEIGHT]
  );


  // Composant de ligne pour react-window v2
  // ✅ react-window v2 passe: index, style, data (qui correspond à rowProps)
  const Row = useCallback(
    ({ index, style, data, rowProps, itemData, columnWidths: rowColumnWidths, ...rest }) => {
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'VirtualizedReservationTable.jsx:175',message:'Row appelé',data:{index,hasData:!!data,hasRowProps:!!rowProps,rowPropsKeys:rowProps?Object.keys(rowProps):null},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'E'})}).catch(()=>{});
      // #endregion
      
      // ✅ Protection: vérifier que toutes les props sont valides
      if (typeof index !== 'number' || index < 0) {
        console.warn('[VirtualizedReservationTable] Row: index invalide', index);
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'VirtualizedReservationTable.jsx:181',message:'Row: index invalide',data:{index},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'E'})}).catch(()=>{});
        // #endregion
        return null;
      }
      
      // ✅ Résoudre la source de données : react-window v2 peut passer rowProps au lieu de data
      const resolvedData = data ?? rowProps ?? itemData ?? (Object.keys(rest).length > 0 ? rest : defaultItemDataRef.current);
      
      // ✅ Protection: garantir que resolvedData est un objet valide
      if (!resolvedData || typeof resolvedData !== 'object' || Array.isArray(resolvedData) || resolvedData === null) {
        console.warn('[VirtualizedReservationTable] Row: data invalide', { data, rowProps, itemData, resolvedData });
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'VirtualizedReservationTable.jsx:200',message:'Row: data invalide',data:{hasData:!!data,hasRowProps:!!rowProps,hasItemData:!!itemData},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'E'})}).catch(()=>{});
        // #endregion
        return null;
      }
      
      // ✅ Protection: garantir que style est un objet valide (peut être null/undefined de react-window)
      const safeStyle = style && typeof style === 'object' && !Array.isArray(style) && style !== null
        ? style
        : {};

      const r = resolvedData?.reservations?.[index] || reservations[index];
      
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'VirtualizedReservationTable.jsx:212',message:'Row: réservation récupérée',data:{index,hasR:!!r,reservationsLength:resolvedData?.reservations?.length||reservations.length,rId:r?.id},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'E'})}).catch(()=>{});
      // #endregion
      
      if (!r) {
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'VirtualizedReservationTable.jsx:217',message:'Row: réservation null',data:{index,reservationsLength:resolvedData?.reservations?.length||reservations.length},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'E'})}).catch(()=>{});
        // #endregion
        return null;
      }

      const status = r.status?.toLowerCase() || 'unknown';
      const isReturn = !!r.is_return;

      // Aucune action pour les statuts terminaux
      const noActionStatuses = [
        'canceled',
        'cancelled',
        'completed',
        'return_completed',
        'rejected',
        'no_show',
      ];
      const hasActions = !noActionStatuses.includes(status);

      const handleRowClick = () => {
        if (resolvedData?.onRowClick) {
          resolvedData.onRowClick(r);
        }
      };

      const handleActionClick = (e) => {
        e.stopPropagation(); // Empêche d'ouvrir le modal en cliquant sur un bouton
      };

      // ✅ Utiliser les largeurs de colonnes mesurées pour l'alignement
      // Les largeurs peuvent venir de rowProps (passées via rowProps.columnWidths) ou directement de columnWidths
      const cellStyle = { 
        display: 'table-cell', 
        paddingTop: 'var(--spacing-sm)',
        paddingBottom: 'var(--spacing-xxs)', // Réduire le padding après le texte
        paddingLeft: 'var(--spacing-xs)',
        paddingRight: 'var(--spacing-xs)',
        verticalAlign: 'middle',
        fontSize: 'var(--font-sm)', // Réduire la taille du texte
        boxSizing: 'border-box',
        borderRight: 'none', // Supprimer les bordures latérales
        borderLeft: 'none',
        borderTop: 'none',
        outline: 'none',
      };
      const widths = (resolvedData && resolvedData.columnWidths) || rowColumnWidths || columnWidths;
      const getCellWidth = (cellIndex) => {
        if (widths && Array.isArray(widths) && widths.length > cellIndex && widths[cellIndex] > 0) {
          return { 
            width: `${widths[cellIndex]}px`, 
            minWidth: `${widths[cellIndex]}px`, 
            maxWidth: `${widths[cellIndex]}px`,
          };
        }
        return {};
      };

      return (
        <div
          style={{
            ...safeStyle,
            display: 'table-row',
            width: '100%',
            cursor: 'pointer',
          }}
          className={styles.virtualizedRow}
          onClick={handleRowClick}
        >
          <div style={{ ...cellStyle, ...getCellWidth(0) }} className={styles.clientCell}>
            {r.client?.full_name || r.client_name || '—'}
          </div>
          <div style={{ ...cellStyle, ...getCellWidth(1) }}>
            {renderBookingDateTime(r)}
          </div>
          <div style={{ ...cellStyle, ...getCellWidth(2) }} className={styles.locationCell}>
            <div>
              <strong>De:</strong> {r.pickup_location || '—'}
            </div>
            <div>
              <strong>À:</strong> {r.dropoff_location || '—'}
            </div>
          </div>
          <div style={{ ...cellStyle, ...getCellWidth(3) }}>
            {Number(r.amount || 0).toFixed(2)} CHF
          </div>
          <div style={{ ...cellStyle, ...getCellWidth(4) }}>
            <span className={`${styles.statusBadge} ${styles[status] || ''}`}>
              {(r.status || '').replace('_', ' ') || status}
            </span>
          </div>
          <div
            style={{ ...cellStyle, ...getCellWidth(5), textAlign: 'right' }}
            className={styles.actionsCell}
            onClick={handleActionClick}
          >
            {!hasActions ? (
              <span className="text-tertiary text-sm italic">Aucune action</span>
            ) : (
              <>
                {status === 'pending' && !isReturn && (
                  <>
                    <button
                      onClick={() => resolvedData?.onAccept?.(r.id)}
                      title="Accepter"
                      className={`${styles.actionButton} ${styles.acceptButton}`}
                    >
                      <FiCheckCircle />
                    </button>
                    <button
                      onClick={() => resolvedData?.onReject?.(r.id)}
                      title="Rejeter"
                      className={`${styles.actionButton} ${styles.rejectButton}`}
                    >
                      <FiXCircle />
                    </button>
                  </>
                )}
                <ReservationActions
                  reservation={r}
                  onSchedule={resolvedData?.onSchedule}
                  onDispatchNow={resolvedData?.onDispatchNow}
                  onAssign={resolvedData?.onAssign}
                  onEdit={resolvedData?.onEdit}
                  onDelete={resolvedData?.onDelete}
                  hideAssign={resolvedData?.hideAssign}
                  hideSchedule={resolvedData?.hideSchedule}
                  hideUrgent={resolvedData?.hideUrgent}
                  hideEdit={resolvedData?.hideEdit}
                  hideDelete={resolvedData?.hideDelete}
                />
              </>
            )}
          </div>
        </div>
      );
    },
    [reservations, columnWidths]
  );

  // Hauteur du conteneur virtualisé
  const VIRTUALIZED_HEIGHT = useMemo(() => {
    if (typeof window !== 'undefined') {
      return Math.min(600, window.innerHeight * 0.6);
    }
    return 600; // Fallback pour SSR
  }, []);

  // ✅ Largeur du conteneur (doit être un nombre, jamais undefined)
  const [containerWidth, setContainerWidth] = useState(() => {
    if (typeof window !== 'undefined') {
      return window.innerWidth;
    }
    return 800;
  });

  // ✅ Mesurer la largeur du conteneur et les largeurs des colonnes du header
  useLayoutEffect(() => {
    const updateWidth = () => {
      if (containerRef.current) {
        const width = containerRef.current.clientWidth;
        if (width > 0) {
          setContainerWidth(width);
        }
      } else if (typeof window !== 'undefined') {
        setContainerWidth(window.innerWidth);
      }
    };

    const updateColumnWidths = () => {
      if (headerRef.current) {
        const thElements = headerRef.current.querySelectorAll('th');
        const widths = Array.from(thElements).map((th) => th.offsetWidth);
        if (widths.length > 0 && widths.some((w) => w > 0)) {
          setColumnWidths(widths);
        }
      }
    };

    updateWidth();
    // Attendre le prochain frame pour que le header soit rendu
    requestAnimationFrame(() => {
      updateColumnWidths();
    });

    if (typeof window !== 'undefined') {
      window.addEventListener('resize', () => {
        updateWidth();
        updateColumnWidths();
      });
      return () => window.removeEventListener('resize', () => {
        updateWidth();
        updateColumnWidths();
      });
    }
  }, []);

  if (reservations.length === 0) {
    return (
      <div className={styles.tableContainer}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>Client</th>
              <th>Date / Heure</th>
              <th>Lieu</th>
              <th>Montant</th>
              <th>Statut</th>
              <th className={styles.actionsCell}>Actions</th>
            </tr>
          </thead>
        </table>
        <div className={styles.emptyMessage}>Aucune réservation à afficher.</div>
      </div>
    );
  }

  return (
    <div className={styles.tableContainer} ref={containerRef}>
      {/* Header fixe */}
      <table className={styles.table} ref={headerRef}>
        <thead>
          <tr>
            <th>Client</th>
            <th>Date / Heure</th>
            <th>Lieu</th>
            <th>Montant</th>
            <th>Statut</th>
            <th className={styles.actionsCell}>Actions</th>
          </tr>
        </thead>
      </table>
      {/* Body virtualisé - Structure hybride pour aligner les colonnes */}
      <div 
        style={{ 
          display: 'table', 
          width: '100%', 
          tableLayout: 'fixed',
          minWidth: columnWidths.length > 0 ? columnWidths.reduce((a, b) => a + b, 0) : '800px'
        }} 
        className={styles.table}
      >
        {/* ✅ Sécurité: Ne rendre List que si reservations est un array non vide */}
        {/* ✅ itemData est TOUJOURS un objet valide grâce à useState initialisé avec defaultItemDataRef */}
        {/* ✅ Protection ultime: useState garantit toujours une valeur valide, même au premier render */}
        {Array.isArray(reservations) && reservations.length > 0 ? (
          (() => {
            // #region agent log
            fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'VirtualizedReservationTable.jsx:418',message:'Rendu List avec données',data:{reservationsLength:reservations.length,isArray:Array.isArray(reservations)},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'C'})}).catch(()=>{});
            // #endregion
            
            // ✅ Protection ultime: vérifier que toutes les props sont valides
            const safeHeight = typeof VIRTUALIZED_HEIGHT === 'number' && !isNaN(VIRTUALIZED_HEIGHT) && VIRTUALIZED_HEIGHT > 0
              ? VIRTUALIZED_HEIGHT
              : 600;
            const safeItemCount = typeof reservations.length === 'number' && reservations.length >= 0
              ? reservations.length
              : 0;
            
            // #region agent log
            fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'VirtualizedReservationTable.jsx:428',message:'Props List calculées',data:{safeHeight,safeItemCount,columnWidthsLength:columnWidths.length},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'C'})}).catch(()=>{});
            // #endregion
            
            // ✅ CRITIQUE: rowProps doit TOUJOURS être un objet (au minimum {})
            // Sinon useMemoizedObject appelle Object.values(undefined) => crash
            const safeItemData = itemData && typeof itemData === 'object' && !Array.isArray(itemData) && itemData !== null
              ? itemData
              : defaultItemDataRef.current;
            
            // ✅ Largeur du conteneur (garantie d'être un nombre, jamais undefined)
            const safeWidth = typeof containerWidth === 'number' && !isNaN(containerWidth) && containerWidth > 0 
              ? containerWidth 
              : 800;
            
            
            // ✅ CRITIQUE: S'assurer que rowProps est TOUJOURS un objet valide
            // react-window v2 appelle useMemoizedObject sur rowProps, qui fait Object.values(rowProps)
            // Si rowProps est undefined/null, cela cause: "Cannot convert undefined or null to object"
            const finalItemData = safeItemData && typeof safeItemData === 'object' && !Array.isArray(safeItemData) && safeItemData !== null
              ? safeItemData
              : defaultItemDataRef.current;
            
            // ✅ API react-window v2: utiliser rowComponent, rowCount, rowHeight, rowProps
            // react-window v2 attend rowComponent (composant) pas children (fonction)
            // react-window v2 attend rowCount, rowHeight, rowProps au lieu de itemCount, itemSize, itemData
            const listProps = {
              listRef: listRef,
              height: safeHeight,
              width: safeWidth,
              rowCount: safeItemCount,
              rowHeight: safeGetItemSize,
              rowProps: {
                ...finalItemData,
                columnWidths,  // Passer les largeurs de colonnes pour l'alignement
              },
              rowComponent: Row,
            };
            
            // #region agent log
            fetch('http://127.0.0.1:7242/ingest/5d8025f1-2a4d-4796-97fe-faa80ad8db74',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'VirtualizedReservationTable.jsx:520',message:'Avant rendu List',data:{rowCount:listProps.rowCount,hasRowComponent:!!listProps.rowComponent,rowPropsKeys:Object.keys(listProps.rowProps)},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'C,D'})}).catch(()=>{});
            // #endregion
            
            return <List {...listProps} />;
          })()
        ) : (
          <div className={styles.emptyMessage}>
            {!Array.isArray(reservations) ? 'Chargement des réservations...' : 'Aucune réservation'}
          </div>
        )}
      </div>
    </div>
  );
};

VirtualizedReservationTable.propTypes = {
  reservations: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
      client: PropTypes.shape({ full_name: PropTypes.string }),
      client_name: PropTypes.string,
      scheduled_time: PropTypes.string,
      pickup_location: PropTypes.string,
      dropoff_location: PropTypes.string,
      amount: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
      status: PropTypes.string,
      is_return: PropTypes.bool,
    })
  ).isRequired,
  onRowClick: PropTypes.func,
  onAccept: PropTypes.func,
  onReject: PropTypes.func,
  onAssign: PropTypes.func,
  onEdit: PropTypes.func,
  onDelete: PropTypes.func,
  onSchedule: PropTypes.func,
  onDispatchNow: PropTypes.func,
  hideAssign: PropTypes.bool,
  hideSchedule: PropTypes.bool,
  hideUrgent: PropTypes.bool,
  hideEdit: PropTypes.bool,
  hideDelete: PropTypes.bool,
};

export default VirtualizedReservationTable;

