import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { toast } from 'sonner';
import { invoiceService } from '../../../../../services/invoiceService';
import { BILLING_SOURCE } from '../../../../../utils/billingRecipient';
import {
  directLineHasSuspectAmount,
  getDirectLinePriceBadges,
  getDisplayedLineAmount,
  roundTo005,
} from '../../../../../utils/directInvoicePricing';
import styles from './ReservationSelector.module.css';

function isAdjustmentDraftDirty(draft) {
  if (!draft) return false;
  const a = String(draft.draftAmountStr ?? '').trim();
  const b = String(draft.initialAmountStr ?? '').trim();
  const n = String(draft.draftNote ?? '').trim();
  const m = String(draft.initialNote ?? '').trim();
  return a !== b || n !== m;
}

const ReservationSelector = ({
  companyId,
  clientId,
  clientName,
  period,
  billToType,
  vatConfig,
  overrides = {},
  onOverrideChange,
  onSelectionChange,
  preselectedIds = [],
  compactMode = false,
  hideClientName = false,
  autoSelectHospitalized = false,
}) => {
  const [reservations, setReservations] = useState([]);
  const [selectedIds, setSelectedIds] = useState([]);
  const [filter, setFilter] = useState('all');
  /** Filtre client-side révision : all | needs_work | suspect_only | corrected */
  const [reviewFilter, setReviewFilter] = useState('all');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hasAutoSelected, setHasAutoSelected] = useState(false);
  const [expandedReservations, setExpandedReservations] = useState(new Set());
  /** Un seul panneau d’ajustement ouvert à la fois → évite les sauts de scroll sur les longues listes */
  const [openAdjustmentId, setOpenAdjustmentId] = useState(null);
  /** Brouillon panneau édition (un seul panneau ouvert) — pas d’écriture parent tant que non Enregistré */
  const [adjustmentDraft, setAdjustmentDraft] = useState(null);
  const [allowManualOverride, setAllowManualOverride] = useState(false);
  const [loadingReservationDetails, setLoadingReservationDetails] = useState(new Set());
  const amountInputRefs = useRef({});
  const focusedAmountInputIdRef = useRef(null);
  const adjustmentDraftRef = useRef(null);
  const saveToastDebounceRef = useRef(null);

  useEffect(() => {
    adjustmentDraftRef.current = adjustmentDraft;
  }, [adjustmentDraft]);

  useEffect(
    () => () => {
      if (saveToastDebounceRef.current) clearTimeout(saveToastDebounceRef.current);
    },
    []
  );
  useEffect(() => {
    if (!reservations.length) return;
    if (!Array.isArray(preselectedIds) || preselectedIds.length === 0) return;
    const normalized = preselectedIds
      .map((id) => Number(id))
      .filter((id) => !Number.isNaN(id) && reservations.some((res) => res.id === id));
    if (normalized.length === 0) return;

    const currentKey = [...selectedIds].sort((a, b) => a - b).join(',');
    const nextKey = [...normalized].sort((a, b) => a - b).join(',');

    if (currentKey !== nextKey) {
      setSelectedIds(normalized);
      setHasAutoSelected(true);
    }
  }, [preselectedIds, reservations, selectedIds]);

  const vatApplicable = Boolean(vatConfig?.applicable);
  const defaultVatRate = vatApplicable
    ? Number.isFinite(Number(vatConfig?.defaultRate))
      ? Number(vatConfig.defaultRate)
      : 0
    : 0;

  useEffect(() => {
    const loadReservations = async () => {
      if (!companyId || !clientId || !period.year || !period.month) return;

      try {
        setLoading(true);
        setError(null);

        const data = await invoiceService.fetchUnbilledReservations(companyId, clientId, {
          year: period.year,
          month: period.month,
          billed_to_type: filter !== 'all' ? filter : (billToType || undefined),
        });

        const list = Array.isArray(data?.reservations) ? data.reservations : [];
        setReservations(list);

        // Retirer les sélections qui n'existent plus
        setSelectedIds((prev) => prev.filter((id) => list.some((res) => res.id === id)));
      } catch (err) {
        console.error('Erreur chargement réservations:', err);
        setError('Erreur lors du chargement des transports');
        setReservations([]);
      } finally {
        setLoading(false);
      }
    };

    loadReservations();
  }, [companyId, clientId, period, filter, billToType, autoSelectHospitalized]);

  // Reset hasAutoSelected sur changement de clientId ou période (évite resets inutiles)
  useEffect(() => {
    if (autoSelectHospitalized) {
      setHasAutoSelected(false);
    }
  }, [clientId, period.year, period.month, autoSelectHospitalized]);

  // Auto-sélection des transports hospitaliers en mode third_party
  useEffect(() => {
    if (autoSelectHospitalized && reservations.length > 0 && !hasAutoSelected) {
      // Auto-sélectionner les transports avec billed_to_type === 'clinic'
      // sauf ceux qui ont un override "facturer patient" (billing_override dans overrides)
      // et sauf ceux qui sont en needs_review ou missing_recipient
      const autoSelectable = reservations.filter((r) => {
        // Si billed_to_type === 'clinic', c'est un transport hospitalier
        const isHospitalized = r.billed_to_type === 'clinic';
        // Vérifier s'il y a un override pour facturer au patient
        const hasPatientOverride = overrides?.[r.id]?.billing_override === 'patient';
        // Ne pas auto-sélectionner si billed_to_type === 'patient'
        const isPatientBilled = r.billed_to_type === 'patient';
        // Safety: ne pas auto-sélectionner si needs_review ou missing_recipient
        const needsReviewCheck = needsReview(r);
        return isHospitalized && !hasPatientOverride && !isPatientBilled && !needsReviewCheck;
      }).map((r) => r.id);
      
      if (autoSelectable.length > 0) {
        // Calculer allAlready dans le setter pour éviter updates inutiles
        setSelectedIds((prev) => {
          // Vérifier si tous les auto-sélectionnables sont déjà sélectionnés
          const allAlready = autoSelectable.every((id) => prev.includes(id));
          if (allAlready) {
            return prev; // Pas de changement nécessaire
          }
          
          // Garder les sélections existantes qui ne sont pas dans autoSelectable
          const manualSelections = prev.filter((id) => !autoSelectable.includes(id));
          // Ajouter les auto-sélectionnables qui ne sont pas déjà sélectionnés
          const newAutoSelections = autoSelectable.filter((id) => !prev.includes(id));
          return [...manualSelections, ...newAutoSelections];
        });
        setHasAutoSelected(true);
      }
    } else if (billToType && reservations.length > 0 && !hasAutoSelected && !autoSelectHospitalized) {
      // Logique originale pour les autres modes
      const matching = reservations.filter((r) => r.billed_to_type === billToType).map((r) => r.id);
      if (matching.length > 0) {
        setSelectedIds(matching);
        setHasAutoSelected(true);
      }
    }
    // Note: needsReview est un useCallback stable, pas besoin dans les deps
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reservations, billToType, hasAutoSelected, autoSelectHospitalized, overrides]);

  useEffect(() => {
    if (openAdjustmentId != null && !selectedIds.includes(openAdjustmentId)) {
      setOpenAdjustmentId(null);
      setAdjustmentDraft(null);
    }
  }, [selectedIds, openAdjustmentId]);

  // ✅ Guard pour éviter boucles/rerenders : comparer le payload avant d'appeler onSelectionChange
  const prevSelectionRef = useRef(null);
  
  useEffect(() => {
    if (!onSelectionChange) return;
    // ✅ Filtrer les réservations sélectionnées, en incluant les objets minimaux si nécessaire
    const selected = selectedIds.map((id) => {
      const found = reservations.find((r) => r.id === id);
      // Si pas trouvé dans reservations, créer un objet minimal (cas "select all" avec IDs uniquement)
      return found || { id };
    });
    
    // ✅ Deep compare léger : comparer les IDs et les versions d'override
    const currentKey = JSON.stringify({
      ids: selected.map((r) => r.id).sort((a, b) => a - b),
      overrideVersions: selected.map((r) => {
        const override = overrides?.[String(r.id)] || overrides?.[r.id] || {};
        return `${r.id}:${override.amount ?? 'null'}:${override.note ?? 'null'}`;
      }).sort(),
    });
    
    const prevKey = prevSelectionRef.current;
    if (prevKey === currentKey) {
      // Pas de changement, éviter l'appel
      return;
    }
    
    prevSelectionRef.current = currentKey;
    onSelectionChange(selected);
    // Note: needsReview est un useCallback stable, pas besoin dans les deps
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedIds, reservations, overrides]);

  const handleToggle = (reservationId) => {
    setSelectedIds((prev) =>
      prev.includes(reservationId)
        ? prev.filter((id) => id !== reservationId)
        : [...prev, reservationId]
    );
  };

  const handleToggleExpand = (reservationId) => {
    setExpandedReservations((prev) => {
      const next = new Set(prev);
      if (next.has(reservationId)) {
        next.delete(reservationId);
      } else {
        next.add(reservationId);
      }
      return next;
    });
  };

  // ✅ Vérifier si une réservation est minimale (contient uniquement l'ID)
  const isMinimalReservation = useCallback((reservation) => {
    return reservation && typeof reservation.id !== 'undefined' && 
           (reservation.amount === undefined || reservation.amount === null) &&
           !reservation.pickup_location && !reservation.dropoff_location;
  }, []);

  // ✅ Charger les détails d'une réservation spécifique (endpoint single pour éviter mismatch période/status)
  const loadReservationDetails = useCallback(async (reservationId) => {
    if (!companyId) return null;
    
    try {
      setLoadingReservationDetails((prev) => new Set(prev).add(reservationId));
      
      // Utiliser l'endpoint single pour récupérer uniquement cette réservation
      const data = await invoiceService.fetchReservationById(companyId, reservationId);
      const found = data?.reservation;
      
      if (found) {
        // Remplacer l'objet minimal par l'objet complet dans l'état
        setReservations((prev) => {
          const index = prev.findIndex((r) => r.id === reservationId);
          if (index >= 0) {
            const updated = [...prev];
            updated[index] = found;
            return updated;
          }
          // Si pas trouvé dans prev, l'ajouter
          return [...prev, found];
        });
      }
      
      return found;
    } catch (err) {
      console.error(`Erreur lors du chargement des détails de la réservation ${reservationId}:`, err);
      return null;
    } finally {
      setLoadingReservationDetails((prev) => {
        const next = new Set(prev);
        next.delete(reservationId);
        return next;
      });
    }
  }, [companyId]);

  const scheduleSaveToast = useCallback(() => {
    if (saveToastDebounceRef.current) clearTimeout(saveToastDebounceRef.current);
    saveToastDebounceRef.current = setTimeout(() => {
      toast.success('Enregistré');
      saveToastDebounceRef.current = null;
    }, 450);
  }, []);

  const getOverrideForId = useCallback(
    (reservationId) => overrides?.[String(reservationId)] || overrides?.[reservationId] || {},
    [overrides]
  );

  const normalizeAmount = useCallback((value) => {
    if (!value || value === '' || value === null || value === undefined) {
      return { normalized: null, formatted: null, isValid: true };
    }
    const normalized = String(value).replace(/,/g, '.').trim().replace(/\s/g, '');
    const numeric = parseFloat(normalized);
    if (Number.isNaN(numeric) || numeric < 0) {
      return { normalized: null, formatted: null, isValid: false };
    }
    const rounded = roundTo005(numeric);
    return { normalized: rounded, formatted: rounded.toFixed(2), isValid: true };
  }, []);

  const openAdjustmentPanel = useCallback(
    (reservationId, reservationRow) => {
      const o = getOverrideForId(reservationId);
      const refHt = getDisplayedLineAmount(reservationRow, o);
      const hasOv =
        o.amount !== undefined && o.amount !== null && o.amount !== '' && Number.isFinite(Number(o.amount));
      const initialAmountStr = hasOv ? String(o.amount) : '';
      const initialNote = o.note != null && String(o.note).trim() !== '' ? String(o.note) : '';
      setAdjustmentDraft({
        reservationId,
        referenceHt: refHt,
        initialAmountStr,
        initialNote,
        draftAmountStr: initialAmountStr,
        draftNote: initialNote,
      });
      setOpenAdjustmentId(reservationId);
    },
    [getOverrideForId]
  );

  const handleAdjustmentButton = async (reservationId, e) => {
    e?.preventDefault?.();
    e?.stopPropagation?.();

    if (openAdjustmentId === reservationId) {
      if (isAdjustmentDraftDirty(adjustmentDraftRef.current)) {
        toast.message('Enregistrez ou annulez avant de fermer.');
        return;
      }
      setOpenAdjustmentId(null);
      setAdjustmentDraft(null);
      return;
    }

    if (openAdjustmentId != null && openAdjustmentId !== reservationId) {
      if (isAdjustmentDraftDirty(adjustmentDraftRef.current)) {
        toast.error(
          'Enregistrez ou annulez les modifications en cours avant d’ouvrir une autre ligne.'
        );
        return;
      }
    }

    let reservation = reservations.find((r) => r.id === reservationId);
    if (reservation && isMinimalReservation(reservation)) {
      const detailed = await loadReservationDetails(reservationId);
      if (!detailed) return;
      reservation = detailed;
    } else if (!reservation) {
      return;
    }

    if (!selectedIds.includes(reservationId)) {
      setSelectedIds((prev) => [...prev, reservationId]);
    }

    openAdjustmentPanel(reservationId, reservation);
  };

  const handleSaveAdjustment = useCallback(
    (reservationId, e) => {
      e?.stopPropagation?.();
      if (!onOverrideChange) return;
      const draft = adjustmentDraftRef.current;
      if (!draft || draft.reservationId !== reservationId) return;

      const result = normalizeAmount(draft.draftAmountStr);
      if (!result.isValid) {
        toast.error('Montant invalide.');
        return;
      }
      const noteVal =
        draft.draftNote != null && String(draft.draftNote).trim() !== ''
          ? String(draft.draftNote).trim()
          : null;
      if (result.normalized === null) {
        onOverrideChange(reservationId, { amount: null, note: noteVal });
      } else {
        onOverrideChange(reservationId, { amount: result.normalized, note: noteVal });
      }
      setAdjustmentDraft(null);
      setOpenAdjustmentId(null);
      scheduleSaveToast();
    },
    [onOverrideChange, normalizeAmount, scheduleSaveToast]
  );

  const handleCancelAdjustment = useCallback((e) => {
    e?.stopPropagation?.();
    setAdjustmentDraft(null);
    setOpenAdjustmentId(null);
  }, []);

  const handleRestoreCatalog = useCallback(
    (reservationId, e) => {
      e?.preventDefault?.();
      e?.stopPropagation?.();
      if (!onOverrideChange) return;
      onOverrideChange(reservationId, { amount: null, note: null });
      setAdjustmentDraft(null);
      setOpenAdjustmentId(null);
      scheduleSaveToast();
    },
    [onOverrideChange, scheduleSaveToast]
  );

  const handleDraftAmountChange = useCallback((reservationId, value) => {
    setAdjustmentDraft((prev) => {
      if (!prev || prev.reservationId !== reservationId) return prev;
      return { ...prev, draftAmountStr: value };
    });
  }, []);

  const handleDraftNoteChange = useCallback((reservationId, value) => {
    setAdjustmentDraft((prev) => {
      if (!prev || prev.reservationId !== reservationId) return prev;
      return { ...prev, draftNote: value };
    });
  }, []);

  const handleAdjustmentKeyDown = useCallback(
    (reservationId, e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        handleSaveAdjustment(reservationId, e);
      }
      if (e.key === 'Escape') {
        e.preventDefault();
        handleCancelAdjustment(e);
      }
    },
    [handleSaveAdjustment, handleCancelAdjustment]
  );

  // Déterminer si un transport nécessite une vérification (ne doit pas être auto-sélectionné)
  // ⚠️ IMPORTANT: Défini AVANT isAutoSelected car utilisé dedans
  const needsReview = useCallback((reservation) => {
    // Vérifier si billing_review_status === 'needs_review'
    if (reservation.billing_review_status === 'needs_review') return true;
    // Vérifier si missing_recipient === true
    if (reservation.missing_recipient === true) return true;
    return false;
  }, []);

  // Déterminer si un transport est auto-sélectionné (hospitalier, non override, pas needs_review)
  const isAutoSelected = useCallback(
    (reservation) => {
    if (!autoSelectHospitalized) return false;
    const isHospitalized = reservation.billed_to_type === 'clinic';
    // ✅ Cohérence: utiliser String(id) comme clé pour les overrides
    const override = overrides?.[String(reservation.id)] || overrides?.[reservation.id] || {};
    const hasPatientOverride = override.billing_override === 'patient';
    const isPatientBilled = reservation.billed_to_type === 'patient';
    const needsReviewCheck = needsReview(reservation);
    return isHospitalized && !hasPatientOverride && !isPatientBilled && !needsReviewCheck && selectedIds.includes(reservation.id);
  }, [autoSelectHospitalized, overrides, selectedIds, needsReview]);

  // Déterminer si un transport doit être facturé au patient (override explicite uniquement)
  const isBilledToPatient = useCallback((reservation) => {
    // ✅ CORRECTION: Ne pas déduire override depuis billed_to_type === 'patient'
    // Vérifier uniquement les overrides explicites
    
    // 1. Override explicite dans les overrides du formulaire
    // ✅ Cohérence: utiliser String(id) comme clé pour les overrides
    const override = overrides?.[String(reservation.id)] || overrides?.[reservation.id] || {};
    if (override.billing_override === 'patient') return true;
    
    // 2. Override manuel détecté via billing_source (utiliser constante centralisée)
    // billing_source === 'manual_override' ET billed_to_type === 'patient' = override explicite
    if (
      reservation.billing_source === BILLING_SOURCE.MANUAL_OVERRIDE &&
      reservation.billed_to_type === 'patient'
    ) {
      return true;
    }
    
    // ❌ NE PAS retourner true si billed_to_type === 'patient' sans override explicite
    // (peut être un fallback, pas un override)
    return false;
  }, [overrides]);

  const handleSelectAll = () => {
    const allIds = reservations.map((r) => r.id);
    setSelectedIds(allIds);
  };

  const handleDeselectAll = () => {
    setSelectedIds([]);
  };

  // ✅ Auto-focus + garder la ligne visible dans la zone scrollable (évite les « sauts »)
  useEffect(() => {
    if (openAdjustmentId == null) return;
    const id = openAdjustmentId;
    const reservation = reservations.find((r) => r.id === id);
    if (!reservation) return;
    const isMinimal =
      reservation &&
      typeof reservation.id !== 'undefined' &&
      (reservation.amount === undefined || reservation.amount === null) &&
      !reservation.pickup_location &&
      !reservation.dropoff_location;
    if (isMinimal) return;

    const raf = requestAnimationFrame(() => {
      const row = document.getElementById(`invoice-adjust-row-${id}`);
      row?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      const inputRef = amountInputRefs.current[id];
      if (inputRef && document.activeElement !== inputRef) {
        inputRef.focus();
        inputRef.select();
      }
    });
    return () => cancelAnimationFrame(raf);
  }, [openAdjustmentId, reservations]);


  const formatDate = (dateString) => {
    if (!dateString) return '-';
    try {
      return new Date(dateString).toLocaleDateString('fr-FR', {
        day: '2-digit',
        month: '2-digit',
        year: 'numeric',
      });
    } catch {
      return '-';
    }
  };

  const _getBillingTypeLabel = (type) => {
    const labels = {
      patient: '👤 Patient',
      clinic: '🏥 Clinique',
      insurance: '🏢 Assurance',
    };
    return labels[type] || type;
  };

  const getBillingTypeLabelShort = (type) => {
    const labels = { patient: 'Patient', clinic: 'Clinique', insurance: 'Assurance' };
    return labels[type] || type;
  };

  const getBillingTypeClass = (type) => type || 'patient';

  const priceBadgeClassByKey = {
    corrected: styles.priceBadgeCorrected,
    catalog: styles.priceBadgeCatalog,
    estimate: styles.priceBadgeEstimate,
    suspect_low: styles.priceBadgeSuspect,
  };

  const computeAmounts = useCallback(
    (reservation) => {
      // ✅ Cohérence avec NewInvoiceModal: utiliser String(id) comme clé pour les overrides
      const override = overrides?.[String(reservation.id)] || overrides?.[reservation.id] || {};
      const amount = getDisplayedLineAmount(reservation, override);
      const vatRate = vatApplicable
        ? Number(reservation.vat_rate ?? reservation.default_vat_rate ?? defaultVatRate)
        : 0;
      const sanitizedAmount = Number.isNaN(amount) ? 0 : amount;
      const sanitizedVatRate = Number.isNaN(vatRate) ? 0 : vatRate;
      const vatValue = vatApplicable
        ? Number(((sanitizedAmount * sanitizedVatRate) / 100).toFixed(2))
        : 0;
      const total = Number((sanitizedAmount + vatValue).toFixed(2));

      return {
        amount: sanitizedAmount,
        vatRate: sanitizedVatRate,
        vatValue,
        total,
        note: override.note || '',
      };
    },
    [overrides, vatApplicable, defaultVatRate]
  );

  const mergedReservationRows = useMemo(() => {
    const loadedIds = new Set(reservations.map((r) => r.id));
    const missingSelectedIds = selectedIds.filter((id) => !loadedIds.has(id));
    return [...reservations, ...missingSelectedIds.map((id) => ({ id }))];
  }, [reservations, selectedIds]);

  const totalRowsInScope = mergedReservationRows.length;

  const displayReservationRows = useMemo(() => {
    const getO = (id) => overrides[String(id)] || overrides[id] || {};
    const lineSuspect = (r) => {
      if (isMinimalReservation(r)) return false;
      return directLineHasSuspectAmount(r, getO(r.id));
    };
    const lineCorrected = (id) => {
      const o = getO(id);
      return o.amount !== undefined && o.amount !== null && o.amount !== '' && Number.isFinite(Number(o.amount));
    };
    const sortBucket = (r) => {
      if (isMinimalReservation(r)) return 3;
      if (lineSuspect(r)) return 0;
      if (needsReview(r)) return 1;
      if (lineCorrected(r.id)) return 2;
      return 3;
    };

    let list = [...mergedReservationRows];
    if (reviewFilter === 'needs_work') {
      list = list.filter((r) => !isMinimalReservation(r) && (lineSuspect(r) || needsReview(r)));
    } else if (reviewFilter === 'suspect_only') {
      list = list.filter((r) => !isMinimalReservation(r) && lineSuspect(r));
    } else if (reviewFilter === 'corrected') {
      list = list.filter((r) => lineCorrected(r.id));
    }

    list.sort((a, b) => {
      const ba = sortBucket(a);
      const bb = sortBucket(b);
      if (ba !== bb) return ba - bb;
      const ta = isMinimalReservation(a) ? 0 : new Date(a.date || 0).getTime();
      const tb = isMinimalReservation(b) ? 0 : new Date(b.date || 0).getTime();
      return tb - ta;
    });
    return list;
  }, [mergedReservationRows, overrides, reviewFilter, needsReview, isMinimalReservation]);

  const selectionSummary = useMemo(() => {
    const sel = reservations.filter((r) => selectedIds.includes(r.id));
    let suspects = 0;
    let corrected = 0;
    let rawSubtotalHt = 0;
    for (const r of sel) {
      const o = getOverrideForId(r.id);
      if (directLineHasSuspectAmount(r, o)) suspects += 1;
      if (
        o.amount !== undefined &&
        o.amount !== null &&
        o.amount !== '' &&
        Number.isFinite(Number(o.amount))
      ) {
        corrected += 1;
      }
      rawSubtotalHt += getDisplayedLineAmount(r, o);
    }
    return {
      selectedCount: selectedIds.length,
      suspects,
      corrected,
      /** Même agrégation que le footer NewInvoiceModal (directFinancialBreakdown) */
      subtotalHt: roundTo005(rawSubtotalHt),
    };
  }, [reservations, selectedIds, getOverrideForId]);

  const formatCurrency = (value) => `${Number(value || 0).toFixed(2)} CHF`;

  if (loading) {
    return <div className={styles.loading}>Chargement des transports...</div>;
  }

  if (error) {
    return <div className={styles.error}>{error}</div>;
  }

  if (reservations.length === 0) {
    return (
      <div className={styles.empty}>
        <div className={styles.emptyIcon}>🚗</div>
        <p>Aucun transport non facturé pour cette période</p>
      </div>
    );
  }

  const shouldShowActions = !compactMode && reservations.length > 1;
  const shouldShowAllFilter = !compactMode && reservations.length > 1;

  return (
    <div className={styles.container}>
      <div
        className={`${styles.headerRow} ${hideClientName ? styles.headerRowFiltersOnly : ''}`}
      >
        <div className={styles.headerLeft}>
          {!hideClientName && (
            <h4 className={styles.clientNameSmall}>{clientName}</h4>
          )}
          <div className={styles.filterButtonsCompact}>
            {shouldShowAllFilter && (
              <button
                type="button"
                className={`${styles.filterBtnCompact} ${filter === 'all' ? styles.active : ''}`}
                onClick={() => setFilter('all')}
              >
                Tous ({reservations.length})
              </button>
            )}
            <button
              type="button"
              className={`${styles.filterBtnCompact} ${filter === 'clinic' ? styles.active : ''}`}
              onClick={() => setFilter('clinic')}
            >
              Clinique
            </button>
            <button
              type="button"
              className={`${styles.filterBtnCompact} ${filter === 'patient' ? styles.active : ''}`}
              onClick={() => setFilter('patient')}
            >
              Patient
            </button>
          </div>
          {!compactMode && reservations.length > 0 && (
            <div
              className={styles.reviewFiltersRow}
              role="group"
              aria-label="Filtres de révision"
            >
              <span className={styles.reviewFiltersLabel}>Révision</span>
              <button
                type="button"
                className={`${styles.reviewFilterBtn} ${reviewFilter === 'all' ? styles.active : ''}`}
                aria-pressed={reviewFilter === 'all'}
                onClick={() => setReviewFilter('all')}
              >
                Toutes
              </button>
              <button
                type="button"
                className={`${styles.reviewFilterBtn} ${reviewFilter === 'needs_work' ? styles.active : ''}`}
                aria-pressed={reviewFilter === 'needs_work'}
                onClick={() => setReviewFilter('needs_work')}
              >
                À corriger
              </button>
              <button
                type="button"
                className={`${styles.reviewFilterBtn} ${reviewFilter === 'suspect_only' ? styles.active : ''}`}
                aria-pressed={reviewFilter === 'suspect_only'}
                onClick={() => setReviewFilter('suspect_only')}
              >
                Montants suspects
              </button>
              <button
                type="button"
                className={`${styles.reviewFilterBtn} ${reviewFilter === 'corrected' ? styles.active : ''}`}
                aria-pressed={reviewFilter === 'corrected'}
                onClick={() => setReviewFilter('corrected')}
              >
                Corrigées
              </button>
            </div>
          )}
        </div>
        {shouldShowActions && (
          <div className={styles.actionsInline}>
            <button type="button" onClick={handleSelectAll} className={styles.actionLink}>
              Tout sélectionner
            </button>
            <button type="button" onClick={handleDeselectAll} className={styles.actionLink}>
              Tout désélectionner
            </button>
          </div>
        )}
      </div>

      {/* Message info et bouton "Modifier la sélection" pour mode auto-sélection */}
      {autoSelectHospitalized && hasAutoSelected && !allowManualOverride && 
       reservations.some((r) => isAutoSelected(r)) && (
        <div className={styles.autoSelectInfo}>
          <span className={styles.autoSelectMessage}>
            🏥 Transport hospitalier — facturé à la clinique par défaut
          </span>
          <button
            type="button"
            className={styles.modifySelectionBtn}
            onClick={() => setAllowManualOverride(true)}
          >
            Modifier la sélection
          </button>
        </div>
      )}

      {!compactMode && reservations.length > 0 && (
        <div className={styles.summaryControlBar} role="status">
          <span>
            <strong>{selectionSummary.selectedCount}</strong> sélectionné
            {selectionSummary.selectedCount > 1 ? 's' : ''}
          </span>
          <span>
            <strong>{selectionSummary.suspects}</strong> montant{selectionSummary.suspects > 1 ? 's' : ''}{' '}
            suspect{selectionSummary.suspects > 1 ? 's' : ''}
          </span>
          <span>
            <strong>{selectionSummary.corrected}</strong> corrigée{selectionSummary.corrected > 1 ? 's' : ''}
          </span>
          <span>
            Sous-total HT <strong>{formatCurrency(selectionSummary.subtotalHt)}</strong>
          </span>
          {reviewFilter !== 'all' && (
            <span className={styles.summaryViewHint}>
              {displayReservationRows.length} affichée
              {displayReservationRows.length > 1 ? 's' : ''} sur {totalRowsInScope}
            </span>
          )}
        </div>
      )}

      <div className={`${styles.reservationsList} ${!compactMode ? styles.reservationsListDense : ''}`}>
        {displayReservationRows.map((reservation) => {
            const isSelected = selectedIds.includes(reservation.id);
            const isExpanded = expandedReservations.has(reservation.id);
            const showAdjust = openAdjustmentId === reservation.id;
            const isLoadingDetails = loadingReservationDetails.has(reservation.id);
            const isMinimal = isMinimalReservation(reservation);
            
            // Pour les objets minimaux, utiliser des valeurs par défaut pour l'affichage
            const figures = isMinimal 
              ? { amount: 0, vatRate: 0, vatValue: 0, total: 0, note: '' }
              : computeAmounts(reservation);
            const baseAmount = isMinimal ? 0 : Number(reservation?.amount ?? 0);
          // ✅ Cohérence: utiliser String(id) comme clé pour les overrides
          const override = overrides?.[String(reservation.id)] || overrides?.[reservation.id] || {};
          const overrideAmount = override.amount;
          const hasOverrideAmount =
            overrideAmount !== undefined
            && overrideAmount !== null
            && Number.isFinite(Number(overrideAmount));
          const adjustment = hasOverrideAmount ? Number(overrideAmount) - baseAmount : 0;
          const overrideNote = override.note;
          const rowSuspect = !isMinimal && directLineHasSuspectAmount(reservation, override);
          const draft =
            adjustmentDraft && adjustmentDraft.reservationId === reservation.id
              ? adjustmentDraft
              : null;
          const catalogAmt = !isMinimal ? Number(reservation?.amount ?? 0) : 0;
          const catalogWasLow =
            Number.isFinite(catalogAmt) && catalogAmt > 0 && catalogAmt < 5;

          const panelPreviewFigures =
            draft && !isMinimal
              ? (() => {
                  const nr = normalizeAmount(draft.draftAmountStr);
                  if (!nr.isValid) return figures;
                  const noteVal =
                    draft.draftNote != null && String(draft.draftNote).trim() !== ''
                      ? String(draft.draftNote).trim()
                      : null;
                  const simOverride = { ...override, note: noteVal };
                  if (nr.normalized !== null) {
                    simOverride.amount = nr.normalized;
                  } else {
                    simOverride.amount = null;
                  }
                  const amount = getDisplayedLineAmount(reservation, simOverride);
                  const vatRate = vatApplicable
                    ? Number(reservation.vat_rate ?? reservation.default_vat_rate ?? defaultVatRate)
                    : 0;
                  const sanitizedAmount = Number.isNaN(amount) ? 0 : amount;
                  const sanitizedVatRate = Number.isNaN(vatRate) ? 0 : vatRate;
                  const vatValue = vatApplicable
                    ? Number(((sanitizedAmount * sanitizedVatRate) / 100).toFixed(2))
                    : 0;
                  const total = Number((sanitizedAmount + vatValue).toFixed(2));
                  return {
                    amount: sanitizedAmount,
                    vatRate: sanitizedVatRate,
                    vatValue,
                    total,
                    note: noteVal || '',
                  };
                })()
              : figures;

          // Mode compact: 1 ligne par défaut
          if (compactMode) {
            const autoSelected = isAutoSelected(reservation);
            const billedToPatient = isBilledToPatient(reservation);
            const needsReviewCheck = needsReview(reservation);
            const checkboxDisabled = autoSelected && !allowManualOverride;
            const showNeedsReviewBadge =
              needsReviewCheck && !rowSuspect && !hasOverrideAmount;
            const linePriceBadges = !isMinimal
              ? getDirectLinePriceBadges(reservation, override)
              : [];
            const hasCorrectedBadge = linePriceBadges.some((b) => b.key === 'corrected');
            const hadSuspectWithCorrected =
              hasCorrectedBadge && linePriceBadges.some((b) => b.key === 'suspect_low');

            return (
              <div
                key={reservation.id}
                className={`${styles.reservationItemCompact} ${isSelected ? styles.selected : ''} ${autoSelected ? styles.autoSelected : ''}`}
              >
                <input
                  type="checkbox"
                  checked={isSelected}
                  onChange={() => handleToggle(reservation.id)}
                  className={styles.checkbox}
                  onClick={(e) => e.stopPropagation()}
                  disabled={checkboxDisabled}
                  title={checkboxDisabled ? 'Transport hospitalier — facturé à la clinique par défaut' : ''}
                />
                <div 
                  className={styles.reservationContentCompact}
                  onClick={() => handleToggleExpand(reservation.id)}
                >
                  <span className={styles.dateCompact}>
                    {isMinimal ? '—' : formatDate(reservation.date)}
                  </span>
                  <span className={styles.routeCompact}>
                    {isMinimal 
                      ? (isLoadingDetails ? 'Chargement...' : '—')
                      : `${reservation.pickup_location} → ${reservation.dropoff_location}`
                    }
                  </span>
                  {showNeedsReviewBadge && (
                    <span className={styles.reviewBadge}>⚠️ À vérifier</span>
                  )}
                  {reservation.status === 'CANCELED' && (
                    <span
                      className={styles.cancellationBadge}
                      title={reservation.cancellation_display_label || 'Réservation annulée (facturée)'}
                    >
                      Annulé
                    </span>
                  )}
                  {billedToPatient && (
                    <span className={styles.patientBadge}>👤 Facturé au patient</span>
                  )}
                  {!billedToPatient && reservation.billed_to_type === 'patient' && (
                    <span 
                      className={styles.patientFallbackBadge} 
                      title={
                        reservation.billing_source === BILLING_SOURCE.DEFAULT_CLIENT
                          ? "Facturation patient par défaut du client (pas d'override explicite ni de séjour hospitalier)"
                          : reservation.billing_source === BILLING_SOURCE.CLIENT_STAY && reservation.billed_to_type === 'patient'
                          ? "Facturation patient malgré séjour hospitalier (à vérifier)"
                          : "Facturation patient (source: " + (reservation.billing_source || 'inconnue') + ")"
                      }
                    >
                      👤 Patient
                    </span>
                  )}
                  {!isMinimal &&
                    linePriceBadges.map((b) => {
                      const isSecondarySuspect = b.key === 'suspect_low' && hasCorrectedBadge;
                      const badgeTitle =
                        b.key === 'corrected' && hadSuspectWithCorrected
                          ? 'Ligne corrigée — montant HT encore très bas (contrôle recommandé)'
                          : b.key === 'corrected' && catalogWasLow
                            ? 'Tarif catalogue initial très bas — ligne corrigée'
                            : b.key === 'suspect_low' && isSecondarySuspect
                              ? 'Montant HT affiché très bas — contrôle recommandé'
                              : b.key === 'suspect_low'
                                ? 'Montant HT affiché très bas — vérifiez la ligne'
                                : undefined;
                      return (
                        <span
                          key={b.key}
                          className={`${styles.badgeCompact} ${
                            isSecondarySuspect
                              ? styles.priceBadgeSuspectSecondary
                              : priceBadgeClassByKey[b.key] || ''
                          }`}
                          title={badgeTitle}
                        >
                          {b.label}
                        </span>
                      );
                    })}
                  <span className={styles.amountCompact}>
                    {isMinimal ? '—' : formatCurrency(figures.total)}
                  </span>
                  <span className={styles.expandIcon}>
                    {isExpanded ? '⌄' : '›'}
                  </span>
                </div>
                {isExpanded && (
                  <div className={styles.reservationDetails}>
                    <div className={styles.detailsRow}>
                      <span className={styles.detailLabel}>Montant HT:</span>
                      <span className={styles.detailValue}>
                        {isMinimal ? '—' : formatCurrency(figures.amount)}
                      </span>
                    </div>
                    {vatApplicable && !isMinimal && figures.vatValue > 0 && (
                      <div className={styles.detailsRow}>
                        <span className={styles.detailLabel}>TVA:</span>
                        <span className={styles.detailValue}>
                          {figures.vatRate.toFixed(2)}% · {formatCurrency(figures.vatValue)}
                        </span>
                      </div>
                    )}
                    {hasOverrideAmount && !isMinimal && Math.abs(adjustment) >= 0.01 && (
                      <div className={styles.detailsRow}>
                        <span className={styles.detailLabel}>Ajustement:</span>
                        <span className={styles.detailValue}>
                          {adjustment >= 0 ? '+' : '-'}{formatCurrency(Math.abs(adjustment))}
                          {overrideNote ? ` — ${overrideNote}` : ''}
                        </span>
                      </div>
                    )}
                    {isSelected && (
                      <div className={styles.adjustmentSection}>
                        {!showAdjust ? (
                          <div className={styles.rowActionsDense}>
                            {rowSuspect && !isMinimal && (
                              <button
                                type="button"
                                className={styles.adjustCatalogQuickBtn}
                                onClick={(e) => handleRestoreCatalog(reservation.id, e)}
                                disabled={isLoadingDetails}
                              >
                                Rétablir le tarif catalogue
                              </button>
                            )}
                            <button
                              type="button"
                              className={styles.adjustActionTextBtn}
                              onClick={(e) => handleAdjustmentButton(reservation.id, e)}
                              title={isMinimal ? 'Chargement des détails...' : undefined}
                              disabled={isLoadingDetails}
                            >
                              {isLoadingDetails ? 'Chargement…' : hasOverrideAmount ? 'Modifier' : 'Corriger'}
                            </button>
                          </div>
                        ) : isMinimal && isLoadingDetails ? (
                          <div className={styles.adjustments}>
                            <div className={styles.adjustRow}>
                              <span className={styles.adjustLabel}>Chargement des détails...</span>
                            </div>
                          </div>
                        ) : (
                          <div
                            className={styles.adjustments}
                            role="form"
                            onKeyDown={(e) => {
                              if (e.key === 'Escape') handleCancelAdjustment(e);
                            }}
                          >
                            <div className={styles.adjustRow}>
                              <span className={styles.adjustLabel}>Montant de référence (HT)</span>
                              <span className={styles.referenceAmountReadonly}>
                                {isMinimal ? '—' : formatCurrency(draft?.referenceHt ?? figures.amount)}
                              </span>
                            </div>
                            <div className={styles.adjustGrid}>
                              <label className={styles.field}>
                                <span>Nouveau montant HT</span>
                                <input
                                  ref={(el) => {
                                    if (el) {
                                      amountInputRefs.current[reservation.id] = el;
                                    } else {
                                      delete amountInputRefs.current[reservation.id];
                                    }
                                  }}
                                  type="text"
                                  inputMode="decimal"
                                  autoComplete="off"
                                  className={styles.input}
                                  value={draft?.draftAmountStr ?? ''}
                                  placeholder={
                                    isMinimal
                                      ? '0.00'
                                      : (draft?.referenceHt ?? figures.amount).toFixed(2)
                                  }
                                  onChange={(e) =>
                                    handleDraftAmountChange(reservation.id, e.target.value)
                                  }
                                  onFocus={() => {
                                    focusedAmountInputIdRef.current = reservation.id;
                                  }}
                                  onBlur={() => {
                                    focusedAmountInputIdRef.current = null;
                                  }}
                                  onKeyDown={(e) => handleAdjustmentKeyDown(reservation.id, e)}
                                  onClick={(e) => e.stopPropagation()}
                                  disabled={isMinimal}
                                />
                              </label>
                            </div>
                            <label className={styles.field}>
                              <span>Motif / note (facultatif)</span>
                              <textarea
                                rows={2}
                                className={styles.noteInput}
                                value={draft?.draftNote ?? ''}
                                placeholder="Ex. Correction tarif transport"
                                onChange={(e) =>
                                  handleDraftNoteChange(reservation.id, e.target.value)
                                }
                                onKeyDown={(e) => handleAdjustmentKeyDown(reservation.id, e)}
                                onClick={(e) => e.stopPropagation()}
                                disabled={isMinimal}
                              />
                            </label>
                            <div className={styles.adjustActions}>
                              <button
                                type="button"
                                className={styles.adjustSecondaryBtn}
                                onClick={(e) => handleCancelAdjustment(e)}
                              >
                                Annuler
                              </button>
                              <button
                                type="button"
                                className={styles.adjustTertiaryBtn}
                                onClick={(e) => handleRestoreCatalog(reservation.id, e)}
                                disabled={isMinimal}
                              >
                                Rétablir le tarif catalogue
                              </button>
                              <button
                                type="button"
                                className={styles.adjustPrimaryBtn}
                                onClick={(e) => handleSaveAdjustment(reservation.id, e)}
                                disabled={isMinimal}
                              >
                                Enregistrer
                              </button>
                            </div>
                            <div className={styles.adjustSummary}>
                              <span>
                                HT <strong>{isMinimal ? '—' : formatCurrency(panelPreviewFigures.amount)}</strong>
                              </span>
                              {vatApplicable && !isMinimal && panelPreviewFigures.vatValue > 0 && (
                                <span>
                                  TVA <strong>{formatCurrency(panelPreviewFigures.vatValue)}</strong>
                                </span>
                              )}
                              <span>
                                TTC{' '}
                                <strong>{isMinimal ? '—' : formatCurrency(panelPreviewFigures.total)}</strong>
                              </span>
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          }

          // Mode détaillé dense (client) — 1 ligne : checkbox, date, trajet, badges, montant. Ajustement = ✏️, inline au clic.
          const autoSelected = isAutoSelected(reservation);
          const billedToPatient = isBilledToPatient(reservation);
          const needsReviewCheck = needsReview(reservation);
          const checkboxDisabled = autoSelected && !allowManualOverride;
          const showNeedsReviewBadge =
            needsReviewCheck && !rowSuspect && !hasOverrideAmount;
          const linePriceBadges = !isMinimal
            ? getDirectLinePriceBadges(reservation, override)
            : [];
          const hasCorrectedBadge = linePriceBadges.some((b) => b.key === 'corrected');
          const hadSuspectWithCorrected =
            hasCorrectedBadge && linePriceBadges.some((b) => b.key === 'suspect_low');
          const routeLabel = `${reservation.pickup_location} → ${reservation.dropoff_location}`;

          return (
            <div
              key={reservation.id}
              id={showAdjust ? `invoice-adjust-row-${reservation.id}` : undefined}
              className={`${styles.reservationItem} ${styles.reservationItemDense} ${isSelected ? styles.selected : ''} ${autoSelected ? styles.autoSelected : ''} ${showAdjust ? styles.rowAdjustOpen : ''}`}
            >
              <input
                type="checkbox"
                checked={isSelected}
                onChange={() => handleToggle(reservation.id)}
                className={`${styles.checkbox} ${styles.checkboxDense}`}
                disabled={checkboxDisabled}
                title={checkboxDisabled ? 'Transport hospitalier — facturé à la clinique par défaut' : ''}
              />

              <div className={`${styles.reservationContent} ${styles.reservationContentDense}`}>
                <div
                  className={styles.reservationSingleLine}
                  onClick={(e) => {
                    if (checkboxDisabled) return;
                    const t = e.target;
                    if (t instanceof Element && t.closest('button, input, textarea, a, select')) return;
                    handleToggle(reservation.id);
                  }}
                >
                  <span className={styles.dateDense}>
                    {isMinimal ? '—' : formatDate(reservation.date)}
                  </span>
                  <span className={styles.lineSep}>•</span>
                  <span className={styles.routeInline} title={isMinimal ? 'Chargement...' : routeLabel}>
                    {isMinimal 
                      ? (isLoadingDetails ? 'Chargement...' : '—')
                      : `${reservation.pickup_location} → ${reservation.dropoff_location}`
                    }
                  </span>
                  <div className={styles.badgesInline}>
                    {showNeedsReviewBadge && (
                      <span className={`${styles.badgeCompact} ${styles.review}`}>À vérifier</span>
                    )}
                    {reservation.status === 'CANCELED' && (
                      <span
                        className={`${styles.badgeCompact} ${styles.cancellation}`}
                        title={reservation.cancellation_display_label || 'Réservation annulée (facturée)'}
                      >
                        Annulé
                      </span>
                    )}
                    {billedToPatient ? (
                      <span className={`${styles.badgeCompact} ${styles.patient}`}>Patient</span>
                    ) : reservation.billed_to_type === 'patient' ? (
                      <span
                        className={`${styles.badgeCompact} ${styles.patientFallback}`}
                        title={
                          reservation.billing_source === BILLING_SOURCE.DEFAULT_CLIENT
                            ? "Facturation patient par défaut du client (pas d'override explicite ni de séjour hospitalier)"
                            : reservation.billing_source === BILLING_SOURCE.CLIENT_STAY && reservation.billed_to_type === 'patient'
                            ? "Facturation patient malgré séjour hospitalier (à vérifier)"
                            : "Facturation patient (source: " + (reservation.billing_source || 'inconnue') + ")"
                        }
                      >
                        Patient
                      </span>
                    ) : reservation.billed_to_type ? (
                      <span
                        className={`${styles.badgeCompact} ${styles[getBillingTypeClass(reservation.billed_to_type)]}`}
                      >
                        {getBillingTypeLabelShort(reservation.billed_to_type)}
                      </span>
                    ) : null}
                    {!isMinimal &&
                      linePriceBadges.map((b) => {
                        const isSecondarySuspect = b.key === 'suspect_low' && hasCorrectedBadge;
                        const badgeTitle =
                          b.key === 'corrected' && hadSuspectWithCorrected
                            ? 'Ligne corrigée — montant HT encore très bas (contrôle recommandé)'
                            : b.key === 'corrected' && catalogWasLow
                              ? 'Tarif catalogue initial très bas — ligne corrigée'
                              : b.key === 'suspect_low' && isSecondarySuspect
                                ? 'Montant HT affiché très bas — contrôle recommandé'
                                : b.key === 'suspect_low'
                                  ? 'Montant HT affiché très bas — vérifiez la ligne'
                                  : undefined;
                        return (
                          <span
                            key={b.key}
                            className={`${styles.badgeCompact} ${
                              isSecondarySuspect
                                ? styles.priceBadgeSuspectSecondary
                                : priceBadgeClassByKey[b.key] || ''
                            }`}
                            title={badgeTitle}
                          >
                            {b.label}
                          </span>
                        );
                      })}
                  </div>
                  <span className={styles.amountHT}>
                    {isMinimal ? '—' : formatCurrency(figures.amount)}
                  </span>
                  {!showAdjust && (
                    <div
                      className={styles.rowActionsDense}
                      onClick={(e) => e.stopPropagation()}
                    >
                      {rowSuspect && !isMinimal && (
                        <button
                          type="button"
                          className={styles.adjustCatalogQuickBtn}
                          onClick={(e) => handleRestoreCatalog(reservation.id, e)}
                          disabled={isLoadingDetails}
                        >
                          Rétablir catalogue
                        </button>
                      )}
                      <button
                        type="button"
                        className={styles.adjustActionTextBtn}
                        onClick={(e) => handleAdjustmentButton(reservation.id, e)}
                        title={isMinimal ? 'Chargement des détails...' : undefined}
                        aria-expanded={showAdjust}
                        aria-controls={`adjust-${reservation.id}`}
                        disabled={isLoadingDetails}
                      >
                        {isLoadingDetails ? 'Chargement…' : hasOverrideAmount ? 'Modifier' : 'Corriger'}
                      </button>
                    </div>
                  )}
                </div>

                {showAdjust && (
                  <div
                    className={styles.adjustInline}
                    id={`adjust-${reservation.id}`}
                    tabIndex={-1}
                    role="form"
                    aria-label="Révision du montant"
                    onClick={(e) => e.stopPropagation()}
                    onKeyDown={(e) => {
                      if (e.key === 'Escape') handleCancelAdjustment(e);
                    }}
                  >
                    {isMinimal && isLoadingDetails ? (
                      <div className={styles.adjustRow}>
                        <span className={styles.adjustLabel}>Chargement des détails...</span>
                      </div>
                    ) : (
                      <>
                        <div className={styles.adjustRow}>
                          <span className={styles.adjustLabel}>Montant de référence (HT)</span>
                          <span className={styles.referenceAmountReadonly}>
                            {isMinimal ? '—' : formatCurrency(draft?.referenceHt ?? figures.amount)}
                          </span>
                        </div>
                        <div className={styles.adjustRow}>
                          <span className={styles.adjustLabel}>Nouveau montant HT</span>
                          <input
                            ref={(el) => {
                              if (el) {
                                amountInputRefs.current[reservation.id] = el;
                              } else {
                                delete amountInputRefs.current[reservation.id];
                              }
                            }}
                            type="text"
                            inputMode="decimal"
                            autoComplete="off"
                            className={styles.adjustInput}
                            value={draft?.draftAmountStr ?? ''}
                            placeholder={
                              isMinimal
                                ? '0.00'
                                : (draft?.referenceHt ?? figures.amount).toFixed(2)
                            }
                            onChange={(e) => handleDraftAmountChange(reservation.id, e.target.value)}
                            onFocus={() => {
                              focusedAmountInputIdRef.current = reservation.id;
                            }}
                            onBlur={() => {
                              focusedAmountInputIdRef.current = null;
                            }}
                            onKeyDown={(e) => handleAdjustmentKeyDown(reservation.id, e)}
                            onClick={(e) => e.stopPropagation()}
                            disabled={isMinimal}
                          />
                          <span className={styles.adjustSuffix}>CHF</span>
                        </div>
                        <div className={styles.adjustRowNote}>
                          <span className={styles.adjustLabel}>Motif / note (facultatif)</span>
                          <input
                            type="text"
                            className={styles.adjustNoteInput}
                            value={draft?.draftNote ?? ''}
                            placeholder="Ex. Correction tarif transport"
                            onChange={(e) => handleDraftNoteChange(reservation.id, e.target.value)}
                            onKeyDown={(e) => handleAdjustmentKeyDown(reservation.id, e)}
                            onClick={(e) => e.stopPropagation()}
                            disabled={isMinimal}
                          />
                        </div>
                        <div className={styles.adjustSummaryLine}>
                          HT <strong>{isMinimal ? '—' : formatCurrency(panelPreviewFigures.amount)}</strong>
                          {vatApplicable && !isMinimal && panelPreviewFigures.vatValue > 0 && (
                            <> • TVA <strong>{formatCurrency(panelPreviewFigures.vatValue)}</strong></>
                          )}
                          {' • '}
                          TTC{' '}
                          <strong>{isMinimal ? '—' : formatCurrency(panelPreviewFigures.total)}</strong>
                        </div>
                        <div className={styles.adjustActions}>
                          <button
                            type="button"
                            className={styles.adjustSecondaryBtn}
                            onClick={(e) => handleCancelAdjustment(e)}
                          >
                            Annuler
                          </button>
                          <button
                            type="button"
                            className={styles.adjustTertiaryBtn}
                            onClick={(e) => handleRestoreCatalog(reservation.id, e)}
                            disabled={isMinimal}
                          >
                            Rétablir le tarif catalogue
                          </button>
                          <button
                            type="button"
                            className={styles.adjustPrimaryBtn}
                            onClick={(e) => handleSaveAdjustment(reservation.id, e)}
                            disabled={isMinimal}
                          >
                            Enregistrer
                          </button>
                        </div>
                      </>
                    )}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {!compactMode && (() => {
        const selected = reservations.filter((r) => selectedIds.includes(r.id));
        const totalTTC = selected.reduce((sum, r) => {
          const figures = computeAmounts(r);
          return sum + figures.total;
        }, 0);
        return (
          <div className={styles.summaryMinimal}>
            {selectedIds.length} sélectionné{selectedIds.length > 1 ? 's' : ''} • TTC {formatCurrency(totalTTC)}
          </div>
        );
      })()}
    </div>
  );
};

export default ReservationSelector;
