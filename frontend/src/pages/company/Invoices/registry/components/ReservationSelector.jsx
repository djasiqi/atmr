import React, { useCallback, useEffect, useRef, useState } from 'react';
import { toast } from 'sonner';
import { invoiceService } from '../../../../../services/invoiceService';
import { BILLING_SOURCE } from '../../../../../utils/billingRecipient';
import styles from './ReservationSelector.module.css';

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
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hasAutoSelected, setHasAutoSelected] = useState(false);
  const [expandedReservations, setExpandedReservations] = useState(new Set());
  /** Un seul panneau d’ajustement ouvert à la fois → évite les sauts de scroll sur les longues listes */
  const [openAdjustmentId, setOpenAdjustmentId] = useState(null);
  const [allowManualOverride, setAllowManualOverride] = useState(false);
  const [loadingReservationDetails, setLoadingReservationDetails] = useState(new Set());
  // ✅ État local pour les valeurs en cours de saisie (évite re-renders pendant la frappe)
  const [localInputValues, setLocalInputValues] = useState({});
  // ✅ Refs pour auto-focus sur les inputs "Montant HT" (un par réservation)
  const amountInputRefs = useRef({});
  // ✅ Ne pas nettoyer localInputValues du champ en cours de saisie (évite de devoir recliquer pour continuer à taper)
  const focusedAmountInputIdRef = useRef(null);
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

  // ✅ Nettoyer localInputValues pour les réservations désélectionnées ou démontées
  useEffect(() => {
    // Nettoyer les valeurs locales pour les IDs qui ne sont plus sélectionnés ou qui n'existent plus
    setLocalInputValues((prev) => {
      const next = { ...prev };
      let changed = false;
      
      // Supprimer les valeurs pour les réservations qui ne sont plus dans la liste
      const validIds = new Set(reservations.map((r) => r.id));
      Object.keys(next).forEach((id) => {
        const numId = Number(id);
        if (!validIds.has(numId) || !selectedIds.includes(numId)) {
          delete next[id];
          changed = true;
        }
      });
      
      return changed ? next : prev;
    });
  }, [reservations, selectedIds]);

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

  const handleToggleAdjustments = async (reservationId, e) => {
    e.stopPropagation();
    
    // Trouver la réservation dans l'état actuel
    const reservation = reservations.find((r) => r.id === reservationId);
    
    // Si la réservation est minimale (uniquement { id }), charger les détails
    if (reservation && isMinimalReservation(reservation)) {
      const detailed = await loadReservationDetails(reservationId);
      if (!detailed) {
        // Erreur de chargement, ne pas ouvrir l'ajustement
        return;
      }
    }
    
    // Si l'item n'est pas sélectionné, le sélectionner d'abord
    if (!selectedIds.includes(reservationId)) {
      setSelectedIds((prev) => [...prev, reservationId]);
    }
    // Ouvrir / fermer (un seul panneau à la fois)
    setOpenAdjustmentId((prev) => (prev === reservationId ? null : reservationId));
  };

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

  const normalizeAmount = useCallback((value) => {
    if (!value || value === '' || value === null || value === undefined) {
      return { normalized: null, formatted: null, isValid: true };
    }
    const normalized = String(value).replace(/,/g, '.').trim().replace(/\s/g, '');
    const numeric = parseFloat(normalized);
    if (Number.isNaN(numeric) || numeric < 0) {
      return { normalized: null, formatted: null, isValid: false };
    }
    const rounded = Math.round(numeric * 100) / 100;
    return { normalized: rounded, formatted: rounded.toFixed(2), isValid: true };
  }, []);

  // ✅ Mise à jour uniquement de l'état local pendant la saisie → saisie fluide (ex. "350")
  // Le parent est mis à jour au blur ; le total TTC se met à jour à la sortie du champ.
  const handleAmountChange = useCallback((reservationId, value) => {
    setLocalInputValues((prev) => ({
      ...prev,
      [reservationId]: value,
    }));
  }, []);

  // ✅ Synchroniser avec le parent lors du blur (quand l'utilisateur quitte le champ)
  const handleAmountBlur = useCallback((reservationId, value, originalAmount) => {
    if (!onOverrideChange) return;
    
    // Normaliser la valeur
    const result = normalizeAmount(value);
    
    if (!result.isValid) {
      // Valeur non vide mais invalide : toast + réinitialiser à la valeur d'origine
      toast.error('Montant invalide. Réinitialisation à la valeur d\'origine.');
      onOverrideChange(reservationId, { amount: originalAmount !== undefined ? originalAmount : null });
      // Nettoyer l'état local
      setLocalInputValues((prev) => {
        const next = { ...prev };
        delete next[reservationId];
        return next;
      });
    } else if (result.normalized === null) {
      // Valeur vide : reset à valeur d'origine SANS toast
      onOverrideChange(reservationId, { amount: originalAmount !== undefined ? originalAmount : null });
      // Nettoyer l'état local
      setLocalInputValues((prev) => {
        const next = { ...prev };
        delete next[reservationId];
        return next;
      });
    } else {
      // Valeur valide : appliquer la valeur normalisée
      onOverrideChange(reservationId, { amount: result.normalized });
      // Afficher la valeur formatée dans l'input (ex: 45,5 -> 45.50)
      // Le nettoyage se fera de manière déterministe via useEffect qui observe overrides
      setLocalInputValues((prev) => ({
        ...prev,
        [reservationId]: result.formatted,
      }));
    }
  }, [onOverrideChange, normalizeAmount]);

  // ✅ Nettoyage déterministe : clear localInputValues quand la valeur parent correspond
  // (sauf pour le champ actuellement focalisé, pour pouvoir taper "35" d'affilée sans recliquer)
  useEffect(() => {
    setLocalInputValues((prev) => {
      const next = { ...prev };
      let changed = false;
      const focusedId = focusedAmountInputIdRef.current;

      Object.keys(next).forEach((idStr) => {
        const id = Number(idStr);
        if (id === focusedId) return; // ne pas toucher au champ en cours de saisie
        const localValue = next[idStr];
        
        // Trouver la réservation correspondante
        const reservation = reservations.find((r) => r.id === id);
        if (!reservation) {
          // Réservation n'existe plus, nettoyer
          delete next[idStr];
          changed = true;
          return;
        }

        // Vérifier la valeur parent (override.amount ou reservation.amount)
        const override = overrides?.[String(id)] || overrides?.[id] || {};
        const parentAmount = override.amount !== undefined && override.amount !== null
          ? override.amount
          : reservation.amount;

        // Si la valeur parent correspond à la valeur locale formatée (arrondie), nettoyer
        if (parentAmount !== undefined && parentAmount !== null) {
          const parentFormatted = Number(parentAmount.toFixed(2));
          const localNumeric = parseFloat(localValue);
          
          if (!Number.isNaN(localNumeric) && Math.abs(parentFormatted - localNumeric) < 0.01) {
            // Valeurs correspondent → nettoyer
            delete next[idStr];
            changed = true;
          }
        }
      });

      return changed ? next : prev;
    });
  }, [overrides, reservations]);

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

  // ✅ Commit aussi sur Enter
  const handleAmountKeyDown = useCallback((e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      e.currentTarget.blur(); // Déclenche handleAmountBlur
    }
  }, []);

  const handleNoteChange = (reservationId, value) => {
    if (!onOverrideChange) return;
    onOverrideChange(reservationId, { note: value?.trim?.() ? value : null });
  };

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

  const computeAmounts = useCallback(
    (reservation) => {
      // ✅ Cohérence avec NewInvoiceModal: utiliser String(id) comme clé pour les overrides
      const override = overrides?.[String(reservation.id)] || overrides?.[reservation.id] || {};
      const amount = Number(override.amount ?? reservation.amount ?? 0);
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
      {!hideClientName && (
        <div className={styles.headerRow}>
          <div className={styles.headerLeft}>
            <h4 className={styles.clientNameSmall}>{clientName}</h4>
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
      )}

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

      <div className={`${styles.reservationsList} ${!compactMode ? styles.reservationsListDense : ''}`}>
        {/* ✅ Afficher aussi les réservations sélectionnées qui ne sont pas encore dans la liste chargée (objets minimaux) */}
        {(() => {
          // Combiner les réservations chargées avec les IDs sélectionnés qui ne sont pas encore chargés
          const loadedIds = new Set(reservations.map((r) => r.id));
          const missingSelectedIds = selectedIds.filter((id) => !loadedIds.has(id));
          const allReservations = [
            ...reservations,
            ...missingSelectedIds.map((id) => ({ id })), // Objets minimaux pour les IDs non chargés
          ];
          
          return allReservations.map((reservation) => {
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

          // Mode compact: 1 ligne par défaut
          if (compactMode) {
            const autoSelected = isAutoSelected(reservation);
            const billedToPatient = isBilledToPatient(reservation);
            const needsReviewCheck = needsReview(reservation);
            const checkboxDisabled = autoSelected && !allowManualOverride;

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
                  {needsReviewCheck && (
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
                          <button
                            type="button"
                            className={styles.adjustLink}
                            onClick={(e) => handleToggleAdjustments(reservation.id, e)}
                            title={isMinimal ? 'Chargement des détails...' : 'Ajuster le montant'}
                            disabled={isLoadingDetails}
                          >
                            {isLoadingDetails ? '⏳' : '✏️'}
                          </button>
                        ) : isMinimal && isLoadingDetails ? (
                          <div className={styles.adjustments}>
                            <div className={styles.adjustRow}>
                              <span className={styles.adjustLabel}>Chargement des détails...</span>
                            </div>
                          </div>
                        ) : (
                          <div className={styles.adjustments}>
                            <div className={styles.adjustGrid}>
                              <label className={styles.field}>
                                <span>Montant HT</span>
                                <input
                                  ref={(el) => {
                                    if (el) {
                                      amountInputRefs.current[reservation.id] = el;
                                    } else {
                                      delete amountInputRefs.current[reservation.id];
                                    }
                                  }}
                                  type="number"
                                  step="0.05"
                                  min="0"
                                  className={styles.input}
                                  value={
                                    localInputValues[reservation.id] !== undefined
                                      ? localInputValues[reservation.id]
                                      : (overrides?.[String(reservation.id)] || overrides?.[reservation.id])?.amount !== undefined
                                        ? (overrides[String(reservation.id)] || overrides[reservation.id]).amount
                                        : ''
                                  }
                                  placeholder={isMinimal ? '0.00' : figures.amount.toFixed(2)}
                                  onChange={(e) => handleAmountChange(reservation.id, e.target.value)}
                                  onFocus={() => { focusedAmountInputIdRef.current = reservation.id; }}
                                  onBlur={(e) => {
                                    focusedAmountInputIdRef.current = null;
                                    handleAmountBlur(reservation.id, e.target.value, reservation.amount);
                                  }}
                                  onKeyDown={handleAmountKeyDown}
                                  onClick={(e) => e.stopPropagation()}
                                  disabled={isMinimal}
                                />
                              </label>
                            </div>
                            <label className={styles.field}>
                              <span>Note d'ajustement (facultatif)</span>
                              <textarea
                                rows={2}
                                className={styles.noteInput}
                                value={(overrides?.[String(reservation.id)] || overrides?.[reservation.id])?.note ?? ''}
                                placeholder="Ex. Ajustement temps d'attente"
                                onChange={(e) => handleNoteChange(reservation.id, e.target.value)}
                                onClick={(e) => e.stopPropagation()}
                                disabled={isMinimal}
                              />
                            </label>
                            <div className={styles.adjustSummary}>
                              <span>
                                HT <strong>{isMinimal ? '—' : formatCurrency(figures.amount)}</strong>
                              </span>
                              {vatApplicable && !isMinimal && figures.vatValue > 0 && (
                                <span>
                                  TVA <strong>{formatCurrency(figures.vatValue)}</strong>
                                </span>
                              )}
                              <span>
                                TTC <strong>{isMinimal ? '—' : formatCurrency(figures.total)}</strong>
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
          const routeLabel = `${reservation.pickup_location} → ${reservation.dropoff_location}`;

          return (
            <label
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
                <div className={styles.reservationSingleLine}>
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
                    {needsReviewCheck && (
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
                    {reservation.is_return && <span className={styles.returnBadgeCompact}>Retour</span>}
                    {reservation.is_urgent && <span className={styles.urgentBadgeCompact}>Urgent</span>}
                    {reservation.medical_facility && (
                      <span className={styles.medicalBadgeCompact} title={reservation.medical_facility}>
                        {reservation.medical_facility}
                      </span>
                    )}
                  </div>
                  <span className={styles.amountHT}>
                    {isMinimal ? '—' : formatCurrency(figures.amount)}
                  </span>
                  {!showAdjust && (
                    <button
                      type="button"
                      className={styles.adjustLink}
                      onClick={(e) => handleToggleAdjustments(reservation.id, e)}
                      title={isMinimal ? 'Chargement des détails...' : 'Ajuster le montant'}
                      aria-expanded={showAdjust}
                      aria-controls={`adjust-${reservation.id}`}
                      disabled={isLoadingDetails}
                    >
                      {isLoadingDetails ? '⏳' : '✏️'}
                    </button>
                  )}
                </div>

                {showAdjust && (
                  <div className={styles.adjustInline} id={`adjust-${reservation.id}`}>
                    {isMinimal && isLoadingDetails ? (
                      <div className={styles.adjustRow}>
                        <span className={styles.adjustLabel}>Chargement des détails...</span>
                      </div>
                    ) : (
                      <>
                        <div className={styles.adjustRow}>
                          <span className={styles.adjustLabel}>Montant HT</span>
                          <input
                            ref={(el) => {
                              if (el) {
                                amountInputRefs.current[reservation.id] = el;
                              } else {
                                delete amountInputRefs.current[reservation.id];
                              }
                            }}
                            type="number"
                            step="0.05"
                            min="0"
                            className={styles.adjustInput}
                            value={
                              localInputValues[reservation.id] !== undefined
                                ? localInputValues[reservation.id]
                                : (overrides?.[String(reservation.id)] || overrides?.[reservation.id])?.amount !== undefined
                                  ? (overrides[String(reservation.id)] || overrides[reservation.id]).amount
                                  : ''
                            }
                            placeholder={isMinimal ? '0.00' : figures.amount.toFixed(2)}
                            onChange={(e) => handleAmountChange(reservation.id, e.target.value)}
                            onFocus={() => { focusedAmountInputIdRef.current = reservation.id; }}
                            onBlur={(e) => {
                              focusedAmountInputIdRef.current = null;
                              handleAmountBlur(reservation.id, e.target.value, reservation.amount);
                            }}
                            onKeyDown={handleAmountKeyDown}
                            onClick={(e) => e.stopPropagation()}
                            disabled={isMinimal}
                          />
                          <span className={styles.adjustSuffix}>CHF</span>
                        </div>
                        <div className={styles.adjustRowNote}>
                          <span className={styles.adjustLabel}>Note (optionnelle)</span>
                          <input
                            type="text"
                            className={styles.adjustNoteInput}
                            value={(overrides?.[String(reservation.id)] || overrides?.[reservation.id])?.note ?? ''}
                            placeholder="Ex. Ajustement temps d'attente"
                            onChange={(e) => handleNoteChange(reservation.id, e.target.value)}
                            onClick={(e) => e.stopPropagation()}
                            disabled={isMinimal}
                          />
                        </div>
                        <div className={styles.adjustSummaryLine}>
                          HT <strong>{isMinimal ? '—' : formatCurrency(figures.amount)}</strong>
                          {vatApplicable && !isMinimal && figures.vatValue > 0 && (
                            <> • TVA <strong>{formatCurrency(figures.vatValue)}</strong></>
                          )}
                          {' • '}
                          TTC <strong>{isMinimal ? '—' : formatCurrency(figures.total)}</strong>
                        </div>
                        <div className={styles.adjustActions}>
                          <button
                            type="button"
                            className={styles.adjustResetBtn}
                            onClick={(e) => {
                              e.stopPropagation();
                              // Réinitialiser: montant = valeur d'origine, note vide
                              if (onOverrideChange) {
                                onOverrideChange(reservation.id, { amount: null, note: null });
                              }
                            }}
                            disabled={isMinimal}
                          >
                            Réinitialiser
                          </button>
                          <button
                            type="button"
                            className={styles.adjustLink}
                            onClick={(e) => handleToggleAdjustments(reservation.id, e)}
                            aria-expanded={true}
                            aria-controls={`adjust-${reservation.id}`}
                          >
                            Fermer
                          </button>
                        </div>
                      </>
                    )}
                  </div>
                )}
              </div>
            </label>
          );
          });
        })()}
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
