import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { toast } from 'sonner';
import styles from './NewInvoiceModal.module.css';
import { formatCurrencyCHF, generateInvoice, invoiceService } from '../../../../../services/invoiceService';
import { setBookingPayer } from '../../../../../services/billingReviewService';
import ReservationSelector from './ReservationSelector';
import PartnerTransferSelector from './PartnerTransferSelector';
import useUrlSearchSync from '../../../../../hooks/useUrlSearchSync';
import useCompanyData from '../../../../../hooks/useCompanyData';

const formatApiError = (err) => {
  const data = err?.response?.data ?? err?.data ?? err;
  const status = err?.response?.status;
  const statusText = err?.response?.statusText;

  const extractMessage = (value, depth = 0) => {
    if (value === null || value === undefined) return null;
    if (typeof value === 'string') return value;
    if (typeof value === 'number' || typeof value === 'boolean') return String(value);
    if (Array.isArray(value)) {
      const parts = value.map((item) => extractMessage(item, depth + 1)).filter(Boolean);
      return parts.length ? parts.join('\n') : null;
    }
    if (typeof value === 'object') {
      const candidates = [
        value.message,
        value.error,
        value.detail,
        value.description,
        value.reason,
        value.title,
      ];
      for (const candidate of candidates) {
        const msg = extractMessage(candidate, depth + 1);
        if (msg) return msg;
      }
      if (value.errors) {
        const msg = extractMessage(value.errors, depth + 1);
        if (msg) return msg;
      }
      if (value.data) {
        const msg = extractMessage(value.data, depth + 1);
        if (msg) return msg;
      }
    }
    return null;
  };

  const main = extractMessage(data) || extractMessage(err?.message);

  // ✅ Marshmallow: préférer les messages dans data.errors si message est générique
  const genericValidation = 'Erreur de validation des données';
  let preferred = main;
  if (data && typeof data === 'object' && data.errors && typeof data.errors === 'object') {
    const flat = [];
    const visit = (o) => {
      if (Array.isArray(o)) o.forEach((x) => { const m = extractMessage(x); if (m) flat.push(m); });
      else if (o && typeof o === 'object') Object.values(o).forEach(visit);
    };
    visit(data.errors);
    const firstError = flat[0];
    if (firstError && (!preferred || preferred === genericValidation)) {
      preferred = flat.length > 1 ? flat.join(' ; ') : firstError;
    }
  }

  const statusLabel = status ? `HTTP ${status}${statusText ? ` ${statusText}` : ''}` : null;
  if (preferred && statusLabel && !preferred.includes(statusLabel)) {
    return `${preferred} (${statusLabel})`;
  }
  return preferred || statusLabel || 'Erreur inconnue';
};

const NewInvoiceModal = ({
  open,
  onClose,
  onInvoiceGenerated,
  companyId,
  initialDraft = null,
  refreshTrigger = 0,
}) => {
  const navigate = useNavigate();
  const { company } = useCompanyData();
  const [billingType, setBillingType] = useState('direct'); // 'direct', 'third_party' ou 'partner'
  const [formData, setFormData] = useState({
    client_id: '',
    client_ids: [],
    bill_to_client_id: '',
    partnership_id: '',
    period_year: new Date().getFullYear(),
    period_month: new Date().getMonth() + 1,
  });
  const [clients, setClients] = useState([]);
  const [clientCache, setClientCache] = useState({});
  const [clientSearch, setClientSearch] = useState('');
  const [clientsLoading, setClientsLoading] = useState(false);
  const [clientsError, setClientsError] = useState(null);
  // ✅ Référence pour garder le focus sur l'input de recherche
  const clientSearchInputRef = useRef(null);
  const wasInputFocusedRef = useRef(false);
  const searchDebounceRef = useRef(null);
  const { initialSearch, shouldFocus, consumeFocus, initialized } = useUrlSearchSync();
  const [institutions, setInstitutions] = useState([]);
  const [partners, setPartners] = useState([]);
  const [partnersLoading, setPartnersLoading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);
  const lastInstitutionRef = useRef(null);
  // ✅ Génération groupée (batch) - génère N factures en une opération (une par patient)
  // Note: Ce n'est PAS une vraie consolidation (1 facture unique pour tous les patients)
  // Pour une vraie consolidation, il faudrait modifier le backend
  const [isConsolidated, setIsConsolidated] = useState(false);
  // ✅ S2: Facture clinique mensuelle unique (1 facture pour tous les patients)
  // En mode third_party, S2 est activé par défaut
  const [isClinicMonthly, setIsClinicMonthly] = useState(false);
  const [includeClientIds, setIncludeClientIds] = useState([]); // Exceptions: limiter à certains patients
  // ✅ Totaux S2 (basés sur tous les transports éligibles, pas la sélection UI)
  const [s2Totals, setS2Totals] = useState({
    total_eligible: 0,
    total_amount_eligible: 0,
    total_invoiced: 0,
    total_amount_invoiced: 0,
    total_excluded: 0,
    total_amount_excluded: 0,
    excluded_bookings: [], // Liste des bookings exclus pour affichage dans l'accordéon
  });
  const [s2TotalsLoading, setS2TotalsLoading] = useState(false);
  // ✅ États pour les accordéons S2 (validation rapide)
  const [showS2Summary, setShowS2Summary] = useState(true); // Résumé (ouvert par défaut)
  const [showS2Exclusions, setShowS2Exclusions] = useState(false); // Exclusions (fermé par défaut)
  const [showDirectTransports, setShowDirectTransports] = useState(false); // Transports à facturer (fermé par défaut, comme S2)
  const [showPartnerSummary, setShowPartnerSummary] = useState(true); // Facture partenaire (ouvert par défaut)
  const [partnerOverrides, setPartnerOverrides] = useState({}); // { transferId: { amount?, note? } }
  const [partnerSelectedTransfers, setPartnerSelectedTransfers] = useState([]); // Transferts sélectionnés pour la facture

  useEffect(() => {
    if (billingType === 'partner') {
      setPartnerOverrides({});
      setPartnerSelectedTransfers([]);
    }
  }, [formData.partnership_id, formData.period_year, formData.period_month, billingType]);
  const [showS2Patients, setShowS2Patients] = useState(false); // Patients inclus (fermé par défaut)
  const [showS2Advanced, setShowS2Advanced] = useState(false); // Options avancées (fermé par défaut)
  const [expandedPatientId, setExpandedPatientId] = useState(null); // Patient dont on affiche les détails
  // ✅ États pour les trajets des patients (S2 - marge d'erreur)
  const [patientBookings, setPatientBookings] = useState({}); // { client_id: [booking_objects] }
  const [patientBookingsLoading, setPatientBookingsLoading] = useState({}); // { client_id: boolean }
  const [bookingOverridesInProgress, setBookingOverridesInProgress] = useState(new Set()); // IDs des bookings en cours d'override
  const [bookingOverrideConfirm, setBookingOverrideConfirm] = useState(null); // { bookingId, clientId } pour la confirmation inline
  const confirmButtonRef = useRef(null); // Référence pour le focus automatique sur le bouton "Confirmer"
  const overrideRequestVersions = useRef({}); // { bookingId: version } pour ignorer les réponses obsolètes
  const overrideInFlightRef = useRef(new Set()); // Sync guard: prevent double submit (state updates async)
  // ✅ S2: Ajustement montant/note par transport avant génération facture clinique (comme côté client)
  const [s2BookingOverrides, setS2BookingOverrides] = useState({}); // { bookingId: { amount?, note? } }
  const [s2AdjustOpenBookingId, setS2AdjustOpenBookingId] = useState(null); // ID du transport dont le panneau d'ajustement est ouvert
  const [s2AmountInputLocal, setS2AmountInputLocal] = useState({}); // { bookingId: string } valeur en cours de saisie

  // NOUVEAU: Gestion des sélections de réservations par client
  const [selectedReservations, setSelectedReservations] = useState({}); // { client_id: [reservation_objects] }
  const [overrides, setOverrides] = useState({});
  const [preselectedReservations, setPreselectedReservations] = useState({});
  useEffect(() => {
    if (!open) return;

    if (initialDraft) {
      const billing = initialDraft.billing_type === 'partner'
        ? 'partner'
        : initialDraft.billing_type === 'third_party'
          ? 'third_party'
          : 'direct';
      setBillingType(billing);

      setFormData({
        client_id: initialDraft.client_id ? String(initialDraft.client_id) : '',
        client_ids:
          billing === 'third_party' && Array.isArray(initialDraft.client_ids)
            ? initialDraft.client_ids
            : [],
        bill_to_client_id: initialDraft.bill_to_client_id
          ? String(initialDraft.bill_to_client_id)
          : '',
        partnership_id: initialDraft.partnership_id
          ? String(initialDraft.partnership_id)
          : '',
        period_year: initialDraft.period_year ?? new Date().getFullYear(),
        period_month: initialDraft.period_month ?? new Date().getMonth() + 1,
      });

      setOverrides(initialDraft.overrides || {});
      if (
        Array.isArray(initialDraft.reservation_ids) &&
        initialDraft.client_id &&
        initialDraft.reservation_ids.length > 0
      ) {
        setPreselectedReservations({
          [initialDraft.client_id]: initialDraft.reservation_ids.map((id) => Number(id)),
        });
      } else {
        setPreselectedReservations({});
      }

      if (initialDraft.client) {
        setClientCache((prev) => ({ ...prev, [initialDraft.client.id]: initialDraft.client }));
        setClients((prev) => {
          if (prev.some((c) => c.id === initialDraft.client.id)) {
            return prev;
          }
          return [...prev, initialDraft.client];
        });
      }

      setSelectedReservations({});
      setClientSearch('');
      const el0 = clientSearchInputRef.current;
      if (el0) el0.value = '';
      if (billing === 'direct') setShowDirectTransports(true);
      return;
    }

    // Réinitialiser les champs pour une création manuelle
    setBillingType('direct');
    setFormData({
      client_id: '',
      client_ids: [],
      bill_to_client_id: '',
      period_year: new Date().getFullYear(),
      period_month: new Date().getMonth() + 1,
    });
    setOverrides({});
    setSelectedReservations({});
    setPreselectedReservations({});
    setShowDirectTransports(false);
    setS2BookingOverrides({});
    setS2AdjustOpenBookingId(null);
    setS2AmountInputLocal({});
    setClientSearch('');
    const el1 = clientSearchInputRef.current;
    if (el1) el1.value = '';
  }, [open, initialDraft]);

  useEffect(() => {
    if (!open || !initialized) return;
    if (initialSearch && initialSearch !== clientSearch) {
      setClientSearch(initialSearch);
      const el = clientSearchInputRef.current;
      if (el) el.value = initialSearch;
    }
    if (shouldFocus) {
      window.scrollTo({ top: 0, behavior: 'smooth' });
      requestAnimationFrame(() => {
        clientSearchInputRef.current?.focus();
      });
      consumeFocus();
    }
  }, [open, initialized, initialSearch, shouldFocus, consumeFocus, clientSearch]);

  const handleSearchChange = useCallback(() => {
    if (searchDebounceRef.current) clearTimeout(searchDebounceRef.current);
    searchDebounceRef.current = setTimeout(() => {
      searchDebounceRef.current = null;
      const el = clientSearchInputRef.current;
      if (el) setClientSearch(el.value);
    }, 280);
  }, []);

  useEffect(() => {
    return () => {
      if (searchDebounceRef.current) clearTimeout(searchDebounceRef.current);
    };
  }, []);
  const [vatConfig, setVatConfig] = useState({
    applicable: false,
    defaultRate: 0,
    label: '',
    number: '',
  });

  const selectedInstitution = useMemo(
    () =>
      institutions.find(
        (inst) => String(inst.id) === String(formData.bill_to_client_id)
      ) || null,
    [institutions, formData.bill_to_client_id]
  );

  const clinicsForS2 = useMemo(
    () => institutions,
    [institutions]
  );

  /** ID de la clinique pour set-payer (billed_to_company_id). Fallback sur id si clinic_company_id absent. Entier >= 1 ou null. */
  const clinicCompanyId = useMemo(() => {
    const v = selectedInstitution?.clinic_company_id ?? selectedInstitution?.id;
    if (v == null || v === undefined) return null;
    const n = Number(v);
    return Number.isFinite(n) && n >= 1 ? Math.floor(n) : null;
  }, [selectedInstitution?.clinic_company_id, selectedInstitution?.id]);

  // ✅ Ne plus forcer third_party quand bill_to_client_id existe : ça empêchait de cliquer
  // sur "Facturation directe" / "Facturation partenaire" en mode facture clinique.

  // ✅ En mode third_party, activer S2 par défaut (facture clinique mensuelle)
  const hasInitializedS2 = useRef(false);
  useEffect(() => {
    if (!open) {
      hasInitializedS2.current = false;
      return;
    }
    if (billingType === 'third_party' && formData.bill_to_client_id && selectedInstitution?.clinic_company_id) {
      // Activer S2 par défaut si on est en mode third_party avec une clinique (une seule fois)
      if (!hasInitializedS2.current) {
        setIsClinicMonthly(true);
        setIsConsolidated(false);
        setShowS2Advanced(false); // Masquer les options avancées par défaut
        hasInitializedS2.current = true;
      }
    } else if (billingType !== 'third_party') {
      // Désactiver S2 si on n'est plus en mode third_party
      setIsClinicMonthly(false);
      setShowS2Advanced(false);
      hasInitializedS2.current = false;
    }
  }, [billingType, formData.bill_to_client_id, selectedInstitution?.clinic_company_id, open]);

  // ✅ Charger les totaux S2 (tous les transports éligibles, pas la sélection UI)
  useEffect(() => {
    if (!open || !companyId) return;
    if (!isClinicMonthly || !formData.bill_to_client_id || !selectedInstitution?.clinic_company_id) {
      setS2Totals({
        total_eligible: 0,
        total_amount_eligible: 0,
        total_excluded: 0,
        total_amount_excluded: 0,
        excluded_bookings: [],
      });
      // Réinitialiser les accordéons
      setShowS2Summary(true);
      setShowS2Exclusions(false);
      setShowS2Patients(false);
      setShowS2Advanced(false);
      return;
    }

    let cancelled = false;

    const fetchS2Totals = async () => {
      try {
        setS2TotalsLoading(true);
        const response = await invoiceService.fetchClinicMonthlyTotals(companyId, {
          year: formData.period_year,
          month: formData.period_month,
          clinic_company_id: selectedInstitution.clinic_company_id,
          include_client_ids: includeClientIds.length > 0 ? includeClientIds : undefined,
        });

        if (!cancelled && response?.data) {
          // ✅ Hardening: safe Number pour éviter NaN
          const safeTotalEligible = Number.isFinite(response.data.total_eligible) ? Math.floor(response.data.total_eligible) : 0;
          const safeTotalAmountEligible = Number.isFinite(response.data.total_amount_eligible) && !Number.isNaN(response.data.total_amount_eligible)
            ? Number(response.data.total_amount_eligible.toFixed(2))
            : 0;
          const safeTotalExcluded = Number.isFinite(response.data.total_excluded) ? Math.floor(response.data.total_excluded) : 0;
          const safeTotalAmountExcluded = Number.isFinite(response.data.total_amount_excluded) && !Number.isNaN(response.data.total_amount_excluded)
            ? Number(response.data.total_amount_excluded.toFixed(2))
            : 0;
          
          const safeTotalInvoiced = Number.isFinite(response.data.total_invoiced) ? Math.floor(response.data.total_invoiced) : 0;
          const safeTotalAmountInvoiced = Number.isFinite(response.data.total_amount_invoiced) && !Number.isNaN(response.data.total_amount_invoiced)
            ? Number(response.data.total_amount_invoiced.toFixed(2))
            : 0;
          const totals = {
            total_eligible: safeTotalEligible,
            total_amount_eligible: safeTotalAmountEligible,
            total_invoiced: safeTotalInvoiced,
            total_amount_invoiced: safeTotalAmountInvoiced,
            total_excluded: safeTotalExcluded,
            total_amount_excluded: safeTotalAmountExcluded,
            excluded_bookings: Array.isArray(response.data.excluded_bookings) ? response.data.excluded_bookings : [],
          };
          setS2Totals(totals);
          // Accordéon exclusions : fermé à l'accès, ouverture/fermeture manuelle
        }
      } catch (err) {
        console.error('❌ [NewInvoiceModal] Erreur lors du chargement des totaux S2:', err);
        if (!cancelled) {
        setS2Totals({
          total_eligible: 0,
          total_amount_eligible: 0,
          total_invoiced: 0,
          total_amount_invoiced: 0,
          total_excluded: 0,
          total_amount_excluded: 0,
          excluded_bookings: [],
        });
        }
      } finally {
        if (!cancelled) {
          setS2TotalsLoading(false);
        }
      }
    };

    fetchS2Totals();

    return () => {
      cancelled = true;
    };
  }, [
    open,
    companyId,
    isClinicMonthly,
    formData.bill_to_client_id,
    formData.period_year,
    formData.period_month,
    selectedInstitution?.clinic_company_id,
    includeClientIds,
    refreshTrigger,
  ]);

  // ✅ Charger les bookings d'un patient pour afficher les trajets (S2 - marge d'erreur)
  const loadPatientBookings = useCallback(async (clientId) => {
    if (!companyId || !clientId || !isClinicMonthly || !formData.period_year || !formData.period_month) {
      return;
    }

    try {
      setPatientBookingsLoading((prev) => ({ ...prev, [clientId]: true }));
      
      const data = await invoiceService.fetchUnbilledReservations(companyId, clientId, {
        year: formData.period_year,
        month: formData.period_month,
        billed_to_type: 'clinic',
        // Inclure les transports déjà facturés pour afficher la vue complète (S2)
        include_invoiced: true,
        clinic_company_id: selectedInstitution?.clinic_company_id,
      });

      const bookings = Array.isArray(data?.reservations) ? data.reservations : [];
      
      // ✅ Filtrer strictement les bookings de la clinique sélectionnée (hardening)
      const clinicBookings = bookings.filter((booking) => {
        // Filtre strict: billed_to_type doit être exactement 'clinic'
        if (booking.billed_to_type !== 'clinic') {
          return false;
        }
        // Si on a une clinique sélectionnée, vérifier aussi billed_to_company_id
        if (selectedInstitution?.clinic_company_id) {
          return booking.billed_to_company_id === selectedInstitution.clinic_company_id;
        }
        return true;
      });

      setPatientBookings((prev) => ({ ...prev, [clientId]: clinicBookings }));
    } catch (err) {
      console.error(`❌ [NewInvoiceModal] Erreur lors du chargement des trajets pour le patient ${clientId}:`, err);
      setPatientBookings((prev) => ({ ...prev, [clientId]: [] }));
    } finally {
      setPatientBookingsLoading((prev) => ({ ...prev, [clientId]: false }));
    }
  }, [companyId, isClinicMonthly, formData.period_year, formData.period_month, selectedInstitution?.clinic_company_id]);

  // ✅ Gérer l'override de facturation d'un booking (S2 - marge d'erreur) avec optimistic update
  const handleBookingBillingOverride = useCallback(async (bookingId, newBilledToType) => {
    const debugBilling = typeof sessionStorage !== 'undefined' && sessionStorage.getItem('debugBilling') === '1';

    if (!companyId) return;
    if (bookingOverridesInProgress.has(bookingId) || overrideInFlightRef.current.has(bookingId)) {
      return;
    }

    if (newBilledToType === 'clinic') {
      if (clinicCompanyId == null || clinicCompanyId < 1) {
        toast.error('Sélectionnez une clinique avant de réintégrer ce transport.', { duration: 5000 });
        setBookingOverrideConfirm(null);
        return;
      }
    }

    overrideInFlightRef.current.add(bookingId);
    setBookingOverridesInProgress((prev) => new Set(prev).add(bookingId));

    const requestVersion = Date.now() + Math.random();
    overrideRequestVersions.current[bookingId] = requestVersion;

    const affectedClientId = Object.keys(patientBookings).find((clientId) =>
      patientBookings[clientId]?.some((b) => b.id === bookingId)
    );
    const previousBooking = affectedClientId 
      ? patientBookings[affectedClientId]?.find((b) => b.id === bookingId)
      : (s2Totals.excluded_bookings || []).find((b) => b.id === bookingId);
    const previousBilledToType = previousBooking?.billed_to_type || 'clinic';
    const previousTotals = { ...s2Totals };

    const doRollback = () => {
      if (affectedClientId && previousBooking) {
        setPatientBookings((prev) => ({
          ...prev,
          [affectedClientId]: prev[affectedClientId].map((b) =>
            b.id === bookingId ? { ...b, billed_to_type: previousBilledToType } : b
          ),
        }));
      }
      setS2Totals(previousTotals);
    };

    const cleanup = () => {
      overrideInFlightRef.current.delete(bookingId);
      setBookingOverridesInProgress((prev) => {
        const next = new Set(prev);
        next.delete(bookingId);
        return next;
      });
      delete overrideRequestVersions.current[bookingId];
    };

    if (affectedClientId && previousBooking) {
      setPatientBookings((prev) => ({
        ...prev,
        [affectedClientId]: prev[affectedClientId].map((b) =>
          b.id === bookingId ? { ...b, billed_to_type: newBilledToType } : b
        ),
      }));
      if (previousBilledToType === 'clinic' && newBilledToType === 'patient') {
        setS2Totals((prev) => ({
          ...prev,
          excluded_bookings: [...(prev.excluded_bookings || []), { ...previousBooking, billed_to_type: 'patient' }],
        }));
      } else if (previousBilledToType === 'patient' && newBilledToType === 'clinic') {
        setS2Totals((prev) => ({
          ...prev,
          excluded_bookings: (prev.excluded_bookings || []).filter((b) => b.id !== bookingId),
        }));
      }
    }
    if (!affectedClientId && previousBooking && previousBilledToType === 'patient' && newBilledToType === 'clinic') {
      setS2Totals((prev) => ({
        ...prev,
        excluded_bookings: (prev.excluded_bookings || []).filter((b) => b.id !== bookingId),
      }));
    }

    try {
      const reason = newBilledToType === 'patient' 
        ? 'Override manuel depuis modal S2: facturation patient'
        : 'Override manuel depuis modal S2: retour facturation clinique';

      const payload = {
        billed_to_type: newBilledToType,
        billing_party_id: null,
        billed_to_company_id: newBilledToType === 'clinic' ? clinicCompanyId : null,
        reason,
      };
      // Check: bookingId doit venir de booking.id, pas d'index / rowIndex
      const selectedBooking = previousBooking ?? { id: bookingId };
      console.log('[NewInvoiceModal] set-payer call', {
        selectedBooking,
        bookingId: selectedBooking?.id ?? bookingId,
        payload,
      });
      const response = await setBookingPayer(bookingId, payload);

      if (overrideRequestVersions.current[bookingId] !== requestVersion) {
        if (debugBilling) console.log(`[NewInvoiceModal] Requête obsolète ignorée booking ${bookingId}`);
        return;
      }

      const rawAmount = response?.data?.amount;
      const rawOld = response?.data?.old_amount;
      const numAmount = Number(rawAmount);
      const numOld = Number(rawOld);
      const valid = Number.isFinite(numAmount) && !Number.isNaN(numAmount)
        && Number.isFinite(numOld) && !Number.isNaN(numOld);

      if (!valid) {
        console.error("[NewInvoiceModal] Réponse API invalide (amount/old_amount manquants ou NaN)", {
          bookingId,
          amount: rawAmount,
          old_amount: rawOld,
        });
        doRollback();
        toast.error('Réponse invalide (montants manquants). Veuillez réessayer.');
        return;
      }

      const safeNewAmount = numAmount;
      const safeOldAmount = numOld;

      if (debugBilling) {
        console.log("[NewInvoiceModal DEBUG] override response", {
          booking_id: bookingId,
          old_amount: rawOld,
          amount: rawAmount,
          clinic_rate: response?.data?.clinic_rate,
          rate_source: response?.data?.rate_source,
        });
      }
      
      // ✅ Mettre à jour bookingOverrideConfirm avec les valeurs strictes de la réponse API
      // ✅ Ces valeurs seront utilisées pour les calculs de delta dans les messages de confirmation
      setBookingOverrideConfirm((prev) =>
        prev?.bookingId === bookingId
          ? { ...prev, newAmount: safeNewAmount, oldAmount: safeOldAmount }
          : prev
      );
      
      // ✅ Mettre à jour le booking avec le nouveau montant dans l'UI
      if (affectedClientId && previousBooking) {
        setPatientBookings((prev) => ({
          ...prev,
          [affectedClientId]: prev[affectedClientId].map((b) =>
            b.id === bookingId
              ? { ...b, billed_to_type: newBilledToType, amount: safeNewAmount }
              : b
          ),
        }));
      }
      
      // ✅ Mettre à jour aussi dans excluded_bookings si présent
      if (previousBooking && !affectedClientId) {
        setS2Totals((prev) => ({
          ...prev,
          excluded_bookings: (prev.excluded_bookings || []).map((b) =>
            b.id === bookingId
              ? { ...b, billed_to_type: newBilledToType, amount: safeNewAmount }
              : b
          ),
        }));
      }

      // ✅ RÈGLE STRICTE : totaux recalculés UNIQUEMENT depuis la réponse API.
      // On ne corrige pas les totaux localement avec delta : on refetch fetchClinicMonthlyTotals
      // (source de vérité). delta = response.amount - response.old_amount sert uniquement pour
      // l'affichage confirmation (message) ; aucun fallback sur booking.amount / bookingOverrideConfirm.

      // ✅ Rafraîchir les totaux S2 depuis le serveur (source de vérité)
      const totalsResponse = await invoiceService.fetchClinicMonthlyTotals(companyId, {
        year: formData.period_year,
        month: formData.period_month,
        clinic_company_id: clinicCompanyId ?? selectedInstitution?.clinic_company_id ?? undefined,
        include_client_ids: includeClientIds.length > 0 ? includeClientIds : undefined,
      });

      if (overrideRequestVersions.current[bookingId] !== requestVersion) {
        if (debugBilling) console.log(`[NewInvoiceModal] Réponse obsolète ignorée (totals) booking ${bookingId}`);
        return;
      }

      if (totalsResponse?.data) {
        // ✅ Hardening: safe Number pour éviter NaN
        const safeTotalEligible = Number.isFinite(totalsResponse.data.total_eligible) ? Math.floor(totalsResponse.data.total_eligible) : 0;
        const safeTotalAmountEligible = Number.isFinite(totalsResponse.data.total_amount_eligible) && !Number.isNaN(totalsResponse.data.total_amount_eligible) 
          ? Number(totalsResponse.data.total_amount_eligible.toFixed(2)) 
          : 0;
        const safeTotalExcluded = Number.isFinite(totalsResponse.data.total_excluded) ? Math.floor(totalsResponse.data.total_excluded) : 0;
        const safeTotalAmountExcluded = Number.isFinite(totalsResponse.data.total_amount_excluded) && !Number.isNaN(totalsResponse.data.total_amount_excluded)
          ? Number(totalsResponse.data.total_amount_excluded.toFixed(2))
          : 0;
        
        const safeTotalInvoiced = Number.isFinite(totalsResponse.data.total_invoiced) ? Math.floor(totalsResponse.data.total_invoiced) : 0;
        const safeTotalAmountInvoiced = Number.isFinite(totalsResponse.data.total_amount_invoiced) && !Number.isNaN(totalsResponse.data.total_amount_invoiced)
          ? Number(totalsResponse.data.total_amount_invoiced.toFixed(2))
          : 0;
        setS2Totals({
          total_eligible: safeTotalEligible,
          total_amount_eligible: safeTotalAmountEligible,
          total_invoiced: safeTotalInvoiced,
          total_amount_invoiced: safeTotalAmountInvoiced,
          total_excluded: safeTotalExcluded,
          total_amount_excluded: safeTotalAmountExcluded,
          excluded_bookings: Array.isArray(totalsResponse.data.excluded_bookings) ? totalsResponse.data.excluded_bookings : [],
        });
      }

      // ✅ Rafraîchir les bookings du patient concerné depuis le serveur
      if (affectedClientId) {
        await loadPatientBookings(Number(affectedClientId));
      }

      // ✅ Toast de succès
      toast.success(
        newBilledToType === 'patient'
          ? 'Transport facturé au patient (exclu de la facture clinique)'
          : 'Transport facturé à la clinique (inclus dans la facture)'
      );
    } catch (err) {
      if (overrideRequestVersions.current[bookingId] !== requestVersion) {
        if (debugBilling) console.log(`[NewInvoiceModal] Erreur requête obsolète ignorée booking ${bookingId}`);
        return;
      }
      const backendPayload = err?.response?.data;
      const status = err?.response?.status;
      const isNotFound = status === 404
        || backendPayload?.error_code === 'not_found'
        || (typeof backendPayload?.message === 'string' && backendPayload.message.toLowerCase().includes('non trouvé'));
      console.error(`❌ [NewInvoiceModal] Erreur override booking ${bookingId}:`, err);
      if (backendPayload != null) {
        console.error(`❌ [NewInvoiceModal] Réponse backend (${status}):`, backendPayload);
      }
      console.error(`❌ [NewInvoiceModal] Payload envoyé:`, { bookingId, billed_to_type: newBilledToType, billed_to_company_id: newBilledToType === 'clinic' ? clinicCompanyId : null });
      doRollback();
      setBookingOverrideConfirm(null);
      if (isNotFound) {
        // Rafraîchir la liste (totaux S2 + bookings patient) et toast explicite
        const refreshTotals = async () => {
          try {
            const totalsResponse = await invoiceService.fetchClinicMonthlyTotals(companyId, {
              year: formData.period_year,
              month: formData.period_month,
              clinic_company_id: clinicCompanyId ?? selectedInstitution?.clinic_company_id ?? undefined,
              include_client_ids: includeClientIds.length > 0 ? includeClientIds : undefined,
            });
            if (totalsResponse?.data) {
              setS2Totals({
                total_eligible: Number.isFinite(totalsResponse.data.total_eligible) ? Math.floor(totalsResponse.data.total_eligible) : 0,
                total_amount_eligible: Number.isFinite(totalsResponse.data.total_amount_eligible) && !Number.isNaN(totalsResponse.data.total_amount_eligible) ? Number(totalsResponse.data.total_amount_eligible.toFixed(2)) : 0,
                total_invoiced: Number.isFinite(totalsResponse.data.total_invoiced) ? Math.floor(totalsResponse.data.total_invoiced) : 0,
                total_amount_invoiced: Number.isFinite(totalsResponse.data.total_amount_invoiced) && !Number.isNaN(totalsResponse.data.total_amount_invoiced) ? Number(totalsResponse.data.total_amount_invoiced.toFixed(2)) : 0,
                total_excluded: Number.isFinite(totalsResponse.data.total_excluded) ? Math.floor(totalsResponse.data.total_excluded) : 0,
                total_amount_excluded: Number.isFinite(totalsResponse.data.total_amount_excluded) && !Number.isNaN(totalsResponse.data.total_amount_excluded) ? Number(totalsResponse.data.total_amount_excluded.toFixed(2)) : 0,
                excluded_bookings: Array.isArray(totalsResponse.data.excluded_bookings) ? totalsResponse.data.excluded_bookings : [],
              });
            }
          } catch (e) {
            /* ignorer */
          }
        };
        refreshTotals();
        if (affectedClientId) loadPatientBookings(Number(affectedClientId));
        toast.error('La course n\'existe plus ou n\'est pas accessible pour cette société.', { duration: 5000 });
      } else {
        toast.error(formatApiError(err) || 'Erreur lors de la modification de la facturation', { duration: 5000 });
      }
    } finally {
      cleanup();
    }
  }, [
    companyId,
    bookingOverridesInProgress,
    clinicCompanyId,
    selectedInstitution?.clinic_company_id,
    formData.period_year,
    formData.period_month,
    includeClientIds,
    patientBookings,
    loadPatientBookings,
    s2Totals,
  ]);

  // ✅ Reset UI state override quand clinic_company_id/year/month change (hardening)
  useEffect(() => {
    if (!isClinicMonthly) return;
    
    // Réinitialiser tous les états UI des overrides
    setExpandedPatientId(null);
    setBookingOverrideConfirm(null);
    setPatientBookings({});
    setPatientBookingsLoading({});
    setShowS2Patients(false);
    setBookingOverridesInProgress(new Set());
    // ✅ Nettoyer les versions de requêtes obsolètes
    overrideRequestVersions.current = {};
  }, [
    selectedInstitution?.clinic_company_id,
    formData.period_year,
    formData.period_month,
    isClinicMonthly,
  ]);

  // ✅ Charger automatiquement les bookings d'un patient quand on l'ouvre (S2 - marge d'erreur)
  useEffect(() => {
    if (!expandedPatientId || !isClinicMonthly) return;
    
    const bookings = patientBookings[expandedPatientId] || [];
    const isLoading = patientBookingsLoading[expandedPatientId] || false;
    
    // Charger les bookings si on vient d'ouvrir et qu'ils ne sont pas déjà chargés
    if (bookings.length === 0 && !isLoading) {
      loadPatientBookings(expandedPatientId);
    }
  }, [expandedPatientId, isClinicMonthly, patientBookings, patientBookingsLoading, loadPatientBookings]);

  // ✅ Focus automatique sur le bouton "Confirmer" quand la confirmation s'affiche
  useEffect(() => {
    if (bookingOverrideConfirm && confirmButtonRef.current) {
      // Petit délai pour s'assurer que le DOM est mis à jour
      setTimeout(() => {
        confirmButtonRef.current?.focus();
      }, 100);
    }
  }, [bookingOverrideConfirm]);

  // ✅ Gestion des touches clavier (Escape = annuler, Enter = confirmer)
  useEffect(() => {
    if (!bookingOverrideConfirm) return;

    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        setBookingOverrideConfirm(null);
      } else if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey && !e.metaKey) {
        // Enter seul (pas Shift+Enter, Ctrl+Enter, etc.)
        const confirmButton = confirmButtonRef.current;
        if (confirmButton && !confirmButton.disabled) {
          e.preventDefault();
          confirmButton.click();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [bookingOverrideConfirm]);

  // Charger la liste des institutions à l'ouverture du modal
  useEffect(() => {
    if (!open || !companyId) return;

    let isMounted = true;

    const loadInstitutions = async () => {
      try {
        setLoading(true);
        const institutionsData = await invoiceService.fetchInstitutions(companyId);
        if (!isMounted) return;
        setInstitutions(institutionsData.institutions || []);
      } catch (err) {
        console.error('Erreur lors du chargement des institutions:', err);
        if (isMounted) {
          setError('Erreur lors du chargement des institutions');
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    };

    loadInstitutions();

    return () => {
      isMounted = false;
    };
  }, [companyId, open]);

  // Charger la liste des partenaires facturables quand le type est "partner"
  useEffect(() => {
    if (!open || !companyId || billingType !== 'partner') return;

    let isMounted = true;

    const loadPartners = async () => {
      try {
        setPartnersLoading(true);
        let response;
        try {
          response = await invoiceService.fetchBillablePartners(companyId);
        } catch (err) {
          throw err;
        }
        if (!isMounted) return;
        // La réponse est {data: [...]}, donc on accède à response.data.data
        const partnersList = response?.data?.data || response?.data || [];
        setPartners(partnersList);
      } catch (err) {
        console.error('Erreur lors du chargement des partenaires:', err);
        if (isMounted) {
          setError('Erreur lors du chargement des partenaires');
        }
      } finally {
        if (isMounted) {
          setPartnersLoading(false);
        }
      }
    };

    loadPartners();

    return () => {
      isMounted = false;
    };
  }, [companyId, open, billingType]);

  useEffect(() => {
    if (!open || !companyId) return;
    let cancelled = false;

    const loadBillingSettings = async () => {
      try {
        const settings = await invoiceService.fetchBillingSettings(companyId);
        if (cancelled || !settings) return;
        setVatConfig({
          applicable: Boolean(settings.vat_applicable),
          defaultRate:
            settings.vat_rate !== undefined && settings.vat_rate !== null
              ? Number(settings.vat_rate)
              : 0,
          label: settings.vat_label ?? '',
          number: settings.vat_number ?? '',
        });
      } catch (err) {
        console.warn('Erreur chargement paramètres TVA:', err);
        if (!cancelled) {
          setVatConfig((prev) => ({
            ...prev,
            applicable: false,
            defaultRate: 0,
          }));
        }
      }
    };

    loadBillingSettings();

    return () => {
      cancelled = true;
    };
  }, [companyId, open]);

  useEffect(() => {
    if (!open) return;
    if (billingType !== 'third_party' || !formData.bill_to_client_id) return;
    const now = new Date();
    const currentYear = now.getFullYear();
    const currentMonth = now.getMonth() + 1;
    setFormData((prev) => {
      const shouldUpdatePeriod =
        prev.period_year !== currentYear || prev.period_month !== currentMonth;
      if (!shouldUpdatePeriod) return prev;
      return {
        ...prev,
        period_year: currentYear,
        period_month: currentMonth,
      };
    });
    setClientSearch('');
    const elSearch = clientSearchInputRef.current;
    if (elSearch) elSearch.value = '';
    if (lastInstitutionRef.current !== formData.bill_to_client_id) {
      lastInstitutionRef.current = formData.bill_to_client_id;
      setOverrides({});
      setSelectedReservations({});
      setFormData((prev) => ({
        ...prev,
        client_ids: [],
      }));
    }
  }, [billingType, formData.bill_to_client_id, open]);

  // Charger les clients éligibles (trajets non facturés) avec recherche
  useEffect(() => {
    if (!open || !companyId) return;

    let cancelled = false;

    const fetchClients = async () => {
      try {
        setClientsLoading(true);
        setClientsError(null);
        const query = clientSearch.trim();

        console.log('🔍 [NewInvoiceModal] fetchEligibleClients appelé avec:', {
          companyId,
          search: query || undefined,
          limit: 120,
          year: formData.period_year,
          month: formData.period_month,
          bill_to_client_id:
            billingType === 'third_party' ? formData.bill_to_client_id : undefined,
          clinic_company_id: selectedInstitution?.clinic_company_id,
          billed_to_type: billingType === 'direct' ? 'patient' : undefined,
        });

        const response = await invoiceService.fetchEligibleClients(companyId, {
          search: query || undefined,
          limit: 120,
          year: formData.period_year,
          month: formData.period_month,
          bill_to_client_id:
            billingType === 'third_party' ? formData.bill_to_client_id : undefined,
          clinic_company_id: selectedInstitution?.clinic_company_id,
          billed_to_type: billingType === 'direct' ? 'patient' : undefined,
        });

        console.log('🔍 [NewInvoiceModal] Réponse reçue:', {
          response,
          responseData: response?.data,
          hasClients: !!(response?.data?.clients),
          clientsType: typeof response?.data?.clients,
          clientsLength: Array.isArray(response?.data?.clients) ? response.data.clients.length : 'N/A',
          clients: response?.data?.clients,
        });

        // Le service retourne response.data, donc response est déjà {clients: [...], total: ...}
        // Mais axios peut avoir une structure imbriquée, donc on vérifie les deux
        const list = Array.isArray(response?.clients) 
          ? response.clients 
          : Array.isArray(response?.data?.clients) 
            ? response.data.clients 
            : [];

        if (!list.length) {
          setClientsError(
            query
              ? 'Aucun client trouvé pour cette recherche.'
              : "Aucun client éligible (courses terminées non facturées) n'a été trouvé pour cette période."
          );
        }

        if (cancelled) return;

        setClients(list);
        setClientCache((prev) => {
          const next = { ...prev };
          list.forEach((client) => {
            if (client && client.id != null) {
              next[client.id] = client;
            }
          });
          return next;
        });
        if (
          billingType === 'third_party' &&
          formData.bill_to_client_id &&
          formData.client_ids.length === 0 &&
          list.length > 0
        ) {
          setFormData((prev) => ({
            ...prev,
            client_ids: list.map((client) => client.id),
          }));
        }
      } catch (err) {
        console.error('❌ [NewInvoiceModal] Erreur lors du chargement des clients éligibles:', {
          error: err,
          message: err?.message,
          response: err?.response,
          responseData: err?.response?.data,
          status: err?.response?.status,
          companyId,
          period: { year: formData.period_year, month: formData.period_month },
          billingType,
          bill_to_client_id: formData.bill_to_client_id,
        });
        if (!cancelled) {
          // ✅ Ne pas vider les clients existants en cas d'erreur pour éviter de perdre les données
          // Seulement définir l'erreur pour informer l'utilisateur
          setClientsError(
            'Impossible de charger les clients à facturer. Vérifiez que votre backend est à jour.'
          );
        }
      } finally {
        if (!cancelled) {
          setClientsLoading(false);
          // ✅ Refocuser l'input après la mise à jour de la liste pour permettre la saisie continue
          // Utiliser requestAnimationFrame pour s'assurer que le DOM est mis à jour
          requestAnimationFrame(() => {
            if (clientSearchInputRef.current && wasInputFocusedRef.current) {
              clientSearchInputRef.current.focus();
              // ✅ Restaurer la position du curseur à la fin du texte
              const length = clientSearchInputRef.current.value.length;
              clientSearchInputRef.current.setSelectionRange(length, length);
            }
          });
        }
      }
    };

    const timer = setTimeout(fetchClients, 100);

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [
    companyId,
    open,
    clientSearch,
    formData.period_year,
    formData.period_month,
    formData.bill_to_client_id,
    formData.client_ids.length,
    billingType,
    selectedInstitution?.clinic_company_id,
    refreshTrigger,
  ]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: name.includes('year') || name.includes('month') ? parseInt(value) : value,
    }));
  };

  const _handleClientToggle = (clientId) => {
    setFormData((prev) => {
      const isSelected = prev.client_ids.includes(clientId);
      const newClientIds = isSelected
        ? prev.client_ids.filter((id) => id !== clientId)
        : [...prev.client_ids, clientId];

      // Si on désélectionne un client, supprimer aussi ses réservations sélectionnées
      if (isSelected) {
        setSelectedReservations((prevReservations) => {
          if (!prevReservations) return {};
          const { [clientId]: _removed, ...rest } = prevReservations;
          return rest;
        });
      }

      return {
        ...prev,
        client_ids: newClientIds,
      };
    });
  };

  // IMPORTANT: Utiliser useCallback pour éviter les re-renders infinis
  const handleReservationSelectionChange = useCallback((clientId, reservations) => {
    setSelectedReservations((prev) => {
      // Vérifier que prev existe, sinon initialiser à {}
      const current = prev || {};

      // Ne mettre à jour que si les réservations ont changé
      const prevIds = (current[clientId] || [])
        .map((r) => r?.id || r)
        .sort()
        .join(',');
      const newIds = (reservations || [])
        .map((r) => r?.id || r)
        .sort()
        .join(',');

      if (prevIds === newIds) {
        return current; // Pas de changement, retourner le même objet
      }

      return {
        ...current,
        [clientId]: reservations || [],
      };
    });
  }, []);

  // ✅ Sélectionner toutes les réservations disponibles pour le client actif (mode direct)
  // Optimisation: utilise l'endpoint IDs-only, ne charge pas les détails (plus rapide)
  // Les totaux utilisent directSummary quand sélection complète
  // Note: activeClientId sera défini plus bas, on utilise formData.client_id directement
  const handleSelectAllReservations = useCallback(async () => {
    const clientId = formData.client_id ? parseInt(formData.client_id, 10) : null;
    if (!companyId || !clientId || !formData.period_year || !formData.period_month) return;
    
    try {
      // ✅ Optimisation: charger uniquement les IDs (plus rapide, pas de détails)
      const idsData = await invoiceService.fetchUnbilledReservationIds(companyId, clientId, {
        year: formData.period_year,
        month: formData.period_month,
        billed_to_type: 'patient', // Mode direct = facturation patient
      });

      const reservationIds = Array.isArray(idsData?.reservation_ids) ? idsData.reservation_ids : [];
      if (reservationIds.length > 0) {
        // Créer des objets minimaux avec uniquement l'ID
        // Les totaux utiliseront directSummaryTTC pour sélection complète
        const minimalReservations = reservationIds.map((id) => ({ id }));
        handleReservationSelectionChange(clientId, minimalReservations);
      }
    } catch (err) {
      console.error('Erreur lors du chargement des IDs de réservations:', err);
      toast.error('Erreur lors du chargement des transports');
    }
  }, [companyId, formData.client_id, formData.period_year, formData.period_month, handleReservationSelectionChange]);

  const handleOverrideChange = useCallback((reservationId, patch) => {
    const key = String(reservationId);
    setOverrides((prev) => {
      const current = prev[key] ? { ...prev[key] } : {};
      let changed = false;

      Object.entries(patch).forEach(([field, value]) => {
        if (value === null || value === undefined || value === '') {
          if (field in current) {
            delete current[field];
            changed = true;
          }
        } else if (current[field] !== value) {
          current[field] = value;
          changed = true;
        }
      });

      const next = { ...prev };
      if (Object.keys(current).length === 0) {
        if (next[key]) {
          delete next[key];
          changed = true;
        }
      } else {
        next[key] = current;
      }

      return changed ? next : prev;
    });
  }, []);

  const selectedClientIds = useMemo(() => {
    const ids = new Set();
    if (formData.client_id) {
      const parsed = parseInt(formData.client_id, 10);
      if (!Number.isNaN(parsed)) ids.add(parsed);
    }
    formData.client_ids.forEach((value) => {
      const parsed = parseInt(value, 10);
      if (!Number.isNaN(parsed)) ids.add(parsed);
    });
    return Array.from(ids);
  }, [formData.client_id, formData.client_ids]);

  const selectedClients = useMemo(() => {
    return selectedClientIds.map((id) => clientCache[id]).filter(Boolean);
  }, [selectedClientIds, clientCache]);

  const allClients = useMemo(() => {
    // Facturation directe : uniquement les clients éligibles (trajets non facturés) retournés par l'API
    if (billingType === 'direct') {
      return [...clients];
    }
    // Tierce-partie / partenaire : sélectionnés d'abord, puis éligibles
    const seen = new Set();
    const ordered = [];
    selectedClients.forEach((client) => {
      if (client && !seen.has(client.id)) {
        seen.add(client.id);
        ordered.push(client);
      }
    });
    clients.forEach((client) => {
      if (client && !seen.has(client.id)) {
        seen.add(client.id);
        ordered.push(client);
      }
    });
    return ordered;
  }, [billingType, clients, selectedClients]);

  useEffect(() => {
    if (
      billingType !== 'direct' ||
      clientsLoading ||
      !formData.client_id
    ) return;
    const id = parseInt(formData.client_id, 10);
    if (Number.isNaN(id)) return;
    const inList = clients.some((c) => c.id === id);
    if (!inList) {
      setFormData((prev) => ({ ...prev, client_id: '' }));
    }
  }, [billingType, clientsLoading, formData.client_id, clients]);

  useEffect(() => {
    const hasPendingPreselection = Object.values(preselectedReservations).some(
      (ids) => Array.isArray(ids) && ids.length > 0
    );
    if (
      hasPendingPreselection &&
      (!selectedReservations || Object.keys(selectedReservations || {}).length === 0)
    ) {
      return;
    }

    const activeIds = new Set();
    Object.values(selectedReservations).forEach((list) => {
      (list || []).forEach((reservation) => {
        if (reservation?.id != null) {
          activeIds.add(String(reservation.id));
        }
      });
    });

    setOverrides((prev) => {
      let changed = false;
      const next = { ...prev };
      Object.keys(prev).forEach((key) => {
        if (!activeIds.has(key)) {
          delete next[key];
          changed = true;
        }
      });
      return changed ? next : prev;
    });
  }, [selectedReservations, preselectedReservations]);

  const directClient = useMemo(() => {
    if (!formData.client_id) return null;
    const target = parseInt(formData.client_id, 10);
    if (Number.isNaN(target)) return null;
    return allClients.find((client) => client.id === target) || null;
  }, [allClients, formData.client_id]);

  const selectedPartner = useMemo(() => {
    if (!formData.partnership_id) return null;
    const id = parseInt(formData.partnership_id, 10);
    if (Number.isNaN(id)) return null;
    return partners.find((p) => p.partnership_id === id) || null;
  }, [formData.partnership_id, partners]);

  const partnerTotalComputed = useMemo(() => {
    if (partnerSelectedTransfers.length === 0) return null;
    return partnerSelectedTransfers.reduce((sum, t) => {
      const ov = partnerOverrides[String(t.id)] || partnerOverrides[t.id] || {};
      const amount = ov.amount ?? t.partner_cost ?? 0;
      return sum + Number(amount);
    }, 0);
  }, [partnerSelectedTransfers, partnerOverrides]);

  // ✅ Vérifier si une réservation est minimale (contient uniquement l'ID)
  const isMinimalReservation = useCallback((reservation) => {
    return reservation && typeof reservation.id !== 'undefined' && 
           (reservation.amount === undefined || reservation.amount === null) &&
           !reservation.pickup_location && !reservation.dropoff_location;
  }, []);

  const computeTotals = useCallback(
    (reservationsList = []) => {
      // ✅ Filtrer les objets minimaux : ne pas utiliser amount=0 pour eux
      const validReservations = reservationsList.filter((r) => !isMinimalReservation(r));
      
      if (validReservations.length === 0 && reservationsList.length > 0) {
        // Toutes les réservations sont minimales : retourner null pour indiquer qu'il faut hydrater
        return null;
      }
      
      return validReservations.reduce(
        (acc, reservation) => {
          const override = overrides[String(reservation?.id)] || {};
          const baseAmount = Number(
            override.amount ?? reservation?.amount ?? reservation?.estimated_amount ?? 0
          );
          const amount = Number.isNaN(baseAmount) ? 0 : baseAmount;
          const vatRate = vatConfig.applicable
            ? Number(
                reservation?.vat_rate ?? reservation?.default_vat_rate ?? vatConfig.defaultRate ?? 0
              )
            : 0;
          const sanitizedRate = Number.isNaN(vatRate) ? 0 : vatRate;
          const vatValue = vatConfig.applicable
            ? Number(((amount * sanitizedRate) / 100).toFixed(2))
            : 0;
          const total = Number((amount + vatValue).toFixed(2));

          acc.base += amount;
          acc.vat += vatValue;
          acc.total += total;
          return acc;
        },
        { base: 0, vat: 0, total: 0 }
      );
    },
    [overrides, vatConfig, isMinimalReservation]
  );

  const activeClientId = formData.client_id ? parseInt(formData.client_id, 10) : null;

  /** Résumé par client (count + total HT) depuis eligible. Utilisé pour l’aperçu sans charger le détail. */
  const directSummary = useMemo(() => {
    if (clientsLoading || !formData.client_id || !directClient) return null;
    const count = directClient.unbilled_count ?? 0;
    const totalAmount = Number(directClient.unbilled_total_amount);
    const total = Number.isFinite(totalAmount) ? totalAmount : 0;
    return { count, totalAmount: total };
  }, [clientsLoading, formData.client_id, directClient]);
  
  /** TTC du résumé (HT + TVA selon vatConfig) pour affichage aperçu. */
  const directSummaryTTC = useMemo(() => {
    if (!directSummary) return null;
    const base = directSummary.totalAmount;
    const vat = vatConfig.applicable
      ? Number(((base * (vatConfig.defaultRate || 0)) / 100).toFixed(2))
      : 0;
    return Number((base + vat).toFixed(2));
  }, [directSummary, vatConfig.applicable, vatConfig.defaultRate]);
  
  const directSelection = useMemo(() => {
    if (!activeClientId) return [];
    return selectedReservations[activeClientId] || [];
  }, [activeClientId, selectedReservations]);
  
  // ✅ Calculer les totaux : utiliser directSummaryTTC si sélection complète sans overrides de montant
  const _directTotals = useMemo(() => {
    const hasOnlyIds = directSelection.length > 0 && directSelection.every(
      (r) => isMinimalReservation(r)
    );

    if (
      directSummary &&
      directSelection.length === directSummary.count &&
      hasOnlyIds
    ) {
      // Vérifier si des overrides de montant existent pour cette sélection (ex: 45 → 35)
      const ids = directSelection.map((r) => r?.id ?? r).filter(Boolean);
      const hasAmountOverrides = ids.some(
        (id) => overrides[String(id)]?.amount !== undefined && overrides[String(id)]?.amount !== null
      );
      if (hasAmountOverrides) {
        // Réappliquer les overrides pour que le footer affiche le bon total TTC
        const base = ids.reduce((sum, id) => {
          const o = overrides[String(id)] ?? overrides[id];
          const amount = o?.amount !== undefined && o?.amount !== null
            ? Number(o.amount)
            : directSummary.totalAmount / directSummary.count;
          return sum + (Number.isFinite(amount) ? amount : 0);
        }, 0);
        const vat = vatConfig.applicable
          ? Number(((base * (vatConfig.defaultRate || 0)) / 100).toFixed(2))
          : 0;
        const total = Number((base + vat).toFixed(2));
        return { base, vat, total };
      }
      // Aucun override : utiliser directSummaryTTC
      return {
        base: directSummary.totalAmount,
        vat: vatConfig.applicable
          ? Number(((directSummary.totalAmount * (vatConfig.defaultRate || 0)) / 100).toFixed(2))
          : 0,
        total: directSummaryTTC || 0,
      };
    }

    const computed = computeTotals(directSelection);
    if (computed === null) {
      return null;
    }
    return computed;
  }, [computeTotals, directSelection, directSummary, directSummaryTTC, vatConfig, isMinimalReservation, overrides]);
  
  // ✅ Détecter sélection partielle: selectedCount < directSummary.count
  const isPartialSelection = useMemo(() => {
    if (!directSummary || !directSelection.length) return false;
    return directSelection.length < directSummary.count;
  }, [directSummary, directSelection.length]);

  // ✅ Détecter si la sélection contient des objets minimaux non hydratés
  const hasUnhydratedMinimals = useMemo(() => {
    if (!directSelection.length) return false;
    return directSelection.some((r) => isMinimalReservation(r));
  }, [directSelection, isMinimalReservation]);

  const consolidatedSelection = useMemo(
    () => Object.values(selectedReservations).reduce((acc, list) => acc.concat(list || []), []),
    [selectedReservations]
  );
  const consolidatedTotals = useMemo(
    () => computeTotals(consolidatedSelection),
    [computeTotals, consolidatedSelection]
  );
  
  // ✅ Détecter si sélection manuelle de trajets (pour affichage conditionnel)
  const hasManualSelection = useMemo(
    () => Object.keys(selectedReservations || {}).some(
      (clientId) => selectedReservations[clientId]?.length > 0
    ),
    [selectedReservations]
  );

  // Calculer les totaux éligibles (tous les transports disponibles, pas seulement sélectionnés)
  const _eligibleReservationsCount = useMemo(() => {
    return formData.client_ids.reduce((total, clientId) => {
      // On ne peut pas calculer exactement sans charger toutes les réservations
      // On utilise le unbilled_count du client si disponible
      const client = clients.find((c) => c.id === clientId);
      return total + (client?.unbilled_count || 0);
    }, 0);
  }, [formData.client_ids, clients]);


  const formatCurrency = useCallback((value) => `${Number(value || 0).toFixed(2)} CHF`, []);

  const buildOverridesPayload = useCallback(
    (reservationsList = []) => {
      const payload = {};
      reservationsList.forEach((reservation) => {
        if (!reservation || reservation.id == null) return;
        const override = overrides[String(reservation.id)];
        if (!override) return;
        const clean = {};
        if (override.amount !== undefined) {
          const amount = Number(override.amount);
          if (!Number.isNaN(amount)) clean.amount = amount;
        }
        if (override.note) {
          clean.note = override.note;
        }
        if (Object.keys(clean).length > 0) {
          payload[reservation.id] = clean;
        }
      });
      return payload;
    },
    [overrides]
  );

  const formatClientLabel = useCallback((client) => {
    if (!client) return 'Client';
    // Pour les clients institution facturés au patient : afficher le nom du patient
    const name =
      (client.display_name && client.display_name.trim()) ||
      (client.full_name && client.full_name.trim()) ||
      `${client.first_name || ''} ${client.last_name || ''}`.trim() ||
      client.username ||
      `Client #${client.id}`;
    const count = client.unbilled_count ?? 0;
    const suffix = count > 1 ? 's' : '';
    return `${name} • ${count} transport${suffix}`;
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();

    // Validation en fonction du type de facturation
    if (billingType === 'direct' && !formData.client_id) {
      setError('Veuillez sélectionner un client');
      return;
    }

    if (billingType === 'third_party') {
      if (formData.client_ids.length === 0) {
        setError('Veuillez sélectionner au moins un patient');
        return;
      }
      if (!formData.bill_to_client_id) {
        setError('Veuillez sélectionner une institution payeuse');
        return;
      }
    }

    if (billingType === 'partner') {
      if (!formData.partnership_id) {
        setError('Veuillez sélectionner un partenaire');
        return;
      }
    }

    try {
      setLoading(true);
      setError(null);
      setSuccessMessage(null);

      let result;

      if (billingType === 'direct') {
        // Facturation directe — zéro friction : on peut générer sans ouvrir "Transports à facturer"
        const clientId = parseInt(formData.client_id);
        const reservs = Array.isArray(selectedReservations?.[clientId])
          ? selectedReservations[clientId]
          : [];
        const reservationIds = reservs.length > 0 ? reservs.map((r) => r?.id || r) : undefined;
        // reservation_ids undefined → backend facture tous les unbilled du client pour la période
        const overridePayload = buildOverridesPayload(reservs);

        const payload = {
          client_id: clientId,
          period_year: formData.period_year,
          period_month: formData.period_month,
          reservation_ids: reservationIds,
        };

        if (Object.keys(overridePayload).length > 0) {
          payload.overrides = overridePayload;
        }

        result = await generateInvoice(companyId, payload);

        // Ouvrir le PDF dans un nouvel onglet
        if (result?.pdf_url) {
          window.open(result.pdf_url, '_blank');
        } else if (result?.data?.pdf_url) {
          // Si la structure est {data: {pdf_url: ...}}
          window.open(result.data.pdf_url, '_blank');
        }

        // Vérifier la structure de la réponse avant de notifier le parent
        if (result?.data) {
          onInvoiceGenerated(result.data);
        } else if (result) {
        onInvoiceGenerated(result);
        }
      } else if (billingType === 'third_party') {
        // Facturation tierce

        // ✅ S2: Facture clinique mensuelle unique
        if (isClinicMonthly && selectedInstitution?.clinic_company_id) {
          const payload = {
            mode: 'clinic_monthly',
            clinic_company_id: selectedInstitution.clinic_company_id,
            period_year: formData.period_year,
            period_month: formData.period_month,
          };

          // Ajouter include_client_ids si sélectionné
          if (includeClientIds.length > 0) {
            payload.include_client_ids = includeClientIds;
          }

          // ✅ S2: Overrides montant/note par transport (ajustements avant génération)
          const s2Overrides = {};
          Object.values(patientBookings || {}).flat().forEach((b) => {
            if (!b?.id) return;
            const o = s2BookingOverrides[String(b.id)] || s2BookingOverrides[b.id];
            if (!o) return;
            const clean = {};
            if (o.amount !== undefined && o.amount !== null && Number.isFinite(Number(o.amount))) clean.amount = Number(o.amount);
            if (o.note != null && String(o.note).trim()) clean.note = String(o.note).trim();
            if (Object.keys(clean).length > 0) s2Overrides[b.id] = clean;
          });
          if (Object.keys(s2Overrides).length > 0) {
            payload.overrides = s2Overrides;
          }

          try {
            result = await generateInvoice(companyId, payload);

            // ✅ Gérer l'erreur "déjà générée" (409) avec option d'ouvrir la facture existante
            if (result?.error || result?.status === 409 || (result?.response?.status === 409)) {
              // Backend retourne { error, existing_invoice_id, existing_invoice_number } dans response.data
              const errorData = result?.data || result?.response?.data || (typeof result?.error === 'object' ? result.error : null) || {};
              const errorDataObj = typeof errorData === 'object' ? errorData : {};
              const existingInvoiceId = errorDataObj.existing_invoice_id;
              const existingInvoiceNumber = errorDataObj.existing_invoice_number || 'N/A';
              const errorMsg = typeof errorDataObj.error === 'string' ? errorDataObj.error : 'Facture déjà générée';

              if (existingInvoiceId) {
                // ✅ UX: Proposer d'ouvrir la facture existante
                const openExisting = window.confirm(
                  `${errorMsg}\n\n` +
                  `Souhaitez-vous ouvrir la facture existante ?`
                );
                if (openExisting) {
                  const openPdfAndNavigate = async () => {
                    try {
                      const res = await invoiceService.getInvoice(companyId, existingInvoiceId);
                      const inv = res?.data ?? res;
                      if (inv?.pdf_url) {
                        window.open(inv.pdf_url, '_blank');
                      }
                    } catch (e) {
                      // Ignorer si la récupération échoue
                    }
                    if (company?.public_id) {
                      const params = new URLSearchParams({
                        search: existingInvoiceNumber,
                        focusSearch: '1',
                        invoice_id: String(existingInvoiceId),
                      });
                      navigate(
                        `/dashboard/company/${company.public_id}/invoices/clients?${params.toString()}`
                      );
                      onClose();
                    } else {
                      setError(
                        `Facture déjà générée (${existingInvoiceNumber}). ` +
                        `ID: ${existingInvoiceId}. Veuillez utiliser le registre des factures pour l'ouvrir.`
                      );
                    }
                  };
                  openPdfAndNavigate();
                } else {
                  setError(errorMsg);
                }
              } else {
                setError(errorMsg);
              }
              setLoading(false);
              return;
            }

            if (result?.pdf_url) {
              window.open(result.pdf_url, '_blank');
            } else if (result?.data?.pdf_url) {
              window.open(result.data.pdf_url, '_blank');
            }

            if (result?.data) {
              onInvoiceGenerated(result.data);
            } else if (result) {
              onInvoiceGenerated(result);
            }

            setSuccessMessage('1 facture clinique mensuelle générée avec succès');
            setLoading(false);
            return;
          } catch (err) {
            const status = err?.response?.status;
            const url = err?.config?.url;
            const method = err?.config?.method;
            const requestData = err?.config?.data;
            const responseData = err?.response?.data;

            console.error('[NewInvoiceModal] generate invoice (clinic_monthly) failed', {
              status,
              url,
              method,
              requestData,
              responseData,
            });

            if (typeof responseData === 'string') {
              console.error('[NewInvoiceModal] responseData (string)', responseData.slice(0, 2000));
            }

            // Gérer les erreurs HTTP (409, 400, 422, etc.)
            if (err?.response?.status === 409) {
              // response.data = { error, existing_invoice_id, existing_invoice_number }
              const errorData = err?.response?.data || {};
              const existingInvoiceId = errorData.existing_invoice_id;
              const existingInvoiceNumber = errorData.existing_invoice_number || 'N/A';
              const errorMsg = typeof errorData.error === 'string' ? errorData.error : 'Facture déjà générée';

              if (existingInvoiceId) {
                const openExisting = window.confirm(
                  `${errorMsg}\n\n` +
                  `Souhaitez-vous ouvrir la facture existante ?`
                );
                if (openExisting) {
                  // ✅ Ouvrir le PDF directement si disponible, puis naviguer vers le registre
                  const openPdfAndNavigate = async () => {
                    try {
                      const invoiceDetail = await invoiceService.getInvoice(companyId, existingInvoiceId);
                      if (invoiceDetail?.pdf_url) {
                        window.open(invoiceDetail.pdf_url, '_blank');
                      }
                    } catch (e) {
                      // Ignorer si la récupération échoue, on navigue quand même
                    }
                    if (company?.public_id) {
                      const params = new URLSearchParams({
                        search: existingInvoiceNumber,
                        focusSearch: '1',
                        invoice_id: String(existingInvoiceId),
                      });
                      navigate(
                        `/dashboard/company/${company.public_id}/invoices/clients?${params.toString()}`
                      );
                      onClose();
                    } else {
                      setError(
                        `Facture déjà générée (${existingInvoiceNumber}). ` +
                        `ID: ${existingInvoiceId}. Veuillez utiliser le registre des factures pour l'ouvrir.`
                      );
                    }
                  };
                  openPdfAndNavigate();
                } else {
                  setError(errorMsg);
                }
              } else {
                setError(errorMsg);
              }
            } else if (err?.response?.status === 422) {
              // ✅ Erreur 422: Aucun transport éligible
              const errorData = err?.response?.data?.error || {};
              setError(errorData.error || 'Aucun transport clinique éligible pour cette période/sélection');
            } else {
              setError(formatApiError(err) || 'Erreur lors de la génération de la facture');
            }
            setLoading(false);
            // ✅ Ne pas réinitialiser les données en cas d'erreur pour permettre à l'utilisateur de corriger
            return;
          }
        }

        // ✅ Préparer le mapping des réservations par client
        const clientReservations = {};
        formData.client_ids.forEach((clientId) => {
          const reservs = selectedReservations?.[clientId];
          if (reservs && Array.isArray(reservs) && reservs.length > 0) {
            clientReservations[clientId] = reservs.map((r) => r?.id || r);
          }
        });

        const overridePayload = buildOverridesPayload(consolidatedSelection);

        if (isConsolidated) {
          // ✅ Mode batch: utiliser le vrai endpoint consolidé (GenerateConsolidatedInvoiceUseCase)
          // Ce use-case génère une facture PAR patient (pas une seule facture pour tous)
          // Toutes les factures sont adressées à bill_to_client_id (institution payeuse)
          
          // Vérifier si sélection manuelle de trajets (avertir l'utilisateur)
          const hasManualSelection = Object.keys(clientReservations).length > 0;
          if (hasManualSelection) {
            const confirmConsolidate = window.confirm(
              '⚠️ Mode génération groupée (batch)\n\n' +
              'Le mode groupé facture automatiquement TOUS les trajets éligibles du mois pour chaque patient.\n\n' +
              '⚠️ Votre sélection manuelle de trajets sera ignorée.\n\n' +
              'Pour conserver votre sélection manuelle, utilisez le mode "factures séparées" (décocher "Génération groupée").\n\n' +
              'Continuer avec la génération automatique ?'
            );
            if (!confirmConsolidate) {
              setLoading(false);
              return;
            }
          }

          // ✅ Utiliser le vrai endpoint consolidé avec client_ids[] + bill_to_client_id
          // Le backend génère une facture par client, toutes adressées à bill_to_client_id
          const payload = {
            client_ids: formData.client_ids.map((id) => parseInt(id)),
            bill_to_client_id: parseInt(formData.bill_to_client_id),
            period_year: formData.period_year,
            period_month: formData.period_month,
            // Note: client_reservations est supporté par le use-case si besoin
            // Mais en mode "groupé", on laisse le backend gérer automatiquement par période
            // pour éviter les incohérences (une facture par client avec ses propres réservations)
          };

          // Si sélection manuelle et utilisateur a confirmé, on peut essayer d'envoyer client_reservations
          // Mais attention: le use-case génère une facture PAR client, donc chaque client_reservations[client_id]
          // doit contenir uniquement les réservations de ce client
          if (hasManualSelection && Object.keys(clientReservations).length > 0) {
            // Vérifier que chaque client_reservations[client_id] contient uniquement les réservations de ce client
            const validClientReservations = {};
            formData.client_ids.forEach((clientId) => {
              const reservs = clientReservations[clientId] || [];
              if (reservs.length > 0) {
                validClientReservations[clientId] = reservs;
              }
            });
            if (Object.keys(validClientReservations).length > 0) {
              payload.client_reservations = validClientReservations;
            }
          }

          if (Object.keys(overridePayload).length > 0) {
            payload.overrides = overridePayload;
          }

          result = await invoiceService.generateConsolidatedInvoice(companyId, payload);

          if (result.invoices && result.invoices.length > 0) {
            // ✅ Mode consolidé: le use-case génère une facture par patient
            const invoiceCount = result.invoices.length;
            const patientNames = formData.client_ids
              .map((id) => {
                const client = clients.find((c) => c.id === parseInt(id));
                return client ? (client.display_name || `${client.first_name} ${client.last_name}`) : `Client ${id}`;
              })
              .join(', ');

            if (invoiceCount === formData.client_ids.length) {
              // Toutes les factures ont été générées avec succès
              setSuccessMessage(
                `${invoiceCount} facture(s) générée(s) en batch pour ${formData.client_ids.length} patient(s): ${patientNames}${
                  result.error_count > 0 ? `, ${result.error_count} erreur(s)` : ''
                }`
              );
            } else {
              // Génération partielle
              setSuccessMessage(
                `${invoiceCount} facture(s) générée(s) sur ${formData.client_ids.length} patient(s) (batch partiel): ${patientNames}${
                  result.error_count > 0 ? `, ${result.error_count} erreur(s)` : ''
                }`
              );
            }

            // Ouvrir les PDFs dans de nouveaux onglets
            result.invoices.forEach((inv) => {
              if (inv.pdf_url) {
                window.open(inv.pdf_url, '_blank');
              }
            });

            // Notifier le parent pour chaque facture
            result.invoices.forEach((inv) => onInvoiceGenerated(inv));
          } else if (result.success_count === 0) {
            // Aucune facture générée (toutes en erreur)
            setError('Aucune facture n\'a pu être générée. Vérifiez les erreurs ci-dessous.');
          }

          if (result?.errors && result.errors.length > 0) {
            const errorMessages = result.errors
              .map((e) => {
                const clientName = clients.find((c) => c.id === e.client_id);
                const name = clientName 
                  ? (clientName.display_name || `${clientName.first_name} ${clientName.last_name}`) 
                  : `Client ${e.client_id}`;
                return `${name}: ${formatApiError(e?.error ?? e)}`;
              })
              .join('\n');
            setError(`Erreurs:\n${errorMessages}`);
          }
        } else {
          // ✅ Mode séparé: 1 facture par patient (comportement actuel)
          // Générer une facture pour chaque client sélectionné
          const invoicePromises = formData.client_ids.map(async (clientId) => {
            const clientReservs = clientReservations[clientId] || [];
            const clientOverrides = {};
            // Extraire les overrides pour ce client uniquement
            Object.keys(overridePayload).forEach((reservationId) => {
              const reservation = consolidatedSelection.find((r) => r.id === parseInt(reservationId));
              if (reservation && reservation.client_id === parseInt(clientId)) {
                clientOverrides[reservationId] = overridePayload[reservationId];
              }
            });

            const payload = {
              client_id: parseInt(clientId),
              bill_to_client_id: parseInt(formData.bill_to_client_id),
              period_year: formData.period_year,
              period_month: formData.period_month,
              reservation_ids: clientReservs.length > 0 ? clientReservs : undefined,
            };

            if (Object.keys(clientOverrides).length > 0) {
              payload.overrides = clientOverrides;
            }

            return generateInvoice(companyId, payload);
          });

          const results = await Promise.all(invoicePromises);
          const successfulInvoices = results.filter((r) => r && !r.error);
          const failedInvoices = results.filter((r) => r && r.error);

          if (successfulInvoices.length > 0) {
            setSuccessMessage(
              `${successfulInvoices.length} facture(s) générée(s) avec succès${
                failedInvoices.length > 0 ? `, ${failedInvoices.length} erreur(s)` : ''
              }`
            );

            // Ouvrir les PDFs dans de nouveaux onglets
            successfulInvoices.forEach((inv) => {
              if (inv?.pdf_url) {
                window.open(inv.pdf_url, '_blank');
              } else if (inv?.data?.pdf_url) {
                window.open(inv.data.pdf_url, '_blank');
              }
            });

            // Notifier le parent pour chaque facture
            successfulInvoices.forEach((inv) => {
              onInvoiceGenerated(inv?.data || inv);
            });
          }

          if (failedInvoices.length > 0) {
            const errorMessages = failedInvoices
              .map((inv) => formatApiError(inv.error))
              .join('\n');
            setError(`Erreurs:\n${errorMessages}`);
          }
        }
      } else if (billingType === 'partner') {
        // Facturation partenaire — avec sélection et overrides si disponibles
        const payload = {
          partnership_id: parseInt(formData.partnership_id),
          period_year: formData.period_year,
          period_month: formData.period_month,
        };
        if (partnerSelectedTransfers.length > 0) {
          payload.transfer_ids = partnerSelectedTransfers.map((t) => t.id);
        }
        if (Object.keys(partnerOverrides).length > 0) {
          payload.overrides = partnerOverrides;
        }

        result = await invoiceService.generatePartnerInvoice(companyId, payload);

        // Ouvrir le PDF dans un nouvel onglet
        if (result?.data?.pdf_url) {
          window.open(result.data.pdf_url, '_blank');
        }

        // Vérifier que result.data existe avant de notifier le parent
        if (result?.data) {
        onInvoiceGenerated(result.data);
        } else if (result) {
          // Si result.data n'existe pas mais result existe, utiliser result directement
          onInvoiceGenerated(result);
        }
      }

      // Fermer le modal si tout s'est bien passé et pas d'erreurs
      if (!result || !result.errors || result.errors.length === 0) {
        // Reset de l'état avant fermeture pour éviter les états fantômes
        setSelectedReservations({});
        setOverrides({});
        setIsConsolidated(false); // ✅ Reset génération groupée (batch)
        setIsClinicMonthly(false); // ✅ Reset S2
        setIncludeClientIds([]); // ✅ Reset exceptions
        // ✅ Reset accordéons S2
        setShowS2Summary(true);
        setShowS2Exclusions(false);
        setShowS2Patients(false);
        setShowS2Advanced(false);
        setExpandedPatientId(null);
        setFormData((prev) => ({
          ...prev,
          client_ids: [],
          client_id: '',
        }));
        setTimeout(() => {
          onClose();
        }, 2000);
      }
    } catch (err) {
      const status = err?.response?.status;
      const url = err?.config?.url;
      const method = err?.config?.method;
      const requestData = err?.config?.data;
      const responseData = err?.response?.data;

      console.error('[NewInvoiceModal] generate invoice failed', {
        status,
        url,
        method,
        requestData,
        responseData,
      });

      if (typeof responseData === 'string') {
        console.error('[NewInvoiceModal] responseData (string)', responseData.slice(0, 2000));
      }

      const errorMessage = formatApiError(err) || 'Erreur lors de la génération de la facture';
      setError(errorMessage);
      
      // ✅ Ne pas réinitialiser les données en cas d'erreur pour permettre à l'utilisateur de corriger
      // Les clients et réservations restent visibles
    } finally {
      setLoading(false);
    }
  };

  const handleClose = () => {
    setError(null);
    setSuccessMessage(null);
    setBillingType('direct');
    setSelectedReservations({});
    setShowDirectTransports(false);
    setShowPartnerSummary(true);
    setIsConsolidated(false); // ✅ Reset génération groupée (batch)
    setIsClinicMonthly(false); // ✅ Reset S2
    setIncludeClientIds([]); // ✅ Reset exceptions
    // ✅ Reset accordéons S2
    setShowS2Summary(true);
    setShowS2Exclusions(false);
    setShowS2Patients(false);
    setShowS2Advanced(false);
    setExpandedPatientId(null);
    setFormData({
      client_id: '',
      client_ids: [],
      bill_to_client_id: '',
      period_year: new Date().getFullYear(),
      period_month: new Date().getMonth() + 1,
    });
    onClose();
  };

  // ✅ Calculer le nombre de patients inclus en mode S2 (avant le return conditionnel)
  const s2PatientsCount = useMemo(() => {
    if (!isClinicMonthly) return 0;
    if (includeClientIds.length > 0) {
      return includeClientIds.length;
    }
    // Si aucun filtre, tous les clients éligibles sont inclus
    return clients.length;
  }, [isClinicMonthly, includeClientIds.length, clients.length]);

  // ✅ Formater le nom du mois (avant le return conditionnel)
  const monthName = useMemo(() => {
    const monthNames = ['Janv', 'Févr', 'Mars', 'Avr', 'Mai', 'Juin', 'Juil', 'Août', 'Sept', 'Oct', 'Nov', 'Déc'];
    return monthNames[formData.period_month - 1] || '';
  }, [formData.period_month]);

  if (!open) return null;

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

  const years = Array.from({ length: 3 }, (_, i) => new Date().getFullYear() - i);

  return (
    <div className="modal-overlay modal-invoice">
      <div className={`modal-content modal-xl ${styles.modalInvoice}`} data-tour-id="invoice-new-modal">
        <div className="modal-header">
          <h2 className="modal-title">Nouvelle facture</h2>
          <button className="modal-close" onClick={handleClose}>
            ✕
          </button>
        </div>

        <form
          onSubmit={handleSubmit}
          className={`${styles.form} ${styles.modalInvoiceForm} ${
            (billingType === 'third_party' && formData.bill_to_client_id) ||
            (billingType === 'direct' && formData.client_id) ||
            (billingType === 'partner' && formData.partnership_id)
              ? styles.formWithStickyFooter
              : ''
          }`}
        >
          <div className={styles.modalBody}>
          {error && <div className="alert alert-error mb-md">{error}</div>}

          {successMessage && <div className={styles.success}>{successMessage}</div>}

          <div
            className={`${styles.formGroup} ${styles.stickyBillingType}`}
            data-tour-id="invoice-modal-billing-type"
          >
            <label className={styles.label}>Type de facturation</label>
            <div className={styles.radioGroup}>
              <label className={styles.radioLabel}>
                <input
                  type="radio"
                  value="direct"
                  checked={billingType === 'direct'}
                  onChange={(e) => setBillingType(e.target.value)}
                  disabled={loading}
                />
                Facturation directe au client
              </label>
              <label className={styles.radioLabel}>
                <input
                  type="radio"
                  value="third_party"
                  checked={billingType === 'third_party'}
                  onChange={(e) => setBillingType(e.target.value)}
                  disabled={loading}
                />
                Facturation tierce (clinique)
              </label>
              <label className={styles.radioLabel}>
                <input
                  type="radio"
                  value="partner"
                  checked={billingType === 'partner'}
                  onChange={(e) => setBillingType(e.target.value)}
                  disabled={loading}
                />
                Facturation partenaire
              </label>
            </div>
          </div>

          {/* Facturation directe — même modèle que facturation clinique */}
          {billingType === 'direct' && (
            <>
              {/* Header direct : client + période (style s2Header) */}
              <div className={styles.s2Header}>
                <div className={styles.s2HeaderRow} data-tour-id="invoice-modal-client">
                  <span className={styles.s2HeaderLabel}>👤 Client</span>
                  <div className={styles.s2HeaderClinique} style={{ flex: 1, maxWidth: '100%' }}>
                    <input
                      ref={clientSearchInputRef}
                      id="clientSearch"
                      type="search"
                      className={styles.searchInput}
                      placeholder="Recherche nom, prénom, email"
                      autoComplete="off"
                      defaultValue={clientSearch}
                      onChange={handleSearchChange}
                      onFocus={() => { wasInputFocusedRef.current = true; }}
                      onBlur={() => { wasInputFocusedRef.current = false; }}
                      style={{ marginBottom: 8 }}
                    />
                    <select
                      id="client_id"
                      name="client_id"
                      value={formData.client_id}
                      onChange={handleInputChange}
                      className={styles.select}
                      required
                      disabled={loading || clientsLoading}
                    >
                      <option value="">Sélectionner un client</option>
                      {allClients.map((client) => (
                        <option key={client.id} value={client.id}>
                          {`${formatClientLabel(client)}${
                            directClient && client.id === directClient.id && clientSearch.trim()
                              ? ' (sélectionné)'
                              : ''
                          }`}
                        </option>
                      ))}
                    </select>
                    {clientsLoading && <small className={styles.hint}>Chargement…</small>}
                    {!clientsLoading && allClients.length === 0 && (
                      <small className={styles.hint}>Aucun client avec transports à facturer.</small>
                    )}
                  </div>
                </div>
                <div className={styles.s2HeaderRow} data-tour-id="invoice-modal-period">
                  <div className={styles.s2HeaderMeta}>
                    <span className={styles.s2HeaderLabel}>Période à facturer</span>
                    {formData.client_id && directSummary && directSummary.count > 0 && (
                      <>
                        <span className={styles.s2HeaderMetaSep}>•</span>
                        <span>{directSummary.count} transport{directSummary.count > 1 ? 's' : ''}</span>
                      </>
                    )}
                  </div>
                  <div className={styles.s2HeaderPeriod}>
                    <select
                      id="direct_period_month"
                      name="period_month"
                      value={formData.period_month}
                      onChange={handleInputChange}
                      className={styles.select}
                      disabled={loading}
                    >
                      {months.map((m) => (
                        <option key={m.value} value={m.value}>{m.label}</option>
                      ))}
                    </select>
                    <select
                      id="direct_period_year"
                      name="period_year"
                      value={formData.period_year}
                      onChange={handleInputChange}
                      className={styles.select}
                      disabled={loading}
                    >
                      {years.map((y) => (
                        <option key={y} value={y}>{y}</option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              {clientsError && <div className="alert alert-error mb-sm">{clientsError}</div>}

              {/* Transports à facturer — résumé = footer sticky + summaryLine sous la liste */}
              {formData.client_id && (
                <div className={styles.formGroup}>
                  <button
                    type="button"
                    onClick={() => setShowDirectTransports(!showDirectTransports)}
                    className={`${styles.accordion} ${styles.accordionInfo} ${showDirectTransports ? styles.isOpen : ''}`}
                  >
                    <span>Transports à facturer ({clientsLoading ? '—' : (directSummary ? directSummary.count : '—')})</span>
                    <span>{showDirectTransports ? '▼' : '▶'}</span>
                  </button>
                  {showDirectTransports && (
                    <div className={`${styles.accordionContent} ${styles.accordionContentInfo}`}>
                      <ReservationSelector
                        companyId={companyId}
                        clientId={parseInt(formData.client_id)}
                        clientName={directClient?.full_name || ''}
                        period={{ year: formData.period_year, month: formData.period_month }}
                        billToType="patient"
                        vatConfig={vatConfig}
                        overrides={overrides}
                        preselectedIds={preselectedReservations[parseInt(formData.client_id, 10)] || []}
                        onOverrideChange={handleOverrideChange}
                        onSelectionChange={(reservations) =>
                          handleReservationSelectionChange(parseInt(formData.client_id), reservations)
                        }
                      />
                    </div>
                  )}
                </div>
              )}
            </>
          )}

          {/* Facturation tierce */}
          {billingType === 'third_party' && (
            <>
                  {/* Header tierce : clinique + période (même modèle que direct/partenaire, plus d’étape intermédiaire) */}
                  <div className={styles.s2Header}>
                    <div className={styles.s2HeaderRow}>
                      <span className={styles.s2HeaderLabel}>🏥 Clinique</span>
                      <div className={styles.s2HeaderClinique}>
                        <select
                          id="s2_bill_to_client_id"
                          name="bill_to_client_id"
                          value={formData.bill_to_client_id}
                          onChange={handleInputChange}
                          className={styles.select}
                          disabled={loading}
                        >
                          <option value="">Sélectionner une clinique</option>
                          {clinicsForS2.map((inst) => (
                            <option key={inst.id} value={inst.id}>
                              {inst.institution_name || 'Clinique'}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>
                    <div className={styles.s2HeaderRow}>
                      <div className={styles.s2HeaderMeta}>
                        <span>👥 {s2PatientsCount} patient{s2PatientsCount > 1 ? 's' : ''}</span>
                        <span className={styles.s2HeaderMetaSep}>•</span>
                        <span className={styles.s2HeaderLabel}>Période à facturer</span>
                      </div>
                      <div className={styles.s2HeaderPeriod}>
                        <select
                          id="s2_period_month"
                          name="period_month"
                          value={formData.period_month}
                          onChange={handleInputChange}
                          className={styles.select}
                          disabled={loading}
                        >
                          {months.map((m) => (
                            <option key={m.value} value={m.value}>{m.label}</option>
                          ))}
                        </select>
                        <select
                          id="s2_period_year"
                          name="period_year"
                          value={formData.period_year}
                          onChange={handleInputChange}
                          className={styles.select}
                          disabled={loading}
                        >
                          {years.map((y) => (
                            <option key={y} value={y}>{y}</option>
                          ))}
                        </select>
                      </div>
                    </div>
                  </div>

                  {formData.bill_to_client_id && selectedInstitution?.clinic_company_id && (
                  <>
                  {/* ✅ SECTION 0: Résumé (ouvert par défaut) */}
                  <div className={styles.formGroup}>
                    <button
                      type="button"
                      onClick={() => setShowS2Summary(!showS2Summary)}
                      className={`${styles.accordion} ${styles.accordionInfo} ${showS2Summary ? styles.isOpen : ''}`}
                    >
                      <span>
                        ✓ Facture clinique mensuelle
                      </span>
                      <span>{showS2Summary ? '▼' : '▶'}</span>
                    </button>
                    {showS2Summary && (
                      <div className={`${styles.accordionContent} ${styles.accordionContentInfo}`}>
                        <div style={{ fontSize: '13px', color: '#475569', lineHeight: '1.8' }}>
                          <div><strong>1 facture unique</strong> pour tous les patients hospitalisés</div>
                          <div style={{ marginTop: '12px' }}>
                            <div>• <strong>Période :</strong> {monthName} {formData.period_year}</div>
                            <div>• <strong>Patients inclus :</strong> {s2PatientsCount}</div>
                            <div>• <strong>Transports inclus :</strong> {s2Totals.total_eligible}</div>
                            <div style={{ marginTop: '8px', fontSize: '15px', fontWeight: 700, color: '#0f172a' }}>
                              • <strong>Total inclus :</strong> {formatCurrencyCHF(s2Totals.total_amount_eligible)}
                            </div>
                            {s2Totals.total_excluded > 0 && (
                              <div style={{ marginTop: '8px', fontSize: '12px', color: '#92400e' }}>
                                • <strong>Exclusions :</strong> {s2Totals.total_excluded} transport{s2Totals.total_excluded > 1 ? 's' : ''} • {formatCurrencyCHF(s2Totals.total_amount_excluded)}
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* ✅ SECTION 1: Patients inclus (information principale - D'ABORD) */}
                  {s2PatientsCount > 0 && (
                    <div className={styles.formGroup}>
                      <button
                        type="button"
                        onClick={() => setShowS2Patients(!showS2Patients)}
                        className={`${styles.accordion} ${styles.accordionInfo} ${showS2Patients ? styles.isOpen : ''}`}
                      >
                        <span>
                          ▶ Patients inclus ({s2PatientsCount})
                        </span>
                        <span>{showS2Patients ? '▼' : '▶'}</span>
                      </button>
                      {showS2Patients && (
                        <div className={`${styles.accordionContent} ${styles.accordionContentInfo}`}>
                          {/* ✅ Helper text S2 */}
                          <div style={{ 
                            marginBottom: '12px', 
                            padding: '10px', 
                            background: '#eff6ff', 
                            border: '1px solid #bfdbfe',
                            borderRadius: '6px',
                            fontSize: '12px',
                            color: '#1e40af',
                            lineHeight: '1.5'
                          }}>
                            <strong>S2 inclut automatiquement tous les transports cliniques éligibles</strong> ; utilisez l'override pour les exceptions.
                          </div>
                          {clients.length === 0 ? (
                            <div style={{ color: '#6b7280', fontSize: '13px', textAlign: 'center', padding: '12px' }}>
                              Chargement des patients...
                            </div>
                          ) : (
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                              {clients
                                .filter((client) => {
                                  // ✅ Filtrer : n'afficher que les patients avec au moins 1 transport clinique
                                  const transportCount = client.unbilled_count || 0;
                                  return transportCount > 0;
                                })
                                .map((client) => {
                                  const clientName = client.display_name || `${client.first_name || ''} ${client.last_name || ''}`.trim() || client.username;
                                  const isExpanded = expandedPatientId === client.id;
                                  const bookings = patientBookings[client.id] || [];
                                  const isLoading = patientBookingsLoading[client.id] || false;
                                  // ✅ S2: n'afficher que les transports pas encore facturés dans la liste
                                  const bookingsToShow = bookings.filter((b) => !(b.invoiced === true || (b.invoice_line_id != null && b.invoice_line_id !== undefined)));
                                  // Compter les transports cliniques vs patients pour ce patient
                                  const patientBookingsCount = bookings.filter((b) => b.billed_to_type === 'patient').length;
                                  const totalBookingsCount = bookings.length;
                                  const invoicedFromBookings = bookings.filter((b) => b.invoiced === true).length;
                                  const totalDisplay = totalBookingsCount > 0 ? totalBookingsCount : (client.unbilled_count || 0) + (client.invoiced_count || 0);
                                  const invoicedDisplay = totalBookingsCount > 0 ? invoicedFromBookings : (client.invoiced_count || 0);
                                  
                                  return (
                                    <div key={client.id} style={{ 
                                      padding: '8px 12px', 
                                      background: '#fff', 
                                      border: '1px solid #e5e7eb', 
                                      borderRadius: '6px'
                                    }}>
                                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flex: 1 }}>
                                          <span style={{ fontSize: '13px', color: '#111827', fontWeight: 500 }}>
                                            {clientName}
                                          </span>
                                          <span style={{ fontSize: '12px', color: '#6b7280' }}>
                                            ({totalDisplay} transport{totalDisplay > 1 ? 's' : ''}
                                            {invoicedDisplay > 0 ? ` • ${invoicedDisplay} déjà facturé${invoicedDisplay > 1 ? 's' : ''}` : ''})
                                          </span>
                                          {totalBookingsCount > 0 && (
                                            <span style={{ 
                                              fontSize: '11px', 
                                              padding: '2px 6px',
                                              borderRadius: '4px',
                                              fontWeight: 500,
                                              ...(patientBookingsCount === 0 
                                                ? { background: '#d1fae5', color: '#065f46' }
                                                : { background: '#fef3c7', color: '#92400e' })
                                            }}>
                                              {patientBookingsCount === 0 
                                                ? '✓ Tout clinique'
                                                : `⚠ ${patientBookingsCount} facturé${patientBookingsCount > 1 ? 's' : ''} au patient`}
                                            </span>
                                          )}
                                        </div>
                                        <button
                                          type="button"
                                          onClick={() => {
                                            const newExpandedId = isExpanded ? null : client.id;
                                            setExpandedPatientId(newExpandedId);
                                            // Fermer la confirmation si on ferme le patient
                                            if (!newExpandedId && bookingOverrideConfirm?.clientId === client.id) {
                                              setBookingOverrideConfirm(null);
                                            }
                                            // Charger les bookings si on ouvre et qu'ils ne sont pas déjà chargés
                                            if (newExpandedId && bookings.length === 0 && !isLoading) {
                                              loadPatientBookings(client.id);
                                            }
                                          }}
                                          className="btn btn-link"
                                          style={{ 
                                            padding: '2px 6px', 
                                            fontSize: '11px', 
                                            color: '#0369a1',
                                            textDecoration: 'underline',
                                            whiteSpace: 'nowrap'
                                          }}
                                        >
                                          {isExpanded ? 'Fermer' : 'Voir trajets'}
                                        </button>
                                      </div>
                                      {isExpanded && (
                                        <div style={{ 
                                          marginTop: '12px', 
                                          padding: '12px', 
                                          background: '#f9fafb', 
                                          border: '1px solid #e5e7eb',
                                          borderRadius: '6px'
                                        }}>
                                          {isLoading ? (
                                            <div style={{ color: '#6b7280', fontSize: '12px', textAlign: 'center', padding: '12px' }}>
                                              Chargement des trajets...
                                            </div>
                                          ) : bookings.length === 0 ? (
                                            <div style={{ color: '#6b7280', fontSize: '12px', textAlign: 'center', padding: '12px' }}>
                                              Aucun trajet trouvé pour cette période.
                                            </div>
                                          ) : bookingsToShow.length === 0 ? (
                                            <div style={{ color: '#6b7280', fontSize: '12px', textAlign: 'center', padding: '12px' }}>
                                              Tous les transports de ce patient sont déjà facturés.
                                            </div>
                                          ) : (
                                            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                                              {bookingsToShow.map((booking) => {
                                                const isPatientBilling = booking.billed_to_type === 'patient';
                                                const isOverrideInProgress = bookingOverridesInProgress.has(booking.id);
                                                const isConfirming = bookingOverrideConfirm?.bookingId === booking.id;
                                                // ✅ Safety rules S2: Vérifier si le booking est modifiable
                                                // Normaliser billing_review_status en lowercase pour comparaison
                                                const normalizedReviewStatus = booking.billing_review_status 
                                                  ? String(booking.billing_review_status).toLowerCase() 
                                                  : null;
                                                const isLocked = normalizedReviewStatus === 'locked';
                                                const isInvoiced = booking.invoiced === true || (booking.invoice_line_id != null && booking.invoice_line_id !== undefined);
                                                const isNoClinic = isPatientBilling && !clinicCompanyId;
                                                const isNotModifiable = isLocked || isInvoiced || isNoClinic;
                                                
                                                // Tooltips spécifiques selon la raison de désactivation
                                                const tooltipMessage = isInvoiced 
                                                  ? 'Inclus dans une facture (ligne existante).'
                                                  : (isLocked 
                                                    ? 'Transport verrouillé (contrôle facturation).'
                                                    : (isNoClinic 
                                                      ? 'Sélectionnez une clinique avant de réintégrer ce transport.'
                                                      : null));
                                                const scheduledDate = booking.scheduled_time ? new Date(booking.scheduled_time) : null;
                                                const dateStr = scheduledDate 
                                                  ? scheduledDate.toLocaleDateString('fr-FR', { day: '2-digit', month: '2-digit', year: 'numeric' }).replace(/\//g, '.')
                                                  : '';
                                                const pickupAddress = booking.pickup_address || '';
                                                const dropoffAddress = booking.dropoff_address || '';
                                                const isCancelled = (booking.status || '').toUpperCase() === 'CANCELED';
                                                const isDelivery = (booking.mission_type || '').toLowerCase() === 'material_delivery';
                                                const deliveryDesc = (booking.delivery_description || '').trim();
                                                const cancellationLabel = booking.cancellation_display_label || null;
                                                let transportLabel = '';
                                                if (isDelivery) {
                                                  const deliveryPart = deliveryDesc ? deliveryDesc + ' – ' : '';
                                                  transportLabel = (pickupAddress && dropoffAddress) ? 'Livraison – ' + deliveryPart + pickupAddress + ' → ' + dropoffAddress : (pickupAddress && dropoffAddress) ? pickupAddress + ' → ' + dropoffAddress : '';
                                                } else {
                                                  transportLabel = (pickupAddress && dropoffAddress) ? pickupAddress + ' → ' + dropoffAddress : '';
                                                }
                                                // ✅ Règle UX stricte : ne JAMAIS afficher booking.amount dans le message de confirmation
                                                // ✅ Avant réponse API : afficher "— (recalcul en cours)"
                                                // ✅ Après réponse API : afficher confirmedNewAmount uniquement
                                                const isRecalculating = isConfirming && isOverrideInProgress;
                                                // ✅ Règle stricte : utiliser uniquement les valeurs de l'API (response.amount et response.old_amount)
                                                // ✅ Aucun fallback sur booking.amount
                                                const confirmedNewAmount = isConfirming && bookingOverrideConfirm?.newAmount != null 
                                                  ? bookingOverrideConfirm.newAmount 
                                                  : null;
                                                const confirmedOldAmount = isConfirming && bookingOverrideConfirm?.oldAmount != null 
                                                  ? bookingOverrideConfirm.oldAmount 
                                                  : null;
                                                
                                                // ✅ HARDENING : dans le confirm dialog, ne JAMAIS rendre booking.amount.
                                                const confirmDisplayAmount = (isRecalculating || confirmedNewAmount == null)
                                                  ? null
                                                  : (Number.isFinite(confirmedNewAmount) && !Number.isNaN(confirmedNewAmount)
                                                    ? confirmedNewAmount
                                                    : null);
                                                let amount;
                                                if (!isConfirming) {
                                                  const overrideAmount = s2BookingOverrides[booking.id]?.amount ?? s2BookingOverrides[String(booking.id)]?.amount;
                                                  const rawAmount = overrideAmount != null ? Number(overrideAmount) : Number(booking.amount || 0);
                                                  amount = Number.isFinite(rawAmount) && !Number.isNaN(rawAmount) ? rawAmount : 0;
                                                } else {
                                                  amount = confirmDisplayAmount;
                                                }
                                                const showS2Adjust = s2AdjustOpenBookingId === booking.id;
                                                const s2Override = s2BookingOverrides[booking.id] || s2BookingOverrides[String(booking.id)] || {};
                                                const s2DisplayAmount = s2Override.amount != null ? Number(s2Override.amount) : (Number(booking.amount || 0) || 0);
                                                
                                                return (
                                                  <div key={booking.id} style={{
                                                    padding: '6px 10px',
                                                    background: isInvoiced ? '#f1f5f9' : '#fff',
                                                    border: '1px solid #e5e7eb',
                                                    borderRadius: '4px',
                                                    fontSize: '12px',
                                                    opacity: isInvoiced ? 0.9 : 1
                                                  }}>
                                                    {!isConfirming ? (
                                                      <>
                                                        {/* ✅ Affichage ultra compact (1 ligne: date + pickup→dropoff + montant + ✏️, switch à droite) */}
                                                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px' }}>
                                                          <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: '10px', minWidth: 0 }}>
                                                            {dateStr && (
                                                              <span style={{ fontWeight: 500, color: '#111827', fontSize: '11px', whiteSpace: 'nowrap', flexShrink: 0 }}>
                                                                {dateStr}
                                                              </span>
                                                            )}
                                                            {transportLabel && (
                                                              <span style={{ color: '#4b5563', fontSize: '11px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1, minWidth: 0 }}>
                                                                {transportLabel}
                                                              </span>
                                                            )}
                                                            <span style={{ fontWeight: 600, color: '#111827', fontSize: '11px', whiteSpace: 'nowrap', flexShrink: 0 }}>
                                                              {formatCurrencyCHF(amount)}
                                                            </span>
                                                            {/* ✅ S2: Ajuster le montant avant génération (comme côté client) */}
                                                            {!isInvoiced && (
                                                              <button
                                                                type="button"
                                                                onClick={(e) => { e.preventDefault(); e.stopPropagation(); setS2AdjustOpenBookingId((prev) => (prev === booking.id ? null : booking.id)); }}
                                                                title="Ajuster le montant"
                                                                aria-expanded={showS2Adjust}
                                                                aria-controls={`s2-adjust-${booking.id}`}
                                                                style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '2px', fontSize: '14px', flexShrink: 0 }}
                                                              >
                                                                ✏️
                                                              </button>
                                                            )}
                                                            {isCancelled && (
                                                              <span
                                                                style={{ fontSize: '10px', color: '#92400e', fontWeight: 500, padding: '2px 6px', background: '#fef3c7', border: '1px solid #d97706', borderRadius: '3px', whiteSpace: 'nowrap', flexShrink: 0, marginLeft: '4px' }}
                                                                title={cancellationLabel || 'Réservation annulée (facturée)'}
                                                              >
                                                                Annulé
                                                              </span>
                                                            )}
                                                            {/* ✅ Safety: Label "Déjà facturé" si booking facturé */}
                                                            {isInvoiced && (
                                                              <span style={{ 
                                                                fontSize: '10px', 
                                                                color: '#dc2626', 
                                                                fontWeight: 500,
                                                                padding: '2px 6px',
                                                                background: '#fee2e2',
                                                                borderRadius: '3px',
                                                                whiteSpace: 'nowrap',
                                                                flexShrink: 0,
                                                                marginLeft: '8px'
                                                              }}>
                                                                Déjà facturé
                                                              </span>
                                                            )}
                                                          </div>
                                                          {/* Switch aligné à droite */}
                                                          <div 
                                                          role="switch"
                                                          aria-checked={isPatientBilling}
                                                          aria-disabled={isOverrideInProgress || isNotModifiable}
                                                          aria-label={isPatientBilling ? 'Facturer le patient' : 'Facturé à la clinique'}
                                                          tabIndex={(isOverrideInProgress || isNotModifiable) ? -1 : 0}
                                                          title={tooltipMessage || (isPatientBilling ? 'Facturer le patient' : 'Facturé à la clinique')}
                                                          style={{
                                                            position: 'relative',
                                                            width: '40px',
                                                            height: '22px',
                                                            borderRadius: '11px',
                                                            background: isPatientBilling ? '#dc2626' : '#3b82f6',
                                                            cursor: (isOverrideInProgress || isNotModifiable) ? 'not-allowed' : 'pointer',
                                                            transition: 'background-color 0.2s',
                                                            flexShrink: 0,
                                                            opacity: (isOverrideInProgress || isNotModifiable) ? 0.5 : 1,
                                                            pointerEvents: (isOverrideInProgress || isNotModifiable) ? 'none' : 'auto'
                                                          }}
                                                          onClick={(e) => {
                                                            if (isOverrideInProgress || isNotModifiable) {
                                                              e.preventDefault();
                                                              e.stopPropagation();
                                                              return;
                                                            }
                                                            e.preventDefault();
                                                            const newState = !isPatientBilling;
                                                            if (newState) {
                                                              // Bascule vers ON (patient) → confirmation
                                                              setBookingOverrideConfirm({ bookingId: booking.id, clientId: client.id, action: 'set_patient' });
                                                            } else {
                                                              // Bascule vers OFF (clinique) → confirmation
                                                              setBookingOverrideConfirm({ bookingId: booking.id, clientId: client.id, action: 'cancel' });
                                                            }
                                                          }}
                                                          onKeyDown={(e) => {
                                                            if (isOverrideInProgress || isNotModifiable) return;
                                                            if (e.key === 'Enter' || e.key === ' ') {
                                                              e.preventDefault();
                                                              const newState = !isPatientBilling;
                                                              if (newState) {
                                                                setBookingOverrideConfirm({ bookingId: booking.id, clientId: client.id, action: 'set_patient' });
                                                              } else {
                                                                setBookingOverrideConfirm({ bookingId: booking.id, clientId: client.id, action: 'cancel' });
                                                              }
                                                            }
                                                          }}
                                                        >
                                                          <div style={{
                                                            position: 'absolute',
                                                            top: '2px',
                                                            left: isPatientBilling ? '20px' : '2px',
                                                            width: '18px',
                                                            height: '18px',
                                                            borderRadius: '50%',
                                                            background: '#fff',
                                                            transition: 'left 0.2s',
                                                            boxShadow: '0 1px 3px rgba(0,0,0,0.2)'
                                                          }} />
                                                        </div>
                                                      </div>
                                                      {showS2Adjust && (
                                                        <div id={`s2-adjust-${booking.id}`} style={{ marginTop: '8px', padding: '10px', background: '#f9fafb', border: '1px solid #e5e7eb', borderRadius: '4px', fontSize: '12px' }}>
                                                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                                                            <span style={{ fontWeight: 500, minWidth: '80px' }}>Montant HT</span>
                                                            <input
                                                              type="number"
                                                              step="0.05"
                                                              min="0"
                                                              value={s2AmountInputLocal[booking.id] !== undefined ? s2AmountInputLocal[booking.id] : (s2DisplayAmount > 0 ? String(s2DisplayAmount) : '')}
                                                              placeholder={Number(booking.amount || 0) ? String(booking.amount) : '0'}
                                                              onChange={(e) => setS2AmountInputLocal((prev) => ({ ...prev, [booking.id]: e.target.value }))}
                                                              onBlur={(e) => {
                                                                const v = e.target.value.trim();
                                                                const num = v === '' ? null : parseFloat(v.replace(/,/g, '.'));
                                                                if (num !== null && !Number.isNaN(num) && num >= 0) {
                                                                  setS2BookingOverrides((prev) => ({ ...prev, [booking.id]: { ...(prev[booking.id] || prev[String(booking.id)] || {}), amount: num } }));
                                                                  setS2AmountInputLocal((prev) => ({ ...prev, [booking.id]: undefined }));
                                                                } else if (v !== '') {
                                                                  setS2BookingOverrides((prev) => ({ ...prev, [booking.id]: { ...(prev[booking.id] || prev[String(booking.id)] || {}), amount: Number(booking.amount || 0) } }));
                                                                  setS2AmountInputLocal((prev) => ({ ...prev, [booking.id]: undefined }));
                                                                }
                                                              }}
                                                              style={{ width: '100px', padding: '4px 8px' }}
                                                            />
                                                            <span>CHF</span>
                                                          </div>
                                                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                                                            <span style={{ fontWeight: 500, minWidth: '80px' }}>Note (optionnelle)</span>
                                                            <input
                                                              type="text"
                                                              value={s2Override.note ?? ''}
                                                              onChange={(e) => setS2BookingOverrides((prev) => ({ ...prev, [booking.id]: { ...(prev[booking.id] || prev[String(booking.id)] || {}), note: e.target.value.trim() || undefined } }))}
                                                              placeholder="Ex. Ajustement temps d'attente"
                                                              style={{ flex: 1, padding: '4px 8px' }}
                                                            />
                                                          </div>
                                                          <div style={{ marginBottom: '8px', fontSize: '11px', color: '#6b7280' }}>
                                                            HT <strong>{formatCurrencyCHF(s2DisplayAmount)}</strong>
                                                          </div>
                                                          <div style={{ display: 'flex', gap: '8px' }}>
                                                            <button
                                                              type="button"
                                                              onClick={() => {
                                                                setS2BookingOverrides((prev) => { const next = { ...prev }; delete next[booking.id]; delete next[String(booking.id)]; return next; });
                                                                setS2AmountInputLocal((prev) => { const next = { ...prev }; delete next[booking.id]; return next; });
                                                                setS2AdjustOpenBookingId(null);
                                                              }}
                                                              className="btn btn-secondary"
                                                              style={{ fontSize: '11px', padding: '4px 8px' }}
                                                            >
                                                              Réinitialiser
                                                            </button>
                                                            <button
                                                              type="button"
                                                              onClick={() => setS2AdjustOpenBookingId(null)}
                                                              className="btn btn-link"
                                                              style={{ fontSize: '11px', padding: '4px 8px' }}
                                                            >
                                                              Fermer
                                                            </button>
                                                          </div>
                                                        </div>
                                                      )}
                                                    </>
                                                    ) : (
                                                      // ✅ Confirmation inline
                                                      <div 
                                                        role="alertdialog"
                                                        aria-labelledby="confirm-title"
                                                        aria-describedby="confirm-description"
                                                        style={{ 
                                                          padding: '12px', 
                                                          background: '#fef3c7', 
                                                          border: '1px solid #fbbf24',
                                                          borderRadius: '6px'
                                                        }}
                                                      >
                                                        <div 
                                                          id="confirm-title"
                                                          aria-live="polite"
                                                          style={{ marginBottom: '12px', fontSize: '13px', color: '#78350f', fontWeight: 500 }}
                                                        >
                                                          {bookingOverrideConfirm.action === 'set_patient' 
                                                            ? 'Facturer ce transport au patient ?'
                                                            : 'Annuler l\'override et facturer à la clinique ?'}
                                                        </div>
                                                        <div 
                                                          id="confirm-description"
                                                          aria-live="polite"
                                                          style={{ marginBottom: '12px', fontSize: '12px', color: '#92400e', lineHeight: '1.6' }}
                                                        >
                                                          {bookingOverrideConfirm.action === 'set_patient' ? (
                                                            <>
                                                              <div style={{ marginBottom: '6px' }}>
                                                                Ce transport sera exclu de la facture clinique mensuelle.
                                                              </div>
                                                              {confirmDisplayAmount != null ? (
                                                                <div style={{ fontSize: '11px', color: '#78350f', fontStyle: 'italic', marginBottom: '4px' }}>
                                                                  Montant patient appliqué : <strong>{formatCurrencyCHF(confirmDisplayAmount)}</strong> (recalculé selon tarif patient)
                                                                </div>
                                                              ) : isOverrideInProgress ? (
                                                                <div style={{ fontSize: '11px', color: '#78350f', fontStyle: 'italic', marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                                                                  <span className={styles.spinner} aria-hidden="true" />
                                                                  Montant patient appliqué : <strong>—</strong> (recalcul en cours)
                                                                </div>
                                                              ) : (
                                                                <div style={{ fontSize: '11px', color: '#78350f', fontStyle: 'italic', marginBottom: '4px' }}>
                                                                  Montant patient appliqué : <strong>—</strong>
                                                                </div>
                                                              )}
                                                              {!isOverrideInProgress && confirmedNewAmount != null && confirmedOldAmount != null && (
                                                                <div style={{ fontSize: '11px', color: '#78350f', fontStyle: 'italic' }}>
                                                                  → Total S2 : <strong>{(() => {
                                                                    // ✅ RÈGLE STRICTE : totaux/delta uniquement depuis response (amount, old_amount). Aucun fallback.
                                                                    // ✅ Passage clinic → patient : on retire du total l’ancien montant (old_amount)
                                                                    const currentTotal = Number.isFinite(s2Totals.total_amount_eligible) && !Number.isNaN(s2Totals.total_amount_eligible) ? s2Totals.total_amount_eligible : 0;
                                                                    const newTotal = Math.max(0, Number((currentTotal - confirmedOldAmount).toFixed(2)));
                                                                    const safe = Number.isFinite(newTotal) && !Number.isNaN(newTotal) ? newTotal : 0;
                                                                    return formatCurrencyCHF(safe);
                                                                  })()}</strong> ({(() => {
                                                                    const currentCount = Number.isFinite(s2Totals.total_eligible) ? s2Totals.total_eligible : 0;
                                                                    const newCount = Math.max(0, Math.floor(currentCount - 1));
                                                                    return newCount;
                                                                  })()} transport{(() => {
                                                                    const currentCount = Number.isFinite(s2Totals.total_eligible) ? s2Totals.total_eligible : 0;
                                                                    const newCount = Math.max(0, Math.floor(currentCount - 1));
                                                                    return newCount > 1 ? 's' : '';
                                                                  })()}) (<span style={{ color: '#dc2626' }}>- {formatCurrencyCHF(confirmedOldAmount)}</span>)
                                                                </div>
                                                              )}
                                                            </>
                                                          ) : (
                                                            <>
                                                              <div style={{ marginBottom: '6px' }}>
                                                                Ce transport sera inclus dans la facture clinique mensuelle.
                                                              </div>
                                                              {confirmDisplayAmount != null ? (
                                                                <div style={{ fontSize: '11px', color: '#78350f', fontStyle: 'italic', marginBottom: '4px' }}>
                                                                  Montant clinique appliqué : <strong>{formatCurrencyCHF(confirmDisplayAmount)}</strong> (recalculé selon tarif clinique préférentiel)
                                                                </div>
                                                              ) : isOverrideInProgress ? (
                                                                <div style={{ fontSize: '11px', color: '#78350f', fontStyle: 'italic', marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                                                                  <span className={styles.spinner} aria-hidden="true" />
                                                                  Montant clinique appliqué : <strong>—</strong> (recalcul en cours)
                                                                </div>
                                                              ) : (
                                                                <div style={{ fontSize: '11px', color: '#78350f', fontStyle: 'italic', marginBottom: '4px' }}>
                                                                  Montant clinique appliqué : <strong>—</strong>
                                                                </div>
                                                              )}
                                                              {!isOverrideInProgress && confirmedNewAmount != null && confirmedOldAmount != null && (
                                                                <div style={{ fontSize: '11px', color: '#78350f', fontStyle: 'italic' }}>
                                                                  → Total S2 : <strong>{(() => {
                                                                    const currentTotal = Number.isFinite(s2Totals.total_amount_eligible) && !Number.isNaN(s2Totals.total_amount_eligible) ? s2Totals.total_amount_eligible : 0;
                                                                    const newTotal = currentTotal + confirmedNewAmount;
                                                                    const safeNewTotal = Number.isFinite(newTotal) && !Number.isNaN(newTotal) ? Number(newTotal.toFixed(2)) : 0;
                                                                    return formatCurrencyCHF(safeNewTotal);
                                                                  })()}</strong> ({(() => {
                                                                    const currentCount = Number.isFinite(s2Totals.total_eligible) ? s2Totals.total_eligible : 0;
                                                                    const newCount = Math.floor(currentCount + 1);
                                                                    return newCount;
                                                                  })()} transport{(() => {
                                                                    const currentCount = Number.isFinite(s2Totals.total_eligible) ? s2Totals.total_eligible : 0;
                                                                    const newCount = Math.floor(currentCount + 1);
                                                                    return newCount > 1 ? 's' : '';
                                                                  })()}) (<span style={{ color: '#059669' }}>+ {formatCurrencyCHF(confirmedNewAmount)}</span>)
                                                                </div>
                                                              )}
                                                            </>
                                                          )}
                                                        </div>
                                                        <div style={{ display: 'flex', gap: '8px' }}>
                                                          <button
                                                            ref={confirmButtonRef}
                                                            type="button"
                                                            onClick={async () => {
                                                              const newBilledToType = bookingOverrideConfirm.action === 'set_patient' ? 'patient' : 'clinic';
                                                              setBookingOverrideConfirm(null);
                                                              await handleBookingBillingOverride(booking.id, newBilledToType);
                                                            }}
                                                            className="btn btn-primary"
                                                            disabled={isOverrideInProgress || isNoClinic}
                                                            style={{ 
                                                              padding: '6px 12px', 
                                                              fontSize: '12px',
                                                              opacity: (isOverrideInProgress || isNoClinic) ? 0.6 : 1
                                                            }}
                                                          >
                                                            {isOverrideInProgress ? 'En cours...' : 'Confirmer'}
                                                          </button>
                                                          <button
                                                            type="button"
                                                            onClick={() => setBookingOverrideConfirm(null)}
                                                            className="btn btn-secondary"
                                                            disabled={isOverrideInProgress}
                                                            style={{ 
                                                              padding: '6px 12px', 
                                                              fontSize: '12px',
                                                              opacity: isOverrideInProgress ? 0.6 : 1
                                                            }}
                                                          >
                                                            Annuler
                                                          </button>
                                                        </div>
                                                      </div>
                                                    )}
                                                  </div>
                                                );
                                              })}
                                            </div>
                                          )}
                                        </div>
                                      )}
                                    </div>
                                  );
                                })}
                              {clients.filter((client) => (client.unbilled_count || 0) === 0 && (client.invoiced_count || 0) === 0).length > 0 && (
                                <div style={{ 
                                  padding: '8px 12px', 
                                  background: '#f9fafb', 
                                  border: '1px solid #e5e7eb', 
                                  borderRadius: '6px',
                                  fontSize: '12px',
                                  color: '#6b7280',
                                  fontStyle: 'italic'
                                }}>
                                  {clients.filter((client) => (client.unbilled_count || 0) === 0 && (client.invoiced_count || 0) === 0).length} patient{clients.filter((client) => (client.unbilled_count || 0) === 0 && (client.invoiced_count || 0) === 0).length > 1 ? 's' : ''} avec 0 transport clinique {clients.filter((client) => (client.unbilled_count || 0) === 0 && (client.invoiced_count || 0) === 0).length > 1 ? 'sont' : 'est'} exclu{clients.filter((client) => (client.unbilled_count || 0) === 0 && (client.invoiced_count || 0) === 0).length > 1 ? 's' : ''} de cette facture
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}

                  {/* ✅ SECTION 2: Exclusions (alerte secondaire - ENSUITE) */}
                  {s2Totals.total_excluded > 0 && (
                    <div className={styles.formGroup}>
                      <button
                        type="button"
                        onClick={() => setShowS2Exclusions(!showS2Exclusions)}
                        className={`${styles.accordion} ${styles.accordionWarning} ${showS2Exclusions ? styles.isOpen : ''}`}
                      >
                        <span>
                          ⚠️ Transports non facturés à la clinique
                        </span>
                        <span>{showS2Exclusions ? '▼' : '▶'}</span>
                      </button>
                      {showS2Exclusions && (
                        <div className={`${styles.accordionContent} ${styles.accordionContentWarning}`}>
                          <div style={{ marginBottom: '12px', lineHeight: '1.6' }}>
                            <strong>{s2Totals.total_excluded} transport{s2Totals.total_excluded > 1 ? 's' : ''} • {formatCurrencyCHF(s2Totals.total_amount_excluded)}</strong>
                          </div>
                          <div style={{ marginBottom: '16px', fontSize: '12px', color: '#92400e' }}>
                            Transports facturés au patient (override). Vous pouvez les réintégrer à la facture clinique.
                          </div>
                          
                          {/* ✅ Liste des transports exclus avec switch pour correction rapide */}
                          {Array.isArray(s2Totals.excluded_bookings) && s2Totals.excluded_bookings.length > 0 ? (
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                              {s2Totals.excluded_bookings.map((booking) => {
                                const isPatientBilling = booking.billed_to_type === 'patient';
                                const isOverrideInProgress = bookingOverridesInProgress.has(booking.id);
                                const isConfirming = bookingOverrideConfirm?.bookingId === booking.id;
                                
                                // ✅ Safety rules S2: Vérifier si le booking est modifiable
                                const normalizedReviewStatus = booking.billing_review_status 
                                  ? String(booking.billing_review_status).toLowerCase() 
                                  : null;
                                const isLocked = normalizedReviewStatus === 'locked';
                                const isInvoiced = booking.invoice_line_id != null && booking.invoice_line_id !== undefined;
                                const isNoClinic = !clinicCompanyId;
                                const isNotModifiable = isLocked || isInvoiced || isNoClinic;
                                
                                // Tooltips spécifiques selon la raison de désactivation
                                const tooltipMessage = isInvoiced 
                                  ? 'Inclus dans une facture (ligne existante).'
                                  : (isLocked 
                                    ? 'Transport verrouillé (contrôle facturation).'
                                    : (isNoClinic 
                                      ? 'Sélectionnez une clinique avant de réintégrer ce transport.'
                                      : null));
                                
                                const scheduledDate = booking.scheduled_time ? new Date(booking.scheduled_time) : null;
                                const dateStr = scheduledDate 
                                  ? scheduledDate.toLocaleDateString('fr-FR', { day: '2-digit', month: '2-digit', year: 'numeric' }).replace(/\//g, '.')
                                  : '';
                                
                                // ✅ Règle UX stricte : ne JAMAIS afficher booking.amount dans le message de confirmation
                                // ✅ Avant réponse API : afficher "— (recalcul en cours)"
                                // ✅ Après réponse API : afficher confirmedNewAmount uniquement
                                const isRecalculating = isConfirming && isOverrideInProgress;
                                // ✅ Règle stricte : utiliser uniquement les valeurs de l'API (response.amount et response.old_amount)
                                // ✅ Aucun fallback sur booking.amount
                                const confirmedNewAmount = isConfirming && bookingOverrideConfirm?.newAmount != null 
                                  ? bookingOverrideConfirm.newAmount 
                                  : null;
                                const confirmedOldAmount = isConfirming && bookingOverrideConfirm?.oldAmount != null 
                                  ? bookingOverrideConfirm.oldAmount 
                                  : null;
                                
                                // ✅ HARDENING : dans le confirm dialog, ne JAMAIS rendre booking.amount.
                                const confirmDisplayAmount = (isRecalculating || confirmedNewAmount == null)
                                  ? null
                                  : (Number.isFinite(confirmedNewAmount) && !Number.isNaN(confirmedNewAmount)
                                    ? confirmedNewAmount
                                    : null);
                                let amount;
                                if (!isConfirming) {
                                  const rawAmount = Number(booking.amount || 0);
                                  amount = Number.isFinite(rawAmount) && !Number.isNaN(rawAmount) ? rawAmount : 0;
                                } else {
                                  amount = confirmDisplayAmount;
                                }
                                
                                // Nom du patient (customer_name ou client info)
                                const patientName = booking.customer_name || booking.client_name || 'Patient inconnu';
                                
                                // Adresses + préfixes Annulation / Livraison
                                const pickupAddress = booking.pickup_address || booking.pickup_location || '';
                                const dropoffAddress = booking.dropoff_address || booking.dropoff_location || '';
                                const isCancelledExcl = (booking.status || '').toUpperCase() === 'CANCELED';
                                const isDeliveryExcl = (booking.mission_type || '').toLowerCase() === 'material_delivery';
                                const deliveryDescExcl = (booking.delivery_description || '').trim();
                                const cancellationLabelExcl = booking.cancellation_display_label || null;
                                let transportLabelExcl = '';
                                if (isDeliveryExcl) {
                                  const deliveryPartExcl = deliveryDescExcl ? deliveryDescExcl + ' – ' : '';
                                  transportLabelExcl = (pickupAddress && dropoffAddress) ? 'Livraison – ' + deliveryPartExcl + pickupAddress + ' → ' + dropoffAddress : (pickupAddress && dropoffAddress) ? pickupAddress + ' → ' + dropoffAddress : '';
                                } else {
                                  transportLabelExcl = (pickupAddress && dropoffAddress) ? pickupAddress + ' → ' + dropoffAddress : '';
                                }
                                
                                return (
                                  <div key={booking.id} style={{
                                    padding: '6px 10px',
                                    background: '#fff',
                                    border: '1px solid #fbbf24',
                                    borderRadius: '4px',
                                    fontSize: '12px'
                                  }}>
                                    {!isConfirming ? (
                                      // ✅ Affichage compact (1 ligne: date • patient • pickup→dropoff • montant, switch à droite)
                                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px' }}>
                                        <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: '8px', minWidth: 0 }}>
                                          {dateStr && (
                                            <span style={{ fontWeight: 500, color: '#111827', fontSize: '11px', whiteSpace: 'nowrap', flexShrink: 0 }}>
                                              {dateStr}
                                            </span>
                                          )}
                                          <span style={{ color: '#92400e', fontSize: '11px', whiteSpace: 'nowrap', flexShrink: 0 }}>
                                            •
                                          </span>
                                          <span style={{ color: '#4b5563', fontSize: '11px', whiteSpace: 'nowrap', flexShrink: 0, maxWidth: '120px', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                            {patientName}
                                          </span>
                                          {transportLabelExcl && (
                                            <>
                                              <span style={{ color: '#92400e', fontSize: '11px', whiteSpace: 'nowrap', flexShrink: 0 }}>
                                                •
                                              </span>
                                              <span style={{ color: '#4b5563', fontSize: '11px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1, minWidth: 0 }}>
                                                {transportLabelExcl}
                                              </span>
                                            </>
                                          )}
                                          <span style={{ fontWeight: 600, color: '#111827', fontSize: '11px', whiteSpace: 'nowrap', flexShrink: 0 }}>
                                            {formatCurrencyCHF(amount)}
                                          </span>
                                          {isCancelledExcl && (
                                            <span
                                              style={{ fontSize: '10px', color: '#92400e', fontWeight: 500, padding: '2px 6px', background: '#fef3c7', border: '1px solid #d97706', borderRadius: '3px', whiteSpace: 'nowrap', flexShrink: 0, marginLeft: '4px' }}
                                              title={cancellationLabelExcl || 'Réservation annulée (facturée)'}
                                            >
                                              Annulé
                                            </span>
                                          )}
                                          {/* Badge "Override patient" */}
                                          <span style={{ 
                                            fontSize: '10px', 
                                            color: '#92400e', 
                                            fontWeight: 500,
                                            padding: '2px 6px',
                                            background: '#fef3c7',
                                            borderRadius: '3px',
                                            whiteSpace: 'nowrap',
                                            flexShrink: 0,
                                            marginLeft: '4px'
                                          }}>
                                            Override patient
                                          </span>
                                          {/* ✅ Safety: Label "Déjà facturé" si booking facturé */}
                                          {isInvoiced && (
                                            <span style={{ 
                                              fontSize: '10px', 
                                              color: '#dc2626', 
                                              fontWeight: 500,
                                              padding: '2px 6px',
                                              background: '#fee2e2',
                                              borderRadius: '3px',
                                              whiteSpace: 'nowrap',
                                              flexShrink: 0,
                                              marginLeft: '4px'
                                            }}>
                                              Déjà facturé
                                            </span>
                                          )}
                                        </div>
                                        {/* Switch aligné à droite (OFF=Clinique, ON=Patient) */}
                                        <div 
                                          role="switch"
                                          aria-checked={isPatientBilling}
                                          aria-disabled={isOverrideInProgress || isNotModifiable}
                                          aria-label={isPatientBilling ? 'Facturer le patient' : 'Facturé à la clinique'}
                                          tabIndex={(isOverrideInProgress || isNotModifiable) ? -1 : 0}
                                          title={tooltipMessage || (isPatientBilling ? 'Facturer le patient' : 'Revenir à la clinique')}
                                          style={{
                                            position: 'relative',
                                            width: '40px',
                                            height: '22px',
                                            borderRadius: '11px',
                                            background: isPatientBilling ? '#dc2626' : '#3b82f6',
                                            cursor: (isOverrideInProgress || isNotModifiable) ? 'not-allowed' : 'pointer',
                                            transition: 'background-color 0.2s',
                                            flexShrink: 0,
                                            opacity: (isOverrideInProgress || isNotModifiable) ? 0.5 : 1,
                                            pointerEvents: (isOverrideInProgress || isNotModifiable) ? 'none' : 'auto'
                                          }}
                                          onClick={(e) => {
                                            if (isOverrideInProgress || isNotModifiable) {
                                              e.preventDefault();
                                              e.stopPropagation();
                                              return;
                                            }
                                            e.preventDefault();
                                            // Pour les exclusions, on veut toujours revenir à la clinique (OFF)
                                            if (isPatientBilling) {
                                              setBookingOverrideConfirm({ bookingId: booking.id, action: 'cancel' });
                                            }
                                          }}
                                          onKeyDown={(e) => {
                                            if (isOverrideInProgress || isNotModifiable) return;
                                            if (e.key === 'Enter' || e.key === ' ') {
                                              e.preventDefault();
                                              if (isPatientBilling) {
                                                setBookingOverrideConfirm({ bookingId: booking.id, action: 'cancel' });
                                              }
                                            }
                                          }}
                                        >
                                          <div style={{
                                            position: 'absolute',
                                            top: '2px',
                                            left: isPatientBilling ? '20px' : '2px',
                                            width: '18px',
                                            height: '18px',
                                            borderRadius: '50%',
                                            background: '#fff',
                                            transition: 'left 0.2s',
                                            boxShadow: '0 1px 3px rgba(0,0,0,0.2)'
                                          }} />
                                        </div>
                                      </div>
                                    ) : (
                                      // ✅ Confirmation inline
                                      <div 
                                        role="alertdialog"
                                        aria-labelledby="confirm-title"
                                        aria-describedby="confirm-description"
                                        style={{ 
                                          padding: '12px', 
                                          background: '#fef3c7', 
                                          border: '1px solid #fbbf24',
                                          borderRadius: '6px'
                                        }}
                                      >
                                        <div 
                                          id="confirm-title"
                                          aria-live="polite"
                                          style={{ marginBottom: '12px', fontSize: '13px', color: '#78350f', fontWeight: 500 }}
                                        >
                                          Inclure ce transport dans la facture clinique mensuelle ?
                                        </div>
                                        <div 
                                          id="confirm-description"
                                          aria-live="polite"
                                          style={{ marginBottom: '12px', fontSize: '12px', color: '#92400e', lineHeight: '1.6' }}
                                        >
                                          <div style={{ marginBottom: '6px' }}>
                                            Ce transport sera inclus dans la facture clinique mensuelle.
                                          </div>
                                          {confirmDisplayAmount != null ? (
                                            <div style={{ fontSize: '11px', color: '#78350f', fontStyle: 'italic', marginBottom: '4px' }}>
                                              Montant clinique appliqué : <strong>{formatCurrencyCHF(confirmDisplayAmount)}</strong> (recalculé selon tarif clinique préférentiel)
                                            </div>
                                          ) : isOverrideInProgress ? (
                                            <div style={{ fontSize: '11px', color: '#78350f', fontStyle: 'italic', marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                                              <span className={styles.spinner} aria-hidden="true" />
                                              Montant clinique appliqué : <strong>—</strong> (recalcul en cours)
                                            </div>
                                          ) : (
                                            <div style={{ fontSize: '11px', color: '#78350f', fontStyle: 'italic', marginBottom: '4px' }}>
                                              Montant clinique appliqué : <strong>—</strong>
                                            </div>
                                          )}
                                          {!isOverrideInProgress && confirmedNewAmount != null && confirmedOldAmount != null && (
                                            <div style={{ fontSize: '11px', color: '#78350f', fontStyle: 'italic' }}>
                                              → Total S2 : <strong>{(() => {
                                                const currentTotal = Number.isFinite(s2Totals.total_amount_eligible) && !Number.isNaN(s2Totals.total_amount_eligible) ? s2Totals.total_amount_eligible : 0;
                                                const newTotal = currentTotal + confirmedNewAmount;
                                                const safeNewTotal = Number.isFinite(newTotal) && !Number.isNaN(newTotal) ? Number(newTotal.toFixed(2)) : 0;
                                                return formatCurrencyCHF(safeNewTotal);
                                              })()}</strong> ({(() => {
                                                const currentCount = Number.isFinite(s2Totals.total_eligible) ? s2Totals.total_eligible : 0;
                                                const newCount = Math.floor(currentCount + 1);
                                                return newCount;
                                              })()} transport{(() => {
                                                const currentCount = Number.isFinite(s2Totals.total_eligible) ? s2Totals.total_eligible : 0;
                                                const newCount = Math.floor(currentCount + 1);
                                                return newCount > 1 ? 's' : '';
                                              })()}) (<span style={{ color: '#059669' }}>+ {formatCurrencyCHF(confirmedNewAmount)}</span>)
                                            </div>
                                          )}
                                        </div>
                                        <div style={{ display: 'flex', gap: '8px' }}>
                                          <button
                                            ref={confirmButtonRef}
                                            type="button"
                                            onClick={async () => {
                                              // ✅ Ne PAS fermer le message immédiatement - le garder ouvert pour afficher les valeurs de l'API
                                              // setBookingOverrideConfirm(null); // ❌ Retiré : ferme le message avant que newAmount soit disponible
                                              await handleBookingBillingOverride(booking.id, 'clinic');
                                              // ✅ Fermer le message APRÈS avoir reçu la réponse de l'API (fait dans handleBookingBillingOverride)
                                            }}
                                            className="btn btn-primary"
                                            disabled={isOverrideInProgress || isNoClinic}
                                            style={{ 
                                              padding: '6px 12px', 
                                              fontSize: '12px',
                                              opacity: (isOverrideInProgress || isNoClinic) ? 0.6 : 1
                                            }}
                                          >
                                            {isOverrideInProgress ? 'En cours...' : 'Confirmer'}
                                          </button>
                                          <button
                                            type="button"
                                            onClick={() => setBookingOverrideConfirm(null)}
                                            className="btn btn-secondary"
                                            disabled={isOverrideInProgress}
                                            style={{ 
                                              padding: '6px 12px', 
                                              fontSize: '12px',
                                              opacity: isOverrideInProgress ? 0.6 : 1
                                            }}
                                          >
                                            Annuler
                                          </button>
                                        </div>
                                      </div>
                                    )}
                                  </div>
                                );
                              })}
                            </div>
                          ) : (
                            <div style={{ color: '#6b7280', fontSize: '12px', textAlign: 'center', padding: '12px' }}>
                              Aucun transport exclu trouvé.
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}

                  {/* ✅ SECTION 3: Options avancées (accordion, fermé par défaut) */}
                  <div className={styles.formGroup}>
                    <button
                      type="button"
                      onClick={() => setShowS2Advanced(!showS2Advanced)}
                      className={`${styles.accordion} ${styles.accordionInfo} ${showS2Advanced ? styles.isOpen : ''}`}
                      style={{ background: showS2Advanced ? '#f0f9ff' : 'transparent' }}
                    >
                      <span style={{ fontWeight: 500, fontSize: '13px' }}>
                        ▶ Options avancées
                      </span>
                      <span>{showS2Advanced ? '▼' : '▶'}</span>
                    </button>
                    {showS2Advanced && (
                      <div className={`${styles.accordionContent} ${styles.accordionContentInfo}`}>
                        {/* Limiter à certains patients */}
                        <div style={{ marginBottom: '16px' }}>
                          <label className={styles.label} style={{ marginBottom: '8px' }}>
                            Limiter à certains patients (optionnel)
                          </label>
                          <small className={styles.hint} style={{ display: 'block', marginBottom: '8px', color: '#6b7280', fontSize: '12px' }}>
                            Par défaut, tous les patients de la clinique sont inclus. Sélectionnez des patients pour créer une facture partielle.
                          </small>
                          <div className={styles.clientsList} style={{ maxHeight: '200px', overflowY: 'auto' }}>
                            {clients.map((client) => (
                              <label key={client.id} className={styles.checkboxLabel}>
                                <input
                                  type="checkbox"
                                  checked={includeClientIds.includes(client.id)}
                                  onChange={(e) => {
                                    if (e.target.checked) {
                                      setIncludeClientIds([...includeClientIds, client.id]);
                                    } else {
                                      setIncludeClientIds(includeClientIds.filter((id) => id !== client.id));
                                    }
                                  }}
                                  disabled={loading}
                                />
                                <span>
                                  {client.display_name || `${client.first_name || ''} ${client.last_name || ''}`.trim() ||
                                    client.username}
                                </span>
                              </label>
                            ))}
                          </div>
                          {includeClientIds.length > 0 && (
                            <small className={styles.hint} style={{ display: 'block', marginTop: '8px', color: '#0369a1', fontSize: '12px' }}>
                              {includeClientIds.length} patient{includeClientIds.length > 1 ? 's' : ''} sélectionné{includeClientIds.length > 1 ? 's' : ''} (facture partielle)
                            </small>
                          )}
                        </div>

                      </div>
                    )}
                  </div>
                </>
                  )}
            </>
          )}

          {/* Facturation partenaire — même modèle que clinique / direct */}
          {billingType === 'partner' && (
            <>
              <div className={styles.s2Header}>
                <div className={styles.s2HeaderRow}>
                  <span className={styles.s2HeaderLabel}>🤝 Partenaire</span>
                  <div className={styles.s2HeaderClinique} style={{ flex: 1, maxWidth: '100%' }}>
                    <select
                      id="partnership_id"
                      name="partnership_id"
                      value={formData.partnership_id}
                      onChange={handleInputChange}
                      className={styles.select}
                      required
                      disabled={loading || partnersLoading}
                    >
                      <option value="">Sélectionner un partenaire</option>
                      {partners.map((partner) => (
                        <option key={partner.partnership_id} value={partner.partnership_id}>
                          {partner.partner_company_name} ({partner.unbilled_transfers_count} transfert{partner.unbilled_transfers_count > 1 ? 's' : ''} • {formatCurrency(partner.total_amount)})
                        </option>
                      ))}
                    </select>
                    {partnersLoading && <small className={styles.hint}>Chargement…</small>}
                    {!partnersLoading && partners.length === 0 && (
                      <small className={styles.hint}>Aucun partenaire avec transferts facturables.</small>
                    )}
                  </div>
                </div>
                <div className={styles.s2HeaderRow}>
                  <div className={styles.s2HeaderMeta}>
                    <span className={styles.s2HeaderLabel}>Période à facturer</span>
                    {selectedPartner && (
                      <>
                        <span className={styles.s2HeaderMetaSep}>•</span>
                        <span>{selectedPartner.unbilled_transfers_count} transfert{selectedPartner.unbilled_transfers_count > 1 ? 's' : ''}</span>
                      </>
                    )}
                  </div>
                  <div className={styles.s2HeaderPeriod}>
                    <select
                      id="partner_period_month"
                      name="period_month"
                      value={formData.period_month}
                      onChange={handleInputChange}
                      className={styles.select}
                      disabled={loading}
                    >
                      {months.map((m) => (
                        <option key={m.value} value={m.value}>{m.label}</option>
                      ))}
                    </select>
                    <select
                      id="partner_period_year"
                      name="period_year"
                      value={formData.period_year}
                      onChange={handleInputChange}
                      className={styles.select}
                      disabled={loading}
                    >
                      {years.map((y) => (
                        <option key={y} value={y}>{y}</option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              {selectedPartner && (
                <div className={styles.formGroup}>
                  <button
                    type="button"
                    onClick={() => setShowPartnerSummary(!showPartnerSummary)}
                    className={`${styles.accordion} ${styles.accordionInfo} ${showPartnerSummary ? styles.isOpen : ''}`}
                  >
                    <span>✓ Liste des transferts et ajustements</span>
                    <span>{showPartnerSummary ? '▼' : '▶'}</span>
                  </button>
                  {showPartnerSummary && (
                    <div className={`${styles.accordionContent} ${styles.accordionContentInfo}`}>
                      <PartnerTransferSelector
                        companyId={companyId}
                        partnershipId={parseInt(formData.partnership_id, 10)}
                        period={{
                          year: formData.period_year,
                          month: formData.period_month,
                        }}
                        overrides={partnerOverrides}
                        onOverrideChange={(transferId, override) => {
                          setPartnerOverrides((prev) => {
                            const next = { ...prev };
                            const key = String(transferId);
                            if (override.amount === undefined && override.note === undefined) {
                              delete next[key];
                            } else {
                              next[key] = { ...(next[key] || {}), ...override };
                              if (next[key].amount === null) delete next[key].amount;
                              if (next[key].note === null) delete next[key].note;
                              if (Object.keys(next[key]).length === 0) delete next[key];
                            }
                            return next;
                          });
                        }}
                        onSelectionChange={(selected) => {
                          setPartnerSelectedTransfers(selected);
                        }}
                      />
                    </div>
                  )}
                </div>
              )}
            </>
          )}

          {/* Période (masquée en S2, direct, partenaire et tierce, déjà dans les headers) */}
          {billingType !== 'direct' && billingType !== 'partner' && billingType !== 'third_party' && (
            <div className={styles.formRow}>
              <div className={styles.formGroup}>
                <label htmlFor="period_year" className={styles.label}>
                  Année
                </label>
                <select
                  id="period_year"
                  name="period_year"
                  value={formData.period_year}
                  onChange={handleInputChange}
                  className={styles.select}
                  disabled={loading}
                >
                  {years.map((year) => (
                    <option key={year} value={year}>
                      {year}
                    </option>
                  ))}
                </select>
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="period_month" className={styles.label}>
                  Mois
                </label>
                <select
                  id="period_month"
                  name="period_month"
                  value={formData.period_month}
                  onChange={handleInputChange}
                  className={styles.select}
                  disabled={loading}
                >
                  {months.map((month) => (
                    <option key={month.value} value={month.value}>
                      {month.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          )}

          </div>

          {/* Footer sticky avec totaux (third_party, direct, partenaire) */}
          {((billingType === 'third_party' && formData.bill_to_client_id) ||
            (billingType === 'direct' && formData.client_id) ||
            (billingType === 'partner' && formData.partnership_id)) && (
            <div className={styles.stickyFooter}>
              <div className={styles.stickyFooterContent}>
                <div className={styles.stickyFooterTotals}>
                  {billingType === 'direct' ? (
                    clientsLoading && formData.client_id ? (
                      <span className={styles.stickyFooterEmpty}>Chargement des totaux…</span>
                    ) : !formData.client_id ? (
                      <span className={styles.stickyFooterEmpty}>Sélectionnez un client</span>
                    ) : directSummary && directSummary.count > 0 ? (
                      directSelection.length === 0 ? (
                        <div className={styles.stickyFooterEmptyWithAction}>
                          <span>Aucun transport sélectionné (0/{directSummary.count})</span>
                          <button
                            type="button"
                            className={styles.stickyFooterActionLink}
                            onClick={handleSelectAllReservations}
                            disabled={loading || clientsLoading}
                          >
                            Tout sélectionner
                          </button>
                        </div>
                      ) : isPartialSelection ? (
                        _directTotals === null || hasUnhydratedMinimals ? (
                          <span className={styles.stickyFooterTotal}>
                            Sélection partielle {directSelection.length}/{directSummary.count} • TTC <strong>—</strong>
                            <span className={styles.stickyFooterWarning} style={{ marginLeft: '8px', fontSize: '0.9em', color: '#ff9800' }}>
                              (Détails requis pour calculer le total)
                            </span>
                          </span>
                        ) : (
                          <span className={styles.stickyFooterTotal}>
                            Sélection partielle {directSelection.length}/{directSummary.count} • TTC <strong>{formatCurrency(_directTotals.total)}</strong>
                          </span>
                        )
                      ) : (
                        <span className={styles.stickyFooterTotal}>
                          1 facture client • {directSummary.count} transport{directSummary.count > 1 ? 's' : ''} • Total TTC <strong>{formatCurrency(_directTotals?.total ?? directSummaryTTC ?? 0)}</strong>
                        </span>
                      )
                    ) : directSummary && directSummary.count === 0 ? (
                      <span className={styles.stickyFooterEmpty}>Aucun transport à facturer</span>
                    ) : (
                      <span className={styles.stickyFooterEmpty}>—</span>
                    )
                  ) : billingType === 'partner' ? (
                    selectedPartner ? (
                      <span className={styles.stickyFooterTotal}>
                        1 facture partenaire •{' '}
                        {partnerSelectedTransfers.length > 0
                          ? `${partnerSelectedTransfers.length} transfert${partnerSelectedTransfers.length > 1 ? 's' : ''}`
                          : `${selectedPartner.unbilled_transfers_count} transfert${selectedPartner.unbilled_transfers_count > 1 ? 's' : ''}`}
                        {' • '}
                        Total{' '}
                        <strong>
                          {partnerTotalComputed != null
                            ? formatCurrency(partnerTotalComputed)
                            : formatCurrency(selectedPartner.total_amount)}
                        </strong>
                      </span>
                    ) : (
                      <span className={styles.stickyFooterEmpty}>Sélectionnez un partenaire</span>
                    )
                  ) : isClinicMonthly ? (
                    // ✅ Mode S2: Résumé facture clinique mensuelle
                    <>
                      {s2TotalsLoading ? (
                        <span className={styles.stickyFooterEmpty}>Chargement des totaux...</span>
                      ) : s2Totals.total_eligible === 0 ? (
                        <span className={styles.stickyFooterEmpty} style={{ color: '#dc2626' }}>
                          ⚠️ Aucun transport clinique éligible{includeClientIds.length > 0 ? ' pour les patients sélectionnés' : ''} sur cette période
                        </span>
                      ) : (
                        <span className={styles.stickyFooterTotal}>
                          1 facture clinique • {s2Totals.total_eligible} transport{s2Totals.total_eligible > 1 ? 's' : ''} à facturer
                          {s2Totals.total_invoiced > 0 ? ` (${s2Totals.total_invoiced} déjà facturés)` : ''}
                          {' • Total '}<strong>{formatCurrencyCHF(s2Totals.total_amount_eligible)}</strong>
                        </span>
                      )}
                    </>
                  ) : isConsolidated && formData.client_ids.length >= 2 ? (
                    <>
                      <span className={styles.stickyFooterTotal}>
                        {formData.client_ids.length} factures • Total <strong>{formatCurrency(consolidatedTotals.total)}</strong>
                      </span>
                      {vatConfig.applicable && consolidatedTotals.vat > 0 && (
                        <span className={styles.stickyFooterVat}>
                          HT: {formatCurrency(consolidatedTotals.base)} • TVA: {formatCurrency(consolidatedTotals.vat)}
                        </span>
                      )}
                    </>
                  ) : consolidatedSelection.length > 0 ? (
                    <>
                      <span className={styles.stickyFooterTotal}>
                        {consolidatedSelection.length} transport{consolidatedSelection.length > 1 ? 's' : ''} • Total <strong>{formatCurrency(consolidatedTotals.total)}</strong>
                      </span>
                      {vatConfig.applicable && consolidatedTotals.vat > 0 && (
                        <span className={styles.stickyFooterVat}>
                          HT: {formatCurrency(consolidatedTotals.base)} • TVA: {formatCurrency(consolidatedTotals.vat)}
                        </span>
                      )}
                    </>
                  ) : (
                    <span className={styles.stickyFooterEmpty}>
                      {isClinicMonthly ? 'Facture clinique mensuelle (tous les transports éligibles seront inclus)' : 'Sélectionnez des transports'}
                    </span>
                  )}
                </div>
                <div className={styles.stickyFooterActions}>
                  <button
                    type="button"
                    onClick={handleClose}
                    className="btn btn-secondary"
                    disabled={loading}
                  >
                    Annuler
                  </button>
                  <button
                    type="submit"
                    className="btn btn-primary"
                    data-tour-id="invoice-modal-submit"
                    disabled={
                      loading ||
                      (billingType === 'direct'
                        ? !formData.client_id || clientsLoading || !directSummary || directSummary.count === 0 || directSelection.length === 0
                        // ✅ Zéro friction : pas besoin d'ouvrir "Transports à facturer", directSummary = eligible
                        // ✅ Mais nécessite au moins une sélection (même si directSummary.count > 0)
                        : billingType === 'partner'
                          ? !formData.partnership_id
                          : isClinicMonthly
                            ? !formData.bill_to_client_id || !selectedInstitution?.clinic_company_id || s2TotalsLoading || s2Totals.total_eligible === 0
                            : formData.client_ids.length === 0 || !formData.bill_to_client_id || consolidatedSelection.length === 0)
                    }
                  >
                    {loading
                      ? 'Génération...'
                      : billingType === 'direct'
                        ? 'Générer la facture'
                        : billingType === 'partner'
                          ? 'Générer la facture'
                          : isClinicMonthly
                            ? 'Générer facture clinique (mois)'
                            : isConsolidated && formData.client_ids.length >= 2
                              ? (hasManualSelection ? 'Générer groupé (ignore la sélection)' : `Générer ${formData.client_ids.length} factures`)
                              : formData.client_ids.length > 1
                                ? `Générer ${formData.client_ids.length} factures`
                                : 'Générer la facture'}
                  </button>
                </div>
              </div>
            </div>
          )}

        </form>
      </div>
    </div>
  );
};

export default NewInvoiceModal;
