import React, { useCallback, useEffect, useState, useMemo, useRef } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  FiX, FiTruck, FiMapPin, FiInfo, FiClock, FiFileText, FiUser, FiPhone,
  FiAlertCircle, FiEdit2, FiPackage, FiHome, FiTrash2,
} from 'react-icons/fi';
import { renderBookingDateTime } from '../../../../utils/formatDate';
import { formatLegTime } from '../../../../utils/formatLegTime';
import { fetchTransportVouchers } from '../../../../services/transportVoucherService';
import AddressAutocomplete from '../../../../components/common/AddressAutocomplete';
import InlineDatePicker from '../../../../components/ui/InlineDatePicker';
import InlineTimePicker from '../../../../components/ui/InlineTimePicker';
import useCompanySocket from '../../../../hooks/useCompanySocket';
import { toast } from 'sonner';
import BookingChat from './BookingChat';
import { buildIdentityFromApi } from '../../../../utils/bookingIdentity';
import { getBookingSourceMeta } from '../../../../constants/bookingSourceLabels';
import {
  completeReservation,
  patchBillingAdjustment,
  fetchBookingChangeEvents,
  acknowledgeBookingChangeEvent,
  respondToChangeRequest,
} from '../../../../services/companyService';
import { fetchClinicBillingMappings } from '../../../../services/settingsService';
import s from './ReservationDetailPanel.module.css';

const STATUS_MAP = {
  pending:            { label: 'En attente',       css: 'statusPending' },
  accepted:           { label: 'Acceptée',         css: 'statusAccepted' },
  assigned:           { label: 'Assignée',         css: 'statusAssigned' },
  en_route:           { label: 'En route',         css: 'statusEnRoute' },
  in_progress:        { label: 'En cours',         css: 'statusInProgress' },
  completed:          { label: 'Terminée',         css: 'statusCompleted' },
  return_completed:   { label: 'Retour terminé',   css: 'statusCompleted' },
  canceled:           { label: 'Annulée',          css: 'statusCancelled' },
  cancelled:          { label: 'Annulée',          css: 'statusCancelled' },
  rejected:           { label: 'Refusée',          css: 'statusCancelled' },
  no_show:            { label: 'Non présente',     css: 'statusCancelled' },
};

const VOUCHER_STATUS_LABELS = {
  draft: 'Brouillon', submitted: 'Soumis', validated: 'Valide', rejected: 'Rejeté', expired: 'Expiré',
};

const VOUCHER_TYPE_LABELS = {
  clinic: 'Clinique', insurance: 'Assurance', other: 'Autre',
};

const INTENT_LABELS = {
  institution: 'Institution', clinic: 'Clinique', patient: 'Patient',
  curator: 'Curateur', spc: 'SPC', other: 'Autre',
};

const formatCurrency = (value) => {
  const n = Number(value);
  if (!Number.isFinite(n)) return '-';
  return `${n.toFixed(2)} CHF`;
};

const fmtShort = (dateStr) => {
  if (!dateStr) return '-';
  return new Date(dateStr).toLocaleString('fr-CH', {
    day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit',
  });
};

const buildTimeline = (r) => {
  if (!r) return [];
  const events = [];
  const driverName = r.driver_name || r.driver?.full_name;

  // Institution request events (if booking came from an institution request)
  const it = r.institution_timeline;
  if (it) {
    const instName = it.institution_name;
    if (it.created_at) {
      events.push({
        event: `Demande créée${it.created_by_name ? ` par ${it.created_by_name}` : ''}${instName ? ` (${instName})` : ''}`,
        date: it.created_at,
      });
    }
    if (it.sent_at) events.push({ event: 'Demande envoyée', date: it.sent_at });
    if (it.accepted_at) {
      events.push({
        event: `Demande acceptée${it.accepted_by_company_name ? ` par ${it.accepted_by_company_name}` : ''}`,
        date: it.accepted_at,
      });
    }
    if (it.converted_at) events.push({ event: 'Réservation créée', date: it.converted_at });
    if (it.cancelled_at) events.push({ event: 'Demande annulée', date: it.cancelled_at });
  } else {
    if (r.created_at) events.push({ event: 'Réservation créée', date: r.created_at });
  }

  // Booking lifecycle events
  if (r.accepted_at && !it?.accepted_at) {
    events.push({ event: `Acceptée${driverName ? ` par ${driverName}` : ''}`, date: r.accepted_at });
  }
  if (r.assigned_at) events.push({ event: `Assignée${driverName ? ` à ${driverName}` : ''}`, date: r.assigned_at });

  // Événements opérationnels : historique consolidé du parcours complet
  // (tous les legs multi-étapes + retours) si disponible, sinon le leg courant.
  const journey = Array.isArray(r.route_journey) ? r.route_journey : null;
  if (journey && journey.length) {
    journey.forEach((ev) => {
      if (ev?.date) events.push({ event: ev.event, date: ev.date, type: ev.type });
    });
  } else {
    if (r.picked_up_at || r.boarded_at) {
      events.push({ event: `Client pris en charge${driverName ? ` par ${driverName}` : ''}`, date: r.picked_up_at || r.boarded_at });
    }
    if (r.completed_at) events.push({ event: 'Course terminée', date: r.completed_at });
  }
  if (r.started_at) events.push({ event: 'Course démarrée', date: r.started_at });
  if ((r.cancelled_at || r.canceled_at) && !it?.cancelled_at) {
    const roleMap = { company: 'Entreprise', driver: 'Chauffeur', admin: 'Admin', system: 'Système' };
    const byLabel = roleMap[r.cancelled_by_role] || '';
    const reasonLabel = r.cancellation_display_label || r.cancellation_reason_code || '';
    const billable = r.is_cancellation_billable;

    let detail = 'Annulée';
    if (byLabel) detail += ` par ${byLabel}`;
    if (reasonLabel) detail += ` — ${reasonLabel}`;
    if (billable === true) detail += ' (facturée)';
    else if (billable === false) detail += ' (non facturée)';

    events.push({ event: detail, date: r.cancelled_at || r.canceled_at, type: 'cancel' });
  }

  return events.sort((a, b) => new Date(b.date) - new Date(a.date));
};

const parseDate = (r) => {
  const raw = r?.scheduled_time || r?.scheduled_date || r?.date;
  if (!raw) return '';
  const dt = new Date(raw);
  if (isNaN(dt.getTime())) return raw.slice(0, 10);
  const pad = (n) => String(n).padStart(2, '0');
  return `${dt.getFullYear()}-${pad(dt.getMonth() + 1)}-${pad(dt.getDate())}`;
};

const parseTime = (r) => {
  const raw = r?.scheduled_time || r?.scheduled_date || r?.date;
  if (!raw) return '';
  const dt = new Date(raw);
  if (isNaN(dt.getTime())) return '';
  const pad = (n) => String(n).padStart(2, '0');
  return `${pad(dt.getHours())}:${pad(dt.getMinutes())}`;
};

function isBillingLockedForAdjust(r) {
  if (!r) return true;
  if (r.billing_locked_at) return true;
  if (r.invoice_line_id != null && r.invoice_line_id !== '') return true;
  return false;
}

/** Statut de réservation fiable (liste dispatch / appels API variés : string, { value }, nombres). */
function getReservationStatusKey(res) {
  if (res == null) return 'unknown';
  const raw = res.status ?? res.booking_status;
  if (raw == null || raw === '') return 'unknown';
  if (typeof raw === 'string') {
    return raw.trim().toLowerCase();
  }
  if (typeof raw === 'object' && raw.value != null) {
    return String(raw.value).trim().toLowerCase();
  }
  return String(raw).trim().toLowerCase();
}

/**
 * Ligne dispatch parfois sans `status` explicite : fallback `scheduled` alors qu’un chauffeur est déjà là.
 * On aligne l’UI (badge + clôture) sur « assignée » dans ce cas.
 */
function inferTripStatusForUi(res) {
  if (!res) return 'unknown';
  const s0 = getReservationStatusKey(res);
  if (['unknown', 'scheduled', ''].includes(s0) && (res.driver_id || res.assignment?.driver_id) && !res.completed_at) {
    return 'assigned';
  }
  return s0;
}

/**
 * Ajustement facturation (patient / clinique) : uniquement saisies manuelles **entreprise** (dispatch),
 * pas invité (Lirie, déjà payé) ni portail client / institution / API.
 * `created_via` : voir `BookingCreatedVia` côté API (serialize).
 */
function allowBillingAdjustByCreatedVia(res) {
  if (res == null) return false;
  const v = String(res.created_via ?? 'legacy').toLowerCase();
  return v === 'dispatcher' || v === 'legacy';
}

const ReservationDetailPanel = ({ reservation, onClose, onSave, onDelete, onReservationUpdated }) => {
  const [vouchers, setVouchers] = useState([]);
  const [loadingVouchers, setLoadingVouchers] = useState(false);
  const [searchParams, setSearchParams] = useSearchParams();
  const lastVoucherErrorRef = useRef(null);
  const changeRequestBannerRef = useRef(null);
  const companySocket = useCompanySocket();

  const [editing, setEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState(null);
  const [form, setForm] = useState({});

  const [billingForm, setBillingForm] = useState({
    amount: '',
    billed_to_type: 'patient',
    billed_to_company_id: '',
    override_reason: '',
  });
  const [savingBilling, setSavingBilling] = useState(false);
  const [billingError, setBillingError] = useState(null);
  const [completeReason, setCompleteReason] = useState('');
  const [savingComplete, setSavingComplete] = useState(false);
  /** Cliniques (Company) issues des mappings paramètres facturation — pour liste + sélecteur. */
  const [clinicRoster, setClinicRoster] = useState([]);
  const [clinicRosterLoading, setClinicRosterLoading] = useState(false);

  const isDateTimeInPast = useMemo(() => {
    if (!form.scheduled_date || !form.scheduled_time) return false;
    // 00:00 = sentinelle "heure à définir", ne pas considérer comme "dans le passé"
    if (form.scheduled_time === "00:00") return false;
    const combined = new Date(`${form.scheduled_date}T${form.scheduled_time}:00`);
    return !isNaN(combined.getTime()) && combined < new Date();
  }, [form.scheduled_date, form.scheduled_time]);

  const buildFormFromReservation = useCallback((r) => {
    if (!r) return {};
    const meta = r.metadata_json || {};
    const routing = meta.routing || {};
    return {
      pickup_location: r.pickup_location || '',
      dropoff_location: r.dropoff_location || '',
      scheduled_date: parseDate(r),
      scheduled_time: parseTime(r),
      amount: r.amount ?? '',
      // Accès départ
      pickup_access_notes: r.pickup_access_notes || '',
      pickup_floor: routing.pickup_floor || r.client?.floor || '',
      pickup_door_code: routing.pickup_door_code || r.client?.door_code || '',
      pickup_entry_point: routing.pickup_entry_point || '',
      // Arrivée — établissement / service / médecin
      dropoff_access_notes: r.dropoff_access_notes || '',
      medical_facility: (r.medical_facility && r.medical_facility !== 'Non spécifié') ? r.medical_facility : '',
      hospital_service: (r.hospital_service && r.hospital_service !== 'Non spécifié') ? r.hospital_service : '',
      doctor_name: (r.doctor_name && r.doctor_name !== 'Non spécifié') ? r.doctor_name : '',
      // Patient & contact
      phone: r.phone || r.client?.contact_phone || r.client?.phone || '',
      notes_medical: (r.notes_medical && r.notes_medical !== 'Aucune note') ? r.notes_medical : '',
      instructions: r.instructions || '',
      external_reference: meta.external_reference || r.external_reference || '',
    };
  }, []);

  useEffect(() => {
    if (reservation) {
      setForm(buildFormFromReservation(reservation));
      setEditing(false);
      const btype = String(
        reservation.billed_to_type || reservation.billing?.billed_to_type || 'patient',
      ).toLowerCase();
      setBillingForm({
        amount: reservation.amount != null && reservation.amount !== '' ? String(reservation.amount) : '',
        billed_to_type: ['patient', 'clinic', 'insurance'].includes(btype) ? btype : 'patient',
        billed_to_company_id:
          reservation.billed_to_company_id != null && reservation.billed_to_company_id !== ''
            ? String(reservation.billed_to_company_id)
            : '',
        override_reason: '',
      });
      setBillingError(null);
      setCompleteReason('');
    }
  }, [reservation, buildFormFromReservation]);

  const returnTo = useMemo(() => {
    const raw = searchParams.get('returnTo');
    if (!raw) return null;
    try {
      const decoded = decodeURIComponent(raw);
      if (decoded.startsWith('/dashboard/company/') || decoded.startsWith('/company/')) return decoded;
      return null;
    } catch { return null; }
  }, [searchParams]);

  useEffect(() => {
    if (searchParams.get('focus') !== 'change_request') return undefined;
    if (reservation?.active_change_request?.status !== 'pending') return undefined;

    const timer = window.setTimeout(() => {
      changeRequestBannerRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      setSearchParams((prev) => {
        const next = new URLSearchParams(prev);
        next.delete('focus');
        return next;
      }, { replace: true });
    }, 350);

    return () => window.clearTimeout(timer);
  }, [
    reservation?.id,
    reservation?.active_change_request?.status,
    searchParams,
    setSearchParams,
  ]);

  const loadVouchers = useCallback(async () => {
    if (!reservation?.id) return;
    try {
      setLoadingVouchers(true);
      const response = await fetchTransportVouchers({ booking_id: reservation.id });
      setVouchers(response?.data || []);
    } catch (e) {
      const errorKey = `${reservation.id}:${e?.response?.status || e?.message || 'unknown'}`;
      if (lastVoucherErrorRef.current !== errorKey) {
        lastVoucherErrorRef.current = errorKey;
      }
    } finally {
      setLoadingVouchers(false);
    }
  }, [reservation?.id]);

  useEffect(() => {
    if (reservation?.id) loadVouchers();
  }, [reservation?.id, loadVouchers]);

  const [institutionChangeEvents, setInstitutionChangeEvents] = useState([]);
  const [respondingChange, setRespondingChange] = useState(false);

  useEffect(() => {
    if (!reservation?.id) {
      setInstitutionChangeEvents([]);
      return;
    }
    const meta = reservation.metadata_json || {};
    const fromInstitution = !!meta.institution_id || !!reservation.institution_timeline;
    if (!fromInstitution) {
      setInstitutionChangeEvents([]);
      return;
    }
    let cancelled = false;
    fetchBookingChangeEvents(reservation.id)
      .then((data) => {
        if (!cancelled) {
          setInstitutionChangeEvents(
            (data?.events || []).filter((ev) => ev.source === 'institution_portal'),
          );
        }
      })
      .catch(() => {
        if (!cancelled) setInstitutionChangeEvents([]);
      });
    return () => { cancelled = true; };
  }, [reservation?.id, reservation?.institution_timeline, reservation?.metadata_json]);

  const allowBillingAdjustByOrigin = useMemo(
    () => (reservation ? allowBillingAdjustByCreatedVia(reservation) : false),
    [reservation],
  );
  /** Vrai = masquer l’ajustement (invité, portail, etc.) — alias de `!allowBillingAdjustByOrigin`. */
  const hidePortalPatientPreRideAdjust = !allowBillingAdjustByOrigin;

  useEffect(() => {
    if (!reservation?.id) return;
    if (hidePortalPatientPreRideAdjust) {
      setClinicRoster([]);
      setClinicRosterLoading(false);
      return;
    }
    let cancelled = false;
    (async () => {
      setClinicRosterLoading(true);
      try {
        const res = await fetchClinicBillingMappings();
        if (cancelled) return;
        const rows = Array.isArray(res?.data) ? res.data : [];
        const byId = new Map();
        for (const m of rows) {
          if (m && m.is_active === false) continue;
          const id = m.clinic_company_id;
          if (id == null) continue;
          const party = m.billing_party_name && String(m.billing_party_name).trim();
          const clinicN = m.clinic_company_name && String(m.clinic_company_name).trim();
          const label = party || clinicN || 'Clinique';
          if (!byId.has(id)) {
            const pr = m.preferential_rate_chf;
            const preferentialRateChf =
              pr != null && pr !== '' && Number.isFinite(Number(pr)) ? Number(pr) : null;
            byId.set(id, {
              id,
              name: label,
              preferential_rate_chf: preferentialRateChf,
            });
          }
        }
        setClinicRoster(
          [...byId.values()].sort((a, b) => a.name.localeCompare(b.name, 'fr-CH')),
        );
      } catch {
        if (!cancelled) setClinicRoster([]);
      } finally {
        if (!cancelled) setClinicRosterLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [reservation?.id, hidePortalPatientPreRideAdjust]);

  /** Noms de cliniques pour liste + <select> (jamais d’ID affiché) : mappings + clinique liée à la réservation. */
  const clinicSelectOptions = useMemo(() => {
    const byId = new Map();
    for (const c of clinicRoster) {
      byId.set(Number(c.id), c.name);
    }
    const add = (id, name) => {
      const n = Number(id);
      if (!Number.isFinite(n) || n < 1) return;
      const label = (name && String(name).trim()) || 'Clinique';
      if (!byId.has(n)) byId.set(n, label);
    };
    const bc = reservation?.billing?.billed_to_company;
    const instLabel = (() => {
      const m = reservation?.medical_facility && String(reservation.medical_facility).trim();
      if (m && m !== 'Non spécifié') return m;
      return null;
    })();
    if (bc && bc.id != null) {
      add(bc.id, instLabel || bc.name);
    }
    const btype = String(
      reservation?.billed_to_type || reservation?.billing?.billed_to_type || '',
    ).toLowerCase();
    if (btype === 'clinic' && reservation?.billed_to_company_id != null) {
      add(reservation.billed_to_company_id, instLabel || bc?.name);
    }
    if (String(billingForm?.billed_to_type || '').toLowerCase() === 'clinic' && billingForm?.billed_to_company_id) {
      const fid = Number(billingForm.billed_to_company_id);
      const nameForForm =
        bc && Number(bc.id) === fid ? (instLabel || bc.name) : undefined;
      add(billingForm.billed_to_company_id, nameForForm);
    }
    return [...byId.entries()]
      .map(([id, name]) => ({ id, name }))
      .sort((a, b) => a.name.localeCompare(b.name, 'fr-CH'));
  }, [clinicRoster, reservation, billingForm.billed_to_type, billingForm.billed_to_company_id]);

  /** Tarif préf. clinique (CHF) : company clinique, sinon client institution, sinon réservation. */
  const getPreferentialRateChfForClinicCompany = useCallback(
    (clinicCompanyId) => {
      const n = Number(clinicCompanyId);
      if (!Number.isFinite(n) || n < 1) return null;
      const fromRoster = clinicRoster.find((c) => Number(c.id) === n);
      if (
        fromRoster
        && fromRoster.preferential_rate_chf != null
        && Number.isFinite(Number(fromRoster.preferential_rate_chf))
      ) {
        return Number(fromRoster.preferential_rate_chf);
      }
      const bc = reservation?.billing?.billed_to_company;
      if (bc
        && Number(bc.id) === n
        && bc.preferential_rate != null
        && Number.isFinite(Number(bc.preferential_rate))) {
        return Number(bc.preferential_rate);
      }
      return null;
    },
    [clinicRoster, reservation?.billing?.billed_to_company],
  );

  const handleChange = (field, value) => {
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  const handleSave = async () => {
    if (!onSave || !reservation?.id) return;
    if (isDateTimeInPast) {
      setSaveError("La date et l'heure de la course sont dans le passé. Ajustez l'horaire pour enregistrer.");
      return;
    }
    try {
      setSaving(true);
      setSaveError(null);
      // Combiner date + heure en ISO local pour le backend (pas de conversion UTC)
      const payload = { ...form };
      if (payload.scheduled_date && payload.scheduled_time) {
        payload.scheduled_time = `${payload.scheduled_date}T${payload.scheduled_time}:00`;
      }
      delete payload.scheduled_date;
      await onSave(reservation.id, payload);
      setEditing(false);
    } catch (err) {
      console.error('Erreur lors de la sauvegarde:', err);
      const errData = err?.response?.data;
      const message = errData?.message || errData?.error || 'Erreur lors de la sauvegarde';
      setSaveError(message);
    } finally {
      setSaving(false);
    }
  };

  const handleCancel = () => {
    setForm(buildFormFromReservation(reservation));
    setEditing(false);
    setSaveError(null);
  };

  const handleManualComplete = async () => {
    if (!reservation?.id) return;
    const st = inferTripStatusForUi(reservation);
    if (st === 'en_route' && !completeReason.trim()) {
      toast.error('Motif obligatoire pour clôturer une course en route.');
      return;
    }
    try {
      setSavingComplete(true);
      const data = await completeReservation(
        reservation.id,
        st === 'en_route' ? { reason: completeReason } : null,
      );
      toast.success('Course clôturée');
      onReservationUpdated?.(data?.reservation || data);
    } catch (e) {
      const err = e?.response?.data || e;
      const msg = err?.error || err?.message || (typeof err === 'string' ? err : 'Échec de la clôture');
      toast.error(String(msg));
    } finally {
      setSavingComplete(false);
    }
  };

  const handleBillingAdjust = async () => {
    if (!reservation?.id) return;
    const v = (billingForm.override_reason || '').trim();
    if (!v) {
      setBillingError('Le motif est obligatoire pour tout ajustement de facturation.');
      return;
    }
    setBillingError(null);
    const amt = parseFloat(String(billingForm.amount).replace(',', '.'), 10);
    if (Number.isNaN(amt) || amt < 0 || (amt > 0 && amt < 0.5)) {
      setBillingError('Montant invalide (0 ou ≥ 0,50 CHF).');
      return;
    }
    const btype = billingForm.billed_to_type;
    let bcomp = null;
    if (btype !== 'patient') {
      const n = parseInt(String(billingForm.billed_to_company_id).trim(), 10);
      if (!Number.isFinite(n) || n <= 0) {
        setBillingError('Identifiant entreprise (tiers payeur) obligatoire pour clinique / assurance.');
        return;
      }
      bcomp = n;
    }
    try {
      setSavingBilling(true);
      const payload = {
        override_reason: v,
        amount: amt,
        billed_to_type: btype,
        billed_to_company_id: btype === 'patient' ? null : bcomp,
      };
      const data = await patchBillingAdjustment(reservation.id, payload);
      toast.success('Facturation mise à jour');
      onReservationUpdated?.(data?.reservation || data);
    } catch (e) {
      const err = e?.response?.data || e;
      const msg = err?.error || err?.message || 'Échec de la mise à jour';
      setBillingError(String(msg));
      toast.error(String(msg));
    } finally {
      setSavingBilling(false);
    }
  };

  if (!reservation) return null;

  const displayTripStatus = inferTripStatusForUi(reservation);
  const status = displayTripStatus;
  const statusInfo = STATUS_MAP[status] || { label: status, css: 'statusPending' };
  const meta = reservation.metadata_json || {};
  const billedToType = reservation.billed_to_type
    || reservation.billing?.billed_to_type
    || meta.billing_resolution_intent
    || null;
  const billingIntent = meta.billing_resolution_intent || billedToType;
  const billingStatusVal = meta.billing_resolution_status;
  const isFailed = billingStatusVal && billingStatusVal.startsWith('failed');
  const isInstitutionBooking = !!meta.institution_id || !!reservation.institution_timeline;
  const isTransferredBooking = !!reservation.is_transferred || !!reservation.active_transfer;
  const isDirectPortalClientBooking =
    !!reservation.client_id && !isInstitutionBooking && !isTransferredBooking;
  const chatBookingId = reservation.is_return && reservation.parent_booking_id
    ? reservation.parent_booking_id
    : reservation.id;
  const isMaterialDelivery = reservation.mission_type === 'material_delivery';

  const originalAmount = reservation?.amount_original ?? reservation?.original_amount ?? reservation?.requested_amount;
  const adjustedDelta = Number.isFinite(Number(originalAmount))
    ? Number(reservation?.amount ?? 0) - Number(originalAmount) : null;

  const bookingIdentity = buildIdentityFromApi(reservation);
  const sourceMeta = getBookingSourceMeta(bookingIdentity.source?.type);
  const passengerBirthDate = isInstitutionBooking
    ? (bookingIdentity.passenger?.birth_date
      || reservation.passenger?.birth_date
      || null)
    : (reservation.client?.birth_date || bookingIdentity.passenger?.birth_date || null);

  const resolveClinicalLabel = (value) => {
    if (!value || value === 'Non spécifié') return null;
    return value;
  };
  const legClinical = reservation.institution_leg || null;
  const arrivalEstablishment = resolveClinicalLabel(reservation.medical_facility)
    || resolveClinicalLabel(legClinical?.establishment);
  const arrivalService = resolveClinicalLabel(reservation.hospital_service)
    || resolveClinicalLabel(legClinical?.service);
  const arrivalDoctor = resolveClinicalLabel(reservation.doctor_name)
    || resolveClinicalLabel(legClinical?.doctor);
  const arrivalClinicalLine = [arrivalEstablishment, arrivalService, arrivalDoctor]
    .filter(Boolean)
    .join(' · ');
  const arrivalAppointmentLabel = (() => {
    if (!legClinical?.appointment_time) return null;
    const d = new Date(legClinical.appointment_time);
    if (Number.isNaN(d.getTime())) return null;
    const pad = (n) => String(n).padStart(2, '0');
    return `RDV ${pad(d.getHours())}:${pad(d.getMinutes())}`;
  })();

  return (
    <div className={s.panel} data-tour-id="ReservationDetailPanel_panel">
      {/* Header */}
      <div className={s.panelHeader}>
        <div className={s.panelTitleRow}>
          <span className={s.panelTitle}>Réservation #{reservation.id}</span>
          <span className={`${s.statusBadge} ${s[statusInfo.css]}`}>{statusInfo.label}</span>
        </div>
        <div className={s.headerActions}>
          {!editing && onSave && (
            <button className={s.editBtn} onClick={() => setEditing(true)} aria-label="Éditer" title="Éditer">
              <FiEdit2 size={14} />
            </button>
          )}
          <button className={s.closeBtn} onClick={onClose} aria-label="Fermer">
            <FiX size={16} />
          </button>
        </div>
      </div>

      {/* Scrollable body */}
      <div className={s.panelBody}>

        {reservation.active_change_request?.status === 'pending' && isInstitutionBooking && (
          <div
            ref={changeRequestBannerRef}
            id="company-change-request-validation"
            style={{
              marginBottom: 12,
              padding: '10px 12px',
              borderRadius: 8,
              border: '1px solid #fcd34d',
              background: '#fffbeb',
            }}
          >
            <p style={{ margin: '0 0 8px', fontSize: 13, fontWeight: 600, color: '#92400e' }}>
              Modification institution — validation requise
            </p>
            <p style={{ margin: '0 0 10px', fontSize: 12, color: '#78350f' }}>
              {reservation.active_change_request.reason || 'Champs modifiés en attente de votre accord.'}
            </p>
            <div style={{ display: 'flex', gap: 8 }}>
              <button
                type="button"
                disabled={respondingChange}
                onClick={async () => {
                  setRespondingChange(true);
                  try {
                    await respondToChangeRequest(
                      reservation.id,
                      reservation.active_change_request.id,
                      'accept'
                    );
                    toast.success('Modification acceptée');
                    onReservationUpdated?.();
                  } catch (e) {
                    toast.error(e?.response?.data?.error || 'Échec');
                  } finally {
                    setRespondingChange(false);
                  }
                }}
                style={{
                  padding: '6px 12px',
                  borderRadius: 6,
                  border: 'none',
                  background: '#059669',
                  color: '#fff',
                  fontSize: 12,
                  fontWeight: 600,
                  cursor: respondingChange ? 'default' : 'pointer',
                }}
              >
                Accepter
              </button>
              <button
                type="button"
                disabled={respondingChange}
                onClick={async () => {
                  setRespondingChange(true);
                  try {
                    await respondToChangeRequest(
                      reservation.id,
                      reservation.active_change_request.id,
                      'refuse'
                    );
                    toast.success('Modification refusée');
                    onReservationUpdated?.();
                  } catch (e) {
                    toast.error(e?.response?.data?.error || 'Échec');
                  } finally {
                    setRespondingChange(false);
                  }
                }}
                style={{
                  padding: '6px 12px',
                  borderRadius: 6,
                  border: '1px solid #cbd5e1',
                  background: '#fff',
                  color: '#334155',
                  fontSize: 12,
                  fontWeight: 600,
                  cursor: respondingChange ? 'default' : 'pointer',
                }}
              >
                Refuser
              </button>
            </div>
          </div>
        )}

        {reservation.route_group_id && (
          <p style={{ margin: '0 0 10px', fontSize: 12, color: '#475569' }}>
            Parcours multi-étapes — acceptation globale
            {reservation.route_sequence_number
              ? ` · étape ${reservation.route_sequence_number}`
              : ''}
            {reservation.route_group_id
              ? ` · #${String(reservation.route_group_id).slice(-6)}`
              : ''}
          </p>
        )}

        {/* ── EDIT MODE ── */}
        {editing ? (
          <div className={s.editForm}>

            {/* Context badge — client + id */}
            <div className={s.editContext}>
              <span className={s.editContextLabel}>
                {bookingIdentity.passengerLabel}
              </span>
              <span className={s.editContextSep} />
              <span className={s.editContextMeta}>#{reservation.id}</span>
            </div>

            {/* ─── Itinéraire ─── */}
            <div className={s.editGroup}>
              <div className={s.editGroupTitle}>
                <FiMapPin size={12} className={s.editGroupIcon} />
                Itinéraire
              </div>

              {/* Route visuelle : dot → ligne → dot */}
              <div className={s.editRoute}>
                <div className={s.editRouteTrack}>
                  <span className={s.editRouteDotA} />
                  <span className={s.editRouteLine} />
                  <span className={s.editRouteDotB} />
                </div>
                <div className={s.editRouteFields}>
                  <div className={s.editInputWrap}>
                    <AddressAutocomplete
                      value={form.pickup_location}
                      onChange={(e) => handleChange('pickup_location', e?.target?.value ?? e)}
                      placeholder="Départ"
                    />
                  </div>
                  <div className={s.editInputWrap}>
                    <AddressAutocomplete
                      value={form.dropoff_location}
                      onChange={(e) => handleChange('dropoff_location', e?.target?.value ?? e)}
                      placeholder="Destination"
                    />
                  </div>
                </div>
              </div>
            </div>

            <div className={s.editDivider} />

            {(() => {
              // Sens du trajet : aller (domicile → hôpital) ou retour (hôpital → domicile).
              // Les champs « domicile » (étage / code porte / accueil + consignes) restent
              // bindés sur pickup_* / pickup_access_notes côté formulaire car ce sont les
              // données stockées sur le client/booking. On inverse uniquement l'ordre
              // d'affichage et les libellés pour refléter le sens réel du trajet.
              const isReturnTrip = !!(reservation?.is_return);
              const homeAccessNotesField = isReturnTrip ? 'dropoff_access_notes' : 'pickup_access_notes';
              const hospitalAccessNotesField = isReturnTrip ? 'pickup_access_notes' : 'dropoff_access_notes';

              const homeBlock = (
                <div className={s.editGroup} key="home-access">
                  <div className={s.editGroupTitle}>
                    <FiMapPin size={12} className={s.editGroupIcon} />
                    {isReturnTrip ? 'Accès domicile · Arrivée' : 'Accès domicile · Départ'}
                  </div>
                  <div className={s.editRowTriple}>
                    <input type="text" className={s.editInput} value={form.pickup_floor}
                      onChange={(e) => handleChange('pickup_floor', e.target.value)} placeholder="Étage" />
                    <input type="text" className={s.editInput} value={form.pickup_door_code}
                      onChange={(e) => handleChange('pickup_door_code', e.target.value)} placeholder="Code" />
                    <input type="text" className={s.editInput} value={form.pickup_entry_point}
                      onChange={(e) => handleChange('pickup_entry_point', e.target.value)} placeholder="Accueil" />
                  </div>
                  <input type="text" className={s.editInput} value={form[homeAccessNotesField]}
                    onChange={(e) => handleChange(homeAccessNotesField, e.target.value)}
                    placeholder={isReturnTrip ? 'Consignes arrivée' : 'Consignes départ'}
                    style={{ marginTop: 6 }} />
                </div>
              );

              const hospitalBlock = (
                <div className={s.editGroup} key="hospital-access">
                  <div className={s.editGroupTitle}>
                    <FiHome size={12} className={s.editGroupIcon} />
                    {isReturnTrip ? 'Lieu de départ · Hôpital' : 'Destination · Hôpital'}
                  </div>
                  <input type="text" className={s.editInput} value={form.medical_facility}
                    onChange={(e) => handleChange('medical_facility', e.target.value)}
                    placeholder="Établissement / lieu" />
                  <div className={s.editRow} style={{ marginTop: 8 }}>
                    <input type="text" className={s.editInput} value={form.hospital_service}
                      onChange={(e) => handleChange('hospital_service', e.target.value)}
                      placeholder="Service" />
                    <input type="text" className={s.editInput} value={form.doctor_name}
                      onChange={(e) => handleChange('doctor_name', e.target.value)}
                      placeholder="Médecin" />
                  </div>
                  <input type="text" className={s.editInput} value={form[hospitalAccessNotesField]}
                    onChange={(e) => handleChange(hospitalAccessNotesField, e.target.value)}
                    placeholder={isReturnTrip ? 'Consignes départ' : 'Consignes arrivée'}
                    style={{ marginTop: 8 }} />
                </div>
              );

              return (
                <>
                  {isReturnTrip ? hospitalBlock : homeBlock}
                  <div className={s.editDivider} />
                  {isReturnTrip ? homeBlock : hospitalBlock}
                </>
              );
            })()}

            <div className={s.editDivider} />

            {/* ─── Horaire ─── */}
            <div className={s.editGroup}>
              <div className={s.editGroupTitle}>
                <FiClock size={12} className={s.editGroupIcon} />
                Horaire
              </div>
              <div className={s.editRow}>
                <div className={s.editField}>
                  <label className={`${s.editLabel} ${isDateTimeInPast ? s.fieldErrorLabel : ''}`}>Date</label>
                  <InlineDatePicker
                    value={form.scheduled_date}
                    onChange={(v) => handleChange('scheduled_date', v)}
                    placeholder="Date"
                  />
                </div>
                <div className={s.editField}>
                  <label className={`${s.editLabel} ${isDateTimeInPast ? s.fieldErrorLabel : ''}`}>Heure</label>
                  <InlineTimePicker
                    value={form.scheduled_time}
                    onChange={(v) => handleChange('scheduled_time', v)}
                    className={isDateTimeInPast ? s.fieldErrorInput : ''}
                  />
                </div>
              </div>
              {isDateTimeInPast && (
                <div className={s.fieldErrorHint}>La date et l'heure sont dans le passé</div>
              )}
            </div>

            <div className={s.editDivider} />

            {/* ─── Montant ─── */}
            <div className={s.editGroup}>
              <div className={s.editGroupTitle}>
                <FiFileText size={12} className={s.editGroupIcon} />
                Montant
              </div>
              <input type="number" step="0.01" className={s.editInput} value={form.amount}
                onChange={(e) => handleChange('amount', e.target.value)} placeholder="CHF" />
            </div>

            <div className={s.editDivider} />

            {/* ─── Patient & notes ─── */}
            {!isMaterialDelivery ? (
              <div className={s.editGroup}>
                <div className={s.editGroupTitle}>
                  <FiUser size={12} className={s.editGroupIcon} />
                  Patient
                </div>
                <div className={s.editRow}>
                  <input
                    type="tel"
                    id={`reservation-${reservation.id}-phone`}
                    name="patient_phone"
                    autoComplete="tel"
                    aria-label="Téléphone du patient"
                    className={s.editInput}
                    value={form.phone}
                    onChange={(e) => handleChange('phone', e.target.value)}
                    placeholder="Téléphone"
                  />
                  <input
                    type="text"
                    id={`reservation-${reservation.id}-external-reference`}
                    name="external_reference"
                    autoComplete="off"
                    aria-label="Référence DPI"
                    className={s.editInput}
                    value={form.external_reference}
                    onChange={(e) => handleChange('external_reference', e.target.value)}
                    placeholder="Ref. DPI"
                  />
                </div>
                <textarea
                  id={`reservation-${reservation.id}-notes-medical`}
                  name="notes_medical"
                  aria-label="Notes médicales"
                  className={s.editTextarea}
                  value={form.notes_medical}
                  onChange={(e) => handleChange('notes_medical', e.target.value)}
                  placeholder="Pathologie, difficultés, mobilité…"
                  rows={2}
                  style={{ marginTop: 8 }}
                />
                <textarea
                  id={`reservation-${reservation.id}-instructions`}
                  name="instructions"
                  aria-label="Instructions chauffeur"
                  className={s.editTextarea}
                  value={form.instructions}
                  onChange={(e) => handleChange('instructions', e.target.value)}
                  placeholder="Instructions chauffeur"
                  rows={2}
                  style={{ marginTop: 8 }}
                />
              </div>
            ) : (
              <div className={s.editGroup}>
                <div className={s.editGroupTitle}>
                  <FiPackage size={12} className={s.editGroupIcon} />
                  Livraison
                </div>
                <textarea className={s.editTextarea} value={form.instructions}
                  onChange={(e) => handleChange('instructions', e.target.value)}
                  placeholder="Description du matériel" rows={3} />
              </div>
            )}

            {/* Erreur de sauvegarde */}
            {saveError && (
              <div className={s.saveErrorBanner}>{saveError}</div>
            )}

            {/* Footer actions — sticky */}
            <div className={s.editFooter}>
              <button type="button" className={s.editCancelBtn} onClick={handleCancel} disabled={saving}>
                Annuler
              </button>
              <button type="button" className={s.editSaveBtn} onClick={handleSave} disabled={saving}>
                {saving ? 'Enregistrement...' : 'Enregistrer'}
              </button>
            </div>
          </div>
        ) : (
          <>
            {/* ── VIEW MODE ── */}

            {/* Informations principales */}
            <div className={s.section}>
              <div className={s.sectionHeader}>
                <div className={`${s.sectionIcon} ${s.sectionIconBrand}`}><FiUser size={13} /></div>
                <h3 className={s.sectionTitle}>Informations</h3>
              </div>
              <div className={s.summaryGrid}>
                <div className={s.summaryItem}>
                  <span className={s.summaryLabel}>Passager</span>
                  <span className={s.summaryValue}>{bookingIdentity.passengerLabel}</span>
                </div>
                {bookingIdentity.source?.name && (
                  <div className={s.summaryItem}>
                    <span className={s.summaryLabel}>Origine</span>
                    <span className={s.summaryValue}>
                      {sourceMeta.label} · {bookingIdentity.source.name}
                      {bookingIdentity.source.code ? ` (${bookingIdentity.source.code})` : ''}
                    </span>
                  </div>
                )}
                {bookingIdentity.requester?.name && (
                  <div className={s.summaryItem}>
                    <span className={s.summaryLabel}>Demandeur</span>
                    <span className={s.summaryValue}>{bookingIdentity.requester.name}</span>
                  </div>
                )}
                {bookingIdentity.ownership?.owner_company_name && (
                  <div className={s.summaryItem}>
                    <span className={s.summaryLabel}>Propriétaire</span>
                    <span className={s.summaryValue}>{bookingIdentity.ownership.owner_company_name}</span>
                  </div>
                )}
                {bookingIdentity.execution?.executing_company_name && (
                  <div className={s.summaryItem}>
                    <span className={s.summaryLabel}>Exécutant</span>
                    <span className={s.summaryValue}>{bookingIdentity.execution.executing_company_name}</span>
                  </div>
                )}
                {bookingIdentity.upstream?.name && (
                  <div className={s.summaryItem}>
                    <span className={s.summaryLabel}>Source amont</span>
                    <span className={s.summaryValue}>
                      {bookingIdentity.upstream.name}
                      {bookingIdentity.upstream.code ? ` (${bookingIdentity.upstream.code})` : ''}
                    </span>
                  </div>
                )}
                <div className={s.summaryItem}>
                  <span className={s.summaryLabel}>Horaire</span>
                  <span className={s.summaryValue}>
                    {reservation.time_confirmed === false
                      ? formatLegTime({ scheduled_time: reservation.scheduled_time, time_confirmed: false })
                      : renderBookingDateTime(reservation)}
                  </span>
                </div>
                <div className={s.summaryItem}>
                  <span className={s.summaryLabel}>Montant</span>
                  <span className={s.summaryValue}>{formatCurrency(reservation.amount)}</span>
                </div>
                {reservation.driver_name && (
                  <div className={s.summaryItem}>
                    <span className={s.summaryLabel}>Chauffeur</span>
                    <span className={s.summaryValue}>{reservation.driver_name}</span>
                  </div>
                )}
                {(reservation.client?.contact_phone || reservation.client?.phone) && !isInstitutionBooking && (
                  <div className={s.summaryItem}>
                    <span className={s.summaryLabel}>Téléphone</span>
                    <span className={s.summaryValue}>{reservation.client.contact_phone || reservation.client.phone}</span>
                  </div>
                )}
                {passengerBirthDate && (
                  <div className={s.summaryItem}>
                    <span className={s.summaryLabel}>Date de naissance</span>
                    <span className={s.summaryValue}>
                      {new Date(passengerBirthDate).toLocaleDateString('fr-CH')}
                    </span>
                  </div>
                )}
                {(reservation.passenger?.external_reference || meta.external_reference || reservation.external_reference) && isInstitutionBooking && (
                  <div className={s.summaryItem}>
                    <span className={s.summaryLabel}>Réf. patient</span>
                    <span className={s.summaryValue}>
                      {reservation.passenger?.external_reference || meta.external_reference || reservation.external_reference}
                    </span>
                  </div>
                )}
              </div>
              {Number.isFinite(Number(originalAmount)) && Number.isFinite(Number(adjustedDelta)) && Math.abs(adjustedDelta) >= 0.01 && (
                <div className={s.adjustedNote}>
                  Montant saisi : {formatCurrency(originalAmount)} — Ajusté : {adjustedDelta >= 0 ? '+' : '-'}{formatCurrency(Math.abs(adjustedDelta))}
                </div>
              )}
              {reservation.active_transfer && (
                <div className={s.adjustedNote}>
                  Course transférée ({reservation.active_transfer.status}) — titulaire : {reservation.active_transfer.owner_company_name || reservation.active_transfer.owner_company_id} / exécutant : {reservation.active_transfer.executing_company_name || reservation.active_transfer.executing_company_id}
                </div>
              )}
            </div>

            {/* Trajet */}
            <div className={s.section}>
              <div className={s.sectionHeader}>
                <div className={`${s.sectionIcon} ${s.sectionIconBrand}`}><FiMapPin size={13} /></div>
                <h3 className={s.sectionTitle}>Trajet</h3>
              </div>
              <div className={s.route}>
                <div className={s.routeTrack}>
                  <div className={`${s.routeDot} ${s.routeDotStart}`} />
                  <div className={s.routeLine} />
                  <div className={`${s.routeDot} ${s.routeDotEnd}`} />
                </div>
                <div className={s.routeStops}>
                  <div className={s.routeStop}>
                    <div className={s.routeStopLabel}>Départ</div>
                    <div className={s.routeStopAddress}>{reservation.pickup_location || '-'}</div>
                    {/* Access details départ */}
                    {(reservation.client?.floor || reservation.client?.door_code || reservation.client?.access_notes || reservation.pickup_access_notes) && (
                      <div className={s.routeStopMeta}>
                        {reservation.client?.floor && <span>Étage {reservation.client.floor}</span>}
                        {reservation.client?.door_code && <span>Code {reservation.client.door_code}</span>}
                        {(reservation.pickup_access_notes || reservation.client?.access_notes) && (
                          <span>{reservation.pickup_access_notes || reservation.client.access_notes}</span>
                        )}
                      </div>
                    )}
                  </div>
                  <div className={s.routeStop}>
                    <div className={s.routeStopLabel}>
                      Arrivée
                      {arrivalAppointmentLabel && (
                        <span className={s.routeStopTime}> · {arrivalAppointmentLabel}</span>
                      )}
                    </div>
                    <div className={s.routeStopAddress}>{reservation.dropoff_location || '-'}</div>
                    {arrivalClinicalLine && (
                      <div className={s.routeStopDetails}>{arrivalClinicalLine}</div>
                    )}
                    {reservation.dropoff_access_notes && (
                      <div className={s.routeStopMeta}>
                        <span>{reservation.dropoff_access_notes}</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
              {reservation.is_return && (
                <div className={s.roundTripBadge}>Aller-retour</div>
              )}
            </div>

            {/* Destination — établissement, service, médecin */}
            {(() => {
              const facility = arrivalEstablishment;
              const service = arrivalService;
              const doctor = arrivalDoctor;
              if (!facility && !service && !doctor) return null;
              return (
                <div className={s.section}>
                  <div className={s.sectionHeader}>
                    <div className={`${s.sectionIcon} ${s.sectionIconBlue}`}><FiHome size={13} /></div>
                    <h3 className={s.sectionTitle}>Destination</h3>
                  </div>
                  <div className={s.detailGrid}>
                    {facility && (
                      <div className={s.detailItem}>
                        <span className={s.detailLabel}>Établissement</span>
                        <span className={s.detailValue}>{facility}</span>
                      </div>
                    )}
                    {service && (
                      <div className={s.detailItem}>
                        <span className={s.detailLabel}>Service</span>
                        <span className={s.detailValue}>{service}</span>
                      </div>
                    )}
                    {doctor && (
                      <div className={s.detailItem}>
                        <span className={s.detailLabel}>Médecin</span>
                        <span className={s.detailValue}>{doctor}</span>
                      </div>
                    )}
                  </div>
                </div>
              );
            })()}

            {/* Patient — pathologie, notes, ref DPI */}
            {(() => {
              const notesMed = reservation.notes_medical && reservation.notes_medical !== 'Aucune note' ? reservation.notes_medical : null;
              const instructions = reservation.instructions || null;
              const phone = reservation.phone || null;
              const extRef = (meta.external_reference || reservation.external_reference) || null;
              const wheelchair = reservation.wheelchair_need || reservation.wheelchair_client_has;
              if (!notesMed && !instructions && !phone && !extRef && !wheelchair) return null;
              return (
                <div className={s.section}>
                  <div className={s.sectionHeader}>
                    <div className={`${s.sectionIcon} ${s.sectionIconWarning}`}><FiInfo size={13} /></div>
                    <h3 className={s.sectionTitle}>Patient et notes</h3>
                  </div>
                  {phone && (
                    <div className={s.infoRow}>
                      <span className={s.infoLabel}><FiPhone size={11} /> Téléphone</span>
                      <span className={s.infoValue}>{phone}</span>
                    </div>
                  )}
                  {extRef && (
                    <div className={s.infoRow}>
                      <span className={s.infoLabel}><FiFileText size={11} /> Ref. DPI</span>
                      <span className={s.infoValue}>{extRef}</span>
                    </div>
                  )}
                  {wheelchair && (
                    <div className={s.infoRow}>
                      <span className={s.infoLabel}><FiTruck size={11} /> Mobilité</span>
                      <span className={s.infoValue}>
                        {reservation.wheelchair_need ? 'Fauteuil requis' : 'Fauteuil client'}
                      </span>
                    </div>
                  )}
                  {notesMed && (
                    <div className={s.notesBlock}>{notesMed}</div>
                  )}
                  {instructions && (
                    <div className={s.notesBlockMuted}>{instructions}</div>
                  )}
                </div>
              );
            })()}

            {/* Facturation (lecture + ajustement dans la même section) */}
            {(() => {
              const stLower = inferTripStatusForUi(reservation);
              const canAdjustBilling = !editing
                && !['canceled', 'cancelled', 'rejected', 'no_show'].includes(stLower)
                && !isBillingLockedForAdjust(reservation)
                && !hidePortalPatientPreRideAdjust;
              const resolvedIntent = billingIntent || billedToType || 'patient';
              const intentLabel = INTENT_LABELS[resolvedIntent] || resolvedIntent;
              const BILLED_TO_LABELS = { patient: 'Patient', clinic: 'Clinique', insurance: 'Assurance' };
              const typeLabel = BILLED_TO_LABELS[billingForm.billed_to_type] || billingForm.billed_to_type;
              const clinicForSummary = clinicSelectOptions.find(
                (c) => String(c.id) === String(billingForm.billed_to_company_id),
              );
              const clinicExtra = billingForm.billed_to_type === 'clinic' && clinicForSummary
                ? ` — ${clinicForSummary.name}`
                : '';
              const summaryAmount = Number.isFinite(Number(billingForm.amount))
                ? Number(billingForm.amount)
                : (Number(reservation?.amount) || 0);

              const billingAdjustForm = (
                <div
                  className={isInstitutionBooking ? s.billingAdjustBlock : undefined}
                >
                  {!isInstitutionBooking && (
                    <div className={s.billingCurrentSummary} role="status">
                      <span className={s.billingCurrentAmount}>{formatCurrency(summaryAmount)}</span>
                      <span className={s.billingCurrentSep} aria-hidden>·</span>
                      <span>
                        Destinataire : {typeLabel}
                        {clinicExtra}
                      </span>
                    </div>
                  )}
                  <p className={`${s.billingMuted} ${s.billingHelpText}`}>
                    Ajustez le montant (CHF) et le payeur. Pour « Clinique », choisissez l’établissement. Toute modification exige un bref motif.
                  </p>
                  <div className={s.billingClinicRoster}>
                    <div className={s.billingClinicRosterTitle}>
                      Cliniques / partenaires (mappings)
                    </div>
                    {clinicRosterLoading ? (
                      <p className={s.billingMuted} style={{ fontSize: 12, margin: 0 }}>Chargement…</p>
                    ) : clinicSelectOptions.length === 0 ? (
                      <p className={s.billingMuted} style={{ fontSize: 12, margin: 0 }}>
                        Aucune clinique connue. Configurez les mappings dans Paramètres → Facturation, ou liez la réservation à une clinique (nom d’entreprise) côté course.
                      </p>
                    ) : (
                      <ul className={s.billingClinicRosterList}>
                        {clinicSelectOptions.map((c) => (
                          <li key={c.id}>
                            <span className={s.billingClinicRosterName}>{c.name}</span>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                  {billingError && <div className={s.saveErrorBanner}>{billingError}</div>}
                  <div className={s.editGroup} style={{ marginTop: 0 }}>
                    <label className={s.editLabel} htmlFor="adj-amount">Montant (CHF)</label>
                    <input
                      id="adj-amount"
                      type="number"
                      step="0.01"
                      min="0"
                      className={s.editInput}
                      value={billingForm.amount}
                      onChange={(e) => setBillingForm((prev) => ({ ...prev, amount: e.target.value }))}
                    />
                    <label className={s.editLabel} style={{ marginTop: 8 }} htmlFor="adj-btype">Destinataire</label>
                    <select
                      id="adj-btype"
                      className={s.editInput}
                      value={billingForm.billed_to_type}
                      onChange={(e) => {
                        const btype = e.target.value;
                        setBillingForm((prev) => {
                          const next = { ...prev, billed_to_type: btype };
                          if (btype === 'clinic' && prev.billed_to_company_id) {
                            const r = getPreferentialRateChfForClinicCompany(prev.billed_to_company_id);
                            if (r != null) {
                              next.amount = r.toFixed(2);
                            }
                          }
                          return next;
                        });
                      }}
                    >
                      <option value="patient">Patient</option>
                      <option value="clinic">Clinique</option>
                      <option value="insurance">Assurance</option>
                    </select>
                    {billingForm.billed_to_type === 'clinic' && (
                      <>
                        <label className={s.editLabel} style={{ marginTop: 8 }} htmlFor="adj-bcomp">
                          Clinique cible
                        </label>
                        {clinicRosterLoading ? (
                          <p className={s.billingMuted} style={{ fontSize: 12, margin: '4px 0 0' }}>Chargement des cliniques…</p>
                        ) : clinicSelectOptions.length > 0 ? (
                          <select
                            id="adj-bcomp"
                            className={s.editInput}
                            value={billingForm.billed_to_company_id}
                            onChange={(e) => {
                              const v = e.target.value;
                              setBillingForm((prev) => {
                                const next = { ...prev, billed_to_company_id: v };
                                if (v && String(prev.billed_to_type).toLowerCase() === 'clinic') {
                                  const r = getPreferentialRateChfForClinicCompany(v);
                                  if (r != null) {
                                    next.amount = r.toFixed(2);
                                  }
                                }
                                return next;
                              });
                            }}
                          >
                            <option value="">Sélectionner une clinique…</option>
                            {clinicSelectOptions.map((c) => (
                              <option key={c.id} value={String(c.id)}>{c.name}</option>
                            ))}
                          </select>
                        ) : (
                          <p className={s.billingMuted} id="adj-bcomp" style={{ fontSize: 12, margin: '4px 0 0' }}>
                            Aucune clinique disponible : ajoutez des mappings ou une clinique liée à la course (voir la liste ci-dessus).
                          </p>
                        )}
                      </>
                    )}
                    {billingForm.billed_to_type === 'insurance' && (
                      <>
                        <label className={s.editLabel} style={{ marginTop: 8 }} htmlFor="adj-bins">
                          ID entreprise (assurance)
                        </label>
                        <input
                          id="adj-bins"
                          type="number"
                          min="1"
                          className={s.editInput}
                          value={billingForm.billed_to_company_id}
                          onChange={(e) => setBillingForm((prev) => ({ ...prev, billed_to_company_id: e.target.value }))}
                          placeholder="ID entreprise cible"
                        />
                      </>
                    )}
                    <label className={s.editLabel} style={{ marginTop: 8 }} htmlFor="adj-reason">Motif (obligatoire)</label>
                    <textarea
                      id="adj-reason"
                      className={s.editTextarea}
                      rows={2}
                      value={billingForm.override_reason}
                      onChange={(e) => setBillingForm((prev) => ({ ...prev, override_reason: e.target.value }))}
                    />
                    <button
                      type="button"
                      className={s.editSaveBtn}
                      style={{ marginTop: 12 }}
                      onClick={handleBillingAdjust}
                      disabled={savingBilling}
                    >
                      {savingBilling ? 'Enregistrement…' : 'Enregistrer'}
                    </button>
                  </div>
                </div>
              );

              return (
                <div className={s.section}>
                  <div className={s.sectionHeader}>
                    <div className={`${s.sectionIcon} ${s.sectionIconWarning}`}><FiFileText size={13} /></div>
                    <h3 className={s.sectionTitle}>Facturation</h3>
                  </div>
                  {(() => {
                    const isCancelled = ['canceled', 'cancelled'].includes(status);
                    const billable = reservation.is_cancellation_billable;

                    if (isCancelled && billable === false) {
                      return (
                        <div className={`${s.billingStatus} ${s.billingStatusCancelled}`}>
                          <FiAlertCircle size={13} />
                          Annulée — non facturée
                          {reservation.cancellation_display_label && (
                            <span className={s.billingMuted} style={{ marginLeft: 4 }}>
                              ({reservation.cancellation_display_label})
                            </span>
                          )}
                        </div>
                      );
                    }

                    const instName = meta.institution_name || reservation.institution_timeline?.institution_name;

                    if (isCancelled && billable === true) {
                      const feeAmount = reservation.cancellation_fee_amount;
                      const feePct = reservation.cancellation_fee_percent;
                      return (
                        <>
                          <div className={`${s.billingStatus} ${s.billingStatusDanger}`}>
                            Annulée — facturée
                            <span className={`${s.billingBadge} ${s.billingBadgeDanger}`}>Facturation maintenue</span>
                          </div>
                          {feeAmount != null && feeAmount > 0 && (
                            <p className={s.billingMuted} style={{ fontWeight: 600 }}>
                              Frais d'annulation : {Number(feeAmount).toFixed(2)} CHF
                              {feePct != null ? ` (${feePct}%)` : ''}
                            </p>
                          )}
                          <p className={s.billingMuted}>
                            Facture à : {intentLabel}
                            {instName ? ` — ${instName}` : ''}
                          </p>
                          {reservation.cancellation_display_label && (
                            <p className={s.billingMuted}>Motif : {reservation.cancellation_display_label}</p>
                          )}
                        </>
                      );
                    }

                    if (isInstitutionBooking) {
                      return (
                        <>
                          <div className={`${s.billingStatus} ${isFailed ? s.billingStatusDanger : s.billingStatusSuccess}`}>
                            Facture à : {intentLabel}
                            {billingStatusVal && (
                              <span className={`${s.billingBadge} ${isFailed ? s.billingBadgeDanger : s.billingBadgeSuccess}`}>
                                {isFailed ? 'Action requise' : 'Résolu'}
                              </span>
                            )}
                          </div>
                          {instName && (
                            <p className={s.billingMuted}>Institution : {instName}</p>
                          )}
                          {isFailed && (
                            <div className={s.billingWarning}>
                              <FiAlertCircle size={13} />
                              <span>Informations de facturation incomplètes. Vérifiez le dossier client.</span>
                            </div>
                          )}
                          {canAdjustBilling && billingAdjustForm}
                        </>
                      );
                    }

                    if (canAdjustBilling) {
                      return billingAdjustForm;
                    }

                    return (
                      <div className={`${s.billingStatus} ${s.billingStatusPatient}`}>
                        Facture à : {intentLabel}
                      </div>
                    );
                  })()}
                  {!editing && isBillingLockedForAdjust(reservation) && (
                    <p className={s.billingMuted} style={{ marginTop: 8 }}>
                      Facturation verrouillée : ajustement du montant ou du destinataire impossible.
                    </p>
                  )}
                </div>
              );
            })()}

            {/* Clôture manuelle (entreprise) — hors PUT opérationnel */}
            {(() => {
              const stLower = inferTripStatusForUi(reservation);
              const canManualComplete = !editing
                && ['accepted', 'assigned', 'in_progress', 'en_route'].includes(stLower);
              if (!canManualComplete) return null;
              return (
                <div className={s.section}>
                  <div className={s.sectionHeader}>
                    <div className={`${s.sectionIcon} ${s.sectionIconBrand}`}><FiClock size={13} /></div>
                    <h3 className={s.sectionTitle}>Clôture de la course</h3>
                  </div>
                  <p className={s.billingMuted} style={{ marginBottom: 8 }}>
                    {stLower === 'en_route'
                      ? 'Clôture par l’entreprise si l’app chauffeur ne permet pas de terminer la course. Un motif est obligatoire.'
                      : (stLower === 'accepted' || stLower === 'assigned'
                        ? 'À utiliser notamment quand le trajet s’est déroulé mais la course n’a pas pu être clôturée côté chauffeur (self-service, facturation clinique, etc.) — pas de motif requis, sauf en course « en route » ci-dessus.'
                        : 'Terminer la course depuis l’espace entreprise si besoin.')}
                  </p>
                  {stLower === 'en_route' && (
                    <textarea
                      className={s.editTextarea}
                      rows={2}
                      placeholder="Motif de la clôture manuelle (obligatoire)"
                      value={completeReason}
                      onChange={(e) => setCompleteReason(e.target.value)}
                      style={{ marginBottom: 8 }}
                    />
                  )}
                  <button
                    type="button"
                    className={s.editSaveBtn}
                    onClick={handleManualComplete}
                    disabled={savingComplete}
                  >
                    {savingComplete ? 'Validation…' : 'Valider la course'}
                  </button>
                </div>
              );
            })()}

            {/* Mini-chat (institution ou partenariat) */}
            {(isInstitutionBooking || isTransferredBooking || isDirectPortalClientBooking) && (
              <BookingChat
                bookingId={chatBookingId}
                socket={companySocket}
                closed={['completed', 'return_completed', 'canceled', 'cancelled'].includes(status)}
              />
            )}

            {/* Historique */}
            {(() => {
              const timeline = buildTimeline(reservation);
              const instEvents = institutionChangeEvents.map((ev) => ({
                event: `Modification institution${ev.severity === 'CRITICAL' ? ' (en route)' : ''}${ev.actor_display_name ? ` — ${ev.actor_display_name}` : ''}`,
                date: ev.created_at,
                type: ev.severity === 'CRITICAL' ? 'institution_critical' : 'institution',
                eventId: ev.id,
                ackRequired: ev.ack_required,
                ackCount: ev.ack_received_count,
              }));
              const merged = [...timeline, ...instEvents].sort(
                (a, b) => new Date(b.date) - new Date(a.date),
              );
              if (merged.length === 0) return null;
              return (
                <div className={s.section}>
                  <div className={s.sectionHeader}>
                    <div className={`${s.sectionIcon} ${s.sectionIconMuted}`}><FiClock size={13} /></div>
                    <h3 className={s.sectionTitle}>Historique</h3>
                  </div>
                  <div className={s.timeline}>
                    {merged.map((item, i) => (
                      <div key={item.eventId || i} className={`${s.timelineItem} ${item.type === 'cancel' ? s.timelineItemCancel : ''}`}>
                        <div className={s.timelineEvent}>
                          {item.event}
                          {item.type === 'institution_critical' && (
                            <span className={s.institutionChangeBadge}> Institution</span>
                          )}
                        </div>
                        <div className={s.timelineDate}>
                          {fmtShort(item.date)}
                          {item.ackRequired && (
                            <button
                              type="button"
                              className={s.ackBtn}
                              onClick={async () => {
                                try {
                                  await acknowledgeBookingChangeEvent(reservation.id, item.eventId);
                                  toast.success('Accusé de réception enregistré');
                                  const data = await fetchBookingChangeEvents(reservation.id);
                                  setInstitutionChangeEvents(
                                    (data?.events || []).filter((ev) => ev.source === 'institution_portal'),
                                  );
                                } catch (e) {
                                  toast.error(e?.response?.data?.error || 'Erreur ACK');
                                }
                              }}
                            >
                              {item.ackCount > 0 ? 'Vu' : 'Accuser réception'}
                            </button>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })()}

            {/* Vouchers */}
            {!loadingVouchers && vouchers.length > 0 && (
              <div className={s.section}>
                <div className={s.sectionHeader}>
                  <div className={`${s.sectionIcon} ${s.sectionIconMuted}`}><FiTruck size={13} /></div>
                  <h3 className={s.sectionTitle}>Bons de transport</h3>
                </div>
                <div className={s.voucherList}>
                  {vouchers.map((v) => (
                    <div key={v.id} className={s.voucherCard}>
                      <div className={s.voucherHeader}>
                        <span className={s.voucherTitle}>Bon #{v.id} - {VOUCHER_TYPE_LABELS[v.type] || v.type}</span>
                        <span className={s.voucherStatus}>{VOUCHER_STATUS_LABELS[v.status] || v.status}</span>
                      </div>
                      {v.external_ref && (
                        <div className={s.voucherInfo}>Ref : {v.external_ref}</div>
                      )}
                      {v.valid_from && (
                        <div className={s.voucherInfo}>
                          Période : {v.valid_to
                            ? `${new Date(v.valid_from).toLocaleDateString('fr-CH')} - ${new Date(v.valid_to).toLocaleDateString('fr-CH')}`
                            : `À partir du ${new Date(v.valid_from).toLocaleDateString('fr-CH')}`
                          }
                        </div>
                      )}
                      {v.files && v.files.length > 0 && (
                        <div className={s.voucherInfo}>
                          {v.files.map((f) => (
                            <a key={f.id} href={f.file_url} target="_blank" rel="noopener noreferrer" className={s.voucherLink}>
                              {f.filename}
                            </a>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Action: Annuler / Supprimer */}
            {onDelete && ['pending', 'accepted', 'assigned', 'en_route'].includes(status) && (
              <div className={s.dangerZone}>
                <button
                  type="button"
                  className={s.deleteBtn}
                  onClick={() => onDelete(reservation)}
                >
                  <FiTrash2 size={13} />
                  {['en_route', 'assigned'].includes(status)
                    ? 'Annuler la course'
                    : 'Supprimer la réservation'}
                </button>
              </div>
            )}

            {/* IN_PROGRESS: annulation indisponible */}
            {status === 'in_progress' && (
              <div className={s.inProgressNotice}>
                <FiAlertCircle size={13} />
                Course en cours — annulation indisponible
              </div>
            )}

            {/* Return link */}
            {returnTo && (
              <button className={s.returnBtn} onClick={() => window.location.assign(returnTo)}>
                Retour au contrôle facturation
              </button>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default ReservationDetailPanel;
