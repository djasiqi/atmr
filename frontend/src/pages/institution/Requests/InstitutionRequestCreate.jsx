// pages/institution/Requests/InstitutionRequestCreate.jsx
/**
 * REFERENCE UI — Institution Transport Request Form
 *
 * This form is the visual baseline for all institution-facing flows.
 * Any deviation from this design pattern must be justified.
 *
 * Design: Card-based layout, Lirie brand tokens, teal palette.
 * See: docs/brand/lirie-brand-guidelines.md
 *
 * Features:
 * - Card 1: Mission type, patient, route (FROM→TO visual), datetime + shortcuts
 * - Card 2: Contextual access details (conditional on trip type)
 * - Card 3: Needs chips, contact, billing
 * - Card 4: Advanced details (accordion)
 * - Footer: Summary + badges + CTA
 * - Pre-fill from patient, phone cleaning, auto-focus
 */

import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { FaArrowLeft, FaSave, FaPaperPlane, FaTimes, FaPlus, FaGripVertical } from 'react-icons/fa';
import AsyncCreatableSelect from 'react-select/async-creatable';
import {
  useCreateRequest,
  useSendRequest,
  useAssignExternalCarrier,
  useInstitutionPatients,
  useInstitutionMe,
  useInstitutionSettings,
} from '../../../hooks/useInstitutionData';
import { listPatients, exportRequestMissionPdf } from '../../../services/institutionService';
import { buildCarrierMailto } from '../../../utils/externalCarrierEmail';
import { canSetRequestBillingOnCreate } from '../../../utils/institutionPermissions';
import AddressAutocomplete from '../../../components/common/AddressAutocomplete';
import PatientFormModal from '../Patients/PatientFormModal';
import { toast } from 'sonner';
import {
  SEND_RETRY_TOAST,
  institutionSubmitBusy,
  institutionSubmitButtonLabel,
  resolveCreatedRequestId,
} from '../../../utils/institutionRequestSubmit';
import { createInstitutionSubmitTrace } from '../../../utils/institutionSubmitTrace';
import InlineDatePicker from '../../../components/ui/InlineDatePicker';
import ChipSelect from '../../../components/ui/ChipSelect';
import styles from './InstitutionRequestForm.module.css';
import {
  buildMultiStopPayloadStops,
  buildReturnStopPayload,
  filterValidMultiStopDestinations,
} from '../../../utils/buildMultiStopLegsPreview';
import DestinationBillingOverride from '../../../components/institution/DestinationBillingOverride';
import MedicalDestinationDetails from '../../../components/institution/MedicalDestinationDetails';
import {
  DESTINATION_TYPE_MEDICAL,
  suggestDestinationTypeFromPlace,
} from '../../../utils/institutionDestinationDetails';
import {
  REQUIRED_FIELDS_TOAST,
  collectInstitutionRequestFormErrors,
  fieldErrorsMap,
  formErrorId,
  scrollToFirstFormError,
} from '../../../utils/institutionRequestFormErrors';
import RouteStepTimeField from '../../../components/institution/RouteStepTimeField';
import ExternalCarrierFields, {
  EMPTY_EXTERNAL_CARRIER_FORM,
  validateExternalCarrierForm,
  buildExternalCarrierPayload,
} from '../../../components/institution/ExternalCarrierFields';
import {
  normalizeMissionDate,
  combineMissionDateTime,
  derivePickupTimeConfirmed,
  applyDepartureToPayload,
  extractHHMM,
  formatLocalDateYMD,
  formatLocalTimeHM,
  sanitizeSchedulePayloadForApi,
} from '../../../utils/missionScheduleForm';
import {
  filterTripTypesForInstitution,
  institutionSupportsDomicilePickupTrip,
  TRIP_TYPE_DOM_TO_DEST,
} from '../../../utils/institutionRouteForm';
import { formatWallClockDateShort, formatWallClockDateTime } from '../../../utils/missionTimeDisplay';
import {
  formatDeliveryDescriptionDisplay,
  formatMissionTypeLabel,
  isMaterialDelivery,
  MISSION_DELIVERY_BADGE,
  PATIENT_TRANSPORT_CREATE_LABEL,
} from '../../../utils/missionTypeDisplay';

const TRAVEL_MINUTES_BETWEEN_STOPS = 20;

const detectScheduleIncoherence = (timesInOrder) => {
  const parsed = timesInOrder
    .filter((t) => t?.time)
    .map((t) => ({
      label: t.label,
      minutes: (() => {
        const [h, m] = t.time.split(':').map(Number);
        return h * 60 + m;
      })(),
    }));
  for (let i = 1; i < parsed.length; i += 1) {
    const prev = parsed[i - 1];
    const cur = parsed[i];
    if (cur.minutes < prev.minutes + TRAVEL_MINUTES_BETWEEN_STOPS) {
      return `Incohérence horaire : ${cur.label} (${timesInOrder[i].time}) trop tôt après ${prev.label} (${timesInOrder[i - 1].time}) (~${TRAVEL_MINUTES_BETWEEN_STOPS} min de trajet).`;
    }
  }
  return null;
};

const BILLING_INTENTS = [
  { value: 'patient', label: 'Patient' },
  { value: 'institution', label: 'Institution' },
];

const TRIP_TYPES = [
  { value: 'inst_to_dest', label: 'Institution → Dest.', pickupType: 'institution', dropoffType: 'other' },
  { value: 'dom_to_dest', label: 'Domicile → Dest.', pickupType: 'domicile', dropoffType: 'other' },
  { value: 'return_home', label: 'Retour domicile', pickupType: 'institution', dropoffType: 'domicile' },
  { value: 'other', label: 'Autre', pickupType: 'other', dropoffType: 'other' },
];

const EMPTY_FORM = {
  mission_type: 'patient_transport',
  patient_id: '',
  external_reference: '',
  mission_date: '',
  pickup_time: '',
  pickup_time_confirmed: false,
  dropoff_time: '',
  dropoff_time_confirmed: false,
  return_time: '',
  return_time_confirmed: false,
  scheduled_time: '',
  scheduled_time_type: 'departure',
  // Trip routing
  trip_type: 'inst_to_dest',
  pickup_type: 'institution',
  dropoff_type: 'other',
  pickup_location: '',
  pickup_floor: '',
  pickup_door_code: '',
  pickup_entry_point: '',
  pickup_instructions: '',
  dropoff_location: '',
  dropoff_floor: '',
  dropoff_door_code: '',
  dropoff_entry_point: '',
  dropoff_instructions: '',
  pickup_establishment: '',
  pickup_service: '',
  pickup_doctor: '',
  dropoff_establishment: '',
  dropoff_service: '',
  dropoff_doctor: '',
  destination_type: DESTINATION_TYPE_MEDICAL,
  floor_elevator_info: '',
  round_trip: false,
  requires_wheelchair: false,
  requires_stretcher: false,
  requires_vehicle_wheelchair: false,
  requires_oxygen: false,
  requires_assistance: false,
  // Contacts
  requester_name: '',
  requester_phone: '',
  requester_service: '',
  onsite_is_different: false,
  onsite_name: '',
  onsite_phone: '',
  notes: '',
  delivery_description: '',
  billing_intent: 'patient',
  dropoff_use_custom_billing: false,
  dropoff_destination_billing_override: 'patient',
  return_stop: { use_custom_billing: false, destination_billing_override: 'patient' },
  is_urgent: false,
  multi_stop: false,
  return_to_institution: true,
  intermediate_stops: [],
};

const InstitutionRequestCreate = ({ onClose, onSuccess }) => {
  const { public_id } = useParams();
  const navigate = useNavigate();
  const isModal = typeof onClose === 'function';

  const { data: meData } = useInstitutionMe();
  const { data: patientsData } = useInstitutionPatients({ per_page: 100 });
  const { data: settingsData } = useInstitutionSettings();
  const createMutation = useCreateRequest();
  const sendMutation = useSendRequest();
  const assignExternalMutation = useAssignExternalCarrier();
  const [createdRequestId, setCreatedRequestId] = useState(null);
  const [submitPhase, setSubmitPhase] = useState(null);

  const institutionRole = meData?.institution_role;
  const institutionType = meData?.institution_type;
  const canSetBillingOnCreate = canSetRequestBillingOnCreate(institutionRole);
  const patientsItems = patientsData?.patients || patientsData?.items;
  const patients = useMemo(() => patientsItems || [], [patientsItems]);

  const availableTripTypes = useMemo(
    () => filterTripTypesForInstitution(TRIP_TYPES, institutionType),
    [institutionType],
  );

  // Refs for focus management
  const destinationRef = useRef(null);
  const datetimeRef = useRef(null);
  const missionDateRef = useRef(null);
  const pickupTimeRef = useRef(null);
  const dropoffTimeRef = useRef(null);
  const returnTimeRef = useRef(null);

  // Form state
  const [formData, setFormData] = useState({ ...EMPTY_FORM });
  const [showQuickPatient, setShowQuickPatient] = useState(false);
  const [selectedPatientOption, setSelectedPatientOption] = useState(null);

  // ── Multi-étapes : réorganisation par glisser-déposer ──
  const [dragIndex, setDragIndex] = useState(null);
  const [dragOverIndex, setDragOverIndex] = useState(null);
  const [fieldErrors, setFieldErrors] = useState({});
  const destinationDetailsRef = useRef(null);

  const clearFieldErrors = useCallback((...keys) => {
    setFieldErrors((prev) => {
      if (!keys.some((k) => prev[k])) return prev;
      const next = { ...prev };
      keys.forEach((k) => { delete next[k]; });
      return next;
    });
  }, []);

  // Réordonne TOUS les points du parcours.
  // index 0 = Départ (origine), 1 = Destination, 2.. = étapes supplémentaires.
  const moveRoutePoint = useCallback((from, to) => {
    setFormData((prev) => {
      const instAddr = meData?.address || '';
      const pickup = {
        kind: 'pickup',
        address: prev.pickup_location || (prev.pickup_type === 'institution' ? instAddr : ''),
        establishment: prev.pickup_establishment || '',
        service: prev.pickup_service || '',
        doctor: prev.pickup_doctor || '',
      };
      const dropoff = {
        kind: 'dropoff',
        address: prev.dropoff_location || (prev.dropoff_type === 'institution' ? instAddr : ''),
        establishment: prev.dropoff_establishment || '',
        service: prev.dropoff_service || '',
        doctor: prev.dropoff_doctor || '',
        destination_type: prev.destination_type || DESTINATION_TYPE_MEDICAL,
      };
      const extras = (prev.intermediate_stops || []).map((s) => ({
        kind: 'extra',
        address: s.dropoff_location || '',
        scheduled_time: s.scheduled_time || '',
        time_confirmed: Boolean(s.time_confirmed),
        establishment: s.dropoff_establishment || '',
        service: s.dropoff_service || '',
        doctor: s.dropoff_doctor || '',
        destination_type: s.destination_type || DESTINATION_TYPE_MEDICAL,
      }));
      const combined = [pickup, dropoff, ...extras];
      if (from === to || from < 0 || to < 0 || from >= combined.length || to >= combined.length) {
        return prev;
      }
      const [moved] = combined.splice(from, 1);
      combined.splice(to, 0, moved);
      const [n0, n1, ...rest] = combined;
      const next = {
        ...prev,
        pickup_location: n0.address || '',
        pickup_establishment: n0.establishment || '',
        pickup_service: n0.service || '',
        pickup_doctor: n0.doctor || '',
        dropoff_location: n1.address || '',
        dropoff_establishment: n1.establishment || '',
        dropoff_service: n1.service || '',
        dropoff_doctor: n1.doctor || '',
        destination_type: n1.destination_type || DESTINATION_TYPE_MEDICAL,
        intermediate_stops: rest.map((r) => ({
          dropoff_location: r.address || '',
          scheduled_time: r.scheduled_time || '',
          time_confirmed: Boolean(r.time_confirmed),
          dropoff_establishment: r.establishment || '',
          dropoff_service: r.service || '',
          dropoff_doctor: r.doctor || '',
          destination_type: r.destination_type || DESTINATION_TYPE_MEDICAL,
        })),
      };
      // Si l'origine change de nature, elle devient une adresse libre.
      if (n0.kind !== 'pickup') {
        next.pickup_type = 'other';
        next.pickup_entry_point = '';
      }
      // Idem pour la destination principale.
      if (n1.kind !== 'dropoff') {
        next.dropoff_type = 'other';
        next.dropoff_entry_point = '';
      }
      return next;
    });
  }, [meData]);

  // ── Patient search (AsyncCreatableSelect) ──
  const formatDob = useCallback((dobStr) => {
    if (!dobStr) return '';
    try {
      const d = new Date(dobStr);
      if (isNaN(d.getTime())) return '';
      return d.toLocaleDateString('fr-CH', { day: '2-digit', month: '2-digit', year: 'numeric' });
    } catch { return ''; }
  }, []);

  const formatPatientOption = useCallback((p) => {
    const dobLabel = formatDob(p.dob);
    const parts = [`${p.last_name} ${p.first_name}`];
    if (dobLabel) parts.push(dobLabel);
    if (p.external_reference) parts.push(`DPI: ${p.external_reference}`);
    return {
      value: p.id,
      label: parts.join(' — '),
      raw: p,
    };
  }, [formatDob]);

  const defaultPatientOptions = useMemo(
    () => patients.map(formatPatientOption),
    [patients, formatPatientOption]
  );

  const loadPatientOptions = useCallback(async (inputValue) => {
    if (!inputValue || inputValue.length < 1) return defaultPatientOptions;
    try {
      const result = await listPatients({ query: inputValue, per_page: 30 });
      // Backend returns { patients: [...], total, page, ... }
      const items = result?.patients || result?.items || result?.data || [];
      const list = Array.isArray(items) ? items : [];
      return list.map(formatPatientOption);
    } catch {
      return [];
    }
  }, [formatPatientOption, defaultPatientOptions]);
  const [executionMode, setExecutionMode] = useState('lirie');
  const [externalCarrierForm, setExternalCarrierForm] = useState(EMPTY_EXTERNAL_CARRIER_FORM);
  const [patientPrefilled, setPatientPrefilled] = useState(false);

  const isLirieSendMode = executionMode === 'lirie';
  const isExternalMode = executionMode === 'external';
  const isDraftMode = executionMode === 'draft';
  const submitBusy = institutionSubmitBusy(submitPhase, {
    createPending: createMutation.isPending,
    sendPending: sendMutation.isPending,
    assignPending: assignExternalMutation.isPending,
  });
  const submitLabel = institutionSubmitButtonLabel({
    phase: submitPhase,
    busy: submitBusy,
    isLirieSendMode,
    isDraftMode,
    hasCreatedRequest: Boolean(createdRequestId),
  });

  // Selected patient object
  const selectedPatient = useMemo(() => {
    if (!formData.patient_id) return null;
    return patients.find(p => String(p.id) === String(formData.patient_id)) || null;
  }, [formData.patient_id, patients]);

  // Derived: institution address from meData
  const institutionAddress = meData?.address || '';

  const multiStopOrigin = useMemo(
    () => formData.pickup_location
      || (formData.pickup_type === 'institution' ? institutionAddress : '')
      || institutionAddress
      || '',
    [formData.pickup_location, formData.pickup_type, institutionAddress],
  );

  // Parcours : la « Destination » principale reste dropoff_location ;
  // les étapes supplémentaires vivent dans intermediate_stops. Aucun « mode ».
  const extraStops = formData.intermediate_stops || [];
  const hasExtraStops = extraStops.length > 0;
  const journeyReturnEnabled = formData.return_to_institution === true;
  // Retour domicile : départ = institution, dernière étape = domicile patient.
  // Les destinations ajoutées s'insèrent obligatoirement entre les deux.
  const isReturnHome = formData.dropoff_type === 'domicile';

  // ── Pre-fill billing_intent + trip type + contacts from institution settings ──
  useEffect(() => {
    if (!settingsData) return;
    const settings = settingsData.settings || {};
    const defaultIntent = settings.default_billing_intent;

    setFormData(prev => {
      const updates = {};

      // Billing intent — uniquement patient | institution
      const allowedDefaults = new Set(BILLING_INTENTS.map((b) => b.value));
      if (!allowedDefaults.has(prev.billing_intent)) {
        updates.billing_intent = allowedDefaults.has(defaultIntent) ? defaultIntent : 'patient';
      } else if (
        defaultIntent
        && allowedDefaults.has(defaultIntent)
        && prev.billing_intent === 'patient'
      ) {
        updates.billing_intent = defaultIntent;
      }

      // Trip type from default_pickup_mode (domicile uniquement pour IMAD / curatelle)
      const mode = settings.default_pickup_mode || 'institution';
      const domicilePickupAllowed = institutionSupportsDomicilePickupTrip(meData?.institution_type);
      if (prev.trip_type === 'inst_to_dest' || prev.trip_type === TRIP_TYPE_DOM_TO_DEST) {
        // Only override at init
        const defaultTrip = mode === 'domicile' && domicilePickupAllowed
          ? TRIP_TYPE_DOM_TO_DEST
          : 'inst_to_dest';
        const def = TRIP_TYPES.find(t => t.value === defaultTrip);
        if (def) {
          updates.trip_type = defaultTrip;
          updates.pickup_type = def.pickupType;
          updates.dropoff_type = def.dropoffType;
          updates.return_to_institution = defaultTrip === 'inst_to_dest';
        }
      }

      // Onsite phone pre-fill from default_contact_phone
      if (!prev.onsite_phone && settings.default_contact_phone) {
        updates.onsite_phone = settings.default_contact_phone;
      }

      // Pickup location pre-fill for institution mode
      if (updates.pickup_type === 'institution' || (!updates.pickup_type && prev.pickup_type === 'institution')) {
        if (!prev.pickup_location && institutionAddress) {
          updates.pickup_location = institutionAddress;
        }
      }

      if (Object.keys(updates).length > 0) {
        return { ...prev, ...updates };
      }
      return prev;
    });
  }, [settingsData, meData, institutionAddress]);

  // ── Pre-fill requester contact from connected user (independent of settingsData) ──
  useEffect(() => {
    if (!meData?.user) return;
    setFormData(prev => {
      const updates = {};
      if (!prev.requester_name) {
        const name = [meData.user.first_name, meData.user.last_name].filter(Boolean).join(' ');
        if (name) updates.requester_name = name;
      }
      if (!prev.requester_phone) {
        const phone = meData.user.phone || meData.contact_phone || '';
        if (phone) updates.requester_phone = phone;
      }
      if (Object.keys(updates).length > 0) {
        return { ...prev, ...updates };
      }
      return prev;
    });
  }, [meData]);

  // ── Pre-fill from patient data when patient is selected ──
  const prefillFromPatient = useCallback((patient) => {
    if (!patient) return;
    setFormData(prev => {
      const updates = {};
      // Build patient address
      const addressParts = [patient.address, patient.postal_code, patient.city]
        .filter(Boolean);
      const patientAddr = addressParts.length > 0 ? addressParts.join(', ') : '';

      // For domicile pickup: fill pickup with patient address
      if (prev.pickup_type === 'domicile' && patientAddr && !prev.pickup_location) {
        updates.pickup_location = patientAddr;
      }
      // For domicile dropoff (return_home): fill dropoff with patient address
      if (prev.dropoff_type === 'domicile' && patientAddr) {
        updates.dropoff_location = patientAddr;
      }
      // For "other" mode: fill pickup like before
      if (prev.pickup_type === 'other' && patientAddr && !prev.pickup_location) {
        updates.pickup_location = patientAddr;
      }

      // Logistics (domicile & other modes)
      if (prev.pickup_type !== 'institution') {
        if (patient.door_code && !prev.pickup_door_code) {
          updates.pickup_door_code = patient.door_code;
        }
        if (patient.floor && !prev.pickup_floor) {
          updates.pickup_floor = patient.floor;
        }
        const accessParts = [];
        if (patient.access_notes) accessParts.push(patient.access_notes);
        if (patient.residence_name) accessParts.push(`Résidence: ${patient.residence_name}`);
        if (accessParts.length > 0 && !prev.floor_elevator_info) {
          updates.floor_elevator_info = accessParts.join(' — ');
        }
      }
      if (!prev.external_reference && patient.external_reference) {
        updates.external_reference = patient.external_reference;
      }

      if (Object.keys(updates).length > 0) {
        return { ...prev, ...updates };
      }
      return prev;
    });
    setPatientPrefilled(true);
  }, []);

  const handlePatientChange = useCallback((patientId, patientObj) => {
    setFormData(prev => ({ ...prev, patient_id: patientId }));
    clearFieldErrors('patient_id', 'pickup_location', 'dropoff_location');
    setPatientPrefilled(false);
    if (patientId) {
      const patient = patientObj || patients.find(p => String(p.id) === String(patientId));
      if (patient) {
        prefillFromPatient(patient);
      }
      // Auto-focus: destination if pickup is institution, else datetime
      setTimeout(() => {
        if (destinationRef.current) {
          destinationRef.current.focus();
        } else if (datetimeRef.current) {
          datetimeRef.current.focus();
        }
      }, 100);
    }
  }, [patients, prefillFromPatient, clearFieldErrors]);

  const handleChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    if (field === 'dropoff_service' || field === 'dropoff_doctor' || field === 'destination_type') {
      clearFieldErrors('medical_principal');
    }
    if (field === 'mission_date') clearFieldErrors('mission_date');
    if (field === 'pickup_time') clearFieldErrors('pickup_time', 'confirmed_time');
    if (field === 'dropoff_time') clearFieldErrors('dropoff_time', 'confirmed_time');
    if (field === 'return_time') clearFieldErrors('return_time', 'confirmed_time');
    if (field === 'pickup_location') clearFieldErrors('pickup_location');
    if (field === 'dropoff_location') clearFieldErrors('dropoff_location');
    if (field === 'delivery_description') clearFieldErrors('delivery_description');
    if (field === 'notes') clearFieldErrors('notes');
    if (field === 'requires_assistance') clearFieldErrors('notes');
    if (field === 'billing_intent') clearFieldErrors('billing');
  };

  // ── Champ « Départ » réutilisable (mode simple + carte parcours) ──
  const renderPickupField = (inputClassName, { editable = false } = {}) => {
    const isFixedType = formData.pickup_type === 'institution' || formData.pickup_type === 'domicile';
    if (isFixedType && !editable) {
      return (
        <input
          type="text"
          id="pickup_location"
          value={formData.pickup_location || (formData.pickup_type === 'institution' ? institutionAddress : '')}
          readOnly
          className={`${inputClassName} ${styles.routeReadonly}`}
          placeholder={formData.pickup_type === 'domicile' && !formData.pickup_location ? 'Sélectionnez un patient' : ''}
          aria-invalid={Boolean(fieldErrors.pickup_location) || undefined}
          aria-describedby={fieldErrors.pickup_location ? formErrorId('pickup_location') : undefined}
        />
      );
    }
    return (
      <AddressAutocomplete
        name="pickup_location"
        inputId="pickup_location"
        value={formData.pickup_location || (editable && formData.pickup_type === 'institution' ? institutionAddress : '')}
        onChange={(e) => handleChange('pickup_location', e.target.value)}
        onSelect={(item) => {
          const address = item.label || item.address || '';
          const placeName = item.name || '';
          const isDoctorPattern = /^(dr\.?|prof\.?|méd\.?|med\.?|docteur|professeur)\s/i;
          setFormData(prev => {
            const updates = { ...prev, pickup_location: address, pickup_establishment: '', pickup_doctor: '' };
            if (placeName && placeName !== item.address) {
              if (isDoctorPattern.test(placeName)) {
                updates.pickup_doctor = placeName;
              } else {
                updates.pickup_establishment = placeName;
              }
            }
            return updates;
          });
          clearFieldErrors('pickup_location');
        }}
        placeholder="Saisir ou choisir l'adresse"
        inputClassName={inputClassName}
        required
        aria-invalid={Boolean(fieldErrors.pickup_location) || undefined}
        aria-describedby={fieldErrors.pickup_location ? formErrorId('pickup_location') : undefined}
      />
    );
  };

  // ── Champ « Destination » réutilisable (destination principale) ──
  const renderDropoffField = (inputClassName) => {
    if (formData.dropoff_type === 'institution' || formData.dropoff_type === 'domicile') {
      return (
        <input
          type="text"
          id="dropoff_location"
          value={formData.dropoff_location || (formData.dropoff_type === 'institution' ? institutionAddress : '')}
          readOnly
          className={`${inputClassName} ${styles.routeReadonly}`}
          placeholder={formData.dropoff_type === 'domicile' && !formData.dropoff_location ? 'Sélectionnez un patient' : ''}
          aria-invalid={Boolean(fieldErrors.dropoff_location) || undefined}
          aria-describedby={fieldErrors.dropoff_location ? formErrorId('dropoff_location') : undefined}
        />
      );
    }
    return (
      <AddressAutocomplete
        name="dropoff_location"
        inputId="dropoff_location"
        value={formData.dropoff_location}
        onChange={(e) => handleChange('dropoff_location', e.target.value)}
        onSelect={(item) => {
          const address = item.label || item.address || '';
          const placeName = item.name || '';
          const isDoctorPattern = /^(dr\.?|prof\.?|méd\.?|med\.?|docteur|professeur)\s/i;
          setFormData(prev => {
            const updates = { ...prev, dropoff_location: address, dropoff_establishment: '', dropoff_doctor: '' };
            if (placeName && placeName !== item.address) {
              if (isDoctorPattern.test(placeName)) {
                updates.dropoff_doctor = placeName;
              } else {
                updates.dropoff_establishment = placeName;
              }
            }
            const suggested = suggestDestinationTypeFromPlace(item)
              || (isDoctorPattern.test(placeName) ? 'medical' : null);
            if (suggested) updates.destination_type = suggested;
            return updates;
          });
          clearFieldErrors('dropoff_location', 'medical_principal');
        }}
        placeholder="Adresse d'arrivée"
        inputClassName={inputClassName}
        required
        aria-invalid={Boolean(fieldErrors.dropoff_location) || undefined}
        aria-describedby={fieldErrors.dropoff_location ? formErrorId('dropoff_location') : undefined}
      />
    );
  };

  // ── Met à jour une étape supplémentaire (destination N) ──
  const setStopAddress = (idx, value) => {
    const nextValue = value?.target?.value ?? value ?? '';
    setFormData((prev) => {
      const next = [...(prev.intermediate_stops || [])];
      next[idx] = { ...next[idx], dropoff_location: nextValue };
      return { ...prev, intermediate_stops: next };
    });
    clearFieldErrors(`medical_extra_${idx}`, `extra_stop_location_${idx}`);
  };

  // Met à jour un champ détail (establishment / service / doctor) d'une étape.
  const setStopField = (idx, field, value) => {
    const nextValue = value?.target?.value ?? value ?? '';
    setFormData((prev) => {
      const next = [...(prev.intermediate_stops || [])];
      next[idx] = { ...next[idx], [field]: nextValue };
      return { ...prev, intermediate_stops: next };
    });
    if (field === 'dropoff_service' || field === 'dropoff_doctor' || field === 'destination_type') {
      clearFieldErrors(`medical_extra_${idx}`);
    }
  };

  const setStopTime = (idx, timeHHMM) => {
    setFormData((prev) => {
      const missionDate = prev.mission_date || prev.scheduled_time?.split('T')[0] || '';
      const iso = combineMissionDateTime(missionDate, timeHHMM);
      const next = [...(prev.intermediate_stops || [])];
      next[idx] = {
        ...next[idx],
        scheduled_time: iso || '',
        time_confirmed: Boolean(timeHHMM?.trim()),
      };
      return { ...prev, intermediate_stops: next };
    });
    clearFieldErrors(`extra_stop_time_${idx}`, 'confirmed_time');
  };

  // Sélection d'adresse pour une étape : auto-remplit établissement / médecin
  // à partir du lieu nommé (même logique que la Destination principale).
  const setStopFromSelection = (idx, item) => {
    const address = item?.label || item?.address || item?.formatted_address || item?.description || '';
    const placeName = item?.name || '';
    const isDoctorPattern = /^(dr\.?|prof\.?|méd\.?|med\.?|docteur|professeur)\s/i;
    setFormData((prev) => {
      const next = [...(prev.intermediate_stops || [])];
      const entry = {
        ...next[idx],
        dropoff_location: address,
        dropoff_establishment: '',
        dropoff_doctor: '',
      };
      if (placeName && placeName !== item?.address) {
        if (isDoctorPattern.test(placeName)) {
          entry.dropoff_doctor = placeName;
        } else {
          entry.dropoff_establishment = placeName;
        }
      }
      const suggested = suggestDestinationTypeFromPlace(item)
        || (isDoctorPattern.test(placeName) ? 'medical' : null);
      if (suggested) entry.destination_type = suggested;
      next[idx] = entry;
      return { ...prev, intermediate_stops: next };
    });
    clearFieldErrors(`extra_stop_location_${idx}`, `medical_extra_${idx}`);
  };

  const addExtraStop = () => {
    setFormData((prev) => ({
      ...prev,
      intermediate_stops: [
        ...(prev.intermediate_stops || []),
        {
          dropoff_location: '',
          scheduled_time: '',
          time_confirmed: false,
          dropoff_establishment: '',
          dropoff_service: '',
          dropoff_doctor: '',
          destination_type: DESTINATION_TYPE_MEDICAL,
          use_custom_billing: false,
          destination_billing_override: 'patient',
        },
      ],
    }));
  };

  const removeExtraStop = (idx) => {
    setFormData((prev) => ({
      ...prev,
      intermediate_stops: (prev.intermediate_stops || []).filter((_, i) => i !== idx),
    }));
  };

  // ── Trip type change handler ──
  const getPatientAddress = useCallback(() => {
    if (!selectedPatient) return '';
    const parts = [selectedPatient.address, selectedPatient.postal_code, selectedPatient.city].filter(Boolean);
    return parts.length > 0 ? parts.join(', ') : '';
  }, [selectedPatient]);

  const handleTripTypeChange = useCallback((tripTypeValue) => {
    const def = TRIP_TYPES.find(t => t.value === tripTypeValue);
    if (!def) return;
    setFormData(prev => {
      const updates = {
        trip_type: tripTypeValue,
        pickup_type: def.pickupType,
        dropoff_type: def.dropoffType,
        destination_type: def.dropoffType === 'other'
          ? DESTINATION_TYPE_MEDICAL
          : def.dropoffType,
      };
      // Retour domicile : facturation par défaut au patient, A/R décoché
      if (tripTypeValue === 'return_home') {
        updates.billing_intent = 'patient';
        updates.round_trip = false;
        updates.return_to_institution = false;
      } else if (tripTypeValue === 'inst_to_dest') {
        updates.return_to_institution = true;
      } else {
        updates.return_to_institution = false;
      }
      // Auto-fill pickup for institution mode
      if (def.pickupType === 'institution' && institutionAddress) {
        updates.pickup_location = institutionAddress;
      }
      // Auto-fill pickup for domicile mode (patient address)
      if (def.pickupType === 'domicile') {
        const addr = getPatientAddress();
        if (addr) updates.pickup_location = addr;
      }
      // Auto-fill dropoff for domicile mode (return home — patient address)
      if (def.dropoffType === 'domicile') {
        const addr = getPatientAddress();
        if (addr) updates.dropoff_location = addr;
      }
      // Auto-fill dropoff for institution mode
      if (def.dropoffType === 'institution' && institutionAddress) {
        updates.dropoff_location = institutionAddress;
      }
      // Clear fields on "other" type
      if (def.dropoffType === 'other') {
        updates.dropoff_location = '';
      }
      if (def.pickupType === 'other') {
        updates.pickup_location = '';
      }
      // Clear irrelevant fields when switching
      if (def.pickupType === 'institution') {
        updates.pickup_floor = '';
        updates.pickup_door_code = '';
      }
      if (def.pickupType !== 'institution') {
        updates.pickup_entry_point = '';
        updates.pickup_instructions = '';
      }
      if (def.dropoffType === 'institution') {
        updates.dropoff_floor = '';
        updates.dropoff_door_code = '';
      }
      if (def.dropoffType !== 'institution') {
        updates.dropoff_entry_point = '';
        updates.dropoff_instructions = '';
      }
      return { ...prev, ...updates };
    });
  }, [institutionAddress, getPatientAddress]);

  // Clinique / hôpital / EMS : pas de segment « Domicile → Dest. »
  useEffect(() => {
    if (institutionSupportsDomicilePickupTrip(institutionType)) return;
    if (formData.trip_type !== TRIP_TYPE_DOM_TO_DEST) return;
    handleTripTypeChange('inst_to_dest');
  }, [institutionType, formData.trip_type, handleTripTypeChange]);

  // ── Datetime shortcuts ──
  const setTimeShortcut = useCallback((minutesFromNow) => {
    const d = new Date();
    d.setMinutes(d.getMinutes() + minutesFromNow);
    const dateVal = formatLocalDateYMD(d);
    const timeVal = formatLocalTimeHM(d);
    setFormData(prev => ({
      ...prev,
      mission_date: dateVal,
      pickup_time: timeVal,
      pickup_time_confirmed: minutesFromNow === 0,
      scheduled_time: `${dateVal}T${timeVal}`,
      is_urgent: minutesFromNow === 0,
      scheduled_time_type: 'departure',
    }));
    clearFieldErrors('mission_date', 'pickup_time', 'confirmed_time');
  }, [clearFieldErrors]);

  const setTimeTomorrow9 = useCallback(() => {
    const d = new Date();
    d.setDate(d.getDate() + 1);
    d.setHours(9, 0, 0, 0);
    const dateVal = formatLocalDateYMD(d);
    setFormData(prev => ({
      ...prev,
      mission_date: dateVal,
      pickup_time: '09:00',
      pickup_time_confirmed: true,
      scheduled_time: `${dateVal}T09:00`,
      scheduled_time_type: 'departure',
    }));
    clearFieldErrors('mission_date', 'pickup_time', 'confirmed_time');
  }, [clearFieldErrors]);

  // ── Phone normalizer (accepts Swiss formats, cleans for storage) ──
  const cleanPhone = (raw) => {
    if (!raw) return '';
    // Remove spaces, dots, dashes, parens
    let cleaned = raw.replace(/[\s.\-()]/g, '');
    // Swiss local → international: 0xx → +41xx
    if (cleaned.startsWith('0') && !cleaned.startsWith('00')) {
      cleaned = '+41' + cleaned.slice(1);
    }
    return cleaned;
  };

  /** Flush synchrone des pickers avant validation/payload (scénarios B/C sans blur). */
  const flushScheduleFields = useCallback(() => {
    const missionDate = missionDateRef.current?.flushPending?.()
      ?? formData.mission_date
      ?? (formData.scheduled_time ? formData.scheduled_time.split('T')[0] : '');
    const pickupTime = pickupTimeRef.current?.flushPending?.() ?? formData.pickup_time;
    const dropoffTime = dropoffTimeRef.current?.flushPending?.() ?? formData.dropoff_time;
    const returnTime = returnTimeRef.current?.flushPending?.() ?? formData.return_time;
    return {
      mission_date: missionDate,
      pickup_time: pickupTime,
      pickup_time_confirmed: derivePickupTimeConfirmed(pickupTime),
      dropoff_time: dropoffTime,
      dropoff_time_confirmed: Boolean(dropoffTime?.trim()),
      return_time: returnTime,
      return_time_confirmed: Boolean(returnTime?.trim()),
    };
  }, [formData]);

  const scheduleIncoherence = useMemo(() => {
    const extraValid = filterValidMultiStopDestinations(formData.intermediate_stops);
    const returnEnabled = formData.return_to_institution === true;
    const isMulti = extraValid.length >= 1 || returnEnabled;
    const times = [];
    if (formData.pickup_time?.trim()) {
      times.push({ label: 'Départ', time: formData.pickup_time.trim() });
    }
    if (formData.dropoff_time?.trim()) {
      times.push({ label: isMulti ? 'Destination 1' : 'RDV', time: formData.dropoff_time.trim() });
    }
    extraValid.forEach((stop, idx) => {
      const t = extractHHMM(stop.scheduled_time) || stop.scheduled_time?.split('T')[1]?.slice(0, 5);
      if (t) times.push({ label: `Destination ${idx + 2}`, time: t });
    });
    if (returnEnabled && formData.return_time?.trim()) {
      times.push({ label: 'Retour', time: formData.return_time.trim() });
    }
    return detectScheduleIncoherence(times);
  }, [formData]);

  // ── Build payload for backend ──
  const buildPayload = (scheduleOverrides = {}) => {
    const missionDate = normalizeMissionDate(
      scheduleOverrides.mission_date
        ?? formData.mission_date
        ?? (formData.scheduled_time ? formData.scheduled_time.split('T')[0] : ''),
    );
    const pickupTime = scheduleOverrides.pickup_time ?? formData.pickup_time;
    const dropoffTime = scheduleOverrides.dropoff_time ?? formData.dropoff_time;
    const returnTime = scheduleOverrides.return_time ?? formData.return_time;
    const pickupIso = combineMissionDateTime(missionDate, pickupTime);
    const dropoffIso = combineMissionDateTime(missionDate, dropoffTime);
    const returnIso = combineMissionDateTime(missionDate, returnTime);

    const payload = {
      mission_type: formData.mission_type,
      patient_id: formData.patient_id ? Number(formData.patient_id) : null,
      mission_date: missionDate,
      pickup_time_confirmed: derivePickupTimeConfirmed(pickupTime),
      pickup_location: formData.pickup_location || (formData.pickup_type === 'institution' ? institutionAddress : ''),
      dropoff_location: formData.dropoff_location || (formData.dropoff_type === 'institution' ? institutionAddress : ''),
      is_round_trip: false,
      is_urgent: formData.is_urgent || false,
      billing_intent: formData.billing_intent,
      notes: formData.notes || null,
    };
    if (payload.pickup_time_confirmed && pickupIso) {
      payload.scheduled_time = pickupIso;
      payload.scheduled_time_type = 'departure';
    }
    if (formData.external_reference?.trim()) {
      payload.external_reference = formData.external_reference.trim();
    }

    // Logistics pickup
    if (formData.pickup_floor) payload.pickup_floor = formData.pickup_floor;
    if (formData.pickup_door_code) payload.pickup_door_code = formData.pickup_door_code;
    if (formData.dropoff_floor) payload.dropoff_floor = formData.dropoff_floor;
    if (formData.dropoff_door_code) payload.dropoff_door_code = formData.dropoff_door_code;
    if (formData.floor_elevator_info) payload.floor_elevator_info = formData.floor_elevator_info;

    // Location types and entry points (direct columns on TransportRequest)
    payload.pickup_type = formData.pickup_type || null;
    payload.dropoff_type = formData.dropoff_type || null;
    payload.destination_type = formData.destination_type || DESTINATION_TYPE_MEDICAL;
    payload.pickup_entry_point = formData.pickup_entry_point || null;
    payload.dropoff_entry_point = formData.dropoff_entry_point || null;
    if (formData.dropoff_establishment?.trim()) {
      payload.dropoff_establishment = formData.dropoff_establishment.trim();
    }
    if (formData.dropoff_service?.trim()) {
      payload.dropoff_service = formData.dropoff_service.trim();
    }
    if (formData.dropoff_doctor?.trim()) {
      payload.dropoff_doctor = formData.dropoff_doctor.trim();
    }

    // Routing details also kept in billing_details.routing for rétrocompatibilité
    payload.billing_details = {
      ...(payload.billing_details || {}),
      routing: {
        pickup_type: formData.pickup_type,
        dropoff_type: formData.dropoff_type,
        pickup_entry_point: formData.pickup_entry_point || null,
        dropoff_entry_point: formData.dropoff_entry_point || null,
        pickup_instructions: formData.pickup_instructions || null,
        dropoff_instructions: formData.dropoff_instructions || null,
        pickup_establishment: formData.pickup_establishment || null,
        pickup_service: formData.pickup_service || null,
        pickup_doctor: formData.pickup_doctor || null,
        dropoff_establishment: formData.dropoff_establishment || null,
        dropoff_service: formData.dropoff_service || null,
        dropoff_doctor: formData.dropoff_doctor || null,
        destination_type: formData.destination_type || DESTINATION_TYPE_MEDICAL,
      },
    };

    // Mobility as structured JSONB
    payload.mobility = {
      wheelchair: formData.requires_wheelchair,
      stretcher: formData.requires_stretcher,
      vehicle_wheelchair: formData.requires_vehicle_wheelchair,
      oxygen: formData.requires_oxygen,
      needs_assistance: formData.requires_assistance,
      walking: !formData.requires_wheelchair && !formData.requires_stretcher,
    };

    // Contact on site (enriched structure, retrocompatible)
    const requesterName = formData.requester_name || 'Contact institution';
    const requesterPhone = cleanPhone(formData.requester_phone);
    const onsitePhone = cleanPhone(formData.onsite_phone);
    payload.contact_on_site = {
      // Retrocompat fields (used by existing code)
      name: formData.onsite_is_different ? (formData.onsite_name || requesterName) : requesterName,
      phone: formData.onsite_is_different ? (onsitePhone || requesterPhone) : requesterPhone,
      // Enriched fields
      requester_name: requesterName,
      requester_phone: requesterPhone,
      requester_service: formData.requester_service || null,
      onsite_is_different: formData.onsite_is_different,
      onsite_name: formData.onsite_is_different ? (formData.onsite_name || '') : '',
      onsite_phone: formData.onsite_is_different ? onsitePhone : '',
    };

    // Delivery
    if (formData.mission_type === 'material_delivery') {
      payload.delivery_description = formData.delivery_description;
    }

    const extraValidStops = filterValidMultiStopDestinations(formData.intermediate_stops);
    const returnEnabled = formData.return_to_institution === true;
    const isMultiRoute = extraValidStops.length >= 1 || returnEnabled;

    if (isMultiRoute) {
      const principalDropoff = (
        formData.dropoff_location
        || (formData.dropoff_type === 'institution' ? institutionAddress : '')
        || ''
      ).trim();
      const principalStop = {
        dropoff_location: principalDropoff,
        ...(dropoffIso ? { scheduled_time: dropoffIso } : {}),
        time_confirmed: Boolean(dropoffTime?.trim()),
        dropoff_establishment: formData.dropoff_establishment || '',
        dropoff_service: formData.dropoff_service || '',
        dropoff_doctor: formData.dropoff_doctor || '',
        destination_type: formData.destination_type || DESTINATION_TYPE_MEDICAL,
        use_custom_billing: Boolean(formData.dropoff_use_custom_billing),
        destination_billing_override: formData.dropoff_destination_billing_override || 'patient',
      };
      payload.multi_stop = true;
      payload.return_to_institution = returnEnabled;
      payload.is_round_trip = false;
      // Retour domicile : domicile = dernière étape, destinations insérées avant.
      const orderedStops = isReturnHome
        ? [...extraValidStops, principalStop]
        : [principalStop, ...extraValidStops];
      payload.intermediate_stops = buildMultiStopPayloadStops(orderedStops, missionDate);
      payload.dropoff_location = isReturnHome
        ? (orderedStops[0]?.dropoff_location || principalDropoff)
        : principalDropoff;
      if (pickupIso) {
        payload.scheduled_time = pickupIso;
        payload.scheduled_time_type = 'departure';
      }
      if (returnEnabled) {
        if (returnIso) payload.return_scheduled_time = returnIso;
        payload.return_time_confirmed = Boolean(returnTime?.trim());
        payload.return_stop = buildReturnStopPayload(formData.return_stop);
      }
    } else {
      const hasPickup = Boolean(pickupTime?.trim());
      const hasDropoff = Boolean(dropoffTime?.trim());
      if (hasPickup && (!hasDropoff || derivePickupTimeConfirmed(pickupTime))) {
        applyDepartureToPayload(payload, { missionDate, pickupTime });
      } else if (hasDropoff) {
        payload.scheduled_time_type = 'arrival';
        payload.pickup_time_confirmed = false;
        if (dropoffIso) payload.scheduled_time = dropoffIso;
        payload.appointment_time_confirmed = Boolean(dropoffTime?.trim());
      }
    }

    return payload;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (submitBusy) {
      return;
    }

    const trace = createInstitutionSubmitTrace();
    trace.mark('institution_submit_click', { execution_mode: executionMode });

    const flushedSchedule = flushScheduleFields();
    const validationErrors = collectInstitutionRequestFormErrors({
      formData,
      flushedSchedule,
      institutionAddress,
      isLirieSendMode,
      isExternalMode,
      billingHasBlockingError: isLirieSendMode && billingWarnings.some((w) => w.level === 'error'),
      externalCarrierError: isExternalMode
        ? validateExternalCarrierForm(externalCarrierForm)
        : null,
    });
    trace.mark('validation_end', { error_count: validationErrors.length });
    if (validationErrors.length > 0) {
      setFieldErrors(fieldErrorsMap(validationErrors));
      toast.error(REQUIRED_FIELDS_TOAST);
      scrollToFirstFormError(validationErrors, {
        mission_date: () => missionDateRef.current?.focus?.(),
        pickup_time: () => pickupTimeRef.current?.focus?.(),
        confirmed_time: () => pickupTimeRef.current?.focus?.(),
        dropoff_time: () => dropoffTimeRef.current?.focus?.(),
        return_time: () => returnTimeRef.current?.focus?.(),
        dropoff_service: () => {
          destinationDetailsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
          document.getElementById('dropoff_service')?.focus?.({ preventScroll: true });
        },
      });
      return;
    }
    setFieldErrors({});

    let requestId = resolveCreatedRequestId(createdRequestId, null);
    try {
      const payload = sanitizeSchedulePayloadForApi(buildPayload(flushedSchedule));
      let result = null;

      if (!requestId) {
        setSubmitPhase('create');
        trace.mark('create_request_start');
        result = await createMutation.mutateAsync(payload);
        requestId = resolveCreatedRequestId(null, result);
        if (requestId) setCreatedRequestId(requestId);
        trace.mark('create_request_end', {
          transport_request_id: requestId,
          reused: false,
        });
      } else {
        trace.mark('create_request_end', {
          transport_request_id: requestId,
          reused: true,
        });
      }

      if (isLirieSendMode) {
        if (!requestId) {
          throw new Error('Demande créée sans identifiant — impossible d’envoyer.');
        }
        setSubmitPhase('send');
        trace.mark('dispatch_start', { transport_request_id: requestId });
        await sendMutation.mutateAsync({ requestId, options: {} });
        trace.mark('dispatch_end', { transport_request_id: requestId });
        toast.success('Demande créée et envoyée');
      } else if (isExternalMode) {
        try {
          const assignResult = await assignExternalMutation.mutateAsync({
            requestId,
            data: buildExternalCarrierPayload(externalCarrierForm),
          });
          toast.success('Demande créée et transporteur externe affecté');

          // Si une adresse e-mail transporteur est fournie : télécharger le bon
          // de transport puis ouvrir le client de messagerie pré-rempli.
          const carrierEmail = (externalCarrierForm.email || '').trim();
          if (carrierEmail) {
            try {
              await exportRequestMissionPdf(requestId, { variant: 'operational' });
              toast.success('Bon téléchargé — joignez-le à l\'e-mail');
            } catch (pdfErr) {
              toast.error(pdfErr?.message || 'Erreur lors de l\'export du bon');
            }
            const requestForEmail = assignResult?.id ? assignResult : result;
            const institutionName = meData?.name || meData?.institution?.name || '';
            window.location.href = buildCarrierMailto(carrierEmail, requestForEmail, {
              institutionName,
              institutionPhone: meData?.contact_phone,
            });
          }
        } catch (assignErr) {
          toast.error(
            assignErr?.response?.data?.error
              || 'La demande a été créée, mais le transporteur externe n\'a pas été affecté.',
          );
          if (isModal && onSuccess) {
            onSuccess(result || { id: requestId });
          } else {
            navigate(`/dashboard/institution/${public_id}/requests/${requestId}`);
          }
          return;
        }
      } else {
        toast.success('Demande créée en brouillon');
      }

      const done = result || { id: requestId };
      trace.mark('submit_ui_complete', { transport_request_id: requestId });
      if (isModal && onSuccess) {
        onSuccess(done);
      } else {
        navigate(`/dashboard/institution/${public_id}/requests/${done.id}`);
      }
    } catch (err) {
      const data = err?.response?.data;
      const fieldErrors = data?.details?.fields || data?.details?.errors || data?.details;
      const firstFieldError = fieldErrors && typeof fieldErrors === 'object'
        ? Object.values(fieldErrors).flat().find((msg) => typeof msg === 'string' && msg.trim())
        : null;
      trace.mark('submit_error', {
        transport_request_id: requestId,
        http_status: err?.response?.status,
        error: data?.error || err?.message || 'unknown',
      });
      if (requestId) {
        toast.error(data?.error || firstFieldError || SEND_RETRY_TOAST);
      } else {
        toast.error(
          data?.message
            || firstFieldError
            || data?.error
            || 'Erreur lors de la création',
        );
      }
    } finally {
      setSubmitPhase(null);
    }
  };

  // ── Details indicator (count filled optional fields in accordion) ──
  const advancedFilledCount = useMemo(() => {
    let count = 0;
    if (formData.pickup_entry_point) count++;
    if (formData.pickup_instructions) count++;
    if (formData.dropoff_entry_point) count++;
    if (formData.dropoff_instructions) count++;
    if (formData.pickup_floor) count++;
    if (formData.pickup_door_code) count++;
    if (formData.floor_elevator_info) count++;
    if (formData.dropoff_floor) count++;
    if (formData.dropoff_door_code) count++;
    if (formData.requester_name) count++;
    if (formData.requester_phone) count++;
    if (formData.requester_service) count++;
    if (formData.onsite_name) count++;
    if (formData.onsite_phone) count++;
    if (formData.external_reference) count++;
    if (formData.notes) count++;
    return count;
  }, [formData.pickup_entry_point, formData.pickup_instructions, formData.dropoff_entry_point, formData.dropoff_instructions, formData.pickup_floor, formData.pickup_door_code, formData.floor_elevator_info, formData.dropoff_floor, formData.dropoff_door_code, formData.requester_name, formData.requester_phone, formData.requester_service, formData.onsite_name, formData.onsite_phone, formData.external_reference, formData.notes]);

  // ── Billing validation warnings ──
  const billingWarnings = useMemo(() => {
    const warnings = [];
    const intent = formData.billing_intent;

    if (intent === 'institution') {
      const instAddress = settingsData?.institution?.billing_address || meData?.address;
      if (!instAddress || !instAddress.trim()) {
        warnings.push({
          level: 'error',
          message: "Votre institution n'a pas d'adresse de facturation configurée. Le transporteur ne pourra pas facturer correctement.",
          action: 'Allez dans Paramètres > Facturation pour renseigner l\'adresse.',
        });
      }
    }

    if (intent === 'patient' && selectedPatient) {
      const hasAddress = selectedPatient.address && selectedPatient.address.trim();
      if (!hasAddress) {
        warnings.push({
          level: 'warning',
          message: `Le patient ${selectedPatient.last_name} n'a pas d'adresse de domicile renseignée. La facture patient pourrait être incomplète.`,
        });
      }
    }

    return warnings;
  }, [formData.billing_intent, settingsData, meData, selectedPatient]);

  const handleClose = () => {
    if (isModal) {
      onClose();
    } else {
      navigate(`/dashboard/institution/${public_id}/requests`);
    }
  };

  return (
    <div className={`${styles.formWrapper} ${isModal ? styles.formWrapperModal : ''}`} data-tour-id="institution-request-create">
      <div className={styles.pageHeader}>
        <div className={styles.pageHeaderLeft}>
          {!isModal && (
            <button className={styles.backLink} onClick={handleClose}>
              <FaArrowLeft /> Demandes
            </button>
          )}
          <h1 className={styles.pageTitle}>Nouvelle demande</h1>
          <p className={styles.pageSubtitle}>Renseignez le trajet, ajoutez les détails si nécessaire.</p>
        </div>
        {isModal && (
          <button type="button" className={styles.closeBtn} onClick={handleClose} aria-label="Fermer">
            <FaTimes />
          </button>
        )}
      </div>

      <form onSubmit={handleSubmit} className={styles.formOuter}>
        <div className={styles.form}>

        {/* ═══ COLONNE GAUCHE — Essentiel ═══ */}
        <div className={styles.columnLeft} data-tour-id="institution-request-form-left">

          {/* Patient selector + Mission type on same row */}
          <div className={styles.missionPatientRow}>
            {/* Patient selector (first = takes remaining space) */}
              <div data-tour-id="institution-request-patient">
              <label htmlFor="patient-select" className={styles.formLabel}>Patient</label>
              <AsyncCreatableSelect
                inputId="patient-select"
                aria-label="Patient"
                cacheOptions
                defaultOptions={defaultPatientOptions}
                loadOptions={loadPatientOptions}
                onChange={(option) => {
                  if (option) {
                    setSelectedPatientOption(option);
                    handlePatientChange(option.value, option.raw);
                  } else {
                    setSelectedPatientOption(null);
                    handlePatientChange('');
                  }
                }}
                onCreateOption={() => setShowQuickPatient(true)}
                value={selectedPatientOption}
                placeholder={formData.mission_type === 'material_delivery' ? 'Patient / destinataire (optionnel)…' : 'Nom, prénom ou date de naissance…'}
                formatCreateLabel={(input) => `+ Nouveau patient "${input}"`}
                noOptionsMessage={({ inputValue }) => inputValue ? 'Aucun patient trouvé' : 'Tapez pour rechercher…'}
                loadingMessage={() => 'Recherche…'}
                isClearable
                classNamePrefix="react-select"
                aria-invalid={Boolean(fieldErrors.patient_id) || undefined}
                aria-describedby={fieldErrors.patient_id ? formErrorId('patient-select') : undefined}
                filterOption={() => true}
                formatOptionLabel={(option, { context }) => {
                  if (option.__isNew__) return option.label;
                  const p = option.raw;
                  if (!p) return option.label;
                  if (context === 'value') {
                    return `${p.last_name} ${p.first_name}`;
                  }
                  return (
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8 }}>
                      <span style={{ fontWeight: 500, color: '#1E293B' }}>
                        {p.last_name} {p.first_name}
                      </span>
                      <span style={{ fontSize: '0.75rem', color: '#94A3B8', whiteSpace: 'nowrap' }}>
                        {formatDob(p.dob)}{p.external_reference ? ` · ${p.external_reference}` : ''}
                      </span>
                    </div>
                  );
                }}
                menuPortalTarget={typeof window !== 'undefined' ? document.body : null}
                menuPosition="fixed"
                styles={{
                  menuPortal: (base) => ({ ...base, zIndex: 'var(--z-modal-popover)' }),
                  control: (base) => fieldErrors.patient_id
                    ? { ...base, borderColor: 'var(--danger, #ef4444)', boxShadow: '0 0 0 1px var(--danger, #ef4444)' }
                    : base,
                }}
              />
              {fieldErrors.patient_id && (
                <p id={formErrorId('patient-select')} className={styles.fieldError} role="alert">
                  {fieldErrors.patient_id}
                </p>
              )}
            </div>

            {/* Mission type segment (right side) */}
            <div className={styles.missionSegment}>
              <button type="button"
                className={`${styles.missionBtn} ${formData.mission_type === 'patient_transport' ? styles.missionBtnActive : ''}`}
                onClick={() => handleChange('mission_type', 'patient_transport')}>
                {PATIENT_TRANSPORT_CREATE_LABEL}
              </button>
              <button type="button"
                className={`${styles.missionBtn} ${formData.mission_type === 'material_delivery' ? styles.missionBtnActive : ''}`}
                onClick={() => handleChange('mission_type', 'material_delivery')}>
                Livraison
              </button>
            </div>
          </div>

          {isMaterialDelivery(formData) && (
            <div className={styles.deliveryBlock} data-testid="institution-create-delivery-block">
              <div className={styles.deliveryBlockBadge}>{MISSION_DELIVERY_BADGE}</div>
              <label htmlFor="delivery_description" className={styles.deliveryBlockTitle}>
                Description de la livraison *
              </label>
              <textarea
                id="delivery_description"
                value={formData.delivery_description}
                onChange={(e) => handleChange('delivery_description', e.target.value)}
                placeholder="Indiquez ce qui doit être transporté et, si nécessaire, les instructions utiles au chauffeur."
                rows={3}
                required
                className={styles.detailsTextarea}
                aria-invalid={Boolean(fieldErrors.delivery_description) || undefined}
                aria-describedby={fieldErrors.delivery_description ? formErrorId('delivery_description') : 'delivery-description-hint'}
              />
              <p id="delivery-description-hint" className={styles.deliveryBlockHint}>
                Indiquez ce qui doit être transporté et, si nécessaire, les instructions utiles au chauffeur.
              </p>
              {fieldErrors.delivery_description && (
                <p id={formErrorId('delivery_description')} className={styles.fieldError} role="alert">
                  {fieldErrors.delivery_description}
                </p>
              )}
            </div>
          )}

          {selectedPatient && patientPrefilled && (
            <div className={styles.prefillTag}>
              Pré-rempli depuis {selectedPatient.first_name} {selectedPatient.last_name}
              <button type="button" className={styles.resetLink} onClick={() => { setFormData(prev => ({ ...prev, pickup_location: '', pickup_floor: '', pickup_door_code: '', floor_elevator_info: '' })); setPatientPrefilled(false); }}>Annuler</button>
            </div>
          )}


          {/* Trip type selector (segment control) */}
          <div className={styles.formGroup}>
            <label className={styles.formLabel}>Type de trajet</label>
            <div className={styles.tripSegment}>
              {availableTripTypes.map(tt => (
                <button key={tt.value} type="button"
                  className={`${styles.tripSegmentBtn} ${formData.trip_type === tt.value ? styles.tripSegmentBtnActive : ''}`}
                  onClick={() => handleTripTypeChange(tt.value)}>
                  {tt.label}
                </button>
              ))}
            </div>
          </div>

          {/* Date mission + raccourcis */}
          <div className={styles.formGroup}>
            <div className={styles.dateTimeLabelRow}>
              <label htmlFor="mission_date" className={styles.formLabel} style={{ margin: 0 }}>Date de mission *</label>
            </div>

            <div className={styles.whenRow} data-tour-id="institution-request-datetime">
              <div className={styles.missionDateField}>
                <InlineDatePicker
                  ref={missionDateRef}
                  inputId="mission_date"
                  value={formData.mission_date || (formData.scheduled_time ? formData.scheduled_time.split('T')[0] : '')}
                  onChange={(dateVal) => handleChange('mission_date', dateVal)}
                  placeholder="Date"
                  invalid={Boolean(fieldErrors.mission_date)}
                  describedBy={fieldErrors.mission_date ? formErrorId('mission_date') : undefined}
                />
              </div>
              <div className={styles.tripPills}>
                <button type="button" className={styles.whenShortcut} onClick={() => setTimeShortcut(0)} aria-label="Urgent"
                  style={formData.is_urgent ? { borderColor: 'var(--danger)', background: '#FEE2E2', color: 'var(--danger)', fontWeight: 600 } : undefined}
                  >🚨 Urgent</button>
                <button type="button" className={styles.whenShortcut} onClick={setTimeTomorrow9} aria-label="Demain 9h">Demain 9h</button>
              </div>
            </div>
            {fieldErrors.mission_date && (
              <p id={formErrorId('mission_date')} className={styles.fieldError} role="alert">
                {fieldErrors.mission_date}
              </p>
            )}

            {scheduleIncoherence && (
              <div className={styles.scheduleCoherenceWarning} role="status">
                ⚠ {scheduleIncoherence}
              </div>
            )}
          </div>

          {/* Addresses with visual route */}
          <div className={styles.formGroup} data-tour-id="institution-request-destination">
            <div className={styles.routeBlock}>
              {/* Départ (origine) — draggable et éditable comme les autres points */}
              <label htmlFor="pickup_location" className={styles.routeLabel}>Départ</label>
              <span className={styles.routeDot} />
              <div
                className={`${styles.routeStepRow} ${dragIndex === 0 ? styles.routeStepDragging : ''} ${dragOverIndex === 0 && dragIndex !== null && dragIndex !== 0 ? styles.routeStepDropTarget : ''}`}
                onDragOver={(e) => {
                  e.preventDefault();
                  if (dragOverIndex !== 0) setDragOverIndex(0);
                }}
                onDrop={(e) => {
                  e.preventDefault();
                  if (dragIndex !== null) moveRoutePoint(dragIndex, 0);
                  setDragIndex(null);
                  setDragOverIndex(null);
                }}
              >
                <span
                  className={styles.routeStepHandle}
                  role="button"
                  tabIndex={0}
                  draggable
                  title="Faire glisser pour réorganiser"
                  aria-label="Réorganiser le départ"
                  onDragStart={() => setDragIndex(0)}
                  onDragEnd={() => { setDragIndex(null); setDragOverIndex(null); }}
                >
                  <FaGripVertical size={11} />
                </span>
                {renderPickupField(styles.routeInput, { editable: true })}
                <RouteStepTimeField
                  ref={pickupTimeRef}
                  inputId="pickup_time"
                  label="Heure de départ"
                  timeValue={formData.pickup_time}
                  invalid={Boolean(fieldErrors.pickup_time || fieldErrors.confirmed_time)}
                  describedBy={
                    (fieldErrors.pickup_time && formErrorId('pickup_time'))
                    || (fieldErrors.confirmed_time && formErrorId('confirmed_time'))
                    || undefined
                  }
                  onTimeChange={(v) => {
                    handleChange('pickup_time', v);
                    handleChange('pickup_time_confirmed', derivePickupTimeConfirmed(v));
                  }}
                />
                <span className={styles.routeStepRemoveSpacer} aria-hidden="true" />
              </div>
              {(fieldErrors.pickup_location || fieldErrors.pickup_time) && (
                <p
                  id={formErrorId(fieldErrors.pickup_location ? 'pickup_location' : 'pickup_time')}
                  className={`${styles.fieldError} ${styles.fieldErrorRoute}`}
                  role="alert"
                >
                  {fieldErrors.pickup_location || fieldErrors.pickup_time}
                </p>
              )}

              {/* Destinations intermédiaires (rendues avant le domicile en mode Retour domicile). */}
              {(() => {
                const extrasNodes = extraStops.map((stop, idx) => {
                  const combinedIdx = idx + 2;
                  const isLastExtra = idx === extraStops.length - 1;
                  // En mode domicile, le domicile reste l'étape finale → jamais dotEnd ici.
                  const dotEnd = !isReturnHome && isLastExtra && !journeyReturnEnabled;
                  // Numérotation : 1..N en mode domicile (pas de destination principale avant), sinon 2..N.
                  const destNumber = isReturnHome ? idx + 1 : idx + 2;
                  const isDragging = dragIndex === combinedIdx;
                  const isDropTarget = dragOverIndex === combinedIdx && dragIndex !== null && dragIndex !== combinedIdx;
                  return (
                    <React.Fragment key={idx}>
                      <span className={styles.routeConnector} />
                      <label className={styles.routeLabel}>Destination {destNumber}</label>
                      <span className={`${styles.routeDot} ${dotEnd ? styles.routeDotEnd : ''}`} />
                      <div
                        className={`${styles.routeStepRow} ${isDragging ? styles.routeStepDragging : ''} ${isDropTarget ? styles.routeStepDropTarget : ''}`}
                        onDragOver={(e) => {
                          e.preventDefault();
                          if (dragOverIndex !== combinedIdx) setDragOverIndex(combinedIdx);
                        }}
                        onDrop={(e) => {
                          e.preventDefault();
                          if (dragIndex !== null) moveRoutePoint(dragIndex, combinedIdx);
                          setDragIndex(null);
                          setDragOverIndex(null);
                        }}
                      >
                        <span
                          className={styles.routeStepHandle}
                          role="button"
                          tabIndex={0}
                          draggable
                          title="Faire glisser pour réorganiser"
                          aria-label={`Réorganiser la destination ${destNumber}`}
                          onDragStart={() => setDragIndex(combinedIdx)}
                          onDragEnd={() => { setDragIndex(null); setDragOverIndex(null); }}
                        >
                          <FaGripVertical size={11} />
                        </span>
                        <AddressAutocomplete
                          name={`intermediate_stop_${idx}`}
                          inputId={`intermediate_stop_${idx}`}
                          value={stop.dropoff_location || ''}
                          onChange={(e) => setStopAddress(idx, e)}
                          onSelect={(place) => setStopFromSelection(idx, place)}
                          placeholder={`Adresse destination ${destNumber}`}
                          inputClassName={styles.routeInput}
                          aria-invalid={Boolean(
                            fieldErrors[`extra_stop_location_${idx}`] || fieldErrors[`extra_stop_time_${idx}`],
                          ) || undefined}
                          aria-describedby={
                            fieldErrors[`extra_stop_location_${idx}`]
                              ? formErrorId(`intermediate_stop_${idx}`)
                              : fieldErrors[`extra_stop_time_${idx}`]
                                ? formErrorId(`intermediate_stop_time_${idx}`)
                                : undefined
                          }
                        />
                      <RouteStepTimeField
                        inputId={`intermediate_stop_time_${idx}`}
                        label={`Heure du rendez-vous ${destNumber}`}
                        timeValue={extractHHMM(stop.scheduled_time) || stop.scheduled_time?.split('T')[1]?.slice(0, 5) || ''}
                        timeConfirmed={Boolean(stop.time_confirmed)}
                        invalid={Boolean(fieldErrors[`extra_stop_time_${idx}`] || fieldErrors.confirmed_time)}
                        describedBy={fieldErrors[`extra_stop_time_${idx}`]
                          ? formErrorId(`intermediate_stop_time_${idx}`)
                          : fieldErrors.confirmed_time
                            ? formErrorId('confirmed_time')
                            : undefined}
                        onTimeChange={(v) => setStopTime(idx, v)}
                      />
                        <button
                          type="button"
                          className={styles.routeStepRemove}
                          title="Supprimer cette destination"
                          aria-label={`Supprimer la destination ${destNumber}`}
                          onClick={() => removeExtraStop(idx)}
                        >
                          <FaTimes size={12} />
                        </button>
                      </div>
                      {(fieldErrors[`extra_stop_location_${idx}`] || fieldErrors[`extra_stop_time_${idx}`]) && (
                        <p
                          id={formErrorId(
                            fieldErrors[`extra_stop_location_${idx}`]
                              ? `intermediate_stop_${idx}`
                              : `intermediate_stop_time_${idx}`,
                          )}
                          className={`${styles.fieldError} ${styles.fieldErrorRoute}`}
                          role="alert"
                        >
                          {fieldErrors[`extra_stop_location_${idx}`] || fieldErrors[`extra_stop_time_${idx}`]}
                        </p>
                      )}
                    </React.Fragment>
                  );
                });

                const dropoffNode = (
                  <React.Fragment key="dropoff">
                    <span className={styles.routeConnector} />
                    <label htmlFor="dropoff_location" className={styles.routeLabel}>
                      {isReturnHome ? 'Domicile' : 'Destination'}
                    </label>
                    <span className={`${styles.routeDot} ${((isReturnHome || (!hasExtraStops && !journeyReturnEnabled))) ? styles.routeDotEnd : ''}`} />
                    <div
                      className={`${styles.routeStepRow} ${dragIndex === 1 ? styles.routeStepDragging : ''} ${dragOverIndex === 1 && dragIndex !== null && dragIndex !== 1 ? styles.routeStepDropTarget : ''}`}
                      onDragOver={(e) => {
                        e.preventDefault();
                        if (dragOverIndex !== 1) setDragOverIndex(1);
                      }}
                      onDrop={(e) => {
                        e.preventDefault();
                        if (dragIndex !== null) moveRoutePoint(dragIndex, 1);
                        setDragIndex(null);
                        setDragOverIndex(null);
                      }}
                    >
                      <span
                        className={styles.routeStepHandle}
                        role="button"
                        tabIndex={0}
                        draggable
                        title="Faire glisser pour réorganiser"
                        aria-label={isReturnHome ? 'Réorganiser le domicile' : 'Réorganiser la destination'}
                        onDragStart={() => setDragIndex(1)}
                        onDragEnd={() => { setDragIndex(null); setDragOverIndex(null); }}
                      >
                        <FaGripVertical size={11} />
                      </span>
                      {renderDropoffField(styles.routeInput)}
                      <RouteStepTimeField
                        ref={dropoffTimeRef}
                        inputId="dropoff_time"
                        label={isReturnHome ? "Heure d'arrivée au domicile" : 'Heure du rendez-vous'}
                        timeValue={formData.dropoff_time}
                        invalid={Boolean(fieldErrors.dropoff_time || fieldErrors.confirmed_time)}
                        describedBy={
                          (fieldErrors.dropoff_time && formErrorId('dropoff_time'))
                          || (fieldErrors.confirmed_time && formErrorId('confirmed_time'))
                          || undefined
                        }
                        onTimeChange={(v) => {
                          handleChange('dropoff_time', v);
                          handleChange('dropoff_time_confirmed', Boolean(v?.trim()));
                        }}
                      />
                      <span className={styles.routeStepRemoveSpacer} aria-hidden="true" />
                    </div>
                    {(fieldErrors.dropoff_location || fieldErrors.dropoff_time) && (
                      <p
                        id={formErrorId(fieldErrors.dropoff_location ? 'dropoff_location' : 'dropoff_time')}
                        className={`${styles.fieldError} ${styles.fieldErrorRoute}`}
                        role="alert"
                      >
                        {fieldErrors.dropoff_location || fieldErrors.dropoff_time}
                      </p>
                    )}
                  </React.Fragment>
                );

                // Retour domicile : étapes intermédiaires AVANT le domicile (étape finale).
                // Sinon : destination principale puis étapes supplémentaires.
                return isReturnHome
                  ? <>{extrasNodes}{dropoffNode}</>
                  : <>{dropoffNode}{extrasNodes}</>;
              })()}

              {/* Retour institution — ligne identique, lecture seule */}
              {journeyReturnEnabled && (
                <>
                  <span className={styles.routeConnector} />
                  <label className={styles.routeLabel}>Retour</label>
                  <span className={`${styles.routeDot} ${styles.routeDotEnd}`} />
                  <div className={styles.routeStepRow}>
                    <span className={styles.routeStepHandleSpacer} aria-hidden="true" />
                    <input
                      type="text"
                      value={multiStopOrigin || ''}
                      readOnly
                      title={multiStopOrigin}
                      className={`${styles.routeInput} ${styles.routeReadonly}`}
                      placeholder="Adresse de l'institution"
                    />
                    <RouteStepTimeField
                      ref={returnTimeRef}
                      inputId="return_time"
                      label="Heure de retour"
                      timeValue={formData.return_time}
                      invalid={Boolean(fieldErrors.return_time || fieldErrors.confirmed_time)}
                      describedBy={
                        (fieldErrors.return_time && formErrorId('return_time'))
                        || (fieldErrors.confirmed_time && formErrorId('confirmed_time'))
                        || undefined
                      }
                      onTimeChange={(v) => {
                        handleChange('return_time', v);
                        handleChange('return_time_confirmed', Boolean(v?.trim()));
                      }}
                    />
                    <button
                      type="button"
                      className={styles.routeStepRemove}
                      title="Retirer le retour à l'institution"
                      aria-label="Retirer le retour à l'institution"
                      onClick={() => handleChange('return_to_institution', false)}
                    >
                      <FaTimes size={12} />
                    </button>
                  </div>
                  {fieldErrors.return_time && (
                    <p id={formErrorId('return_time')} className={`${styles.fieldError} ${styles.fieldErrorRoute}`} role="alert">
                      {fieldErrors.return_time}
                    </p>
                  )}
                </>
              )}
            </div>

            {fieldErrors.confirmed_time && (
              <div id="confirmed_time" className={styles.fieldErrorRouteTimes}>
                <p id={formErrorId('confirmed_time')} className={styles.fieldError} role="alert">
                  {fieldErrors.confirmed_time}
                </p>
              </div>
            )}

            {/* Actions itinéraire : ajouter une destination / retour — sans changement de mode */}
            <div className={styles.routeActions}>
              <button type="button" className={styles.addStepBtn} onClick={addExtraStop}>
                <span className={styles.journeyAddIcon}><FaPlus size={11} /></span>
                Ajouter une destination
              </button>
              <button
                type="button"
                className={`${styles.routeReturnBtn} ${journeyReturnEnabled ? styles.routeReturnBtnActive : ''}`}
                aria-pressed={journeyReturnEnabled}
                title="Aller / retour : ajoute le retour à l'institution en fin de parcours"
                onClick={() => handleChange('return_to_institution', !journeyReturnEnabled)}
              >
                ⇄ A/R
              </button>
            </div>
          </div>

          {/* Billing inline (left column, compact) */}
          <div className={styles.formGroup}>
            <label htmlFor="billing_intent" className={styles.formLabel}>Facturé à</label>
            <ChipSelect
              id="billing_intent"
              options={BILLING_INTENTS}
              value={formData.billing_intent}
              onChange={(val) => handleChange('billing_intent', val)}
              placeholder="Facturé à"
              disabled={!canSetBillingOnCreate}
            />
            {!canSetBillingOnCreate && <span className={styles.billingHint}>Géré par l'institution</span>}
            {fieldErrors.billing && (
              <p id={formErrorId('billing_intent')} className={styles.fieldError} role="alert">
                {fieldErrors.billing}
              </p>
            )}
          </div>

          {/* Contact demandeur */}
          <div className={styles.formGroup}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
              <label htmlFor="requester_name" className={styles.formLabel} style={{ margin: 0 }}>Contact</label>
              <button type="button"
                onClick={() => {
                  const next = !formData.onsite_is_different;
                  handleChange('onsite_is_different', next);
                  if (next) setFormData(prev => ({ ...prev, onsite_name: '', onsite_phone: '' }));
                }}
                style={{
                  display: 'inline-flex', alignItems: 'center', gap: 4,
                  padding: '3px 10px', border: '1px solid var(--border)', borderRadius: 'var(--radius-pill, 20px)',
                  background: formData.onsite_is_different ? 'var(--brand-light, #E0F2F1)' : 'transparent',
                  color: formData.onsite_is_different ? 'var(--brand, #00796B)' : 'var(--text-muted, #94A3B8)',
                  fontSize: '0.72rem', fontWeight: 500, cursor: 'pointer', lineHeight: 1,
                  transition: 'all 0.15s ease',
                }}>
                {formData.onsite_is_different ? '✕ Annuler' : '✎ Modifier'}
              </button>
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              {formData.onsite_is_different ? (
                <>
                  <input type="text" id="onsite_name" value={formData.onsite_name} onChange={(e) => handleChange('onsite_name', e.target.value)}
                    placeholder="Nom" className={styles.inputFull} style={{ flex: 1 }} />
                  <input type="tel" id="onsite_phone" value={formData.onsite_phone} onChange={(e) => handleChange('onsite_phone', e.target.value)}
                    placeholder="Tél. (+41...)" className={styles.inputFull} style={{ flex: 1 }} />
                  <input type="text" id="requester_service" value={formData.requester_service} onChange={(e) => handleChange('requester_service', e.target.value)}
                    placeholder="Service" className={styles.inputFull} style={{ flex: 1 }} />
                </>
              ) : (
                <>
                  <input type="text" id="requester_name" value={formData.requester_name} readOnly
                    placeholder="Nom" className={`${styles.inputFull} ${styles.readonlyInput}`} style={{ flex: 1 }} />
                  <input type="tel" id="requester_phone" value={formData.requester_phone} readOnly
                    placeholder="Tél." className={`${styles.inputFull} ${styles.readonlyInput}`} style={{ flex: 1 }} />
                  <input type="text" id="requester_service" value={formData.requester_service} onChange={(e) => handleChange('requester_service', e.target.value)}
                    placeholder="Service" className={styles.inputFull} style={{ flex: 1 }} />
                </>
              )}
            </div>
          </div>
        </div>

        {/* ═══ COLONNE DROITE — Détails & contexte ═══ */}
        <div className={styles.columnRight} data-tour-id="institution-request-form-tooltip">
          <div className={styles.detailsPanel} ref={destinationDetailsRef}>

            {/* ═══ SECTION 1 — Infos départ ═══ */}
            <h2 className={styles.detailsPanelTitle}>📍 Départ</h2>

            {/* Départ institution → Service / Bâtiment */}
            {formData.pickup_type === 'institution' && (
              <div className={styles.detailsGroup}>
                <label htmlFor="pickup_entry_point" className={styles.detailsLabel}>Service / Bâtiment</label>
                <input type="text" list="entry_points_list" value={formData.pickup_entry_point}
                  onChange={(e) => handleChange('pickup_entry_point', e.target.value)}
                  placeholder="Ex: Réception, Bât. C, Étage 3" className={styles.detailsInput} id="pickup_entry_point" />
                <datalist id="entry_points_list">
                  {(settingsData?.settings?.entry_points || []).map((ep, i) => (<option key={i} value={ep} />))}
                </datalist>
              </div>
            )}

            {/* Départ externe (après swap ou type "autre") → Établissement / Service / Médecin */}
            {formData.pickup_type === 'other' && (
              <>
                <div className={styles.detailsGroup}>
                  <label htmlFor="pickup_establishment" className={styles.detailsLabel}>Établissement / Lieu</label>
                  <input type="text" id="pickup_establishment" value={formData.pickup_establishment || ''}
                    onChange={(e) => handleChange('pickup_establishment', e.target.value)}
                    placeholder="Ex: HUG, Clinique des Grangettes" className={styles.detailsInput} />
                </div>
                <div className={styles.detailsGroup}>
                  <label htmlFor="pickup_service" className={styles.detailsLabel}>Service</label>
                  <input type="text" id="pickup_service" value={formData.pickup_service || ''}
                    onChange={(e) => handleChange('pickup_service', e.target.value)}
                    placeholder="Ex: Radiologie, Urgences, Cardiologie" className={styles.detailsInput} />
                </div>
                <div className={styles.detailsGroup}>
                  <label htmlFor="pickup_doctor" className={styles.detailsLabel}>Médecin</label>
                  <input type="text" id="pickup_doctor" value={formData.pickup_doctor || ''}
                    onChange={(e) => handleChange('pickup_doctor', e.target.value)}
                    placeholder="Ex: Dr. Martin, Prof. Dupont" className={styles.detailsInput} />
                </div>
              </>
            )}

            {/* Départ domicile → Accès domicile */}
            {formData.pickup_type === 'domicile' && (
              <div className={styles.detailsGroup}>
                <label className={styles.detailsLabel}>Accès domicile</label>
                <div className={styles.detailsRow}>
                  <input type="text" id="pickup_floor" value={formData.pickup_floor} onChange={(e) => handleChange('pickup_floor', e.target.value)}
                    placeholder="Étage" className={styles.detailsInputSm} />
                  <input type="text" id="pickup_door_code" value={formData.pickup_door_code} onChange={(e) => handleChange('pickup_door_code', e.target.value)}
                    placeholder="Code porte" className={styles.detailsInputSm} />
                  <input type="text" id="pickup_entry_point_dom" value={formData.pickup_entry_point} onChange={(e) => handleChange('pickup_entry_point', e.target.value)}
                    placeholder="Accueil" className={styles.detailsInputSm} />
                </div>
                <input type="text" id="floor_elevator_info" value={formData.floor_elevator_info} onChange={(e) => handleChange('floor_elevator_info', e.target.value)}
                  placeholder="Ex: concierge, ascenseur gauche" className={styles.detailsInput} style={{ marginTop: 6 }} />
              </div>
            )}

            <hr className={styles.detailsDivider} />

            {/* ═══ SECTION 2 — Infos arrivée (destination principale) ═══ */}
            <h2 className={styles.detailsPanelTitle}>
              {formData.dropoff_type === 'domicile' ? '🏠 Arrivée — Domicile' : '🏥 Arrivée'}
            </h2>

            {/* Arrivée institution → Service / Bâtiment */}
            {formData.dropoff_type === 'institution' && (
              <div className={styles.detailsGroup}>
                <label htmlFor="dropoff_entry_point" className={styles.detailsLabel}>Service / Bâtiment</label>
                <input type="text" list="entry_points_list_dropoff" value={formData.dropoff_entry_point}
                  onChange={(e) => handleChange('dropoff_entry_point', e.target.value)}
                  placeholder="Ex: Réception, Bât. C, Étage 3" className={styles.detailsInput} id="dropoff_entry_point" />
                <datalist id="entry_points_list_dropoff">
                  {(settingsData?.settings?.entry_points || []).map((ep, i) => (<option key={i} value={ep} />))}
                </datalist>
              </div>
            )}

            {/* Arrivée externe (destination standard) → Établissement / Service / Médecin */}
            {formData.dropoff_type === 'other' && (
              <MedicalDestinationDetails
                establishmentId="dropoff_establishment"
                serviceId="dropoff_service"
                doctorId="dropoff_doctor"
                establishment={formData.dropoff_establishment}
                service={formData.dropoff_service}
                doctor={formData.dropoff_doctor}
                destinationType={formData.destination_type || DESTINATION_TYPE_MEDICAL}
                onDestinationTypeChange={(value) => handleChange('destination_type', value)}
                onEstablishmentChange={(e) => handleChange('dropoff_establishment', e.target.value)}
                onServiceChange={(e) => handleChange('dropoff_service', e.target.value)}
                onDoctorChange={(e) => handleChange('dropoff_doctor', e.target.value)}
                  showError={Boolean(fieldErrors.medical_principal)}
                  requireServiceOrDoctor={formData.mission_type !== 'material_delivery'}
                />
              )}

            {/* Arrivée domicile → Accès domicile */}
            {formData.dropoff_type === 'domicile' && (
              <>
                <div className={styles.detailsGroup}>
                  <label className={styles.detailsLabel}>Accès domicile</label>
                  <div className={styles.detailsRow}>
                    <input type="text" id="dropoff_floor" value={formData.dropoff_floor} onChange={(e) => handleChange('dropoff_floor', e.target.value)}
                      placeholder="Étage" className={styles.detailsInputSm} />
                    <input type="text" id="dropoff_door_code" value={formData.dropoff_door_code} onChange={(e) => handleChange('dropoff_door_code', e.target.value)}
                      placeholder="Code porte" className={styles.detailsInputSm} />
                    <input type="text" id="dropoff_entry_point_dom" value={formData.dropoff_entry_point} onChange={(e) => handleChange('dropoff_entry_point', e.target.value)}
                      placeholder="Accueil" className={styles.detailsInputSm} />
                  </div>
                </div>
                <div className={styles.detailsGroup}>
                  <label htmlFor="dropoff_instructions_dom" className={styles.detailsLabel}>Consignes arrivée</label>
                  <input type="text" id="dropoff_instructions_dom" value={formData.dropoff_instructions || ''}
                    onChange={(e) => handleChange('dropoff_instructions', e.target.value)}
                    placeholder="Ex: concierge, ascenseur gauche, sonner 2x" className={styles.detailsInput} />
                </div>
              </>
            )}

            {canSetBillingOnCreate && (hasExtraStops || journeyReturnEnabled) && (
              <DestinationBillingOverride
                idPrefix="dropoff-billing"
                useCustomBilling={formData.dropoff_use_custom_billing}
                billingOverride={formData.dropoff_destination_billing_override}
                onUseCustomBillingChange={(checked) => handleChange('dropoff_use_custom_billing', checked)}
                onBillingOverrideChange={(val) => handleChange('dropoff_destination_billing_override', val)}
                disabled={!canSetBillingOnCreate}
              />
            )}

            {/* ═══ SECTION 2bis — Détails des destinations supplémentaires ═══ */}
            {extraStops.map((stop, idx) => (
              <React.Fragment key={`extra-details-${idx}`}>
                <hr className={styles.detailsDivider} />
                <h2 className={styles.detailsPanelTitle}>🏥 Destination {idx + 2}</h2>
                <MedicalDestinationDetails
                  establishmentId={`stop_establishment_${idx}`}
                  serviceId={`stop_service_${idx}`}
                  doctorId={`stop_doctor_${idx}`}
                  establishment={stop.dropoff_establishment}
                  service={stop.dropoff_service}
                  doctor={stop.dropoff_doctor}
                  destinationType={stop.destination_type || DESTINATION_TYPE_MEDICAL}
                  onDestinationTypeChange={(value) => setStopField(idx, 'destination_type', value)}
                  onEstablishmentChange={(e) => setStopField(idx, 'dropoff_establishment', e.target.value)}
                  onServiceChange={(e) => setStopField(idx, 'dropoff_service', e.target.value)}
                  onDoctorChange={(e) => setStopField(idx, 'dropoff_doctor', e.target.value)}
                  showError={Boolean(fieldErrors[`medical_extra_${idx}`])}
                  requireServiceOrDoctor={formData.mission_type !== 'material_delivery'}
                />
                {canSetBillingOnCreate && (
                  <DestinationBillingOverride
                    idPrefix={`stop-billing-${idx}`}
                    useCustomBilling={stop.use_custom_billing}
                    billingOverride={stop.destination_billing_override}
                    onUseCustomBillingChange={(checked) => setStopField(idx, 'use_custom_billing', checked)}
                    onBillingOverrideChange={(val) => setStopField(idx, 'destination_billing_override', val)}
                    disabled={!canSetBillingOnCreate}
                  />
                )}
              </React.Fragment>
            ))}

            {canSetBillingOnCreate && journeyReturnEnabled && (
              <>
                <hr className={styles.detailsDivider} />
                <h2 className={styles.detailsPanelTitle}>⇄ Retour institution</h2>
                <DestinationBillingOverride
                  idPrefix="return-billing"
                  useCustomBilling={formData.return_stop?.use_custom_billing}
                  billingOverride={formData.return_stop?.destination_billing_override}
                  onUseCustomBillingChange={(checked) => setFormData((prev) => ({
                    ...prev,
                    return_stop: { ...prev.return_stop, use_custom_billing: checked },
                  }))}
                  onBillingOverrideChange={(val) => setFormData((prev) => ({
                    ...prev,
                    return_stop: { ...prev.return_stop, destination_billing_override: val },
                  }))}
                  disabled={!canSetBillingOnCreate}
                />
              </>
            )}


            <hr className={styles.detailsDivider} />

            {/* ═══ SECTION 3 — Infos patient ═══ */}
            <h2 className={styles.detailsPanelTitle}>
              {isMaterialDelivery(formData) ? 'Bénéficiaire & contact' : '👤 Patient & contact'}
            </h2>

            {formData.mission_type === 'patient_transport' && (
              <div className={styles.needsChips} style={{ marginBottom: 10, flexWrap: 'nowrap' }}>
                {/* Fauteuil / Prendre chaise — mutuellement exclusifs */}
                <button type="button" aria-pressed={formData.requires_wheelchair} style={{ flex: 1 }}
                  className={`${styles.needsChip} ${formData.requires_wheelchair ? styles.needsChipActive : ''}`}
                  onClick={() => setFormData(prev => ({ ...prev, requires_wheelchair: !prev.requires_wheelchair, requires_vehicle_wheelchair: false }))}>
                  ♿ Fauteuil
                </button>
                <button type="button" aria-pressed={formData.requires_vehicle_wheelchair} style={{ flex: 1 }}
                  className={`${styles.needsChip} ${formData.requires_vehicle_wheelchair ? styles.needsChipActive : ''}`}
                  onClick={() => setFormData(prev => ({ ...prev, requires_vehicle_wheelchair: !prev.requires_vehicle_wheelchair, requires_wheelchair: false }))}>
                  🏥 Prendre chaise
                </button>
                {/* Assistance — indépendant */}
                <button type="button" aria-pressed={formData.requires_assistance} style={{ flex: 1 }}
                  className={`${styles.needsChip} ${formData.requires_assistance ? styles.needsChipActive : ''}`}
                  onClick={() => handleChange('requires_assistance', !formData.requires_assistance)}>Assistance</button>
              </div>
            )}

            {isMaterialDelivery(formData) ? (
              <div className={styles.detailsGroup}>
                <label htmlFor="patient_notes" className={styles.detailsLabel}>Notes complémentaires</label>
                <textarea
                  id="patient_notes"
                  value={formData.notes}
                  onChange={(e) => handleChange('notes', e.target.value)}
                  placeholder="Notes libres, distinctes de la description de livraison…"
                  rows={2}
                  className={styles.detailsTextarea}
                />
              </div>
            ) : (
              <div className={styles.detailsGroup}>
                <label htmlFor="patient_notes" className={styles.detailsLabel}>
                  Pathologie / Difficultés {formData.requires_assistance && <span style={{ color: 'var(--danger, #e53935)', fontWeight: 600 }}>*</span>}
                </label>
                <textarea
                  id="patient_notes"
                  value={formData.notes}
                  onChange={(e) => handleChange('notes', e.target.value)}
                  placeholder={formData.requires_assistance ? 'Obligatoire — décrivez le besoin d\'assistance…' : 'Ex: patient anxieux, mobilité réduite, sous perfusion…'}
                  rows={2}
                  className={styles.detailsTextarea}
                  required={formData.requires_assistance}
                  aria-invalid={Boolean(fieldErrors.notes) || undefined}
                  aria-describedby={fieldErrors.notes ? formErrorId('patient_notes') : undefined}
                />
                {fieldErrors.notes && (
                  <p id={formErrorId('patient_notes')} className={styles.fieldError} role="alert">
                    {fieldErrors.notes}
                  </p>
                )}
              </div>
            )}

            <div className={styles.detailsGroup}>
              <label htmlFor="external_reference" className={styles.detailsLabel}>Réf. DPI / dossier</label>
              <input type="text" id="external_reference" value={formData.external_reference} onChange={(e) => handleChange('external_reference', e.target.value)}
                placeholder="Ex: 2024-12345" className={styles.detailsInput} />
            </div>


          </div>
        </div>
        </div>

        {/* Champs transporteur externe (au-dessus du footer) */}
        {isExternalMode && (
          <div className={styles.externalFieldsWrap} id="external-carrier">
            <ExternalCarrierFields
              value={externalCarrierForm}
              onChange={(next) => {
                setExternalCarrierForm(next);
                clearFieldErrors('external_carrier');
              }}
              idPrefix="create-external-carrier"
            />
            {fieldErrors.external_carrier && (
              <p id={formErrorId('external-carrier')} className={styles.fieldError} role="alert">
                {fieldErrors.external_carrier}
              </p>
            )}
          </div>
        )}

        {/* ═══ Footer (ManualBookingForm pattern) ═══ */}
        <div className={styles.formFooter}>
          <div className={styles.footerBody}>
            <div className={styles.footerLeft}>
              <button type="button" className={styles.btnGhost} onClick={handleClose}>Annuler</button>
              <span className={styles.footerSummaryText}>
                {isMaterialDelivery(formData)
                  ? `${formatMissionTypeLabel(formData.mission_type)} · ${formatDeliveryDescriptionDisplay(formData)}`
                  : (selectedPatient ? `${selectedPatient.first_name} ${selectedPatient.last_name}` : 'Patient non sélectionné')}
                {formData.mission_date
                  ? ` · ${formatWallClockDateShort(formData.mission_date)}`
                  : (formData.scheduled_time
                    ? (() => {
                        const { date, time } = formatWallClockDateTime(formData.scheduled_time);
                        return ` · ${date}${time ? ` ${time}` : ''}`;
                      })()
                    : '')}
              </span>
              <span className={styles.footerBadges}>
                {isMaterialDelivery(formData) && (
                  <span className={styles.footerBadge}>{MISSION_DELIVERY_BADGE}</span>
                )}
                {isLirieSendMode && <span className={styles.footerBadge}>Envoi auto</span>}
                {isExternalMode && <span className={styles.footerBadge}>Externe</span>}
                {advancedFilledCount > 0 && <span className={styles.footerBadge}>{advancedFilledCount} détail{advancedFilledCount > 1 ? 's' : ''}</span>}
              </span>
            </div>
            <div className={styles.footerRight}>
              <div className={styles.executionModeOptions} role="radiogroup" aria-label="Mode d'exécution">
                <label className={styles.executionModeOption}>
                  <input
                    type="radio"
                    name="execution_mode"
                    value="draft"
                    checked={isDraftMode}
                    disabled={submitBusy}
                    onChange={() => setExecutionMode('draft')}
                  />
                  <span>Brouillon</span>
                </label>
                <label className={styles.executionModeOption}>
                  <input
                    type="radio"
                    name="execution_mode"
                    value="lirie"
                    checked={isLirieSendMode}
                    disabled={submitBusy}
                    onChange={() => setExecutionMode('lirie')}
                  />
                  <span>LIRIE</span>
                </label>
                <label className={styles.executionModeOption}>
                  <input
                    type="radio"
                    name="execution_mode"
                    value="external"
                    checked={isExternalMode}
                    disabled={submitBusy}
                    onChange={() => setExecutionMode('external')}
                  />
                  <span>Externe</span>
                </label>
              </div>
              <button
                type="submit"
                className={styles.btnPrimary}
                disabled={submitBusy}
                data-tour-id="institution-request-submit"
              >
                {submitLabel === 'Créer le brouillon' && <><FaSave /> Créer le brouillon</>}
                {submitLabel !== 'Créer le brouillon' && submitLabel !== 'Enregistrer' && (
                  <><FaPaperPlane /> {submitLabel}</>
                )}
                {submitLabel === 'Enregistrer' && submitLabel}
              </button>
            </div>
          </div>
        </div>
      </form>

      {/* ══════ Modal création patient (hors du <form> pour éviter form imbriqué) ══════ */}
      {showQuickPatient && (
        <PatientFormModal
          onClose={() => setShowQuickPatient(false)}
          onSaved={(patient) => {
            if (patient) {
              setSelectedPatientOption(formatPatientOption(patient));
              handlePatientChange(patient.id, patient);
              toast.success(`Patient ${patient.last_name} ${patient.first_name} ajouté`);
            }
          }}
        />
      )}


      
    </div>
  );
};

export default InstitutionRequestCreate;
