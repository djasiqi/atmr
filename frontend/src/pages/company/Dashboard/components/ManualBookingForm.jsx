// frontend/src/pages/company/Dashboard/components/ManualBookingForm.jsx (fixed)

import React, { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import NewClientModal from '../../Clients/components/NewClientModal';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import {
  createManualBooking,
  previewManualBookingPricing,
  searchClients,
  createClient,
  fetchClientActiveStay,
  createClientStay,
  linkClientBillingParty,
} from '../../../../services/companyService';
import Input from './ui/Input';
import Label from './ui/Label';
import {
  useIsolatedField,
  IsolatedTextarea,
} from './ui/IsolatedFormField';
import { buildCanonicalReservationPayload } from './manualBookingCanonicalPayload';
import CompanyRouteBuilder, { CompanyRouteDetails } from './CompanyRouteBuilder';
import {
  createDestinationPoint,
  createInitialRoute,
  reorderRoutePoints,
  segmentLabels,
} from './companyRouteModel';
import apiClient from '../../../../utils/apiClient';
import { fetchBillingSettings, simulatePricing } from '../../../../services/settingsService';

// ⬇️ Assure-toi que ces chemins correspondent à ta structure réelle
import { extractMedicalServiceInfo } from '../../../../utils/medicalExtract';
import { shortAddress } from './formatAddress';
import { toast } from 'sonner';
import { getApiErrorMessage } from '../../../../utils/apiErrorMessage';
import styles from './ManualBookingForm.module.css';
import InlineDatePicker from '../../../../components/ui/InlineDatePicker';
import ManualBookingClientSelect from './ManualBookingClientSelect';
import { useBookingFormFocusGuard } from './useBookingFormFocusGuard';

/** Trim + vide → null. Réutilisable pour champs optionnels (medical_facility, hospital_service, notes_medical, etc.). */
function cleanOptionalText(s) {
  if (s == null) return null;
  const t = String(s).trim();
  return t === '' ? null : t;
}

const ensureIsoDatetimeWithSeconds = (value) => {
  if (!value || typeof value !== 'string') {
    return value;
  }

  const trimmed = value.trim();

  if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/.test(trimmed)) {
    return `${trimmed}:00`;
  }

  if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$/.test(trimmed)) {
    return trimmed;
  }

  return trimmed;
};

// ⚡ Helper pour combiner date et time en datetime ISO 8601 (comme ensureIsoDatetimeWithSeconds)
const combineDateAndTime = (dateStr, timeStr) => {
  if (!dateStr || !dateStr.trim()) return undefined;
  if (!timeStr || !timeStr.trim()) return undefined;

  // 🔍 Debug
  console.log('[combineDateAndTime] Input - dateStr:', dateStr, 'timeStr:', timeStr);

  // Nettoyer dateStr : extraire uniquement YYYY-MM-DD (au cas où c'est déjà un datetime)
  const dateMatch = dateStr.trim().match(/^(\d{4}-\d{2}-\d{2})/);
  if (!dateMatch) {
    console.warn('[combineDateAndTime] Format de date invalide:', dateStr);
    return undefined;
  }
  const cleanDate = dateMatch[1]; // YYYY-MM-DD
  console.log('[combineDateAndTime] cleanDate:', cleanDate);

  // Nettoyer timeStr : extraire uniquement HH:mm (au cas où c'est déjà un datetime complet)
  // Supprimer toute date qui pourrait être présente dans timeStr
  let cleanTime = String(timeStr).trim();

  // ⚡ Si timeStr contient un 'T', c'est probablement un datetime complet
  // Extraire seulement la partie time après le dernier 'T'
  if (cleanTime.includes('T')) {
    // Utiliser lastIndexOf pour trouver le dernier 'T' et prendre tout ce qui suit
    const lastTIndex = cleanTime.lastIndexOf('T');
    if (lastTIndex !== -1 && lastTIndex < cleanTime.length - 1) {
      cleanTime = cleanTime.substring(lastTIndex + 1);
    }
  }

  // Extraire uniquement HH:mm (supprimer les secondes, millisecondes, timezone, etc.)
  // Regex pour extraire HH:mm même si d'autres éléments sont présents
  const timeExtractMatch = cleanTime.match(
    /^(\d{1,2}):(\d{2})(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?$/
  );
  if (timeExtractMatch) {
    const [, hours, minutes] = timeExtractMatch;
    // S'assurer que les heures et minutes sont valides
    const h = parseInt(hours, 10);
    const m = parseInt(minutes, 10);
    if (h >= 0 && h <= 23 && m >= 0 && m <= 59) {
      cleanTime = `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`;
    } else {
      console.warn('[combineDateAndTime] Heure ou minutes invalides:', h, m);
      return undefined;
    }
  }

  console.log('[combineDateAndTime] cleanTime après nettoyage:', cleanTime);

  // Vérifier que cleanTime est au format HH:mm valide
  const timeMatch = cleanTime.match(/^(\d{2}):(\d{2})$/);
  if (!timeMatch) {
    console.warn(
      '[combineDateAndTime] Format de time invalide après nettoyage:',
      cleanTime,
      '(original:',
      timeStr,
      ')'
    );
    return undefined;
  }

  const [, hours, minutes] = timeMatch;

  // Format final : YYYY-MM-DDTHH:mm:00 (identique à ensureIsoDatetimeWithSeconds)
  const result = `${cleanDate}T${hours}:${minutes}:00`;
  console.log('[combineDateAndTime] Résultat final:', result);
  return result;
};

const SIM_CACHE_TTL_MS = 60 * 1000;
const SIM_DEBOUNCE_MS = 180;
const COORD_PRECISION = 5;

const roundCoord = (value) => {
  if (value == null || Number.isNaN(Number(value))) return null;
  return Number(value).toFixed(COORD_PRECISION);
};

const isValidCoordPair = (coords) =>
  coords &&
  coords.lat != null &&
  coords.lon != null &&
  Number.isFinite(Number(coords.lat)) &&
  Number.isFinite(Number(coords.lon));

const isValidPreferentialAmount = (value) => {
  if (value == null) return false;
  const amount = Number(value);
  return Number.isFinite(amount) && amount > 0;
};

const toIsoWithTimezone = (isoLocalString) => {
  if (!isoLocalString) return undefined;
  const parsed = new Date(isoLocalString);
  if (Number.isNaN(parsed.getTime())) return undefined;
  return parsed.toISOString();
};

const pickupAtBucket = (value) => {
  const raw = String(value || '').trim();
  if (!raw) return '';
  if (raw.length >= 16 && raw.includes('T')) {
    return raw.slice(0, 16);
  }
  const parsed = new Date(raw);
  if (Number.isNaN(parsed.getTime())) return raw;
  const pad = (v) => String(v).padStart(2, '0');
  return `${parsed.getFullYear()}-${pad(parsed.getMonth() + 1)}-${pad(parsed.getDate())}T${pad(
    parsed.getHours()
  )}:${pad(parsed.getMinutes())}`;
};

const routePointsSignature = (routePoints) => {
  if (!Array.isArray(routePoints) || routePoints.length < 2) return '';
  const first = routePoints[0] || {};
  const last = routePoints[routePoints.length - 1] || {};
  const fLat = roundCoord(first.lat);
  const fLon = roundCoord(first.lng ?? first.lon);
  const lLat = roundCoord(last.lat);
  const lLon = roundCoord(last.lng ?? last.lon);
  return `${routePoints.length}:${fLat || ''}:${fLon || ''}:${lLat || ''}:${lLon || ''}`;
};

export default function ManualBookingForm({ onSuccess, onClose, onSubmitStart }) {
  const queryClient = useQueryClient();
  const formRef = useRef(null);
  useBookingFormFocusGuard(formRef);

  const [points, setPoints] = useState(() => createInitialRoute());
  const [selectedStepKey, setSelectedStepKey] = useState(null);

  const setPickupLocation = useCallback((location) => {
    setPoints((prev) => {
      if (!prev[0]) return prev;
      const next = [...prev];
      const current = prev[0].location || '';
      next[0] = {
        ...prev[0],
        location: typeof location === 'function' ? location(current) : location,
      };
      return next;
    });
  }, []);

  const setPickupCoords = useCallback((coords) => {
    setPoints((prev) => {
      if (!prev[0]) return prev;
      const next = [...prev];
      next[0] = { ...prev[0], lat: coords?.lat ?? null, lon: coords?.lon ?? null };
      return next;
    });
  }, []);

  const patchRoutePoint = useCallback((key, patch) => {
    setPoints((prev) => prev.map((point) => (point.key === key ? { ...point, ...patch } : point)));
  }, []);
  const [estimatedDuration, setEstimatedDuration] = useState(null); // Durée estimée en minutes
  const [routePointsForPricing, setRoutePointsForPricing] = useState([]);

  // --- Client
  const [selectedClient, setSelectedClient] = useState(null);
  const [showClientModal, setShowClientModal] = useState(false);
  const [activeStay, setActiveStay] = useState(null); // Séjour actif avec infos clinique
  const [billToPatient, setBillToPatient] = useState(false); // Override: facturation patient

  // --- Date/heure de l'aller (séparés pour UX)
  const [scheduledDate, setScheduledDate] = useState('');
  const [scheduledHour, setScheduledHour] = useState('');

  // --- Aller-retour
  const [isRoundTrip, setIsRoundTrip] = useState(true);
  const [requesterName, setRequesterName] = useState('');
  const [requesterPhone, setRequesterPhone] = useState('');
  const [pricingPreview, setPricingPreview] = useState(null);
  const [isUrgent, setIsUrgent] = useState(false);
  const [pricingMode, setPricingMode] = useState('automatic');
  const [segmentAmounts, setSegmentAmounts] = useState(['']);
  const [needsAssistance, setNeedsAssistance] = useState(false);
  const idempotencyKeyRef = React.useRef(
    typeof crypto !== 'undefined' && crypto.randomUUID
      ? crypto.randomUUID()
      : `mission-${Date.now()}`
  );

  const scheduledTime = scheduledDate && scheduledHour
    ? `${scheduledDate}T${scheduledHour}` : '';

  const pickupPoint = points[0] || { location: '', lat: null, lon: null, accessNotes: '' };
  const destinationPoints = points.slice(1);
  const finalDestination = destinationPoints[destinationPoints.length - 1] || null;
  const pickupLocation = pickupPoint.location || '';
  const dropoffLocation = finalDestination?.location || '';
  const pickupLat = pickupPoint.lat ?? null;
  const pickupLon = pickupPoint.lon ?? null;
  const dropoffLat = finalDestination?.lat ?? null;
  const dropoffLon = finalDestination?.lon ?? null;
  const pickupCoords = useMemo(
    () => ({ lat: pickupLat, lon: pickupLon }),
    [pickupLat, pickupLon]
  );
  const dropoffCoords = useMemo(
    () => ({ lat: dropoffLat, lon: dropoffLon }),
    [dropoffLat, dropoffLon]
  );
  const routeSegmentLabels = segmentLabels(destinationPoints.length, isRoundTrip);

  const segmentCount = Math.max(1, destinationPoints.length) + (isRoundTrip ? 1 : 0);
  useEffect(() => {
    setSegmentAmounts((prev) => {
      if (prev.length === segmentCount) return prev;
      const next = prev.slice(0, segmentCount);
      while (next.length < segmentCount) next.push('');
      return next;
    });
  }, [segmentCount]);

  useEffect(() => {
    setSelectedStepKey((current) => {
      const stillThere = points.some(
        (point) => point.key === current && point.role === 'destination'
      );
      if (stillThere) return current;
      return points.find((point) => point.role === 'destination')?.key || null;
    });
  }, [points]);

  useEffect(() => {
    if (!scheduledDate || !scheduledHour) return;

    const dateMatch = String(scheduledDate).match(/^(\d{4})-(\d{2})-(\d{2})$/);
    const timeMatch = String(scheduledHour).match(/^(\d{2}):(\d{2})$/);
    if (!dateMatch || !timeMatch) return;

    const [, y, m, d] = dateMatch;
    const [, hh, mm] = timeMatch;
    const selected = new Date(
      Number(y),
      Number(m) - 1,
      Number(d),
      Number(hh),
      Number(mm),
      0,
      0
    );
    if (Number.isNaN(selected.getTime())) return;

    const now = new Date();
    const isToday =
      Number(y) === now.getFullYear() &&
      Number(m) === now.getMonth() + 1 &&
      Number(d) === now.getDate();
    if (!isToday) return;

    if (selected.getTime() >= now.getTime()) return;

    const corrected = new Date(now.getTime() + 5 * 60 * 1000);
    const pad = (n) => String(n).padStart(2, '0');
    const correctedDate = `${corrected.getFullYear()}-${pad(corrected.getMonth() + 1)}-${pad(
      corrected.getDate()
    )}`;
    const correctedHour = `${pad(corrected.getHours())}:${pad(corrected.getMinutes())}`;

    // Si l'heure saisie est dans le passe pour aujourd'hui, repositionner automatiquement a maintenant + 5 min.
    if (scheduledDate !== correctedDate) setScheduledDate(correctedDate);
    if (scheduledHour !== correctedHour) setScheduledHour(correctedHour);
  }, [scheduledDate, scheduledHour]);

  // --- Livraison matériel
  const [isMaterialDelivery, setIsMaterialDelivery] = useState(false);
  const [deliveryDescription, setDeliveryDescription] = useState('');

  // --- Récurrence
  const [isRecurring, setIsRecurring] = useState(false);
  const [recurrenceType, setRecurrenceType] = useState('weekly'); // daily, weekly, custom
  const [recurrenceEndDate, setRecurrenceEndDate] = useState('');
  const [selectedDays, setSelectedDays] = useState([]); // Pour la récurrence personnalisée
  const [occurrences, setOccurrences] = useState(4); // Nombre d'occurrences

  const {
    valueRef: notesMedicalRef,
    sync: syncNotesMedical,
    externalValue: notesMedicalExternal,
  } = useIsolatedField('');
  const [wheelchairOptions, setWheelchairOptions] = useState({
    clientHasWheelchair: false,
    needWheelchair: false,
  });


  const buildClinicAddress = useCallback((clinic) => {
    if (!clinic) return '';
    const parts = [];
    const hasStructured =
      clinic.domicile_address_line1 || clinic.domicile_zip || clinic.domicile_city;

    if (hasStructured && clinic.domicile_address_line1) {
      parts.push(clinic.domicile_address_line1);
      if (clinic.domicile_address_line2) {
        parts.push(clinic.domicile_address_line2);
      }
    } else if (clinic.address) {
      parts.push(clinic.address);
    }

    const postalCity = [clinic.domicile_zip, clinic.domicile_city].filter(Boolean).join(' ');
    const base = parts.join(', ');
    if (postalCity && !base.includes(postalCity)) {
      parts.push(postalCity);
    } else {
      if (clinic.domicile_zip && !base.includes(clinic.domicile_zip)) {
        parts.push(clinic.domicile_zip);
      }
      if (clinic.domicile_city && !base.includes(clinic.domicile_city)) {
        parts.push(clinic.domicile_city);
      }
    }

    return parts.filter(Boolean).join(', ');
  }, []);

  const getClinicPickupAddress = useCallback(
    (clinic) => {
      if (!clinic) return '';
      const structured = buildClinicAddress(clinic);
      if (structured) return structured;

      const directFallbacks = [
        clinic.address,
        clinic.domicile_address,
        clinic.billing_address,
        clinic.name,
      ];
      return directFallbacks.find((v) => typeof v === 'string' && v.trim())?.trim() || '';
    },
    [buildClinicAddress]
  );

  // Durée sur le réseau routier (OSRM), ajustée par l'historique des courses.
  // Pas de vol d'oiseau et pas de service d'itinéraire externe.
  React.useEffect(() => {
    const calculateDuration = async () => {
      if (pickupCoords.lat && pickupCoords.lon && dropoffCoords.lat && dropoffCoords.lon) {
        setRoutePointsForPricing([]);
        try {
          const response = await apiClient.get('/osrm/route', {
            params: {
              pickup_lat: pickupCoords.lat,
              pickup_lon: pickupCoords.lon,
              dropoff_lat: dropoffCoords.lat,
              dropoff_lon: dropoffCoords.lon,
            },
            timeout: 9000,
          });

          const data = response.data || {};
          const typicalSeconds = Number(data.duration_typical);
          const roadRoute = !data.fallback && Number.isFinite(typicalSeconds) && typicalSeconds > 0;
          if (!roadRoute) {
            setEstimatedDuration(null);
            setRoutePointsForPricing([]);
            return;
          }

          setEstimatedDuration(Math.max(1, Math.round(typicalSeconds / 60)));
          const routePoints = Array.isArray(data.route)
            ? data.route
                .filter((pair) => Array.isArray(pair) && pair.length >= 2)
                .map((pair) => ({
                  lat: Number(pair[0]),
                  lng: Number(pair[1]),
                }))
                .filter((pt) => Number.isFinite(pt.lat) && Number.isFinite(pt.lng))
            : [];
          setRoutePointsForPricing(routePoints);
        } catch (error) {
          console.warn('Itinéraire routier indisponible:', error?.message || error);
          setEstimatedDuration(null);
          setRoutePointsForPricing([]);
        }
      } else {
        setEstimatedDuration(null);
        setRoutePointsForPricing([]);
      }
    };

    calculateDuration();
  }, [pickupCoords.lat, pickupCoords.lon, dropoffCoords.lat, dropoffCoords.lon]);

  // Helper: min pour <input type="datetime-local"> au format local (pas UTC)
  const _minLocalDatetime = (() => {
    const d = new Date(Date.now() + 5 * 60 * 1000);
    const pad = (n) => String(n).padStart(2, '0');
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(
      d.getHours()
    )}:${pad(d.getMinutes())}`;
  })();

  const applyDateTimePreset = (preset) => {
    const pad = (n) => String(n).padStart(2, '0');
    const now = new Date();
    let target;
    switch (preset) {
      case 'now30':
        target = new Date(now.getTime() + 30 * 60 * 1000);
        break;
      case 'now1h':
        target = new Date(now.getTime() + 60 * 60 * 1000);
        break;
      case 'tomorrow9':
        target = new Date(now);
        target.setDate(target.getDate() + 1);
        target.setHours(9, 0, 0, 0);
        break;
      default:
        return;
    }
    setScheduledDate(`${target.getFullYear()}-${pad(target.getMonth() + 1)}-${pad(target.getDate())}`);
    setScheduledHour(`${pad(target.getHours())}:${pad(target.getMinutes())}`);
  };

  // === State montant + pricing auto ===
  const [amount, setAmount] = useState('');
  const [amountSource, setAmountSource] = useState(null); // preferential | simulated | manual
  const [amountLocked, setAmountLocked] = useState(false);
  const [lastPricingUpdateAt, setLastPricingUpdateAt] = useState(null);
  const [pricingWarning, setPricingWarning] = useState('');
  const [isSimulatingPricing, setIsSimulatingPricing] = useState(false);
  const [activePricingProfileVersionId, setActivePricingProfileVersionId] = useState(null);

  const suppressManualAmountRef = useRef(false);
  const simulateRequestSeqRef = useRef(0);
  const simulateKeyRef = useRef('');
  const simulateAbortRef = useRef(null);
  const simulateDebounceRef = useRef(null);
  const simCacheRef = useRef(new Map());

  const applySystemAmount = useCallback((value, source) => {
    suppressManualAmountRef.current = true;
    setAmount(value != null ? String(value) : '');
    setAmountSource(source || null);
    setLastPricingUpdateAt(new Date());
    queueMicrotask(() => {
      suppressManualAmountRef.current = false;
    });
  }, []);

  // === Gestion des jours de la semaine pour récurrence ===
  // ⚠️ IDs correspondent à Python weekday() : 0=Lundi, 1=Mardi, etc.
  const weekDays = [
    { id: 0, label: 'Lundi', short: 'L' },
    { id: 1, label: 'Mardi', short: 'Ma' },
    { id: 2, label: 'Mercredi', short: 'Me' },
    { id: 3, label: 'Jeudi', short: 'J' },
    { id: 4, label: 'Vendredi', short: 'V' },
    { id: 5, label: 'Samedi', short: 'S' },
    { id: 6, label: 'Dimanche', short: 'D' },
  ];

  const toggleDay = (dayId) => {
    setSelectedDays((prev) =>
      prev.includes(dayId) ? prev.filter((d) => d !== dayId) : [...prev, dayId]
    );
  };

  // === Charger les clients par défaut au montage ===
  const [defaultClientOptions, setDefaultClientOptions] = useState([]);

  React.useEffect(() => {
    const loadDefaultClients = async () => {
      try {
        const clients = await searchClients('', { limit: 20 });
        const options = clients.map((c) => {
          let label = `Client #${c.id}`;
          if (c.is_institution && c.institution_name) {
            label = `🏥 ${c.institution_name}`;
          } else if (c.full_name && c.full_name !== 'Nom non renseigné') {
            label = c.full_name;
          } else {
            const firstName = c.first_name || '';
            const lastName = c.last_name || '';
            if (firstName || lastName) {
              label = `${firstName} ${lastName}`.trim();
            }
          }
          return {
            value: c.id,
            label: label,
            raw: c,
          };
        });

        setDefaultClientOptions(options);
        console.log('📋 Options par défaut disponibles:', options.length);
      } catch (error) {
        console.error('❌ Erreur chargement clients par défaut:', error);
      }
    };

    loadDefaultClients();
  }, []);

  const pricingTriggerKey = useMemo(() => {
    if (!isValidCoordPair(pickupCoords) || !isValidCoordPair(dropoffCoords)) {
      return '';
    }
    const pickupLat = roundCoord(pickupCoords.lat);
    const pickupLon = roundCoord(pickupCoords.lon);
    const dropoffLat = roundCoord(dropoffCoords.lat);
    const dropoffLon = roundCoord(dropoffCoords.lon);
    if (!pickupLat || !pickupLon || !dropoffLat || !dropoffLon) {
      return '';
    }
    return [
      pickupLat,
      pickupLon,
      dropoffLat,
      dropoffLon,
      String(Boolean(isRoundTrip)),
      String(activePricingProfileVersionId || ''),
      pickupAtBucket(scheduledTime),
      routePointsSignature(routePointsForPricing),
    ].join('|');
  }, [
    pickupCoords,
    dropoffCoords,
    isRoundTrip,
    activePricingProfileVersionId,
    scheduledTime,
    routePointsForPricing,
  ]);

  const getPreferentialAmount = useCallback(
    ({ client, stay, patientBillingOverride }) => {
      const clinicRate = stay?.clinic?.preferential_rate;
      if (!patientBillingOverride && isValidPreferentialAmount(clinicRate)) {
        return { amount: Number(clinicRate).toFixed(2), source: 'preferential' };
      }
      const clientRate = client?.preferential_rate;
      if (isValidPreferentialAmount(clientRate)) {
        return { amount: Number(clientRate).toFixed(2), source: 'preferential' };
      }
      return null;
    },
    []
  );

  const runPricingSimulation = useCallback(async () => {
    if (!pricingTriggerKey || amountLocked || amountSource === 'preferential') {
      return;
    }
    if (!activePricingProfileVersionId) {
      setPricingWarning('Profil tarifaire actif introuvable. Saisissez un montant ou contactez le support.');
      return;
    }

    const cached = simCacheRef.current.get(pricingTriggerKey);
    if (cached && Date.now() - cached.cachedAt <= SIM_CACHE_TTL_MS) {
      if (!amountLocked && amountSource !== 'preferential' && simulateKeyRef.current === pricingTriggerKey) {
        applySystemAmount(Number(cached.amount).toFixed(2), 'simulated');
        setPricingWarning(cached.warning || '');
      }
      return;
    }

    setIsSimulatingPricing(true);
    setPricingWarning('');
    simulateRequestSeqRef.current += 1;
    const requestSeq = simulateRequestSeqRef.current;
    simulateKeyRef.current = pricingTriggerKey;
    if (simulateAbortRef.current) {
      simulateAbortRef.current.abort();
    }
    const abortController = new AbortController();
    simulateAbortRef.current = abortController;

    const payload = {
      pricing_profile_version_id: activePricingProfileVersionId,
      booking: {
        pickup_at: toIsoWithTimezone(ensureIsoDatetimeWithSeconds(scheduledTime)) || new Date().toISOString(),
        is_round_trip: Boolean(isRoundTrip),
        pickup_lat: Number(pickupCoords.lat),
        pickup_lng: Number(pickupCoords.lon),
        dropoff_lat: Number(dropoffCoords.lat),
        dropoff_lng: Number(dropoffCoords.lon),
        route_points: Array.isArray(routePointsForPricing) && routePointsForPricing.length > 1
          ? routePointsForPricing
          : undefined,
      },
    };

    try {
      const response = await simulatePricing(payload, { signal: abortController.signal });
      const warningList = Array.isArray(response?.warnings)
        ? response.warnings
        : Array.isArray(response?.breakdown?.warnings)
          ? response.breakdown.warnings
          : [];
      const blockingReasons = Array.isArray(response?.blocking_reasons)
        ? response.blocking_reasons
        : [];
      const confidence = String(response?.confidence || '').toLowerCase();
      const hasDistanceUnavailable = warningList.includes('distance_unavailable');
      const hasZoneUnresolved = warningList.includes('zone_unresolved');
      const modelUsed = String(response?.breakdown?.model_used || '').toLowerCase();
      const isDistanceDependentModel = modelUsed === 'distance' || modelUsed === 'hybrid_stack';
      const responseAmount = Number(response?.amount);

      if (
        requestSeq !== simulateRequestSeqRef.current ||
        pricingTriggerKey !== simulateKeyRef.current ||
        amountLocked ||
        amountSource === 'preferential'
      ) {
        return;
      }

      if ((confidence === 'blocked' || blockingReasons.length > 0) && !Number.isFinite(responseAmount)) {
        if (blockingReasons.includes('zone_unresolved')) {
          setPricingWarning('Calcul indisponible: zonage introuvable pour ce trajet. Saisissez un montant ou réessayez.');
          return;
        }
        if (blockingReasons.includes('zone_unresolved_timeout')) {
          setPricingWarning('Calcul indisponible: délai de calcul zonage dépassé. Réessayez dans quelques secondes.');
          return;
        }
        if (blockingReasons.includes('distance_unavailable')) {
          setPricingWarning('Calcul indisponible (OSRM). Saisissez un montant ou réessayez.');
          return;
        }
        setPricingWarning('Calcul précis temporairement indisponible. Réessayez.');
        return;
      }

      if (hasDistanceUnavailable && isDistanceDependentModel) {
        setPricingWarning('Calcul indisponible (OSRM). Saisissez un montant ou réessayez.');
        return;
      }

      if (hasZoneUnresolved && !Number.isFinite(responseAmount)) {
        setPricingWarning('Calcul indisponible: zonage introuvable pour ce trajet. Saisissez un montant ou réessayez.');
        return;
      }

      if (Number.isFinite(responseAmount) && responseAmount > 0) {
        applySystemAmount(responseAmount.toFixed(2), 'simulated');
        if (warningList.includes('zone_unresolved_fallback')) {
          setPricingWarning('Zonage partiellement résolu: calcul appliqué avec fallback conservateur.');
        } else {
            setPricingWarning('');
        }
        simCacheRef.current.set(pricingTriggerKey, {
          amount: responseAmount,
          warning: warningList.includes('zone_unresolved_fallback')
            ? 'Zonage partiellement résolu: calcul appliqué avec fallback conservateur.'
            : '',
          cachedAt: Date.now(),
        });
      }
    } catch (error) {
      if (error?.name !== 'CanceledError' && error?.name !== 'AbortError') {
        const apiWarnings = Array.isArray(error?.response?.data?.warnings)
          ? error.response.data.warnings
          : [];
        const blockingReasons = Array.isArray(error?.response?.data?.blocking_reasons)
          ? error.response.data.blocking_reasons
          : [];
        const isDistanceBlocking = apiWarnings.includes('distance_unavailable')
          || blockingReasons.includes('distance_unavailable');
        if (isDistanceBlocking) {
          setPricingWarning('Calcul indisponible (OSRM). Saisissez un montant ou réessayez.');
        } else {
          setPricingWarning('Calcul temporairement indisponible. Saisissez un montant ou réessayez.');
        }
      }
    } finally {
      if (requestSeq === simulateRequestSeqRef.current) {
        setIsSimulatingPricing(false);
      }
    }
  }, [
    pricingTriggerKey,
    amountLocked,
    amountSource,
    activePricingProfileVersionId,
    isRoundTrip,
    pickupCoords,
    dropoffCoords,
    routePointsForPricing,
    scheduledTime,
    applySystemAmount,
  ]);

  // === Clients ===
  const handleSelectClient = useCallback(async (clientObj) => {
    console.log('👤 Client sélectionné:', clientObj);
    setSelectedClient(clientObj);
    setActiveStay(null); // Réinitialiser le séjour actif
    setBillToPatient(false); // Réinitialiser l'override
    setAmountLocked(false);
    setAmountSource(null);
    setAmount('');
    setPricingWarning('');

    const client = clientObj?.raw;
    const clientId = client?.id || clientObj?.value?.id;

    if (!clientId) {
      console.warn('⚠️ Pas d\'ID client disponible');
      return;
    }

    // 🏥 Vérifier si le client a un séjour actif
    try {
      const stayResponse = await fetchClientActiveStay(clientId);
      const stayData = stayResponse?.data;

      if (stayData && stayData.clinic) {
        setActiveStay(stayData);
        const clinic = stayData.clinic;
        // Préremplir l'établissement médical avec la clinique d'hospitalisation
        // Set pickup to clinic address (structured > raw > clinic name as last resort)
        const clinicAddress = getClinicPickupAddress(clinic);
        if (clinicAddress) {
          setPickupLocation(clinicAddress);
          setPickupCoords({
            lat: clinic.latitude || null,
            lon: clinic.longitude || null,
          });
        }

        return;
      }
    } catch (error) {
      console.warn('⚠️ Erreur lors de la récupération du séjour actif');
      // Continuer avec l'adresse du client en cas d'erreur
    }

    // Pas de séjour actif ou erreur → utiliser l'adresse du client
    // 📍 Récupérer l'adresse exacte du client avec priorités
    let homeAddress = '';
    let homeGPS = { lat: null, lon: null };

    // Priorité 1: domicile (adresse structurée complète + GPS)
    if (client?.domicile?.address) {
      // Construire l'adresse complète : Rue, Numéro, Code postal, Ville
      const parts = [client.domicile.address, client.domicile.zip, client.domicile.city].filter(
        Boolean
      );
      homeAddress = parts.join(', ');

      // 📍 IMPORTANT: Charger aussi les GPS du domicile !
      if (client.domicile.lat && client.domicile.lon) {
        homeGPS = {
          lat: client.domicile.lat,
          lon: client.domicile.lon,
        };
      }
    }
    // Priorité 2: billing_address (adresse de facturation)
    else if (client?.billing_address) {
      homeAddress = client.billing_address;

      // Charger GPS de facturation si disponibles
      if (client.billing_lat && client.billing_lon) {
        homeGPS = {
          lat: client.billing_lat,
          lon: client.billing_lon,
        };
      }
    }
    // Priorité 3: adresse utilisateur (peut être un nom de résidence)
    else if (client?.user?.address) {
      homeAddress = client.user.address;
    }
    // Priorité 4: adresse client directe
    else if (client?.address) {
      homeAddress = client.address;
    }

    if (homeAddress) {
      setPickupLocation(homeAddress);
      setPickupCoords(homeGPS); // ✅ Charger les GPS du client
    }

  }, [getClinicPickupAddress, setPickupCoords, setPickupLocation]);

  const handleCreateClientOption = useCallback(() => {
    setShowClientModal(true);
  }, []);

  const handleBillToPatientChange = useCallback((checked) => {
    setBillToPatient(checked);
  }, []);

  // Recherche client débouncée (300-400ms) + annulation de la requête précédente
  const clientSearchDebounceRef = useRef(null);
  const clientSearchAbortRef = useRef(null);

  const mapClientsToOptions = useCallback((clients) => {
    return clients.map((c) => {
      let label = `Client #${c.id}`;
      if (c.is_institution && c.institution_name) {
        label = `🏥 ${c.institution_name}`;
      } else if (c.full_name && c.full_name !== 'Nom non renseigné') {
        label = c.full_name;
      } else {
        const firstName = c.first_name || '';
        const lastName = c.last_name || '';
        if (firstName || lastName) {
          label = `${firstName} ${lastName}`.trim();
        }
      }
      return {
        value: c.id,
        label: label,
        raw: c,
      };
    });
  }, []);

  const loadClientOptions = useCallback((q) => {
    const query = String(q || '').trim().slice(0, 100);

    return new Promise((resolve) => {
      if (clientSearchDebounceRef.current) {
        clearTimeout(clientSearchDebounceRef.current);
        clientSearchDebounceRef.current = null;
      }

      // Recherche serveur : minimum 2 caractères (1 caractère → options par défaut)
      if (query.length === 1) {
        resolve(defaultClientOptions);
        return;
      }

      clientSearchDebounceRef.current = setTimeout(async () => {
        if (clientSearchAbortRef.current) {
          try {
            clientSearchAbortRef.current.abort();
          } catch {
            // annulation silencieuse
          }
        }
        const controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
        clientSearchAbortRef.current = controller;
        try {
          const clients = await searchClients(query, { limit: 20, signal: controller?.signal });
          resolve(mapClientsToOptions(clients));
        } catch (e) {
          resolve([]);
        }
      }, 350);
    });
  }, [defaultClientOptions, mapClientsToOptions]);

  useEffect(() => {
    return () => {
      if (clientSearchDebounceRef.current) {
        clearTimeout(clientSearchDebounceRef.current);
      }
      if (clientSearchAbortRef.current) {
        try {
          clientSearchAbortRef.current.abort();
        } catch {
          // annulation silencieuse
        }
      }
    };
  }, []);

  useEffect(() => {
    let mounted = true;
    const loadBillingContext = async () => {
      try {
        const response = await fetchBillingSettings();
        const payload = response?.data || response || {};
        if (!mounted) return;
        setActivePricingProfileVersionId(payload.active_pricing_profile_version_id || null);
        if (!payload.active_pricing_profile_version_id) {
          setPricingWarning('Profil tarifaire actif introuvable. Le montant restera manuel.');
        }
      } catch (error) {
        console.warn('[ManualBookingForm] Impossible de charger la version pricing active', error);
      }
    };
    loadBillingContext();
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    if (!activeStay?.clinic || billToPatient) {
      return;
    }
    const clinic = activeStay.clinic;
    const clinicAddress = getClinicPickupAddress(clinic);
    if (!clinicAddress) {
      return;
    }
    setPickupLocation(clinicAddress);
    setPickupCoords({
      lat: clinic.latitude || null,
      lon: clinic.longitude || null,
    });
  }, [activeStay, billToPatient, getClinicPickupAddress, setPickupCoords, setPickupLocation]);

  useEffect(() => {
    const preferential = getPreferentialAmount({
      client: selectedClient?.raw || null,
      stay: activeStay,
      patientBillingOverride: billToPatient,
    });
    if (preferential) {
      if (!amountLocked) {
        applySystemAmount(preferential.amount, preferential.source);
        setPricingWarning('');
      }
      return;
    }
    if (!amountLocked && amountSource === 'preferential') {
      applySystemAmount('', null);
    }
  }, [
    selectedClient,
    activeStay,
    billToPatient,
    amountLocked,
    amountSource,
    getPreferentialAmount,
    applySystemAmount,
  ]);

  useEffect(() => {
    if (simulateDebounceRef.current) {
      clearTimeout(simulateDebounceRef.current);
      simulateDebounceRef.current = null;
    }
    if (!pricingTriggerKey || amountLocked || amountSource === 'preferential') {
      return;
    }
    simulateDebounceRef.current = setTimeout(() => {
      runPricingSimulation();
    }, SIM_DEBOUNCE_MS);
    return () => {
      if (simulateDebounceRef.current) {
        clearTimeout(simulateDebounceRef.current);
      }
    };
  }, [pricingTriggerKey, amountLocked, amountSource, runPricingSimulation]);

  useEffect(
    () => () => {
      if (simulateAbortRef.current) {
        simulateAbortRef.current.abort();
      }
      if (simulateDebounceRef.current) {
        clearTimeout(simulateDebounceRef.current);
      }
    },
    []
  );

  // === Notes médicales → extraction
  function handleNotesMedicalBlur(e) {
    const value = e.target.value;
    const extracted = extractMedicalServiceInfo(value);
    const targetKey = selectedStepKey || points.find((point) => point.role === 'destination')?.key;
    if (
      targetKey &&
      (extracted.medical_facility || extracted.hospital_service || extracted.doctor_name)
    ) {
      patchRoutePoint(targetKey, {
        destinationKind: 'medical',
        ...(extracted.medical_facility ? { establishment: extracted.medical_facility } : {}),
        ...(extracted.hospital_service ? { service: extracted.hospital_service } : {}),
        ...(extracted.doctor_name ? { doctor: extracted.doctor_name } : {}),
      });
    }
    const hasTextInNotes = (notes, text) => {
      if (!notes || !text) return false;
      const normalize = (t) => t.toLowerCase().replace(/[🏢\s]/g, '').trim();
      return normalize(notes).includes(normalize(text));
    };
    let notes = value || '';
    if (extracted.building && !hasTextInNotes(notes, extracted.building)) {
      notes += (notes ? '\n' : '') + extracted.building;
    }
    if (extracted.floor && !hasTextInNotes(notes, extracted.floor)) {
      notes += (notes ? '\n' : '') + extracted.floor;
    }
    if (notes !== value) syncNotesMedical(notes);
  }

  // P2.1: Résumé enrichi — confirm bar (summaryText + badges séparés)
  const buildFooterSummary = () => {
    const badges = [];
    if (isRoundTrip) badges.push('AR');
    if (isRecurring) badges.push('Récurrente');
    if (isMaterialDelivery) badges.push('Livraison');

    if (isMaterialDelivery) {
      const desc = deliveryDescription?.trim();
      const descShort = desc && desc.length > 40 ? desc.slice(0, 39) + '…' : desc;
      const routeLabel = descShort
        ? `Livraison · ${descShort}`
        : 'Livraison · Description manquante';
      return {
        summaryText: routeLabel,
        summaryBadges: badges,
      };
    }
    if (!selectedClient) {
      return { summaryText: 'Client non sélectionné', summaryBadges: badges };
    }
    if (!scheduledTime) {
      return { summaryText: 'Date/heure manquante', summaryBadges: badges };
    }
    if (!pickupLocation || !dropoffLocation) {
      return { summaryText: 'Trajet incomplet', summaryBadges: badges };
    }
    const client = selectedClient?.label || 'Client';
    let dateStr = '—';
    try {
      const d = new Date(scheduledTime);
      dateStr = d.toLocaleDateString('fr-CH', {
        day: '2-digit',
        month: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      /* ignore */
    }
    const pickup = shortAddress(pickupLocation) || '…';
    const dropoff = shortAddress(dropoffLocation) || '…';
    const routeLabel = `${pickup} → ${dropoff}`;
    const summaryText = `${client} · ${dateStr} · ${routeLabel}`;
    return { summaryText, summaryBadges: badges };
  };

  const footerSummary = buildFooterSummary();

  // === Mutations API ===
  // Gestion de la création de client
  const handleSaveNewClient = async (clientData, { existingClient } = {}) => {
    let createdClient = existingClient || null;
    try {
      const { hospitalization, billing_party_link, ...clientPayload } = clientData || {};
      const newClient = createdClient || (await createClient(clientPayload));
      createdClient = newClient;

      const createdClientId = newClient?.id || newClient?.data?.id || newClient?.client?.id;
      if (!createdClientId) {
        throw new Error('Client créé mais identifiant introuvable.');
      }

      if (hospitalization) {
        await createClientStay(createdClientId, {
          company_id: parseInt(hospitalization.company_id, 10),
          start_date: hospitalization.start_date,
          end_date: hospitalization.end_date || null,
          notes: hospitalization.notes || null,
        });
      }

      if (billing_party_link) {
        await linkClientBillingParty(createdClientId, {
          billing_party_id: billing_party_link.billing_party_id,
          role: billing_party_link.role || null,
          is_default: !!billing_party_link.is_default,
          contact_name: billing_party_link.contact_name || null,
          contact_email: billing_party_link.contact_email || null,
          contact_phone: billing_party_link.contact_phone || null,
        });
      }

      const clientOption = {
        value: newClient.id,
        label: `${newClient.user?.first_name ?? newClient.first_name ?? ''} ${
          newClient.user?.last_name ?? newClient.last_name ?? ''
        }`.trim(),
        raw: newClient,
      };

      await handleSelectClient(clientOption);
      queryClient.invalidateQueries(['clients']);
      setShowClientModal(false);
      toast.success('Client créé !');

      if (newClient.billing_address || newClient.domicile?.address) {
        const homeAddress =
          newClient.billing_address ||
          [newClient.domicile?.address, newClient.domicile?.zip, newClient.domicile?.city]
            .filter(Boolean)
            .join(', ');
        setPickupLocation(homeAddress);
      }
    } catch (err) {
      console.error('API createClient error:', err?.response?.data || err);
      const errorMessage = err?.response?.data?.error || err.message || 'Erreur création client';
      if (createdClient || err?.createdClient) {
        toast.error(`Client créé, mais erreur sur les informations complémentaires: ${errorMessage}`);
        const wrappedError = new Error(errorMessage);
        wrappedError.createdClient = createdClient || err?.createdClient;
        throw wrappedError;
      }
      toast.error(errorMessage);
      throw err; // Pour que NewClientModal affiche l'erreur
    }
  };

  const bookingMutation = useMutation({
    mutationFn: createManualBooking,
    onSuccess: (data) => {
      toast.success('Réservation créée !');
      idempotencyKeyRef.current =
        typeof crypto !== 'undefined' && crypto.randomUUID
          ? crypto.randomUUID()
          : `mission-${Date.now()}`;
      // ✅ Reset billToPatient après succès
      setBillToPatient(false);
      onSuccess?.(data);
    },
    onError: (err) => {
      // ⚡ Ne pas logger les 401 temporaires qui sont gérés par le refresh automatique
      const is401Refresh =
        err?.response?.status === 401 && err?.config?._retryAfterRefresh === undefined;
      if (!is401Refresh) {
        // 🆕 Logger toute la structure de l'erreur pour debug
        console.error('❌ createManualBooking error:', err);
        console.error('📋 err.response:', err?.response);
        console.error('📋 err.response?.data:', err?.response?.data);
        console.error('📋 err.message:', err?.message);
        console.error('📋 err.toString():', err?.toString());

        // 🆕 Afficher les détails des erreurs de validation
        const errorData = err?.response?.data || err?.data || err;
        console.error("📋 Structure complète de l'erreur:", JSON.stringify(errorData, null, 2));

        if (errorData?.errors) {
          console.error('Détails des erreurs de validation:', errorData.errors);

          // 🔍 Extraire récursivement tous les champs en erreur (gérer la structure nested)
          const extractErrors = (obj, prefix = '', depth = 0) => {
            const extracted = [];
            if (!obj || typeof obj !== 'object' || depth > 10) return extracted; // Protection contre récursion infinie

            for (const [key, value] of Object.entries(obj)) {
              // Ignorer les clés spéciales et les structures "errors" nested qui sont juste des wrappers
              if (key === 'message' || key.startsWith('_')) continue;

              // ⚡ Si on trouve "errors" nested, descendre directement dedans sans ajouter au préfixe
              if (key === 'errors' && value && typeof value === 'object' && !Array.isArray(value)) {
                extracted.push(...extractErrors(value, prefix, depth + 1));
                continue;
              }

              const fieldPath = prefix ? `${prefix}.${key}` : key;

              if (Array.isArray(value)) {
                // Liste de messages directement - c'est un champ réel en erreur
                extracted.push({ field: fieldPath, messages: value });
              } else if (value && typeof value === 'object' && !Array.isArray(value)) {
                // Objet nested, extraire récursivement
                extracted.push(...extractErrors(value, fieldPath, depth + 1));
              } else if (value) {
                // Message unique
                extracted.push({ field: fieldPath, messages: [String(value)] });
              }
            }
            return extracted;
          };

          const allErrors = extractErrors(errorData.errors);

          if (allErrors.length > 0) {
            // Construire un message détaillé avec tous les champs en erreur
            const errorMessages = allErrors.map(({ field, messages }) => {
              const msgList = Array.isArray(messages) ? messages : [String(messages)];
              return `• ${field}: ${msgList.join(', ')}`;
            });
            toast.error(`Erreur de validation:\n${errorMessages.join('\n')}`, { duration: 10000 });
            return;
          }
        }
      }

      // Message d'erreur générique (messages métier lisibles via getApiErrorMessage)
      const errorMessage = getApiErrorMessage(
        err,
        `Erreur création réservation : ${err.message || 'Erreur inconnue'}`
      );

      toast.error(errorMessage, {
        duration:
          err?.response?.data?.error_code === 'billing_access_restricted'
            ? 12000
            : 6000,
      });
    },
  });


  const buildMissionInput = useCallback(({ preview }) => {
    const missionDateForStep = (value) => value || scheduledDate || '';
    const destinations = points.slice(1);
    const pickup = points[0] || {};
    const finalDest = destinations[destinations.length - 1] || {};
    const intermediates = destinations.slice(0, -1);
    return {
      clientId: selectedClient?.value,
      pickupLocation: pickup.location || '',
      pickupCoords: { lat: pickup.lat ?? null, lon: pickup.lon ?? null },
      dropoffLocation: finalDest.location || '',
      dropoffCoords: { lat: finalDest.lat ?? null, lon: finalDest.lon ?? null },
      scheduledTime: ensureIsoDatetimeWithSeconds(scheduledTime),
      destinationArrival: combineDateAndTime(
        missionDateForStep(finalDest.arrivalDate),
        finalDest.arrivalTime
      ),
      destinationDeparture: isRoundTrip
        ? combineDateAndTime(missionDateForStep(finalDest.departureDate), finalDest.departureTime)
        : null,
      extraStops: intermediates.map((stop) => {
        const departure = combineDateAndTime(
          missionDateForStep(stop.departureDate),
          stop.departureTime
        );
        return {
          location: stop.location,
          coords: { lat: stop.lat, lon: stop.lon },
          arrival:
            combineDateAndTime(missionDateForStep(stop.arrivalDate), stop.arrivalTime) || departure,
          departure,
          destinationKind: stop.destinationKind,
          establishment: stop.establishment,
          service: stop.service,
          doctor: stop.doctor,
          accessNotes: stop.accessNotes,
        };
      }),
      isRoundTrip,
      returnArrival: isRoundTrip
        ? combineDateAndTime(missionDateForStep(finalDest.departureDate), finalDest.departureTime)
        : null,
      pricingMode: isMaterialDelivery ? 'automatic' : pricingMode,
      manualAmounts: pricingMode === 'manual' ? segmentAmounts : [],
      preferentialAmount: amount,
      idempotencyKey: preview ? 'preview' : idempotencyKeyRef.current,
      requesterName,
      requesterPhone,
      isUrgent,
      needsAssistance,
      isMaterialDelivery,
      deliveryDescription,
      pickupAccessNotes: cleanOptionalText(pickup.accessNotes),
      dropoffAccessNotes: cleanOptionalText(finalDest.accessNotes),
      destinationKind: finalDest.destinationKind,
      establishment: cleanOptionalText(finalDest.establishment),
      service: cleanOptionalText(finalDest.service),
      doctor: cleanOptionalText(finalDest.doctor),
      notesMedical: cleanOptionalText(notesMedicalRef.current),
      wheelchairClientHas: wheelchairOptions.clientHasWheelchair,
      wheelchairNeed: wheelchairOptions.needWheelchair,
      billToPatient,
      isRecurring,
      recurrenceType,
      occurrences,
      recurrenceDays: selectedDays,
      recurrenceEndDate,
    };
  }, [
    amount,
    billToPatient,
    deliveryDescription,
    isMaterialDelivery,
    isRecurring,
    isRoundTrip,
    isUrgent,
    needsAssistance,
    notesMedicalRef,
    occurrences,
    points,
    pricingMode,
    recurrenceEndDate,
    recurrenceType,
    requesterName,
    requesterPhone,
    scheduledDate,
    scheduledTime,
    segmentAmounts,
    selectedClient,
    selectedDays,
    wheelchairOptions,
  ]);

  useEffect(() => {
    const built = buildCanonicalReservationPayload(buildMissionInput({ preview: true }));
    if (built.error || !built.payload) {
      setPricingPreview(null);
      return undefined;
    }
    const timer = setTimeout(() => {
      previewManualBookingPricing(built.payload)
        .then((data) => setPricingPreview(data))
        .catch(() => setPricingPreview(null));
    }, 400);
    return () => clearTimeout(timer);
  }, [buildMissionInput]);

  // === Soumission ===
  const handleSubmit = (e) => {
    e.preventDefault();

    if (!selectedClient) {
      toast.error('Veuillez sélectionner un client');
      return;
    }

    // Vérifier que la date et l'heure de départ sont définies
    if (!scheduledTime) {
      toast.error('Veuillez sélectionner la date & heure de départ');
      return;
    }

    if (!isMaterialDelivery) {
      const missingEstablishment = points
        .slice(1)
        .findIndex(
          (point) =>
            point.destinationKind === 'medical' && !String(point.establishment || '').trim()
        );
      if (missingEstablishment >= 0) {
        const several = points.length > 2;
        toast.error(
          several
            ? `Veuillez indiquer l'établissement de la destination ${missingEstablishment + 1}`
            : "Veuillez indiquer l'établissement"
        );
        return;
      }
      const missingServiceOrDoctor = points
        .slice(1)
        .findIndex(
          (point) =>
            point.destinationKind === 'medical' &&
            !String(point.service || '').trim() &&
            !String(point.doctor || '').trim()
        );
      if (missingServiceOrDoctor >= 0) {
        const several = points.length > 2;
        toast.error(
          several
            ? `Veuillez indiquer le service ou le médecin de la destination ${missingServiceOrDoctor + 1}`
            : 'Veuillez indiquer le service ou le médecin'
        );
        return;
      }
    }

    if (
      pricingMode === 'manual' &&
      !isMaterialDelivery &&
      segmentAmounts.some((value) => !(Number(value) > 0))
    ) {
      toast.error('Veuillez saisir un montant pour chaque tronçon');
      return;
    }
    if (pricingMode === 'preferential' && (!amount || parseFloat(amount) <= 0)) {
      toast.error('Le forfait doit être strictement positif');
      return;
    }

    // Livraison matériel : description obligatoire
    if (isMaterialDelivery && !deliveryDescription?.trim()) {
      toast.error('Veuillez saisir la description de la livraison');
      return;
    }

    if (needsAssistance && !String(notesMedicalRef.current || '').trim()) {
      toast.error("Veuillez indiquer les notes d'assistance");
      return;
    }

    // 🔍 Debug coordonnées GPS
    console.log('📍 Coordonnées pickup:', pickupCoords);
    console.log('📍 Coordonnées dropoff:', dropoffCoords);

    const built = buildCanonicalReservationPayload(buildMissionInput({ preview: false }));
    if (built.error) {
      toast.error(built.error);
      return;
    }
    const payload = built.payload;

    console.log('[ManualBookingForm] payload:', payload);
    console.log('🔄 Récurrence activée:', isRecurring);
    console.log('📅 Type de récurrence:', recurrenceType);
    console.log('🗓️ Jours sélectionnés:', selectedDays);
    console.log("🔢 Nombre d'occurrences:", occurrences);
    // 🔍 Trace pour debug facturation
    console.log('💳 [Facturation] billToPatient:', billToPatient);
    console.log('💳 [Facturation] activeStay:', activeStay);
    console.log('💳 [Facturation] bill_to_patient dans payload:', payload.bill_to_patient);

    // Fermer la modale immédiatement apres validation locale pour eviter la latence percue.
    onSubmitStart?.(payload);
    bookingMutation.mutate(payload);
  };

  return (
    <div
      className={styles.formWrapper}
      data-testid="manual-booking-form-wrapper"
      data-tour-id="manual-booking-form"
    >
      <div className={styles.modalHeader}>
        <h2 className={styles.modalTitle}>Créer une réservation</h2>
        <p className={styles.modalSubtitle}>
          Construisez le parcours : départ, destinations, retour.
        </p>
      </div>
      <form ref={formRef} onSubmit={handleSubmit} className={styles.form}>
        <div className={styles.formScrollBody}>
          {/* COLONNE GAUCHE */}
          <div className={styles.columnLeft} data-tour-id="booking-left-panel">
          <div className={styles.clientMissionRow}>
          <ManualBookingClientSelect
            selectedClient={selectedClient}
            defaultClientOptions={defaultClientOptions}
            loadClientOptions={loadClientOptions}
            onChange={handleSelectClient}
            onCreateOption={handleCreateClientOption}
            activeStay={activeStay}
            billToPatient={billToPatient}
            onBillToPatientChange={handleBillToPatientChange}
            getClinicPickupAddress={getClinicPickupAddress}
          />

          <div className={`${styles.formGroup} ${styles.clientMissionType}`}>
            <Label>Type de mission</Label>
            <div
              className={styles.missionTypeGroup}
              role="group"
              aria-label="Type de mission"
              data-testid="mission-type"
            >
              <button
                type="button"
                className={`${styles.missionTypeBtn} ${!isMaterialDelivery ? styles.isActive : ''}`}
                aria-pressed={!isMaterialDelivery}
                onClick={() => {
                  setIsMaterialDelivery(false);
                  setDeliveryDescription('');
                  setIsRoundTrip(true);
                }}
              >
                Transport de personne
              </button>
              <button
                type="button"
                className={`${styles.missionTypeBtn} ${isMaterialDelivery ? styles.isActive : ''}`}
                aria-pressed={isMaterialDelivery}
                onClick={() => {
                  if (!isMaterialDelivery && amount && parseFloat(amount) > 0) {
                    toast.info('Montant ignoré pour les livraisons (prix fixe appliqué).');
                  }
                  setIsMaterialDelivery(true);
                  setIsRoundTrip(false);
                }}
              >
                Livraison
              </button>
            </div>
          </div>
          </div>

          {isMaterialDelivery && (
            <div className={`${styles.formGroup} ${styles.deliveryDescriptionGroup}`}>
              <Label htmlFor="delivery_description">Description de la livraison *</Label>
              <Input
                id="delivery_description"
                type="text"
                name="delivery_description"
                value={deliveryDescription}
                onChange={(e) => setDeliveryDescription(e.target.value)}
                placeholder="Ex: Livraison médicament, Oxygène, Documents…"
                required={isMaterialDelivery}
              />
            </div>
          )}

          <div className={styles.formGroup} data-tour-id="booking-datetime">
            <Label>Date de mission *</Label>
            <div className={styles.missionDateRow}>
              <div className={styles.missionDateField}>
                <InlineDatePicker
                  value={scheduledDate}
                  onChange={(v) => setScheduledDate(v)}
                  placeholder="Date"
                />
              </div>
              <div className={styles.datePresetGroup} data-tour-id="booking-time-presets">
                <button
                  type="button"
                  className={`${styles.datePresetBtn} ${styles.urgentBtn} ${isUrgent ? styles.isActive : ''}`}
                  aria-pressed={isUrgent}
                  onClick={() => setIsUrgent((value) => !value)}
                >
                  Urgent
                </button>
                <button
                  type="button"
                  className={styles.datePresetBtn}
                  data-tour-id="booking-plus30"
                  onClick={() => applyDateTimePreset('now30')}
                >
                  +30 min
                </button>
                <button
                  type="button"
                  className={styles.datePresetBtn}
                  onClick={() => applyDateTimePreset('now1h')}
                >
                  +1h
                </button>
                <button
                  type="button"
                  className={styles.datePresetBtn}
                  onClick={() => applyDateTimePreset('tomorrow9')}
                >
                  Demain 9h
                </button>
              </div>
            </div>
          </div>

          <CompanyRouteBuilder
            points={points}
            missionDate={scheduledDate}
            departureTime={scheduledHour}
            estimatedDuration={estimatedDuration}
            isRoundTrip={isRoundTrip}
            selectedKey={selectedStepKey}
            onSelect={setSelectedStepKey}
            onPatch={patchRoutePoint}
            onReorder={(from, to) => setPoints((prev) => reorderRoutePoints(prev, from, to))}
            onAddDestination={() =>
              setPoints((prev) => [...prev, createDestinationPoint(scheduledDate)])
            }
            onRemoveDestination={(key) =>
              setPoints((prev) => {
                if (prev.length <= 2) return prev;
                const target = prev.find((point) => point.key === key);
                if (!target || target.role === 'pickup') return prev;
                return prev.filter((point) => point.key !== key);
              })
            }
            onToggleRoundTrip={() => setIsRoundTrip((value) => !value)}
            onDepartureTime={setScheduledHour}
          />

          <div className={styles.routeOptions}>
            <span className={styles.routeOptionsLabel}>Options trajet</span>
            <button
              type="button"
              className={`${styles.routeReturnBtn} ${isRecurring ? styles.isActive : ''}`}
              aria-pressed={isRecurring}
              data-testid="route-recurring"
              onClick={() => setIsRecurring((value) => !value)}
            >
              Récurrente
            </button>
          </div>

            {isRecurring && (
              <div className={styles.recurrenceConfig}>
                <div className={styles.recurrenceGroup}>
                  <div className={styles.recurrenceTypeHead}>
                    <Label id="recurrence_type_label">Type de récurrence</Label>
                    {recurrenceType === 'custom' && (
                      <div className={styles.daysSelector} role="group" aria-label="Jours de la semaine">
                        {weekDays.map((day) => (
                          <button
                            key={day.id}
                            type="button"
                            className={`${styles.dayButton} ${
                              selectedDays.includes(day.id) ? styles.daySelected : ''
                            }`}
                            aria-pressed={selectedDays.includes(day.id)}
                            onClick={() => toggleDay(day.id)}
                            title={day.label}
                          >
                            {day.short}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                  <div
                    className={styles.recurrenceTypeGroup}
                    role="group"
                    aria-labelledby="recurrence_type_label"
                  >
                    {[
                      { value: 'daily', label: 'Tous les jours' },
                      { value: 'weekly', label: 'Toutes les semaines' },
                      { value: 'custom', label: 'Jours personnalisés' },
                    ].map((mode) => (
                      <button
                        key={mode.value}
                        type="button"
                        className={`${styles.recurrenceTypeBtn} ${
                          recurrenceType === mode.value ? styles.isActive : ''
                        }`}
                        aria-pressed={recurrenceType === mode.value}
                        onClick={() => setRecurrenceType(mode.value)}
                      >
                        {mode.label}
                      </button>
                    ))}
                  </div>
                  {recurrenceType === 'custom' && selectedDays.length === 0 && (
                    <div className={styles.recurrenceWarning}>
                      Sélectionnez au moins un jour
                    </div>
                  )}
                </div>

                <div className={styles.recurrenceGroup}>
                  <div className={styles.recurrenceFieldRow}>
                    <Label>Répétitions</Label>
                    <Input
                      type="number"
                      min="1"
                      max="52"
                      value={occurrences}
                      onChange={(e) => setOccurrences(parseInt(e.target.value, 10) || 1)}
                      placeholder="4"
                      aria-label="Nombre de répétitions"
                    />
                  </div>
                  <div className={styles.recurrenceHint}>
                    {occurrences > 0 && (
                      <span>
                        {recurrenceType === 'custom' && selectedDays.length > 0 ? (
                          <>
                            {occurrences} × {selectedDays.length} jour
                            {selectedDays.length > 1 ? 's' : ''} ={' '}
                            {occurrences * selectedDays.length} réservation
                            {occurrences * selectedDays.length > 1 ? 's' : ''}
                            {isRoundTrip && (
                              <>
                                {' '}
                                (×2 avec aller-retour = {occurrences * selectedDays.length * 2} au
                                total)
                              </>
                            )}
                          </>
                        ) : (
                          <>
                            {occurrences} réservation
                            {occurrences > 1 ? 's' : ''}
                            {isRoundTrip && (
                              <> (×2 avec aller-retour = {occurrences * 2} au total)</>
                            )}
                          </>
                        )}
                      </span>
                    )}
                  </div>
                </div>

                <div className={styles.recurrenceFieldRow}>
                  <Label>Jusqu&apos;au</Label>
                  <InlineDatePicker
                    value={recurrenceEndDate}
                    onChange={(v) => setRecurrenceEndDate(v)}
                    placeholder="Date de fin"
                    ariaLabel="Date de fin de récurrence, optionnelle"
                  />
                </div>
              </div>
            )}

          {/* Montant (masqué pour livraison : prix fixe entreprise) */}
          <div className={styles.formGroup} data-tour-id="booking-amount">
            <div className={styles.amountLabel}>
              <Label id="pricing_mode_label" className={styles.routeOptionsLabel}>
                Tarification
              </Label>
            </div>
            <div
              id="pricing_mode"
              className={styles.pricingModeGroup}
              role="group"
              aria-labelledby="pricing_mode_label"
            >
              {[
                { value: 'automatic', label: 'Automatique' },
                { value: 'manual', label: 'Manuel' },
                { value: 'preferential', label: 'Préférentiel' },
              ].map((mode) => {
                const active = (isMaterialDelivery ? 'automatic' : pricingMode) === mode.value;
                return (
                  <button
                    key={mode.value}
                    type="button"
                    className={`${styles.pricingModeBtn} ${active ? styles.isActive : ''}`}
                    aria-pressed={active}
                    disabled={isMaterialDelivery}
                    onClick={() => {
                      setPricingMode(mode.value);
                      setAmountLocked(mode.value !== 'automatic');
                      setAmountSource(mode.value === 'automatic' ? null : mode.value);
                    }}
                  >
                    {mode.label}
                  </button>
                );
              })}
            </div>
            <div className={styles.segmentList} data-testid="route-segment-pricing">
              {routeSegmentLabels.map((label, index) => (
                <div key={label} className={styles.segmentRow}>
                  <span>{label}</span>
                  {pricingMode === 'manual' && !isMaterialDelivery ? (
                    <input
                      type="number"
                      step="0.01"
                      min="0"
                      className={styles.segmentAmount}
                      aria-label={`Montant ${label}`}
                      value={segmentAmounts[index] || ''}
                      onChange={(event) => {
                        const next = [...segmentAmounts];
                        next[index] = event.target.value;
                        setSegmentAmounts(next);
                        setAmountSource('manual');
                        setAmountLocked(true);
                        setPricingWarning('');
                      }}
                    />
                  ) : (
                    <span
                      className={
                        pricingPreview?.segments?.[index]?.amount != null
                          ? styles.segmentAmountValue
                          : styles.segmentAmountRead
                      }
                    >
                      {pricingPreview?.segments?.[index]?.amount != null
                        ? `${pricingPreview.segments[index].amount} CHF`
                        : 'À calculer'}
                    </span>
                  )}
                </div>
              ))}
              <div className={styles.segmentTotal}>
                <span>Total</span>
                <strong>
                  {pricingMode === 'manual' && !isMaterialDelivery
                    ? `${segmentAmounts
                        .reduce((sum, value) => sum + (Number(value) || 0), 0)
                        .toFixed(2)} CHF`
                    : pricingMode === 'preferential'
                      ? `${Number(amount || 0).toFixed(2)} CHF`
                      : pricingPreview?.total
                        ? `${pricingPreview.total} CHF`
                        : 'À calculer'}
                </strong>
              </div>
              {isRecurring && pricingPreview?.series_total ? (
                <div className={styles.segmentRow}>
                  <span>Série</span>
                  <strong>{pricingPreview.series_total} CHF</strong>
                </div>
              ) : null}
            </div>
            {pricingMode === 'preferential' && !isMaterialDelivery && (
              <div className={styles.forfaitRow}>
                <Label htmlFor="amount">Forfait *</Label>
                <Input
                  id="amount"
                  type="number"
                  name="amount"
                  step="0.01"
                  value={amount}
                  onChange={(e) => {
                    const nextValue = e.target.value;
                    setAmount(nextValue);
                    if (!suppressManualAmountRef.current) {
                      setAmountSource('preferential');
                      setAmountLocked(true);
                      setPricingWarning('');
                    }
                  }}
                  placeholder="Ex: 150.00"
                />
              </div>
            )}

            {isMaterialDelivery && (
              <span className={styles.amountHint}>
                Montant géré automatiquement pour les livraisons matériel.
              </span>
            )}
            {!isMaterialDelivery && amountLocked && (
              <span className={styles.amountHint}>
                Montant verrouillé manuellement.
                <button
                  type="button"
                  className={styles.datePresetBtn}
                  style={{ marginLeft: 8 }}
                  onClick={() => {
                    setAmountLocked(false);
                    setAmountSource(null);
                    runPricingSimulation();
                  }}
                >
                  Recalculer
                </button>
              </span>
            )}
            {!isMaterialDelivery && isSimulatingPricing && (
              <span className={styles.amountHint}>
                {amountSource === 'simulated' && amount
                  ? 'Mise à jour du montant exact en cours…'
                  : 'Calcul du montant en cours…'}
              </span>
            )}
            {!isMaterialDelivery && pricingWarning && (
              <span className={styles.amountHint}>{pricingWarning}</span>
            )}
            {isRecurring && Array.isArray(pricingPreview?.occurrences) && (
              <span className={styles.amountHint}>
                {pricingPreview.occurrences
                  .map((row) => `${row.occurrence_date} : ${row.total}`)
                  .join(' · ')}
              </span>
            )}
            {!isMaterialDelivery && lastPricingUpdateAt && (
              <span className={styles.amountHint}>
                Dernière mise à jour: {lastPricingUpdateAt.toLocaleTimeString('fr-CH')}
              </span>
            )}
          </div>
        </div>

        {/* COLONNE DROITE - Informations médicales (section secondaire) */}
        <div className={styles.columnRight}>
          <div className={styles.medicalSection} data-tour-id="booking-medical-section">
            <CompanyRouteDetails
              points={points}
              isMaterialDelivery={isMaterialDelivery}
              onSelect={setSelectedStepKey}
              onPatch={patchRoutePoint}
            />

            <section className={styles.detailSection} data-testid="mobility-section">
              <h3 className={styles.detailTitle}>Mobilité</h3>
              <div className={styles.mobilityGroup} role="group" aria-label="Mobilité">
                <button
                  type="button"
                  className={`${styles.mobilityBtn} ${wheelchairOptions.clientHasWheelchair ? styles.isActive : ''}`}
                  onClick={() => setWheelchairOptions({
                    clientHasWheelchair: !wheelchairOptions.clientHasWheelchair,
                    needWheelchair: false,
                  })}
                  aria-pressed={wheelchairOptions.clientHasWheelchair}
                >
                  En chaise
                </button>
                <button
                  type="button"
                  className={`${styles.mobilityBtn} ${wheelchairOptions.needWheelchair ? styles.isActive : ''}`}
                  onClick={() => setWheelchairOptions({
                    needWheelchair: !wheelchairOptions.needWheelchair,
                    clientHasWheelchair: false,
                  })}
                  aria-pressed={wheelchairOptions.needWheelchair}
                >
                  Fournir chaise
                </button>
                <button
                  type="button"
                  className={`${styles.mobilityBtn} ${needsAssistance ? styles.isActive : ''}`}
                  onClick={() => setNeedsAssistance((value) => !value)}
                  aria-pressed={needsAssistance}
                >
                  Assistance
                </button>
              </div>
              <div className={styles.notesBlock}>
                <Label htmlFor="notes_medical">
                  Notes{needsAssistance ? ' *' : ''}
                </Label>
                <IsolatedTextarea
                  id="notes_medical"
                  name="notes_medical"
                  externalValue={notesMedicalExternal}
                  valueRef={notesMedicalRef}
                  onBlur={handleNotesMedicalBlur}
                  placeholder="Instructions particulières, bâtiment, étage…"
                  rows={1}
                  required={needsAssistance}
                  aria-required={needsAssistance || undefined}
                  className={`${styles.textarea} ${styles.textareaCompact}`}
                />
              </div>
            </section>

            <section className={styles.detailSection}>
              <h3 className={styles.detailTitle}>Contact</h3>
              <div className={styles.detailAccessRow}>
                <label className={styles.detailLabel} htmlFor="requester_name">
                  Nom
                </label>
                <Input
                  id="requester_name"
                  name="requester_name"
                  className={styles.detailInput}
                  value={requesterName}
                  onChange={(e) => setRequesterName(e.target.value)}
                  placeholder="Nom du contact"
                />
              </div>
              <div className={styles.detailAccessRow}>
                <label className={styles.detailLabel} htmlFor="requester_phone">
                  Téléphone
                </label>
                <Input
                  id="requester_phone"
                  name="requester_phone"
                  className={styles.detailInput}
                  value={requesterPhone}
                  onChange={(e) => setRequesterPhone(e.target.value)}
                  placeholder="Téléphone"
                />
              </div>
            </section>

          </div>
        </div>
        </div>

        {/* Footer fixe en bas : résumé + Annuler à gauche, CTA à droite */}
        <div className={styles.footerActions} data-testid="footer-actions-sticky">
          <div className={styles.footerSummary} aria-live="polite">
            <span className={styles.footerSummaryText}>{footerSummary.summaryText}</span>
            <span className={styles.footerSummaryBadges}>
              {footerSummary.summaryBadges.map((badge) => (
                <span key={badge} className={styles.summaryBadge}>
                  {badge}
                </span>
              ))}
            </span>
          </div>
          <div className={styles.footerBody}>
            <div className={styles.footerLeft}>
              {onClose && (
                <button type="button" className={styles.footerCancel} onClick={onClose}>
                  Annuler
                </button>
              )}
            </div>
            <div className={styles.footerRight}>
              <button
                type="submit"
                data-tour-id="booking-submit"
                className={styles.submitButton}
                disabled={
                  bookingMutation.isLoading ||
                  (isMaterialDelivery && !deliveryDescription?.trim())
                }
              >
                {bookingMutation.isLoading ? '⏳ Création…' : 'Créer la réservation'}
              </button>
            </div>
          </div>
        </div>
      </form>

      {showClientModal && (
        <NewClientModal onClose={() => setShowClientModal(false)} onSave={handleSaveNewClient} />
      )}
    </div>
  );
}
