export type ApiListResponse<T> = {
  data?: T[];
  items?: T[];
  results?: T[];
  bookings?: T[];
  total?: number;
  page?: number;
  per_page?: number;
};

/** Réponse `POST /clients/me/indicative-fare/estimate` (hors preview réservation). */
export type IndicativeFareEstimateResponse = {
  distance_m: number;
  duration_s: number;
  indicative_amount_chf: number;
  currency: string;
  config_version: number;
  is_contractual: false;
};

export type ClientProfile = {
  id?: number;
  first_name?: string | null;
  last_name?: string | null;
  full_name?: string | null;
  contact_email?: string | null;
  phone?: string | null;
  domicile?: {
    address?: string | null;
    zip?: string | null;
    city?: string | null;
    /** Présents si l’adresse a été géocodée côté serveur (sélection / fiabilité métier). */
    lat?: number | null;
    lon?: number | null;
    lng?: number | null;
  };
  user?: {
    public_id?: string;
    email?: string | null;
    first_name?: string | null;
    last_name?: string | null;
  };
};

export type Booking = {
  id: number;
  status?: string | null;
  amount?: number | null;
  scheduled_time?: string | null;
  pickup_location?: string | null;
  dropoff_location?: string | null;
  is_return?: boolean;
  is_round_trip?: boolean;
  return_trip?: {
    amount?: number | null;
  } | null;
  return_booking?: {
    amount?: number | null;
  } | null;
  company_name?: string | null;
  payment_required?: boolean;
  payment_status?: PaymentStatus;
  payment_amount?: number | null;
  currency?: string | null;
  payment_provider?: string | null;
  payment_reference?: string | null;
  last_payment_error?: string | null;
  online_payment?: {
    payment_id?: number;
    status?: string | null;
    provider?: string | null;
    has_pending_session?: boolean;
  } | null;
  client?: {
    id?: number | null;
    first_name?: string | null;
    last_name?: string | null;
    email?: string | null;
  };
  price_amount?: number | null;
  price_breakdown_json?: Record<string, unknown> | null;
  pricing_status?: PricingStatus | null;
};

export type PaymentStatus =
  | "not_required"
  | "required"
  | "pending_verification"
  | "paid"
  | "failed"
  | "cancelled"
  | "unknown";

export type PricingStatus =
  | "unavailable"
  | "indicative"
  | "estimated"
  | "confirmed"
  | "adjusted";

export type SaferpayInitializeResponse = {
  payment_id: number;
  booking_id: number;
  redirect_url: string;
  order_id?: string;
  payment_provider?: string;
  payment_status?: string;
  payment_amount?: number;
  currency?: string;
};

export type SaferpayAssertResponse = {
  status?: string;
  payment_id?: number;
  booking_id?: number;
  payment_provider?: string;
  payment_status?: PaymentStatus;
  booking_status?: string;
  pending_verification?: boolean;
  detail?: string | null;
};

export type ClientApiError = {
  status: number | null;
  code: string;
  message: string;
  actionHint?: string;
  retryable?: boolean;
};

export type AddressAutocompleteSuggestion = {
  source?: string;
  label: string;
  address?: string;
  lat?: number | null;
  lon?: number | null;
  lng?: number | null;
  place_id?: string;
};

export type RecurrenceType = "daily" | "weekly" | "custom";

export type BookingDraftPayload = {
  customer_name?: string;
  pickup_location: string;
  dropoff_location: string;
  scheduled_time?: string | null;
  asap?: boolean;
  is_round_trip?: boolean;
  return_time?: string | null;
  return_date?: string | null;
  occurrences?: number;
  client_note?: string;
  is_recurring?: boolean;
  recurrence_type?: RecurrenceType | null;
  recurrence_days?: number[] | null;
  recurrence_end_date?: string | null;
  recurrence_series_length?: number | null;
};

export type BookingPreviewResponse = {
  success?: boolean;
  contracts?: {
    status_dictionary_version?: string;
    pricing_contract_version?: string;
    canonical_address_contract_version?: string;
    preview_contract_version?: string;
    medical_fields_contract_version?: string;
  };
  pricing?: {
    amount: number;
    currency?: string | null;
    distance_meters?: number | null;
    duration_seconds?: number | null;
    pricing_profile_id?: number | null;
    pricing_profile_version_id?: number | null;
    pricing_status?: PricingStatus | null;
    breakdown?: Record<string, unknown> | null;
  };
  canonical_addresses?: {
    pickup?: {
      label?: string;
      lat?: number | null;
      lon?: number | null;
      lng?: number | null;
      admin_token?: string | null;
      place_id?: string | null;
      osm_id?: string | null;
      photon_id?: string | null;
      canonical_hash?: string | null;
      precision_level?: CanonicalAddressPrecisionLevel | null;
    };
    dropoff?: {
      label?: string;
      lat?: number | null;
      lon?: number | null;
      lng?: number | null;
      admin_token?: string | null;
      place_id?: string | null;
      osm_id?: string | null;
      photon_id?: string | null;
      canonical_hash?: string | null;
      precision_level?: CanonicalAddressPrecisionLevel | null;
    };
  };
  workflow?: {
    payment_required?: boolean;
    transmission_requires_client_action?: boolean;
  };
  validation?: {
    canonical_precision_acceptance_matrix?: Partial<
      Record<CanonicalAddressPrecisionLevel, "allow" | "warn" | "block">
    >;
  };
};

export type CanonicalAddressPrecisionLevel =
  | "rooftop"
  | "entrance"
  | "street"
  | "locality"
  | "approximate";

export type CreateClientBookingPayload = BookingDraftPayload & {
  customer_name: string;
  amount: number;
  medical_facility?: string;
  doctor_name?: string;
  /** Service ou unité hospitalière (ex. urgences, cardiologie) — max 100 côté API. */
  hospital_service?: string;
  preview_amount?: number;
};

export type CreateClientBookingResponse = {
  booking_id?: number;
  trace_id?: string;
  booking?: {
    id?: number;
    amount?: number;
    status?: string;
    billed_to_type?: string;
    pricing_adjustment_reason?: string | null;
    price_amount?: number | null;
    pricing_status?: PricingStatus | null;
    pricing_contract_version?: string | null;
  };
  message?: string;
};
