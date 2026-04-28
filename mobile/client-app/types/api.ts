export type ApiListResponse<T> = {
  data?: T[];
  items?: T[];
  results?: T[];
  requests?: T[];
  patients?: T[];
  bookings?: T[];
  total?: number;
  page?: number;
  per_page?: number;
  pages?: number;
};

export type PaginatedResult<T> = {
  items: T[];
  total: number;
  page: number;
  perPage: number;
  pages: number;
};

export type Booking = {
  id: number;
  public_id?: string;
  status?: string;
  pickup_address?: string | null;
  destination_address?: string | null;
  scheduled_time?: string | null;
  company_name?: string | null;
  driver_name?: string | null;
  estimated_price?: number | null;
  amount?: number | null;
  payment_status?: string | null;
  payment_provider?: string | null;
  payment_required?: boolean;
  notes?: string | null;
};

export type InstitutionRequest = {
  id: number;
  public_id?: string;
  institution_id?: number;
  patient_id?: number | null;
  patient?: Patient | null;
  mission_type?: string | null;
  delivery_description?: string | null;
  external_reference?: string | null;
  status?: string;
  is_editable?: boolean;
  is_cancellable?: boolean;
  scheduled_time?: string | null;
  scheduled_time_type?: string | null;
  created_by_name?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  sent_at?: string | null;
  cancelled_at?: string | null;
  accepted_at?: string | null;
  converted_at?: string | null;
  pickup_location?: string | null;
  pickup_lat?: number | null;
  pickup_lng?: number | null;
  pickup_floor?: string | null;
  pickup_door_code?: string | null;
  dropoff_location?: string | null;
  dropoff_lat?: number | null;
  dropoff_lng?: number | null;
  dropoff_floor?: string | null;
  dropoff_door_code?: string | null;
  pickup_type?: string | null;
  dropoff_type?: string | null;
  pickup_entry_point?: string | null;
  dropoff_entry_point?: string | null;
  is_round_trip?: boolean;
  return_time?: string | null;
  mobility?: {
    wheelchair?: boolean;
    stretcher?: boolean;
    oxygen?: boolean;
    walking?: boolean;
    needs_assistance?: boolean;
  } | null;
  floor_elevator_info?: string | null;
  contact_on_site?: string | null;
  notes?: string | null;
  billing_intent?: string | null;
  billing_details?: Record<string, unknown> | null;
  booking_id?: number | null;
  accepted_by_company?: {
    id?: number;
    name?: string;
  } | null;
  booking_summary?: {
    id?: number;
    status?: string;
    scheduled_time?: string | null;
    amount?: number | null;
    company_name?: string | null;
    total_amount?: number | null;
    overall_status?: string | null;
    return_booking?: {
      id?: number;
      status?: string;
      scheduled_time?: string | null;
    } | null;
  } | null;
};

export type Patient = {
  id: number;
  public_id?: string;
  external_reference?: string | null;
  first_name?: string | null;
  last_name?: string | null;
  full_name?: string | null;
  dob?: string | null;
  gender?: string | null;
  address?: string | null;
  city?: string | null;
  postal_code?: string | null;
  phone?: string | null;
  email?: string | null;
  door_code?: string | null;
  floor?: string | null;
  access_notes?: string | null;
  residence_name?: string | null;
  avs_number?: string | null;
  insurance_name?: string | null;
  insurance_number?: string | null;
  has_guardianship?: boolean;
  guardianship_type?: string | null;
  guardian_name?: string | null;
  guardian_organization?: string | null;
  guardian_phone?: string | null;
  guardian_email?: string | null;
  guardian_address?: string | null;
  curator_team_id?: number | null;
  data_source_flags?: Record<string, unknown> | null;
  notes?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
};

export type InstitutionMe = {
  id: number;
  public_id?: string;
  name: string;
  institution_type?: string | null;
  address?: string | null;
  contact_email?: string | null;
  contact_phone?: string | null;
  notes?: string | null;
  institution_role?: string | null;
  user?: {
    id?: number;
    public_id?: string;
    username?: string;
    email?: string | null;
    first_name?: string | null;
    last_name?: string | null;
    phone?: string | null;
  } | null;
};

export type InstitutionSettings = {
  id?: number;
  institution_id?: number;
  timeout_same_day_minutes?: number;
  timeout_default_minutes?: number;
  default_billing_intent?: string;
  default_vat_rate?: number | null;
  default_payment_terms_days?: number;
  notification_emails?: string[];
  notify_request_sent?: boolean;
  notify_offer_accepted?: boolean;
  notify_request_expired?: boolean;
  timezone?: string;
  default_pickup_mode?: string;
  entry_points?: string[];
  default_contact_phone?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
};

export type CreateInstitutionRequestPayload = {
  external_reference: string;
  scheduled_time: string;
  pickup_location: string;
  dropoff_location: string;
  mission_type?: string;
  patient_id?: number | null;
  patient_external_reference?: string | null;
  pickup_lat?: number | null;
  pickup_lng?: number | null;
  pickup_floor?: string | null;
  pickup_door_code?: string | null;
  dropoff_lat?: number | null;
  dropoff_lng?: number | null;
  dropoff_floor?: string | null;
  dropoff_door_code?: string | null;
  pickup_type?: string | null;
  dropoff_type?: string | null;
  pickup_entry_point?: string | null;
  dropoff_entry_point?: string | null;
  is_round_trip?: boolean;
  return_time?: string | null;
  mobility?: {
    wheelchair?: boolean;
    stretcher?: boolean;
    oxygen?: boolean;
    walking?: boolean;
    needs_assistance?: boolean;
  };
  floor_elevator_info?: string | null;
  contact_on_site?: string | null;
  notes?: string | null;
  billing_intent?: 'patient' | 'institution' | 'curator' | 'spc' | 'other';
};

export type CreatePatientPayload = {
  first_name: string;
  last_name: string;
  dob?: string | null;
  gender?: string | null;
  address?: string | null;
  phone?: string | null;
  door_code?: string | null;
  floor?: string | null;
  access_notes?: string | null;
  notes?: string | null;
  external_reference?: string | null;
};

export type ClientProfile = {
  id?: number;
  public_id?: string;
  user_id?: number;
  first_name?: string | null;
  last_name?: string | null;
  email?: string | null;
  phone?: string | null;
  address?: string | null;
  user?: {
    public_id?: string;
    email?: string | null;
    first_name?: string | null;
    last_name?: string | null;
  };
};

export type SaferpayInitializeResponse = {
  redirect_url?: string;
  payment_id?: number;
  transaction_id?: string;
  status?: string;
};

export type SaferpayAssertResponse = {
  status?: string;
  payment_id?: number;
  booking_status?: string;
  message?: string;
};
