import { api } from '@/services/api';
import type {
  ApiListResponse,
  CreateInstitutionRequestPayload,
  CreatePatientPayload,
  InstitutionMe,
  InstitutionRequest,
  InstitutionSettings,
  PaginatedResult,
  Patient,
} from '@/types/api';

function normalizeList<T>(data: unknown): T[] {
  if (Array.isArray(data)) return data as T[];
  const obj = data as ApiListResponse<T> | undefined;
  if (!obj || typeof obj !== 'object') return [];
  return obj.items ?? obj.results ?? obj.data ?? obj.requests ?? obj.patients ?? [];
}

function normalizePaginated<T>(
  data: unknown,
  fallbackPerPage = 20,
): PaginatedResult<T> {
  const items = normalizeList<T>(data);
  const obj = data as ApiListResponse<T> | undefined;
  if (!obj || typeof obj !== 'object') {
    return {
      items,
      total: items.length,
      page: 1,
      perPage: fallbackPerPage,
      pages: 1,
    };
  }
  const total = Number(obj.total ?? items.length);
  const page = Number(obj.page ?? 1);
  const perPage = Number(obj.per_page ?? fallbackPerPage);
  const pages = Number(
    obj.pages ?? (perPage > 0 ? Math.max(1, Math.ceil(total / perPage)) : 1),
  );
  return { items, total, page, perPage, pages };
}

function unwrapData<T>(data: unknown): T {
  const obj = data as { data?: T } | T;
  if (obj && typeof obj === 'object' && 'data' in (obj as { data?: T })) {
    return ((obj as { data?: T }).data ?? {}) as T;
  }
  return obj as T;
}

export async function getInstitutionMe(): Promise<InstitutionMe> {
  const res = await api.get<InstitutionMe>('/institutions/me');
  return res.data;
}

export async function listRequests(params?: {
  status?: string;
  external_reference?: string;
  patient_id?: number;
  date_from?: string;
  date_to?: string;
  page?: number;
  per_page?: number;
}): Promise<PaginatedResult<InstitutionRequest>> {
  const res = await api.get<InstitutionRequest[] | ApiListResponse<InstitutionRequest>>(
    '/institutions/requests',
    { params },
  );
  return normalizePaginated<InstitutionRequest>(res.data, params?.per_page ?? 20);
}

export async function getRequest(requestId: number): Promise<InstitutionRequest> {
  const res = await api.get<{ data?: InstitutionRequest } | InstitutionRequest>(
    `/institutions/requests/${requestId}`,
  );
  return unwrapData<InstitutionRequest>(res.data);
}

export async function listPatients(params?: {
  query?: string;
  external_reference?: string;
  page?: number;
  per_page?: number;
}): Promise<PaginatedResult<Patient>> {
  const res = await api.get<Patient[] | ApiListResponse<Patient>>('/institutions/patients', {
    params,
  });
  return normalizePaginated<Patient>(res.data, params?.per_page ?? 20);
}

export async function getPatient(patientId: number): Promise<Patient> {
  const res = await api.get<{ data?: Patient } | Patient>(`/institutions/patients/${patientId}`);
  return unwrapData<Patient>(res.data);
}

export async function createRequest(
  payload: CreateInstitutionRequestPayload,
): Promise<InstitutionRequest> {
  const res = await api.post<{ data?: InstitutionRequest } | InstitutionRequest>(
    '/institutions/requests',
    payload,
  );
  return unwrapData<InstitutionRequest>(res.data);
}

export async function sendRequest(requestId: number): Promise<InstitutionRequest> {
  const res = await api.post<{ data?: InstitutionRequest } | InstitutionRequest>(
    `/institutions/requests/${requestId}/send`,
    {},
  );
  return unwrapData<InstitutionRequest>(res.data);
}

export async function cancelRequest(requestId: number): Promise<InstitutionRequest> {
  const res = await api.post<{ data?: InstitutionRequest } | InstitutionRequest>(
    `/institutions/requests/${requestId}/cancel`,
    {},
  );
  return unwrapData<InstitutionRequest>(res.data);
}

export async function createPatient(payload: CreatePatientPayload): Promise<Patient> {
  const res = await api.post<{ data?: Patient } | Patient>('/institutions/patients', payload);
  return unwrapData<Patient>(res.data);
}

export async function updatePatient(
  patientId: number,
  payload: Partial<CreatePatientPayload>,
): Promise<Patient> {
  const res = await api.put<{ data?: Patient } | Patient>(
    `/institutions/patients/${patientId}`,
    payload,
  );
  return unwrapData<Patient>(res.data);
}

export async function getInstitutionSettings(): Promise<InstitutionSettings> {
  const res = await api.get<{ settings?: InstitutionSettings } | InstitutionSettings>(
    '/institutions/settings',
  );
  const data = res.data as { settings?: InstitutionSettings } | InstitutionSettings;
  if (data && typeof data === 'object' && 'settings' in data) {
    return (data.settings ?? {}) as InstitutionSettings;
  }
  return data as InstitutionSettings;
}

export async function updateInstitutionSettings(
  payload: Partial<InstitutionSettings>,
): Promise<InstitutionSettings> {
  const res = await api.put<{ settings?: InstitutionSettings } | InstitutionSettings>(
    '/institutions/settings',
    payload,
  );
  const data = res.data as { settings?: InstitutionSettings } | InstitutionSettings;
  if (data && typeof data === 'object' && 'settings' in data) {
    return (data.settings ?? {}) as InstitutionSettings;
  }
  return data as InstitutionSettings;
}
