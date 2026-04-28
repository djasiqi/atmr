export const queryKeys = {
  clientPublicId: ['client', 'public-id'] as const,
  clientProfile: ['client', 'profile'] as const,
  bookings: ['bookings'] as const,
  booking: (bookingId: number | string) => ['booking', String(bookingId)] as const,
  institutionMe: ['institution', 'me'] as const,
  institutionRequests: (params?: {
    status?: string;
    external_reference?: string;
    patient_id?: number;
    date_from?: string;
    date_to?: string;
    page?: number;
    per_page?: number;
  }) =>
    [
      'institution',
      'requests',
      params?.status ?? '',
      params?.external_reference ?? '',
      String(params?.patient_id ?? ''),
      params?.date_from ?? '',
      params?.date_to ?? '',
      String(params?.page ?? 1),
      String(params?.per_page ?? 20),
    ] as const,
  institutionRequest: (requestId: number | string) =>
    ['institution', 'request', String(requestId)] as const,
  institutionPatients: (query: string) =>
    ['institution', 'patients', query.trim().toLowerCase()] as const,
  institutionPatientsPage: (params?: {
    query?: string;
    page?: number;
    per_page?: number;
  }) =>
    [
      'institution',
      'patients',
      params?.query?.trim().toLowerCase() ?? '',
      String(params?.page ?? 1),
      String(params?.per_page ?? 20),
    ] as const,
  institutionPatient: (patientId: number | string) =>
    ['institution', 'patient', String(patientId)] as const,
  institutionSettings: ['institution', 'settings'] as const,
};
