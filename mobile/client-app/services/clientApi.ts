import { fetchAuthMe } from '@/services/auth';
import { api } from '@/services/api';
import type { ApiListResponse, Booking, ClientProfile } from '@/types/api';

let cachedClientPublicId: string | null = null;

function pickClientPublicIdFromAuthMe(me: Record<string, unknown>): string | null {
  const candidates = [
    (me.client as { public_id?: string } | undefined)?.public_id,
    (me.profile as { public_id?: string } | undefined)?.public_id,
    (me.user as { public_id?: string } | undefined)?.public_id,
    me.public_id as string | undefined,
  ];
  for (const candidate of candidates) {
    const value = String(candidate ?? '').trim();
    if (value) return value;
  }
  return null;
}

async function validateClientPublicId(publicId: string): Promise<string | null> {
  try {
    const res = await api.get<ClientProfile>(`/clients/${publicId}`);
    const resolved =
      String(res.data?.public_id ?? res.data?.user?.public_id ?? publicId).trim() || null;
    return resolved;
  } catch {
    return null;
  }
}

async function resolveFromClientsMe(): Promise<string | null> {
  try {
    const res = await api.get<ClientProfile>('/clients/me');
    return String(res.data?.public_id ?? res.data?.user?.public_id ?? '').trim() || null;
  } catch {
    return null;
  }
}

export async function resolveClientPublicId(): Promise<string> {
  if (cachedClientPublicId) {
    return cachedClientPublicId;
  }

  const me = (await fetchAuthMe()) as unknown as Record<string, unknown>;
  const fromMe = pickClientPublicIdFromAuthMe(me);
  if (fromMe) {
    const validated = await validateClientPublicId(fromMe);
    cachedClientPublicId = validated ?? fromMe;
    return cachedClientPublicId;
  }

  const fromClientsMe = await resolveFromClientsMe();
  if (fromClientsMe) {
    cachedClientPublicId = fromClientsMe;
    return cachedClientPublicId;
  }

  throw new Error("Impossible de résoudre l'identifiant client (public_id).");
}

function normalizeBookings(data: unknown): Booking[] {
  if (Array.isArray(data)) return data as Booking[];
  const obj = data as ApiListResponse<Booking> | undefined;
  if (!obj || typeof obj !== 'object') return [];
  return (
    obj.items ??
    obj.results ??
    obj.data ??
    obj.bookings ??
    []
  );
}

export async function getClientProfile(): Promise<ClientProfile> {
  const publicId = await resolveClientPublicId();
  const res = await api.get<ClientProfile>(`/clients/${publicId}`);
  return res.data;
}

export async function getClientBookings(): Promise<Booking[]> {
  const publicId = await resolveClientPublicId();
  const res = await api.get<Booking[] | ApiListResponse<Booking>>(`/clients/${publicId}/bookings`);
  return normalizeBookings(res.data);
}

export async function getBooking(bookingId: number): Promise<Booking> {
  const res = await api.get<{ data?: Booking } | Booking>(`/bookings/${bookingId}`);
  if ('data' in (res.data as { data?: Booking })) {
    return ((res.data as { data?: Booking }).data ?? {}) as Booking;
  }
  return res.data as Booking;
}
