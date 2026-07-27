/**
 * Client registre de sessions tracking (plan v5 Annexe A.2).
 * POST /driver/me/tracking/sessions — idempotent, offline-first.
 */

import { apiClient } from "../../../core/api/apiClient";

export type TrackingSessionRegisterRequest = {
  tracking_session_id: string;
  tracking_session_started_at: string;
};

export type TrackingSessionRegisterResponse = {
  tracking_session_id: string;
  session_generation: number;
  first_sequence_id: number;
  status: "active" | "superseded" | "closed" | "expired";
};

export type TrackingSessionCloseResponse = {
  tracking_session_id: string;
  status: "closed";
  closed_at: string;
  final_sequence_id: number | null;
};

export type TrackingWatermarkResponse = {
  ack_status: "persisted";
  tracking_session_id: string;
  session_generation: number;
  contiguous_persisted_through: number;
  out_of_order_persisted: Array<{
    sequence_id: number;
    location_event_id: string;
  }>;
  missing_ranges: Array<[number, number]>;
  next_cursor: string | null;
};

export async function registerTrackingSession(
  payload: TrackingSessionRegisterRequest
): Promise<TrackingSessionRegisterResponse> {
  const { data } = await apiClient.post<TrackingSessionRegisterResponse>(
    "/driver/me/tracking/sessions",
    payload
  );
  return data;
}

export async function closeTrackingSession(
  trackingSessionId: string,
  finalSequenceId?: number | null
): Promise<TrackingSessionCloseResponse> {
  const { data } = await apiClient.post<TrackingSessionCloseResponse>(
    `/driver/me/tracking/sessions/${encodeURIComponent(trackingSessionId)}/close`,
    { final_sequence_id: finalSequenceId ?? null }
  );
  return data;
}

export async function fetchTrackingWatermark(
  trackingSessionId: string,
  cursor?: string | null
): Promise<TrackingWatermarkResponse> {
  const params = new URLSearchParams({
    tracking_session_id: trackingSessionId,
  });
  if (cursor) {
    params.set("cursor", cursor);
  }
  const { data } = await apiClient.get<TrackingWatermarkResponse>(
    `/driver/me/tracking/watermark?${params.toString()}`
  );
  return data;
}
