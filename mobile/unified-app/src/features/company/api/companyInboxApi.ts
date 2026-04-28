import { apiClient } from "../../../core/api/client";
import { AxiosError } from "axios";

export type CompanyInboxNotification = {
  id: number;
  company_id?: number;
  event_type: string;
  title: string;
  message: string;
  is_read: boolean;
  created_at: string;
  metadata?: Record<string, unknown>;
};

export type CompanyInboxResponse = {
  notifications: CompanyInboxNotification[];
  unread_count: number;
  total: number;
};

function isForbidden(error: unknown): boolean {
  return error instanceof AxiosError && error.response?.status === 403;
}

/**
 * In-app notifications entreprise (`GET /api/v1/companies/notifications`).
 * Nécessite un JWT avec claim `company_id` (même rôle qu’en web).
 */
export async function getCompanyInboxNotifications(
  options: { limit?: number } = {}
): Promise<CompanyInboxResponse> {
  try {
    const { data } = await apiClient.get<CompanyInboxResponse>("/companies/notifications", {
      params: { limit: options.limit ?? 30 },
    });
    return {
      notifications: Array.isArray(data?.notifications) ? data.notifications : [],
      unread_count: typeof data?.unread_count === "number" ? data.unread_count : 0,
      total: typeof data?.total === "number" ? data.total : 0,
    };
  } catch (e) {
    if (isForbidden(e)) {
      return { notifications: [], unread_count: 0, total: 0 };
    }
    throw e;
  }
}

export async function markCompanyNotificationRead(notificationId: number): Promise<void> {
  await apiClient.put(`/companies/notifications/${notificationId}/read`);
}

export async function markAllCompanyNotificationsRead(): Promise<void> {
  await apiClient.put("/companies/notifications/read-all");
}
