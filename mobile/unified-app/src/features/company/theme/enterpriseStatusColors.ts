/**
 * Couleurs de statut alignées sur operations (RideSnippetCard + ride-details).
 */
const statusColors = {
  pending: { bg: "#fef3c7", text: "#f59e0b" },
  proposed: { bg: "#e0e7ff", text: "#4f46e5" },
  accepted: { bg: "#dbeafe", text: "#3b82f6" },
  assigned: { bg: "#dbeafe", text: "#3b82f6" },
  en_route: { bg: "#fef3c7", text: "#f59e0b" },
  arrived: { bg: "#dbeafe", text: "#3b82f6" },
  in_progress: { bg: "#fef3c7", text: "#f59e0b" },
  completed: { bg: "#dcfce7", text: "#16a34a" },
  return_completed: { bg: "#dcfce7", text: "#16a34a" },
  cancelled: { bg: "#f3f4f6", text: "#6b7280" },
  canceled: { bg: "#f3f4f6", text: "#6b7280" },
} as const;

export function getEnterpriseStatusColors(status?: string) {
  if (!status) return statusColors.pending;
  const n = String(status).toLowerCase().trim();
  if (n in statusColors) {
    return statusColors[n as keyof typeof statusColors];
  }
  return statusColors.pending;
}
