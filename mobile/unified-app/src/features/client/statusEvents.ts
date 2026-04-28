export function trackClientKpiEvent(
  name: string,
  payload: Record<string, unknown> = {}
) {
  console.info(`[client-kpi] ${name}`, payload);
}

