import { useMemo } from 'react';

export default function usePlanningKPIs(items = [], mode = 'day') {
  return useMemo(() => {
    const total = items.length;
    const planned = items.filter((s) => String(s.status).toLowerCase() === 'planned').length;
    const active = items.filter((s) => String(s.status).toLowerCase() === 'active').length;
    const completed = items.filter((s) => String(s.status).toLowerCase() === 'completed').length;

    const minutes = (isoA, isoB) =>
      Math.max(0, Math.round((new Date(isoB) - new Date(isoA)) / 60000));
    const totalMinutes = items.reduce((m, s) => m + minutes(s.start_local, s.end_local), 0);

    return {
      total,
      planned,
      active,
      completed,
      totalMinutes,
      mode,
    };
  }, [items, mode]);
}
