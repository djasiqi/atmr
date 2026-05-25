import { useEffect, useState } from "react";

/**
 * Évite le clignotement UI quand une condition booléenne oscille (sync réseau).
 */
export function useDebouncedFlag(active: boolean, delayMs: number): boolean {
  const [debounced, setDebounced] = useState(active);

  useEffect(() => {
    const id = setTimeout(() => setDebounced(active), delayMs);
    return () => clearTimeout(id);
  }, [active, delayMs]);

  return debounced;
}
