import { useEffect, useRef, useState } from "react";
import { beginTransitionPress, commitTransitionPress } from "./driverTransitionPress";

/**
 * Un geste = une transition.
 * Le bouton se verrouille dès l'envoi, puis reste silencieux jusqu'à un nouvel appui.
 */
export function useSingleTransitionPress<T extends string>(pending: boolean, target: T | null) {
  const captured = useRef<string | null>(null);
  const [quietUntil, setQuietUntil] = useState(0);

  useEffect(() => {
    if (quietUntil <= Date.now()) return;
    const timer = setTimeout(() => setQuietUntil(0), quietUntil - Date.now());
    return () => clearTimeout(timer);
  }, [quietUntil]);

  const locked = pending || quietUntil > Date.now();

  const onPressIn = () => {
    captured.current = beginTransitionPress({
      target,
      pending,
      now: Date.now(),
      quietUntil,
    });
  };

  const onPress = (): T | null => {
    const decision = commitTransitionPress({
      captured: captured.current,
      target,
      pending,
      now: Date.now(),
      quietUntil,
    });
    captured.current = null;
    if (!decision.target) return null;
    const chosen = decision.target as T;
    setQuietUntil(decision.quietUntil);
    return chosen;
  };

  return { onPressIn, onPress, locked };
}
