import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";

/**
 * KPI temps-réel publiés par le shell chat à destination du LayoutDebugOverlay.
 * - composerKbGap : `effectiveKeyboardTopY - footerBottom` (cible 8 px ±4).
 * - shellBottomGap : hauteur empilée au-dessus du footer (quick actions + suggestions + safe area).
 */
export type ChatLayoutKpis = {
  composerKbGap: number | null;
  shellBottomGap: number | null;
  footerHeight: number | null;
};

type ChatLayoutKpisContextValue = ChatLayoutKpis & {
  publish: (kpis: ChatLayoutKpis) => void;
};

const EMPTY: ChatLayoutKpis = {
  composerKbGap: null,
  shellBottomGap: null,
  footerHeight: null,
};

const ChatLayoutKpisContext = createContext<ChatLayoutKpisContextValue>({
  ...EMPTY,
  publish: () => {},
});

export function ChatLayoutKpisProvider({ children }: { children: ReactNode }) {
  const [kpis, setKpis] = useState<ChatLayoutKpis>(EMPTY);
  const publish = useCallback((next: ChatLayoutKpis) => {
    setKpis((prev) =>
      prev.composerKbGap === next.composerKbGap &&
      prev.shellBottomGap === next.shellBottomGap &&
      prev.footerHeight === next.footerHeight
        ? prev
        : next
    );
  }, []);
  const value = useMemo<ChatLayoutKpisContextValue>(
    () => ({ ...kpis, publish }),
    [kpis, publish]
  );
  return (
    <ChatLayoutKpisContext.Provider value={value}>{children}</ChatLayoutKpisContext.Provider>
  );
}

/** Hook consommé par le LayoutDebugOverlay pour afficher les KPI chat. */
export function useChatLayoutKpis(): ChatLayoutKpis {
  const ctx = useContext(ChatLayoutKpisContext);
  return {
    composerKbGap: ctx.composerKbGap,
    shellBottomGap: ctx.shellBottomGap,
    footerHeight: ctx.footerHeight,
  };
}

/** Hook consommé par le shell chat pour publier les KPI mesurés. */
export function useChatLayoutKpisPublisher(): (kpis: ChatLayoutKpis) => void {
  return useContext(ChatLayoutKpisContext).publish;
}
