import React, {
  createContext,
  useContext,
  useMemo,
  useState,
  ReactNode,
} from "react";
import dayjs from "dayjs";

export type EnterpriseMode = "manual" | "semi_auto" | "fully_auto";

type EnterpriseContextValue = {
  selectedDate: string;
  setSelectedDate: (isoDate: string) => void;
  minDate: string;
  maxDate: string;
  mode: EnterpriseMode;
  setMode: (mode: EnterpriseMode) => void;
  autoPaused: boolean;
  setAutoPaused: (paused: boolean) => void;
};

const EnterpriseContext = createContext<EnterpriseContextValue | undefined>(
  undefined
);

type EnterpriseProviderProps = {
  children: ReactNode;
  initialMode?: EnterpriseMode;
};

export const EnterpriseProvider = ({
  children,
  initialMode = "semi_auto",
}: EnterpriseProviderProps) => {
  const today = dayjs().format("YYYY-MM-DD");
  const tomorrow = dayjs().add(1, "day").format("YYYY-MM-DD");
  const [selectedDate, setSelectedDate] = useState<string>(today);
  const [mode, setMode] = useState<EnterpriseMode>(initialMode);
  const [autoPaused, setAutoPaused] = useState<boolean>(false);

  const value = useMemo(
    () => ({
      selectedDate,
      setSelectedDate,
      minDate: today,
      maxDate: tomorrow,
      mode,
      setMode,
      autoPaused,
      setAutoPaused,
    }),
    [selectedDate, mode, autoPaused]
  );

  return (
    <EnterpriseContext.Provider value={value}>
      {children}
    </EnterpriseContext.Provider>
  );
};

export const useEnterpriseContext = () => {
  const ctx = useContext(EnterpriseContext);
  if (!ctx) {
    throw new Error(
      "useEnterpriseContext doit être utilisé à l'intérieur d'un EnterpriseProvider"
    );
  }
  return ctx;
};
