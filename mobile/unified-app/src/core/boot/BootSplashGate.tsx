import type { ReactNode } from "react";
import { Platform } from "react-native";

type Props = { children: ReactNode };

const Gate =
  Platform.OS === "web"
    ? // eslint-disable-next-line @typescript-eslint/no-require-imports
      require("./BootSplashGate.web").BootSplashGate
    : // eslint-disable-next-line @typescript-eslint/no-require-imports
      require("./BootSplashGate.native").BootSplashGate;

export function BootSplashGate({ children }: Props) {
  return <Gate>{children}</Gate>;
}
