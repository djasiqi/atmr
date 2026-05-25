import { QueryClientProvider } from "@tanstack/react-query";
import { createInstrumentedQueryClient } from "./observability/instrumentedQueryClient";
// eslint-disable-next-line @typescript-eslint/no-require-imports
const ReactRuntime: any = require("react");
type PropsWithChildren<P = object> = P & { children?: any };

export function QueryProvider({ children }: PropsWithChildren) {
  const [queryClient] = ReactRuntime.useState(() => createInstrumentedQueryClient());

  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
}
