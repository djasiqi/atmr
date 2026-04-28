import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { QUERY_STALE_TIME_MS } from "./queryStaleTimes";
// eslint-disable-next-line @typescript-eslint/no-require-imports
const ReactRuntime: any = require("react");
type PropsWithChildren<P = object> = P & { children?: any };

export function QueryProvider({ children }: PropsWithChildren) {
  const [queryClient] = ReactRuntime.useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: QUERY_STALE_TIME_MS.default,
            retry: 1,
          },
          mutations: {
            retry: 1,
          },
        },
      })
  );

  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
}
