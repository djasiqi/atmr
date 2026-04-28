import { QueryClient, QueryClientProvider as TanstackProvider } from '@tanstack/react-query';
import { useState } from 'react';

function createQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 30_000,
        retry: 1,
        refetchOnWindowFocus: false,
      },
    },
  });
}

export function AppQueryClientProvider({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(() => createQueryClient());
  return <TanstackProvider client={queryClient}>{children}</TanstackProvider>;
}
