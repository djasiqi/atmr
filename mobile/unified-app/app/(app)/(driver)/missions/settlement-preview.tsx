import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useRouter } from "expo-router";
import { apiClient } from "../../../../src/core/api/client";
import { Screen } from "../../../../src/design/responsive";
import { SettlementPreviewBody } from "../../../../src/features/driver/settlement/SettlementPreviewBody";
import { presentServerDecision } from "../../../../src/features/driver/settlement/driverSettlementContract";
import { useSettlementRecipeAccess } from "../../../../src/features/driver/settlement/MissionSettlementSlot";

type Catalog = { scenarios?: { id: string; label: string }[] };

export default function SettlementPreviewScreen() {
  const allowed = useSettlementRecipeAccess();
  if (!allowed) {
    return (
      <Screen>
        <SettlementPreviewBody
          allowed={false}
          scenarios={[]}
          selected={null}
          loading={false}
          offline={false}
          error={null}
          onSelect={() => undefined}
          onBack={() => undefined}
        />
      </Screen>
    );
  }
  return <SettlementPreviewConnected />;
}

function SettlementPreviewConnected() {
  const router = useRouter();
  const [scenarioId, setScenarioId] = useState<string | null>(null);
  const catalog = useQuery({
    queryKey: ["driver-settlement-simulation"],
    queryFn: async () => {
      const { data } = await apiClient.get<Catalog>("/driver/me/settlement-simulation");
      return data?.scenarios ?? [];
    },
    retry: false,
  });
  const scenario = useQuery({
    queryKey: ["driver-settlement-simulation", scenarioId],
    queryFn: async () => {
      const { data } = await apiClient.get(`/driver/me/settlement-simulation/${scenarioId}`);
      return data;
    },
    enabled: Boolean(scenarioId),
    retry: false,
  });
  return (
    <Screen scroll>
      <SettlementPreviewBody
        allowed
        scenarios={(catalog.data ?? []).map((item) => ({
          id: item.id,
          label: item.label,
          contract: null,
        }))}
        selected={presentServerDecision(scenario.data, { preview: true })}
        loading={scenario.isLoading && Boolean(scenarioId)}
        offline={false}
        error={catalog.isError || scenario.isError ? "lecture" : null}
        onSelect={setScenarioId}
        onBack={() => router.back()}
      />
    </Screen>
  );
}
