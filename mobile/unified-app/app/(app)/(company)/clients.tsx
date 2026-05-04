import { useEffect, useMemo, useState } from "react";
import { Pressable, View } from "react-native";
import { useLocalSearchParams } from "expo-router";
import { PermissionGuard } from "../../../src/core/guards";
import { isFeatureEnabled } from "../../../src/core/featureFlags/registry";
import {
  useCompanyClientReadonlyDetailQuery,
  useCompanyClientsReadonlyQuery,
  useCompanyInvoicesReadonlyQuery,
} from "../../../src/features/company/hooks";
import {
  AppButton,
  AppCard,
  AppEmptyState,
  AppInput,
  AppSpinner,
  AppText,
  brandPrimary,
  brandSurfaceSoft,
  brandTextMuted,
  Screen,
  useAppViewport,
  useResponsiveTokens,
} from "../../../src/design/responsive";

function extractClientId(row: Record<string, unknown>): number | null {
  const candidate = row.client_id ?? row.id;
  if (typeof candidate === "number" && Number.isFinite(candidate)) return candidate;
  if (typeof candidate === "string" && candidate.trim().length > 0) {
    const parsed = Number.parseInt(candidate, 10);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function extractClientLabel(row: Record<string, unknown>): string {
  const candidate =
    row.full_name ?? row.name ?? row.display_name ?? row.label ?? row.email ?? row.phone ?? "Client";
  return String(candidate);
}

function extractInvoiceValue(row: Record<string, unknown>, ...keys: string[]): string {
  for (const key of keys) {
    const value = row[key];
    if (value == null) continue;
    if (typeof value === "string" && value.trim().length > 0) return value;
    if (typeof value === "number" && Number.isFinite(value)) return String(value);
  }
  return "n/a";
}

function resolveUnpaid(row: Record<string, unknown>): boolean {
  const statusRaw = String(row.status ?? row.payment_status ?? "").toLowerCase();
  return statusRaw === "unpaid" || statusRaw === "pending" || statusRaw === "overdue";
}

type CompanyReadonlySection = "clients" | "invoices";

export default function CompanyClientsScreen() {
  const params = useLocalSearchParams<{ section?: string }>();
  const requestedSection = params.section === "invoices" ? "invoices" : "clients";
  const clientsReadonlyEnabled = isFeatureEnabled("company_mobile_clients_readonly_enabled");
  const invoicesReadonlyEnabled = isFeatureEnabled("company_mobile_invoices_readonly_enabled");
  const [activeSection, setActiveSection] = useState<CompanyReadonlySection>(requestedSection);
  const [clientSearch, setClientSearch] = useState("");
  const [invoiceSearch, setInvoiceSearch] = useState("");
  const [selectedClientId, setSelectedClientId] = useState<number | null>(null);
  const clientsQuery = useCompanyClientsReadonlyQuery({ q: clientSearch, limit: 50 });
  const invoicesQuery = useCompanyInvoicesReadonlyQuery({ q: invoiceSearch, limit: 50 });
  const detailQuery = useCompanyClientReadonlyDetailQuery(selectedClientId);
  const rows = useMemo(() => clientsQuery.data ?? [], [clientsQuery.data]);
  const invoiceRows = useMemo(() => invoicesQuery.data ?? [], [invoicesQuery.data]);
  const unpaidCount = useMemo(
    () => invoiceRows.filter((row) => resolveUnpaid(row)).length,
    [invoiceRows]
  );
  const t = useResponsiveTokens();
  const { horizontalPadding } = useAppViewport();

  useEffect(() => {
    setActiveSection(requestedSection);
  }, [requestedSection]);

  if (!clientsReadonlyEnabled && !invoicesReadonlyEnabled) {
    return (
      <PermissionGuard permission="company:dashboard:read">
        <View style={{ flex: 1, padding: horizontalPadding, justifyContent: "center" }}>
          <AppEmptyState
            title="Clients & Facturation"
            description="Les surfaces en lecture seule sont désactivées via feature flags."
          />
        </View>
      </PermissionGuard>
    );
  }

  const contentPad = {
    paddingHorizontal: horizontalPadding,
    paddingVertical: t.spacingMd,
    gap: t.pageGap,
  };

  return (
    <PermissionGuard permission="company:dashboard:read">
      <Screen scroll backgroundColor={brandSurfaceSoft} withHorizontalPadding={false} contentContainerStyle={contentPad}>
        <AppText variant="screenTitle">Clients & Facturation</AppText>
        <AppText variant="bodyMuted">Vue unifiée pour consulter les clients et les factures en lecture seule.</AppText>

        <View
          style={{
            flexDirection: "row",
            gap: t.spacingSm,
            backgroundColor: "#F2F4F7",
            borderRadius: t.radiusMd,
            padding: t.spacingXs,
          }}
        >
          <Pressable
            onPress={() => setActiveSection("clients")}
            style={{
              flex: 1,
              borderRadius: t.radiusSm,
              paddingVertical: t.spacingSm,
              alignItems: "center",
              backgroundColor: activeSection === "clients" ? "#FFFFFF" : "transparent",
            }}
          >
            <AppText
              variant="body"
              style={{
                fontWeight: "700",
                color: activeSection === "clients" ? brandPrimary : brandTextMuted,
              }}
            >
              Clients
            </AppText>
          </Pressable>
          <Pressable
            onPress={() => setActiveSection("invoices")}
            style={{
              flex: 1,
              borderRadius: t.radiusSm,
              paddingVertical: t.spacingSm,
              alignItems: "center",
              backgroundColor: activeSection === "invoices" ? "#FFFFFF" : "transparent",
            }}
          >
            <AppText
              variant="body"
              style={{
                fontWeight: "700",
                color: activeSection === "invoices" ? brandPrimary : brandTextMuted,
              }}
            >
              Facturation
            </AppText>
          </Pressable>
        </View>

        {activeSection === "clients" ? (
          <>
            {!clientsReadonlyEnabled ? (
              <AppCard variant="surface">
                <AppText variant="sectionTitle">Clients (lecture seule)</AppText>
                <AppText variant="bodyMuted" style={{ marginTop: t.fieldGap }}>
                  Cette section est désactivée par feature flag.
                </AppText>
              </AppCard>
            ) : (
              <>
                <AppInput
                  value={clientSearch}
                  onChangeText={setClientSearch}
                  placeholder="Rechercher un client"
                />
                {clientsQuery.isLoading ? <AppSpinner size="small" /> : null}
                {rows.map((row, index) => {
                  const id = extractClientId(row);
                  if (id == null) return null;
                  const selected = selectedClientId === id;
                  return (
                    <AppCard
                      key={`${id}-${index}`}
                      variant="interactive"
                      onPress={() => setSelectedClientId(id)}
                      style={{
                        borderColor: selected ? brandPrimary : undefined,
                        backgroundColor: selected ? "#F5FBFF" : "#FFFFFF",
                      }}
                    >
                      <AppText
                        variant="body"
                        style={{ fontWeight: "700", color: selected ? brandPrimary : "#101828" }}
                      >
                        {extractClientLabel(row)}
                      </AppText>
                      <AppText variant="bodyMuted">Client ID: {id}</AppText>
                    </AppCard>
                  );
                })}
                {!clientsQuery.isLoading && rows.length === 0 ? (
                  <AppEmptyState title="Aucun client" description="Aucun client ne correspond à la recherche." />
                ) : null}
                {clientsQuery.error ? (
                  <AppText variant="error">
                    {clientsQuery.error instanceof Error
                      ? clientsQuery.error.message
                      : "Chargement clients impossible."}
                  </AppText>
                ) : null}
                {selectedClientId != null ? (
                  <AppCard variant="surface">
                    <AppText variant="sectionTitle">Fiche client #{selectedClientId}</AppText>
                    {detailQuery.isLoading ? <AppSpinner size="small" /> : null}
                    {detailQuery.data ? (
                      <>
                        <AppText variant="body">
                          Nom: {String((detailQuery.data.full_name ?? detailQuery.data.name ?? "n/a") as string)}
                        </AppText>
                        <AppText variant="body">Email: {String((detailQuery.data.email ?? "n/a") as string)}</AppText>
                        <AppText variant="body">Téléphone: {String((detailQuery.data.phone ?? "n/a") as string)}</AppText>
                        <AppText variant="body">Notes: {String((detailQuery.data.notes ?? "n/a") as string)}</AppText>
                      </>
                    ) : null}
                    {detailQuery.error ? (
                      <AppText variant="error">
                        {detailQuery.error instanceof Error
                          ? detailQuery.error.message
                          : "Chargement fiche client impossible."}
                      </AppText>
                    ) : null}
                    <AppButton title="Fermer la fiche" variant="secondary" onPress={() => setSelectedClientId(null)} />
                  </AppCard>
                ) : null}
              </>
            )}
          </>
        ) : (
          <>
            {!invoicesReadonlyEnabled ? (
              <AppCard variant="surface">
                <AppText variant="sectionTitle">Factures (lecture seule)</AppText>
                <AppText variant="bodyMuted" style={{ marginTop: t.fieldGap }}>
                  Cette section est désactivée par feature flag.
                </AppText>
              </AppCard>
            ) : (
              <>
                <AppInput
                  value={invoiceSearch}
                  onChangeText={setInvoiceSearch}
                  placeholder="Rechercher une facture"
                />
                <View style={{ flexDirection: "row", gap: t.spacingSm }}>
                  <AppCard variant="compact" style={{ flex: 1 }}>
                    <AppText variant="caption">Total</AppText>
                    <AppText variant="sectionTitle" style={{ marginTop: 4 }}>
                      {invoiceRows.length}
                    </AppText>
                  </AppCard>
                  <AppCard
                    variant="compact"
                    style={{
                      flex: 1,
                      borderColor: "#FECACA",
                      backgroundColor: "#FFF7F7",
                    }}
                  >
                    <AppText variant="caption" style={{ color: "#B42318" }}>
                      Impayées
                    </AppText>
                    <AppText variant="sectionTitle" style={{ marginTop: 4, color: "#B42318" }}>
                      {unpaidCount}
                    </AppText>
                  </AppCard>
                </View>
                {invoicesQuery.isLoading ? <AppSpinner size="small" /> : null}
                {invoiceRows.map((row, index) => {
                  const unpaid = resolveUnpaid(row);
                  return (
                    <AppCard
                      key={`${extractInvoiceValue(row, "id", "invoice_id", "number")}-${index}`}
                      variant="surface"
                      style={{
                        borderColor: unpaid ? "#FECACA" : undefined,
                        backgroundColor: unpaid ? "#FFF7F7" : "#FFFFFF",
                      }}
                    >
                      <AppText variant="body" style={{ fontWeight: "700" }}>
                        Facture {extractInvoiceValue(row, "number", "invoice_number", "id", "invoice_id")}
                      </AppText>
                      <AppText variant="body">Statut: {extractInvoiceValue(row, "status", "payment_status")}</AppText>
                      <AppText variant="body">Montant: {extractInvoiceValue(row, "total_amount", "amount_total", "amount")}</AppText>
                      <AppText variant="body">Mission: {extractInvoiceValue(row, "booking_id", "mission_id", "reservation_id")}</AppText>
                      {unpaid ? (
                        <AppText variant="error" style={{ fontWeight: "700" }}>
                          Impayée
                        </AppText>
                      ) : null}
                    </AppCard>
                  );
                })}
                {!invoicesQuery.isLoading && invoiceRows.length === 0 ? (
                  <AppEmptyState title="Aucune facture" description="Aucune facture ne correspond à la recherche." />
                ) : null}
                {invoicesQuery.error ? (
                  <AppText variant="error">
                    {invoicesQuery.error instanceof Error
                      ? invoicesQuery.error.message
                      : "Chargement factures impossible."}
                  </AppText>
                ) : null}
              </>
            )}
          </>
        )}
      </Screen>
    </PermissionGuard>
  );
}
