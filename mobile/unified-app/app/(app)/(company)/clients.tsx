import { useEffect, useMemo, useState } from "react";
import { Pressable, ScrollView, Text, View } from "react-native";
import { useLocalSearchParams } from "expo-router";
import { PermissionGuard } from "../../../src/core/guards";
import { isFeatureEnabled } from "../../../src/core/featureFlags/registry";
import {
  useCompanyClientReadonlyDetailQuery,
  useCompanyClientsReadonlyQuery,
  useCompanyInvoicesReadonlyQuery,
} from "../../../src/features/company/hooks";
import { Button, InputField, Loader } from "../../../src/components/ui";

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

  useEffect(() => {
    setActiveSection(requestedSection);
  }, [requestedSection]);

  if (!clientsReadonlyEnabled && !invoicesReadonlyEnabled) {
    return (
      <PermissionGuard permission="company:dashboard:read">
        <View style={{ flex: 1, padding: 24, justifyContent: "center", gap: 8 }}>
          <Text style={{ fontWeight: "700", fontSize: 18 }}>Clients & Facturation</Text>
          <Text style={{ color: "#666" }}>
            Les surfaces en lecture seule sont désactivées via feature flags.
          </Text>
        </View>
      </PermissionGuard>
    );
  }

  return (
    <PermissionGuard permission="company:dashboard:read">
      <ScrollView contentContainerStyle={{ padding: 24, gap: 12 }}>
        <Text style={{ fontWeight: "700", fontSize: 22 }}>Clients & Facturation</Text>
        <Text style={{ color: "#667085" }}>
          Vue unifiée pour consulter les clients et les factures en lecture seule.
        </Text>

        <View
          style={{
            flexDirection: "row",
            gap: 8,
            backgroundColor: "#F2F4F7",
            borderRadius: 12,
            padding: 4,
          }}
        >
          <Pressable
            onPress={() => setActiveSection("clients")}
            style={{
              flex: 1,
              borderRadius: 10,
              paddingVertical: 10,
              alignItems: "center",
              backgroundColor: activeSection === "clients" ? "#FFFFFF" : "transparent",
            }}
          >
            <Text style={{ fontWeight: "700", color: activeSection === "clients" ? "#0A7EA4" : "#667085" }}>
              Clients
            </Text>
          </Pressable>
          <Pressable
            onPress={() => setActiveSection("invoices")}
            style={{
              flex: 1,
              borderRadius: 10,
              paddingVertical: 10,
              alignItems: "center",
              backgroundColor: activeSection === "invoices" ? "#FFFFFF" : "transparent",
            }}
          >
            <Text style={{ fontWeight: "700", color: activeSection === "invoices" ? "#0A7EA4" : "#667085" }}>
              Facturation
            </Text>
          </Pressable>
        </View>

        {activeSection === "clients" ? (
          <>
            {!clientsReadonlyEnabled ? (
              <View style={{ borderWidth: 1, borderColor: "#E4E7EC", borderRadius: 10, padding: 14 }}>
                <Text style={{ fontWeight: "700", fontSize: 16 }}>Clients (lecture seule)</Text>
                <Text style={{ color: "#667085", marginTop: 6 }}>
                  Cette section est désactivée par feature flag.
                </Text>
              </View>
            ) : (
              <>
                <InputField
                  value={clientSearch}
                  onChangeText={setClientSearch}
                  placeholder="Rechercher un client"
                />
                {clientsQuery.isLoading ? <Loader /> : null}
                {rows.map((row, index) => {
                  const id = extractClientId(row);
                  if (id == null) return null;
                  const selected = selectedClientId === id;
                  return (
                    <Pressable
                      key={`${id}-${index}`}
                      onPress={() => setSelectedClientId(id)}
                      style={{
                        borderWidth: 1,
                        borderColor: selected ? "#0A7EA4" : "#E4E7EC",
                        borderRadius: 12,
                        padding: 12,
                        backgroundColor: selected ? "#F5FBFF" : "#FFFFFF",
                        gap: 4,
                      }}
                    >
                      <Text style={{ fontWeight: "700", color: selected ? "#0A7EA4" : "#101828" }}>
                        {extractClientLabel(row)}
                      </Text>
                      <Text style={{ color: "#667085" }}>Client ID: {id}</Text>
                    </Pressable>
                  );
                })}
                {!clientsQuery.isLoading && rows.length === 0 ? (
                  <Text style={{ color: "#667085" }}>Aucun client trouvé.</Text>
                ) : null}
                {clientsQuery.error ? (
                  <Text style={{ color: "#B42318" }}>
                    {clientsQuery.error instanceof Error
                      ? clientsQuery.error.message
                      : "Chargement clients impossible."}
                  </Text>
                ) : null}
                {selectedClientId != null ? (
                  <View
                    style={{
                      borderWidth: 1,
                      borderColor: "#E4E7EC",
                      borderRadius: 12,
                      padding: 12,
                      gap: 6,
                    }}
                  >
                    <Text style={{ fontWeight: "700" }}>Fiche client #{selectedClientId}</Text>
                    {detailQuery.isLoading ? <Loader /> : null}
                    {detailQuery.data ? (
                      <>
                        <Text>
                          Nom: {String((detailQuery.data.full_name ?? detailQuery.data.name ?? "n/a") as string)}
                        </Text>
                        <Text>Email: {String((detailQuery.data.email ?? "n/a") as string)}</Text>
                        <Text>Téléphone: {String((detailQuery.data.phone ?? "n/a") as string)}</Text>
                        <Text>Notes: {String((detailQuery.data.notes ?? "n/a") as string)}</Text>
                      </>
                    ) : null}
                    {detailQuery.error ? (
                      <Text style={{ color: "#B42318" }}>
                        {detailQuery.error instanceof Error
                          ? detailQuery.error.message
                          : "Chargement fiche client impossible."}
                      </Text>
                    ) : null}
                    <Button label="Fermer la fiche" onPress={() => setSelectedClientId(null)} />
                  </View>
                ) : null}
              </>
            )}
          </>
        ) : (
          <>
            {!invoicesReadonlyEnabled ? (
              <View style={{ borderWidth: 1, borderColor: "#E4E7EC", borderRadius: 10, padding: 14 }}>
                <Text style={{ fontWeight: "700", fontSize: 16 }}>Factures (lecture seule)</Text>
                <Text style={{ color: "#667085", marginTop: 6 }}>
                  Cette section est désactivée par feature flag.
                </Text>
              </View>
            ) : (
              <>
                <InputField
                  value={invoiceSearch}
                  onChangeText={setInvoiceSearch}
                  placeholder="Rechercher une facture"
                />
                <View style={{ flexDirection: "row", gap: 8 }}>
                  <View
                    style={{
                      flex: 1,
                      borderWidth: 1,
                      borderColor: "#E4E7EC",
                      borderRadius: 10,
                      padding: 10,
                    }}
                  >
                    <Text style={{ color: "#667085" }}>Total</Text>
                    <Text style={{ fontWeight: "700", fontSize: 18 }}>{invoiceRows.length}</Text>
                  </View>
                  <View
                    style={{
                      flex: 1,
                      borderWidth: 1,
                      borderColor: "#FECACA",
                      borderRadius: 10,
                      padding: 10,
                      backgroundColor: "#FFF7F7",
                    }}
                  >
                    <Text style={{ color: "#B42318" }}>Impayées</Text>
                    <Text style={{ fontWeight: "700", fontSize: 18, color: "#B42318" }}>{unpaidCount}</Text>
                  </View>
                </View>
                {invoicesQuery.isLoading ? <Loader /> : null}
                {invoiceRows.map((row, index) => {
                  const unpaid = resolveUnpaid(row);
                  return (
                    <View
                      key={`${extractInvoiceValue(row, "id", "invoice_id", "number")}-${index}`}
                      style={{
                        borderWidth: 1,
                        borderColor: unpaid ? "#FECACA" : "#E4E7EC",
                        borderRadius: 12,
                        padding: 12,
                        gap: 4,
                        backgroundColor: unpaid ? "#FFF7F7" : "#FFFFFF",
                      }}
                    >
                      <Text style={{ fontWeight: "700" }}>
                        Facture {extractInvoiceValue(row, "number", "invoice_number", "id", "invoice_id")}
                      </Text>
                      <Text>Statut: {extractInvoiceValue(row, "status", "payment_status")}</Text>
                      <Text>Montant: {extractInvoiceValue(row, "total_amount", "amount_total", "amount")}</Text>
                      <Text>Mission: {extractInvoiceValue(row, "booking_id", "mission_id", "reservation_id")}</Text>
                      {unpaid ? <Text style={{ color: "#B42318", fontWeight: "700" }}>Impayée</Text> : null}
                    </View>
                  );
                })}
                {!invoicesQuery.isLoading && invoiceRows.length === 0 ? (
                  <Text style={{ color: "#667085" }}>Aucune facture trouvée.</Text>
                ) : null}
                {invoicesQuery.error ? (
                  <Text style={{ color: "#B42318" }}>
                    {invoicesQuery.error instanceof Error
                      ? invoicesQuery.error.message
                      : "Chargement factures impossible."}
                  </Text>
                ) : null}
              </>
            )}
          </>
        )}
      </ScrollView>
    </PermissionGuard>
  );
}
