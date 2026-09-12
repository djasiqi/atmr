import {
  buildCompanyBillingSummary,
  buildCompanyProfileViewModel,
  COMPANY_DISPATCH_MODE_OPTIONS,
  formatDispatchModeFr,
  formatServiceAreaLabel,
  resolveCompanyLogoUrl,
  resolveCompanyRealtimeLabel,
  resolveUserDisplayName,
} from "./companySettingsPresentation";

describe("companySettingsPresentation", () => {
  it("formate le mode dispatch en français", () => {
    expect(formatDispatchModeFr("semi_auto")).toBe("Semi-automatique");
    expect(COMPANY_DISPATCH_MODE_OPTIONS.find((m) => m.id === "manual")?.selectable).toBe(true);
    expect(COMPANY_DISPATCH_MODE_OPTIONS.find((m) => m.id === "semi_auto")?.selectable).toBe(false);
  });

  it("formate la zone de service", () => {
    expect(
      formatServiceAreaLabel('{"v":1,"mode":"canton","tokens":["canton:GE"]}')
    ).toBe("Canton de Genève");
  });

  it("résume la facturation par défaut", () => {
    const summary = buildCompanyBillingSummary({
      default_billed_to_type: "Patient",
      default_billed_to_contact: "Clinique ABC",
    });
    expect(summary.label).toBe("Patient");
    expect(summary.detail).toBe("Clinique ABC");
  });

  it("construit le profil entreprise", () => {
    const vm = buildCompanyProfileViewModel(
      {
        name: "Transport Lirie",
        address: "Rue du Test 1",
        contact_email: "contact@lirie.ch",
        is_approved: true,
        dispatch_enabled: true,
        vehicles: [{ id: 1 }, { id: 2 }],
      },
      null,
      "https://api.lirie.ch/api/v1"
    );
    expect(vm.displayName).toBe("Transport Lirie");
    expect(vm.badges.some((b) => b.label === "Compte validé")).toBe(true);
    expect(vm.vehicleCount).toBe(2);
    expect(vm.sections.length).toBeGreaterThan(0);
  });

  it("résout le nom utilisateur", () => {
    expect(resolveUserDisplayName({ full_name: "Jean Dupont", email: "j@lirie.ch" })).toBe(
      "Jean Dupont"
    );
  });

  it("refuse les URL de logo dangereuses", () => {
    expect(resolveCompanyLogoUrl("javascript:alert(1)", "https://api.lirie.ch/api/v1")).toBeNull();
    expect(
      resolveCompanyLogoUrl("data:text/html,<script>1</script>", "https://api.lirie.ch/api/v1")
    ).toBeNull();
    expect(resolveCompanyLogoUrl("//evil.example/x.png", "https://api.lirie.ch/api/v1")).toBeNull();
  });

  it("accepte un logo https et un upload", () => {
    expect(resolveCompanyLogoUrl("https://cdn.example.com/logo.png", "https://api.lirie.ch/api/v1")).toBe(
      "https://cdn.example.com/logo.png"
    );
    expect(
      resolveCompanyLogoUrl("/uploads/company_logos/logo.png", "https://api.lirie.ch/api/v1")
    ).toBe("https://api.lirie.ch/uploads/company_logos/logo.png");
  });

  it("résout le statut temps réel", () => {
    expect(resolveCompanyRealtimeLabel("healthy")).toBe("Connecté");
    expect(resolveCompanyRealtimeLabel("failed")).toBe("Hors ligne");
  });
});
