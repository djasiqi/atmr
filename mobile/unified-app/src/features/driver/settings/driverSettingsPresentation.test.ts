import {
  buildDriverProfileViewModel,
  driverSettingsInitials,
  formatProfileDate,
  formatProfileField,
  resolveDriverGpsStatus,
  resolveDriverNotificationStatus,
  resolveDriverProfileIdentity,
  resolveDriverWeeklyHoursMessage,
  resolveNotificationsEnabled,
} from "./driverSettingsPresentation";

describe("driverSettingsPresentation", () => {
  it("formatProfileField renvoie le fallback si vide", () => {
    expect(formatProfileField("  ", "—")).toBe("—");
    expect(formatProfileField("06 12 34 56 78")).toBe("06 12 34 56 78");
  });

  it("formatProfileDate formate en fr-CH", () => {
    expect(formatProfileDate("1993-07-24")).toMatch(/24\.07\.1993/);
  });

  it("buildDriverProfileViewModel aligne les sections web", () => {
    const vm = buildDriverProfileViewModel(
      {
        first_name: "Emmenez",
        last_name: "Moi",
        birth_date: "1993-07-24",
        nationality: "Suisse",
        avs_number: "756.4040.4040.40",
        email: "emmenez-moi@emmenez-moi.ch",
        phone: "+41762034041",
        address: "Avenue Ernest-Pictet 9, 1203 Genève",
        vehicle_assigned: "Renault Kangoo - GE963822",
        license_valid_until: "2009-01-01",
        medical_valid_until: "2026-06-01",
        employment_start_date: "2026-06-01",
        emergency_contact_name: "Mirjete Osmani",
        emergency_contact_phone: "+41762002000",
        contract_type: "CDI",
        is_active: true,
        weekly_hours: 42,
      },
      null,
      "Lirie Transport"
    );
    expect(vm.displayName).toBe("Emmenez Moi");
    expect(vm.badges.map((b) => b.label)).toEqual(["Actif", "CDI"]);
    expect(vm.sections).toHaveLength(4);
    expect(vm.sections[0]?.rows.some((r) => r.label === "N° AVS")).toBe(true);
    expect(vm.sections[1]?.rows[0]?.value).toContain("Renault Kangoo");
  });

  it("resolveDriverProfileIdentity priorise prénom/nom du profil", () => {
    const identity = resolveDriverProfileIdentity(
      {
        first_name: "Karim",
        last_name: "Ali",
        email: "k@example.com",
        phone: "079 000 00 00",
        weekly_hours: 42,
      },
      { full_name: "Fallback", email: "user@example.com" }
    );
    expect(identity.displayName).toBe("Karim Ali");
    expect(identity.weeklyHours).toBe(42);
    expect(identity.phone).toBe("079 000 00 00");
  });

  it("resolveDriverWeeklyHoursMessage sans horaire", () => {
    expect(resolveDriverWeeklyHoursMessage(null)).toContain("Aucun horaire configuré");
  });

  it("resolveDriverGpsStatus actif si permissions complètes", () => {
    expect(
      resolveDriverGpsStatus({
        foregroundGranted: true,
        backgroundGranted: true,
        servicesEnabled: true,
      }).label
    ).toBe("Actif");
  });

  it("resolveDriverGpsStatus désactivé si GPS coupé", () => {
    expect(
      resolveDriverGpsStatus({
        foregroundGranted: true,
        backgroundGranted: true,
        servicesEnabled: false,
      }).key
    ).toBe("disabled");
  });

  it("resolveNotificationsEnabled", () => {
    expect(resolveNotificationsEnabled("ok", false)).toBe(true);
    expect(resolveNotificationsEnabled("permission_denied", false)).toBe(false);
  });

  it("resolveDriverNotificationStatus ok", () => {
    expect(resolveDriverNotificationStatus("ok", false).tone).toBe("ok");
  });

  it("driverSettingsInitials", () => {
    expect(driverSettingsInitials("Karim Ali")).toBe("KA");
  });
});
