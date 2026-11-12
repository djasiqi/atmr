/**
 * Aperçu non enregistré. L'institution désigne le payeur.
 * L'entreprise configure les moyens, sans les activer.
 */

export function settlementFormPreview(portal) {
  const dormant = {
    records: false,
    driverScreen: false,
    paymentMethodChoices: [],
  };
  if (portal === "guest") {
    return { ...dormant, visible: false, reason: "guest_flow_unchanged" };
  }
  if (portal === "private") {
    return {
      ...dormant,
      visible: true,
      payerChoices: [],
      onlinePaymentUnchanged: true,
      modalitiesChosenByCompany: true,
      readOnlyUntilCompanyKnown: true,
    };
  }
  if (portal === "institution") {
    return {
      ...dormant,
      visible: true,
      payerChoices: ["patient", "institution", "other_authorized"],
      onSiteRequiresPayerLock: true,
      cannotInferLockFromControlValidation: true,
      mutablePayerMessage: "administrative_billing",
      modalitiesChosenByCompany: true,
      inventsMethodsBeforeCompany: false,
    };
  }
  if (portal === "company") {
    return companySettlementSettingsPreview();
  }
  return { ...dormant, visible: false, reason: "unknown_portal" };
}

export function companySettlementSettingsPreview() {
  return {
    records: false,
    activates: false,
    driverScreen: false,
    visible: true,
    paymentMethodChoices: [],
    tapToPay: false,
    cash: false,
    directPatientInvoice: false,
    periodicAccounts: false,
    invoiceSignatureApproved: false,
    mayLockInstitutionPayer: false,
    patientNoticeAutoChecked: false,
  };
}
