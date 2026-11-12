import {
  companySettlementSettingsPreview,
  settlementFormPreview,
} from "./settlementPolicyPreview";

describe("aperçu de politique de règlement", () => {
  it("n'enregistre aucun portail et ne montre pas l'écran chauffeur", () => {
    for (const portal of ["private", "institution", "company", "guest"]) {
      const preview = settlementFormPreview(portal);
      expect(preview.records).toBe(false);
      expect(preview.driverScreen).toBe(false);
      expect(preview.paymentMethodChoices).toEqual([]);
    }
  });

  it("laisse à l'institution le payeur, pas les moyens de l'entreprise", () => {
    const institution = settlementFormPreview("institution");
    expect(institution.payerChoices).toEqual([
      "patient",
      "institution",
      "other_authorized",
    ]);
    expect(institution.modalitiesChosenByCompany).toBe(true);
    expect(institution.inventsMethodsBeforeCompany).toBe(false);
    expect(institution.cannotInferLockFromControlValidation).toBe(true);
    expect(institution.onSiteRequiresPayerLock).toBe(true);
  });

  it("garde les réglages entreprise éteints et la confirmation non cochée", () => {
    const company = companySettlementSettingsPreview();
    expect(company.activates).toBe(false);
    expect(company.tapToPay).toBe(false);
    expect(company.cash).toBe(false);
    expect(company.directPatientInvoice).toBe(false);
    expect(company.periodicAccounts).toBe(false);
    expect(company.invoiceSignatureApproved).toBe(false);
    expect(company.patientNoticeAutoChecked).toBe(false);
    expect(company.mayLockInstitutionPayer).toBe(false);
    expect(settlementFormPreview("private").onlinePaymentUnchanged).toBe(true);
    expect(settlementFormPreview("guest").visible).toBe(false);
  });
});
