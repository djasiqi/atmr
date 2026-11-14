import {
  buildSignPayload,
  canCollectSignatureDuringTransport,
  deferredBillingAuthorizedFromGate,
} from "./conventionRealContract";

describe("conventionRealContract", () => {
  it("autorise la collecte seulement en attente de signature", () => {
    expect(canCollectSignatureDuringTransport("awaiting_signature")).toBe(true);
    expect(canCollectSignatureDuringTransport("applicable")).toBe(false);
    expect(canCollectSignatureDuringTransport("draft")).toBe(false);
  });

  it("refuse le différé réel sans effet juridique", () => {
    expect(
      deferredBillingAuthorizedFromGate({
        authorizes_deferred_billing: false,
        legally_effective: false,
      })
    ).toBe(false);
    expect(
      deferredBillingAuthorizedFromGate({
        authorizes_deferred_billing: true,
        legally_effective: true,
      })
    ).toBe(true);
  });

  it("construit un payload de signature valide", () => {
    const payload = buildSignPayload({
      conventionId: 12,
      signerKind: "client",
      signerName: "Ada Client",
      documentSha256: "a".repeat(64),
      trace: [{ x: 1, y: 2 }],
      idempotencyKey: "sign-key-001",
    });
    expect(payload).toMatchObject({
      conventionId: 12,
      acknowledged: true,
      signerKind: "client",
    });
  });
});
