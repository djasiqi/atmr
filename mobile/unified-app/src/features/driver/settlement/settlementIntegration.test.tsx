import { createElement, useEffect, useState } from "react";
import TestRenderer, { act } from "react-test-renderer";
import { readFileSync, readdirSync } from "fs";
import { join } from "path";
import { Pressable, StyleSheet, Text, View } from "react-native";
import { SettlementPreviewBody } from "./SettlementPreviewBody";
import { presentServerDecision } from "./driverSettlementContract";
import { MissionSettlementEntry } from "./MissionSettlementEntry";
import { missionPanelState } from "./missionPanelState";
import {
  settlementClosed,
  settlementLayout,
  settlementMomentOpen,
  settlementNeedsHalt,
  settlementRecipeAccessAllowed,
  statusAfterSimulatedSettlement,
} from "./settlementAccess";

const openRecipe = {
  dev: true,
  appVariant: "dev",
  recipeDatabase: "atmr_test",
  writableRecipe: true,
  serverPanelEnabled: true,
};

function texts(tree: TestRenderer.ReactTestRenderer): string {
  return tree.root
    .findAllByType(Text)
    .map((node) => (typeof node.props.children === "string" ? node.props.children : ""))
    .join(" ");
}

function press(tree: TestRenderer.ReactTestRenderer, label: string) {
  const node = tree.root.findAll(
    (item) => item.props?.accessibilityLabel === label && typeof item.props.onPress === "function"
  )[0];
  act(() => {
    node.props.onPress();
  });
}

function DestinationHarness({ payload }: { payload: unknown }) {
  const [status, setStatus] = useState("IN_PROGRESS");
  const [halt, setHalt] = useState(false);
  const [opened, setOpened] = useState(false);
  useEffect(() => {
    setHalt(false);
    setOpened(false);
  }, [status]);
  const state = missionPanelState({
    allowed: true,
    operationalStatus: status,
    haltConfirmed: halt,
    opened,
    offline: false,
    loading: false,
    failed: false,
    payload,
  });
  return createElement(
    View,
    null,
    createElement(Text, null, status),
    createElement(MissionSettlementEntry, {
      state,
      onConfirmHalt: () => setHalt(true),
      onOpen: () => setOpened(true),
      onClose: () => setOpened(false),
    }),
    createElement(
      Pressable,
      {
        accessibilityLabel: "Terminer",
        onPress: () => setStatus("COMPLETED"),
      },
      createElement(Text, null, "Terminer")
    )
  );
}

describe("intégration du panneau chauffeur", () => {
  it("ferme la recette hors développement, hors base de test et en production", () => {
    expect(settlementRecipeAccessAllowed(openRecipe)).toBe(true);
    expect(settlementRecipeAccessAllowed({ ...openRecipe, dev: false })).toBe(false);
    expect(settlementRecipeAccessAllowed({ ...openRecipe, serverPanelEnabled: false })).toBe(false);
    expect(settlementRecipeAccessAllowed({ ...openRecipe, appVariant: "prod" })).toBe(false);
    expect(settlementRecipeAccessAllowed({ ...openRecipe, recipeDatabase: "atmr" })).toBe(false);
    expect(settlementRecipeAccessAllowed({ ...openRecipe, writableRecipe: false })).toBe(false);
  });

  it("retire les moyens d'une décision réelle même si elle prétend autoriser un paiement", () => {
    const shown = presentServerDecision(
      {
        kind: "collection_authorized",
        reason: "on_site_authorized",
        modes: ["tap_to_pay", "cash", "invoice"],
        amount_chf: "45.00",
        simulation: false,
        source: "backend",
        payment_executable: true,
      },
      { preview: false }
    );
    expect(shown?.modes).toEqual([]);
    expect(shown?.payment_executable).toBe(false);
    expect(shown?.kind).toBe("administrative_review");
    const hidden = missionPanelState({
      allowed: false,
      operationalStatus: "IN_PROGRESS",
      haltConfirmed: true,
      opened: true,
      offline: false,
      loading: false,
      failed: false,
      payload: { kind: "collection_authorized", modes: ["cash"], source: "backend" },
    });
    expect(hidden.visible).toBe(false);
    expect(hidden.showSettlementAction).toBe(false);
  });

  it("n'affiche les simulations que lorsque la prévisualisation est autorisée", () => {
    const contract = presentServerDecision(
      {
        kind: "collection_authorized",
        modes: ["tap_to_pay"],
        amount_chf: "45.00",
        simulation: true,
        source: "simulation",
        preview_screen: "tap-interrupted",
      },
      { preview: true }
    );
    let tree: TestRenderer.ReactTestRenderer;
    act(() => {
      tree = TestRenderer.create(
        createElement(SettlementPreviewBody, {
          allowed: false,
          scenarios: [{ id: "tap-to-pay", label: "Patient éligible à Tap to Pay", contract }],
          selected: contract,
          loading: false,
          offline: false,
          error: null,
          onSelect: () => undefined,
          onBack: () => undefined,
        })
      );
    });
    expect(texts(tree!)).toContain("Rubrique indisponible");
    expect(texts(tree!)).not.toContain("Tap to Pay");
    act(() => {
      tree = TestRenderer.create(
        createElement(SettlementPreviewBody, {
          allowed: true,
          scenarios: [{ id: "tap-to-pay", label: "Patient éligible à Tap to Pay", contract: null }],
          selected: contract,
          loading: false,
          offline: false,
          error: null,
          onSelect: () => undefined,
          onBack: () => undefined,
        })
      );
    });
    expect(texts(tree!)).toContain("SIMULATION — AUCUN PAIEMENT RÉEL");
    expect(texts(tree!)).toContain("interrompu");
  });

  it("laisse la clôture et l'attestation en dehors du règlement", () => {
    const root = join(__dirname, "..", "..", "..", "..");
    const mission = readFileSync(join(root, "app/(app)/(driver)/missions/[missionId].tsx"), "utf8");
    const preview = readFileSync(
      join(root, "app/(app)/(driver)/missions/settlement-preview.tsx"),
      "utf8"
    );
    const confirm = readFileSync(
      join(root, "src/features/driver/components/ConfirmCompletionModal.tsx"),
      "utf8"
    );
    const attestation = readFileSync(
      join(root, "app/(app)/(driver)/missions/[missionId]/attestation.tsx"),
      "utf8"
    );
    expect(mission).not.toContain("MissionSettlementSlot");
    expect(mission).not.toContain("StatusSwitch");
    expect(mission).not.toContain("handleMissionTransition");
    expect(mission).not.toContain("submitDriverAttestation");
    expect(mission).not.toContain("/attestation");
    expect(mission).not.toContain("Je suis à l'arrêt");
    expect(mission).not.toContain("DriverSettlementPanel");
    expect(mission).not.toContain("ARRIVED_DESTINATION");
    expect(mission).not.toContain("DashboardHandoverSettlement");
    expect(mission).toContain('permission="mission:read"');
    const dashboard = readFileSync(join(root, "app/(app)/(driver)/index.tsx"), "utf8");
    const active = readFileSync(
      join(root, "src/features/driver/components/DashboardActiveMission.tsx"),
      "utf8"
    );
    const cashHost = readFileSync(
      join(root, "src/features/driver/settlement/DriverCashPrepHost.tsx"),
      "utf8"
    );
    const patientHost = readFileSync(
      join(root, "src/features/driver/settlement/DriverPatientPrepHost.tsx"),
      "utf8"
    );
    expect(dashboard).toContain("onMissionTransitionFromDashboard");
    expect(dashboard).toContain("ConfirmCompletionModal");
    expect(active).toContain("useDashboardHandover");
    expect(active).toContain("TERMINER");
    expect(active).not.toContain("Je suis à l'arrêt à destination");
    const stepperAt = active.indexOf("<MissionProgressStepper");
    const panelAt = active.indexOf("{handover.panel}");
    const finishAt = active.indexOf("transitionLabel(forwardTransition");
    expect(stepperAt).toBeGreaterThan(0);
    expect(panelAt).toBeGreaterThan(stepperAt);
    expect(panelAt).toBeLessThan(finishAt);
    expect(cashHost).toContain("if (status === 404) return null");
    expect(cashHost).not.toContain("const created = await apiClient.post");
    expect(patientHost).toContain("if (status === 404) return null");
    expect(patientHost).not.toContain("const created = await apiClient.post");
    expect(confirm).not.toContain("Settlement");
    expect(attestation).not.toContain("Settlement");
    expect(preview).toContain("useSettlementRecipeAccess");
    expect(preview).toContain("allowed={false}");
  });

  it("n'ouvre pas le règlement pendant la conduite, puis le propose à l'arrêt", () => {
    const payload = {
      kind: "collection_authorized",
      modes: ["tap_to_pay", "cash", "invoice"],
      amount_chf: "45.00",
      simulation: true,
      source: "simulation",
    };
    const base = {
      allowed: true,
      operationalStatus: "IN_PROGRESS",
      haltConfirmed: false,
      opened: false,
      offline: false,
      loading: false,
      failed: false,
      payload,
    };
    const driving = missionPanelState(base);
    const stopped = missionPanelState({ ...base, haltConfirmed: true });
    const aboard = missionPanelState({ ...base, haltConfirmed: true, opened: true });
    const doneClosed = missionPanelState({
      ...base,
      operationalStatus: "COMPLETED",
    });
    const done = missionPanelState({
      ...base,
      operationalStatus: "COMPLETED",
      opened: true,
    });
    const enRoute = missionPanelState({
      ...base,
      operationalStatus: "EN_ROUTE",
      haltConfirmed: true,
      opened: true,
    });
    const dropped = missionPanelState({
      ...base,
      haltConfirmed: true,
      opened: true,
      offline: true,
      payload: null,
    });
    expect(settlementNeedsHalt("IN_PROGRESS")).toBe(true);
    expect(settlementNeedsHalt("COMPLETED")).toBe(false);
    expect(settlementMomentOpen("ARRIVED")).toBe(false);
    expect(driving.visible).toBe(false);
    expect(driving.showHaltPrompt).toBe(true);
    expect(driving.showSettlementAction).toBe(false);
    expect(stopped.visible).toBe(false);
    expect(stopped.showSettlementAction).toBe(true);
    expect(aboard.visible).toBe(true);
    expect(aboard.contract?.modes).toEqual(["tap_to_pay", "cash", "invoice"]);
    expect(aboard.consultation).toBe(false);
    expect(doneClosed.visible).toBe(false);
    expect(doneClosed.showSettlementAction).toBe(true);
    expect(done.visible).toBe(true);
    expect(done.consultation).toBe(true);
    expect(settlementClosed("COMPLETED")).toBe(true);
    expect(enRoute.visible).toBe(false);
    expect(enRoute.showHaltPrompt).toBe(false);
    expect(dropped.visible).toBe(false);
    expect(dropped.contract).toBeNull();
    expect(statusAfterSimulatedSettlement("IN_PROGRESS")).toBe("IN_PROGRESS");
    expect(statusAfterSimulatedSettlement("COMPLETED")).toBe("COMPLETED");
  });

  it("n'affiche aucun moyen sur une décision réelle", () => {
    const cases: { reason: string; kind: string; amount: string | null }[] = [
      { kind: "collection_forbidden", reason: "institution_payer", amount: "45.00" },
      { kind: "administrative_review", reason: "payer_still_mutable", amount: "45.00" },
      { kind: "administrative_review", reason: "established_account", amount: "45.00" },
      { kind: "collection_forbidden", reason: "guest_flow_unchanged", amount: "45.00" },
      { kind: "already_paid", reason: "payment_confirmed", amount: "90.00" },
      { kind: "no_collection", reason: "settlement_policy_missing", amount: null },
      { kind: "administrative_review", reason: "collection_disabled", amount: "45.00" },
      { kind: "administrative_review", reason: "company_not_owner", amount: null },
      { kind: "administrative_review", reason: "driver_unassigned", amount: null },
    ];
    for (const item of cases) {
      const state = missionPanelState({
        allowed: true,
        operationalStatus: "IN_PROGRESS",
        haltConfirmed: true,
        opened: true,
        offline: false,
        loading: false,
        failed: false,
        payload: {
          kind: item.kind,
          reason: item.reason,
          modes: ["tap_to_pay", "cash", "invoice"],
          amount_chf: item.amount,
          simulation: false,
          source: "backend",
          payment_executable: true,
        },
      });
      expect(state.visible).toBe(false);
      expect(state.contract?.modes).toEqual([]);
      expect(state.notice).not.toContain("Tap to Pay");
    }
    const durable = missionPanelState({
      allowed: true,
      operationalStatus: "COMPLETED",
      haltConfirmed: true,
      opened: true,
      offline: false,
      loading: false,
      failed: false,
      payload: {
        kind: "collection_authorized",
        modes: ["cash"],
        amount_chf: "45.00",
        simulation: true,
        source: "simulation",
        authorization_durable: true,
      },
    });
    expect(durable.visible).toBe(false);
    expect(durable.notice).toContain("vérifier");
  });

  it("revient à la mission sans changer son statut", () => {
    const payload = {
      kind: "collection_authorized",
      modes: ["tap_to_pay", "cash", "invoice"],
      amount_chf: "45.00",
      simulation: true,
      source: "simulation",
    };
    let tree: TestRenderer.ReactTestRenderer;
    act(() => {
      tree = TestRenderer.create(createElement(DestinationHarness, { payload }));
    });
    const body = () => texts(tree);
    expect(body()).toContain("IN_PROGRESS");
    expect(body()).toContain("Je suis à l'arrêt à destination");
    expect(body()).not.toContain("Tap to Pay");
    expect(body()).not.toContain("Règlement du transport");
    press(tree!, "Je suis à l'arrêt à destination");
    expect(body()).toContain("Règlement du transport");
    expect(body()).not.toContain("Tap to Pay");
    press(tree!, "Règlement du transport");
    expect(body()).toContain("SIMULATION — AUCUN PAIEMENT RÉEL");
    expect(body()).toContain("Tap to Pay");
    expect(body()).toContain("IN_PROGRESS");
    press(tree!, "Retour à la mission");
    expect(body()).not.toContain("Tap to Pay");
    expect(body()).toContain("IN_PROGRESS");
    press(tree!, "Terminer");
    expect(body()).toContain("COMPLETED");
    expect(body()).not.toContain("Tap to Pay");
    press(tree!, "Règlement du transport");
    expect(body()).toContain("COMPLETED");
    expect(body()).toContain("Consultation après la mission");
    expect(body()).toContain("Tap to Pay");
    expect(body()).not.toContain("Je suis à l'arrêt à destination");
  });

  it("garde les boutons utilisables à 360, 390 et 412 px", () => {
    const root = join(__dirname, "..", "..", "..", "..");
    const slot = readFileSync(
      join(root, "src/features/driver/settlement/MissionSettlementSlot.tsx"),
      "utf8"
    );
    expect(slot).toContain("/driver/me/bookings/${missionId}/settlement-read");
    expect(slot).not.toContain("ARRIVED_DESTINATION");
    const directory = join(root, "src/features/driver/settlement");
    for (const name of readdirSync(directory)) {
      if (!name.endsWith(".ts") && !name.endsWith(".tsx")) continue;
      if (name.includes(".test.")) continue;
      const body = readFileSync(join(directory, name), "utf8");
      expect(body).not.toContain("ARRIVED_DESTINATION");
      expect(body).not.toContain("expo-location");
    }
    let tree: TestRenderer.ReactTestRenderer;
    act(() => {
      tree = TestRenderer.create(
        createElement(MissionSettlementEntry, {
          state: missionPanelState({
            allowed: true,
            operationalStatus: "IN_PROGRESS",
            haltConfirmed: false,
            opened: false,
            offline: false,
            loading: false,
            failed: false,
            payload: null,
          }),
          onConfirmHalt: () => undefined,
          onOpen: () => undefined,
          onClose: () => undefined,
        })
      );
    });
    const button = tree!.root.findByProps({ accessibilityLabel: "Je suis à l'arrêt à destination" });
    const styled = button.findAll((node) => typeof node.props?.style === "function");
    expect(styled.length).toBeGreaterThan(0);
    const flat = StyleSheet.flatten(styled[0].props.style({ pressed: false }));
    expect(flat.minHeight).toBeGreaterThanOrEqual(48);
  });

  it("reste lisible à 360, 390 et 412 px", () => {
    for (const width of [360, 390, 412]) {
      const layout = settlementLayout(width);
      expect(layout.overflows).toBe(false);
      expect(layout.contentWidth).toBeGreaterThan(300);
      expect(layout.buttonMinHeight).toBeGreaterThanOrEqual(48);
      expect(layout.amountFontSize).toBeGreaterThanOrEqual(24);
    }
  });
});
