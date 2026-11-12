import { describe, expect, it, jest } from "@jest/globals";
import { createElement } from "react";
import TestRenderer, { act } from "react-test-renderer";
import { Text } from "react-native";
import { LirieStatusBadge } from "./LirieStatusBadge";
import { LirieTransportCard } from "./LirieTransportCard";
import { LirieButton } from "./LirieButton";

jest.mock("../navigation/useReduceMotion", () => ({
  useReduceMotion: () => false,
}));

function textValues(tree: TestRenderer.ReactTestRenderer): string[] {
  return tree.root.findAllByType(Text).map((node) => String(node.props.children));
}

describe("fondations partagées", () => {
  it("laisse les libellés entiers, y compris à grande police", () => {
    let tree: TestRenderer.ReactTestRenderer;
    act(() => {
      tree = TestRenderer.create(
        createElement(LirieButton, { label: "Confirmer la course maintenant" })
      );
    });
    const label = tree!.root.findByType(Text);
    expect(label.props.numberOfLines).toBeUndefined();
    expect(label.props.maxFontSizeMultiplier).toBe(2);
  });

  it("affiche Assignée sans troncature", () => {
    let tree: TestRenderer.ReactTestRenderer;
    act(() => {
      tree = TestRenderer.create(createElement(LirieStatusBadge, { status: "assigned" }));
    });
    const label = tree!.root.findByType(Text);
    expect(label.props.children).toBe("Assignée");
    expect(label.props.numberOfLines).toBeUndefined();
  });

  it("place l’heure avant le statut en lecture opérationnelle", () => {
    let tree: TestRenderer.ReactTestRenderer;
    act(() => {
      tree = TestRenderer.create(
        createElement(LirieTransportCard, {
          reading: "operational",
          time: "18:00",
          status: "assigned",
          title: "Idris Fictif",
          address: "Rue du Théâtre 4, 1003 Lausanne",
          actionLabel: "Ouvrir la course",
        })
      );
    });
    const values = textValues(tree!);
    expect(values.indexOf("18:00")).toBeLessThan(values.indexOf("Assignée"));
    expect(values.indexOf("Assignée")).toBeLessThan(values.indexOf("Rue du Théâtre 4, 1003 Lausanne"));
    expect(values).toContain("Idris Fictif");
  });
});
