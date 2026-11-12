import { createElement } from "react";
import TestRenderer, { act } from "react-test-renderer";
import { Text } from "react-native";
import { SettlementChannelNotice } from "./SettlementChannelNotice";

function texts(tree: TestRenderer.ReactTestRenderer): string {
  return tree.root
    .findAllByType(Text)
    .map((node) => (typeof node.props.children === "string" ? node.props.children : ""))
    .join(" ");
}

test("la simulation reste distincte des modules réels verrouillés", () => {
  let tree: TestRenderer.ReactTestRenderer;
  act(() => {
    tree = TestRenderer.create(createElement(SettlementChannelNotice));
  });
  const value = texts(tree!);
  expect(value).toContain("Simulation de recette");
  expect(value).toContain("Module espèces réel verrouillé");
  expect(value).toContain("Module facture patient réel verrouillé");
});
