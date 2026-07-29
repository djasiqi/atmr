import { tripsExpandMaxHeightFromContent } from "./tripsExpandHeight";

describe("tripsExpandMaxHeightFromContent", () => {
  it("n’impose pas de plafond 520", () => {
    expect(tripsExpandMaxHeightFromContent(800)).toBe(800);
    expect(tripsExpandMaxHeightFromContent(520)).toBe(520);
    expect(tripsExpandMaxHeightFromContent(0)).toBe(1);
  });
});
