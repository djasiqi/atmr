import dayjs from "dayjs";
import { computeUrgentDatetime, isPickupSentinel } from "../urgentTime";

describe("isPickupSentinel", () => {
  it("null/undefined/empty → true", () => {
    expect(isPickupSentinel(null)).toBe(true);
    expect(isPickupSentinel(undefined)).toBe(true);
    expect(isPickupSentinel("")).toBe(true);
  });
  it("00:00:00 → true", () => {
    expect(isPickupSentinel("2026-01-28T00:00:00")).toBe(true);
    expect(isPickupSentinel("2026-01-28T00:00:00.000Z")).toBe(true);
  });
  it("09:30 / 23:59 → false", () => {
    expect(isPickupSentinel("2026-01-28T09:30:00")).toBe(false);
    expect(isPickupSentinel("2026-01-28T23:59:00")).toBe(false);
  });
});

describe("computeUrgentDatetime", () => {
  it("now fixe + 15 min → ISO YYYY-MM-DDTHH:mm:ss", () => {
    const now = dayjs("2026-01-28T14:00:00");
    const got = computeUrgentDatetime(now, 15);
    expect(got).toBe("2026-01-28T14:15:00");
  });

  it("défaut 15 min si minutes non fourni", () => {
    const now = dayjs("2026-01-28T10:00:00");
    expect(computeUrgentDatetime(now)).toBe("2026-01-28T10:15:00");
  });

  it("format ISO sans Z (local)", () => {
    const now = dayjs("2026-01-28T09:45:00");
    const got = computeUrgentDatetime(now, 20);
    expect(got).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$/);
    expect(got).toBe("2026-01-28T10:05:00");
  });
});
