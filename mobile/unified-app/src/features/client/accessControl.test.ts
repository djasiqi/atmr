import { describe, expect, it } from "@jest/globals";
import { bookingBelongsToActiveClient } from "./accessControl";

describe("booking ownership access control", () => {
  it("allows booking when ids match", () => {
    const allowed = bookingBelongsToActiveClient(
      { id: 1, client: { id: 7 } },
      { id: 7 }
    );
    expect(allowed).toBe(true);
  });

  it("denies booking when ids mismatch", () => {
    const allowed = bookingBelongsToActiveClient(
      { id: 1, client: { id: 8 } },
      { id: 7 }
    );
    expect(allowed).toBe(false);
  });

  it("defaults to allow when data is incomplete", () => {
    const allowed = bookingBelongsToActiveClient({ id: 1 }, { id: 7 });
    expect(allowed).toBe(true);
  });
});
