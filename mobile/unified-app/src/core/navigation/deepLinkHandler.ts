export type DriverDeepLinkResolution = {
  route: string;
  missionId: number | null;
};

export type CompanyDeepLinkResolution = {
  route: string;
  rideId: number | null;
  threadId?: string | null;
};

const QUICK_ACTIONS = ["accept", "reject", "start", "complete"] as const;
const SCHEMES = ["atmr://", "lirie://"] as const;

function removeAnySchemePrefix(input: string, prefixes: readonly string[]): string | null {
  const lowered = input.toLowerCase();
  for (const prefix of prefixes) {
    if (lowered.startsWith(prefix)) {
      return input.slice(prefix.length);
    }
  }
  return null;
}

function parseQuery(query: string): Map<string, string> {
  const pairs = query.split("&").map((chunk) => chunk.split("="));
  const map = new Map<string, string>();
  pairs.forEach(([key, value]) => {
    if (key) map.set(decodeURIComponent(key), decodeURIComponent(value ?? ""));
  });
  return map;
}

export function resolveDriverDeepLink(input: string | null | undefined): DriverDeepLinkResolution | null {
  if (!input) return null;
  const normalized = input.toLowerCase();
  const bookingTail = removeAnySchemePrefix(input, SCHEMES.map((scheme) => `${scheme}booking/`));
  if (bookingTail != null) {
    const tail = bookingTail;
    const [idRaw, actionRaw] = tail.split("/");
    const missionId = Number(idRaw);
    if (Number.isFinite(missionId)) {
      const actionFromPath =
        actionRaw && actionRaw.length > 0 ? actionRaw.toLowerCase() : null;
      if (actionFromPath && QUICK_ACTIONS.includes(actionFromPath as (typeof QUICK_ACTIONS)[number])) {
        return {
          route: `/quick-action?missionId=${missionId}&action=${actionFromPath}`,
          missionId,
        };
      }
      return {
        route: `/(app)/(driver)/missions/${missionId}`,
        missionId,
      };
    }
  }
  const missionTail = removeAnySchemePrefix(input, SCHEMES.map((scheme) => `${scheme}mission/`));
  if (missionTail != null) {
    const idRaw = missionTail.split("?")[0];
    const missionId = Number(idRaw);
    if (Number.isFinite(missionId)) {
      return {
        route: `/(app)/(driver)/missions/${missionId}`,
        missionId,
      };
    }
  }
  if (normalized.startsWith("atmr://bookings") || normalized.startsWith("lirie://bookings")) {
    return {
      route: "/(app)/(driver)/missions",
      missionId: null,
    };
  }
  const chatThreadTail = removeAnySchemePrefix(
    input,
    SCHEMES.map((scheme) => `${scheme}chat/thread/`)
  );
  if (chatThreadTail != null) {
    const threadId = chatThreadTail.split("?")[0];
    if (threadId.length > 0) {
      return {
        route: `/(app)/(driver)/messages/${encodeURIComponent(threadId)}`,
        missionId: null,
      };
    }
  }
  if (normalized.startsWith("atmr://chat") || normalized.startsWith("lirie://chat")) {
    return {
      route: "/(app)/(driver)/chat",
      missionId: null,
    };
  }
  if (normalized.startsWith("atmr://quick-action") || normalized.startsWith("lirie://quick-action")) {
    const [, query = ""] = input.split("?");
    const map = parseQuery(query);
    const missionId = Number(map.get("missionId") ?? map.get("bookingId") ?? "");
    const action = (map.get("action") ?? "accept").toLowerCase();
    if (Number.isFinite(missionId)) {
      return {
        route: `/quick-action?missionId=${missionId}&action=${action}`,
        missionId,
      };
    }
  }
  return null;
}

export function resolveCompanyDeepLink(input: string | null | undefined): CompanyDeepLinkResolution | null {
  if (!input) return null;
  const normalized = input.toLowerCase();

  const transferTail = removeAnySchemePrefix(input, SCHEMES.map((scheme) => `${scheme}transfer/`));
  if (transferTail != null) {
    const idRaw = transferTail.split("?")[0];
    const rideId = Number(idRaw);
    if (Number.isFinite(rideId)) {
      return {
        route: `/(app)/(company)/ride-details?rideId=${rideId}`,
        rideId,
      };
    }
  }

  const chatTail = removeAnySchemePrefix(input, SCHEMES.map((scheme) => `${scheme}chat/`));
  if (chatTail != null) {
    const threadId = chatTail.split("?")[0];
    if (threadId.length > 0) {
      return {
        route: `/(app)/(company)/chat?threadId=${encodeURIComponent(threadId)}`,
        rideId: null,
        threadId,
      };
    }
  }

  if (normalized.startsWith("atmr://dashboard") || normalized.startsWith("lirie://dashboard")) {
    return {
      route: "/(app)/(company)/dashboard",
      rideId: null,
    };
  }

  if (normalized.startsWith("atmr://rides") || normalized.startsWith("lirie://rides")) {
    const [, query = ""] = input.split("?");
    const map = parseQuery(query);
    const filter = map.get("filter");
    const filterQuery = filter ? `?filter=${encodeURIComponent(filter)}` : "";
    return {
      route: `/(app)/(company)/rides${filterQuery}`,
      rideId: null,
    };
  }

  const missionTail = removeAnySchemePrefix(input, SCHEMES.map((scheme) => `${scheme}mission/`));
  if (missionTail != null) {
    const idRaw = missionTail.split("?")[0];
    const rideId = Number(idRaw);
    if (Number.isFinite(rideId)) {
      return {
        route: `/(app)/(company)/ride-details?rideId=${rideId}`,
        rideId,
      };
    }
  }

  return null;
}
