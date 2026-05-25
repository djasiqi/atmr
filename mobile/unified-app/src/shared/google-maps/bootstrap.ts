/**
 * Chargeur Google Maps JS local au runtime mobile unified-app.
 * Evite tout couplage direct avec frontend/src.
 */
export const GOOGLE_MAPS_SCRIPT_ID = "google-maps-script";

type GoogleMapsImportLibrary = (name: string) => Promise<unknown>;

type GoogleMapsNamespace = {
  Map?: unknown;
  importLibrary?: GoogleMapsImportLibrary;
};

type GmAuthFailure = (() => void) | undefined;

type BrowserWindow = Window & {
  google?: { maps?: GoogleMapsNamespace };
  gm_authFailure?: GmAuthFailure;
};

function getBrowserWindow(): BrowserWindow | undefined {
  if (typeof window === "undefined") return undefined;
  return window as BrowserWindow;
}

/**
 * Parse "maps,marker routes" -> ["maps","marker","routes"].
 */
export function parseGoogleMapsLibraryList(raw: string | undefined): string[] {
  const value = (raw ?? "marker").trim();
  if (!value) return ["marker"];
  return value.split(/[\s,]+/).filter(Boolean);
}

/**
 * SDK pret quand importLibrary + Map sont disponibles.
 */
export function isGoogleMapsSdkReady(): boolean {
  const win = getBrowserWindow();
  if (!win) return false;
  const maps = win.google?.maps;
  return typeof maps?.importLibrary === "function" && typeof maps?.Map === "function";
}

const NAMESPACE_WAIT_MS = 15000;
const NAMESPACE_POLL_MS = 50;

function waitForGoogleMapsNamespace(): Promise<void> {
  return new Promise((resolve, reject) => {
    const startedAt = Date.now();

    const tick = () => {
      const maps = getBrowserWindow()?.google?.maps;
      if (maps && typeof maps.importLibrary === "function") {
        resolve();
        return;
      }
      if (Date.now() - startedAt > NAMESPACE_WAIT_MS) {
        reject(
          new Error(
            "SDK Google Maps: importLibrary indisponible apres timeout - verifier URL script (v=weekly, loading=async)."
          )
        );
        return;
      }
      window.setTimeout(tick, NAMESPACE_POLL_MS);
    };

    tick();
  });
}

function buildLibrariesQuery(libraryList: string[]): string {
  const ordered = ["maps"];
  for (const name of libraryList) {
    if (name && name !== "maps" && !ordered.includes(name)) {
      ordered.push(name);
    }
  }
  return ordered.join(",");
}

async function bootstrapGoogleMapsLibraries(libraryList: string[]): Promise<void> {
  const maps = getBrowserWindow()?.google?.maps;
  if (!maps) {
    throw new Error("google.maps absent apres chargement du script");
  }
  if (typeof maps.importLibrary !== "function") {
    throw new Error(
      "Google Maps: importLibrary indisponible - exiger script officiel v=weekly avec loading=async."
    );
  }
  await maps.importLibrary("maps");
  const extra = libraryList.filter((lib) => lib !== "maps");
  await Promise.all(extra.map((lib) => maps.importLibrary!(lib)));
}

let inFlight: Promise<void> | null = null;

/**
 * Charge Google Maps JS une seule fois et garantit importLibrary disponible.
 */
export function loadGoogleMapsScriptWithKey(
  apiKey: string,
  opts: { libraryList?: string[] } = {}
): Promise<void> {
  if (typeof window === "undefined" || typeof document === "undefined") {
    return Promise.resolve();
  }

  const libraryList =
    opts.libraryList && opts.libraryList.length > 0
      ? opts.libraryList
      : parseGoogleMapsLibraryList("marker");

  if (!apiKey) {
    return Promise.reject(new Error("API key manquante"));
  }
  if (isGoogleMapsSdkReady()) {
    return Promise.resolve();
  }
  if (inFlight) {
    return inFlight;
  }

  inFlight = new Promise((resolve, reject) => {
    const win = getBrowserWindow();
    const previousGmf = win?.gm_authFailure;

    const cleanupGmf = (replacement: GmAuthFailure) => {
      const current = getBrowserWindow();
      if (current && current.gm_authFailure === replacement) {
        current.gm_authFailure = previousGmf;
      }
    };

    const fail = (err: Error) => {
      inFlight = null;
      cleanupGmf(gmf);
      reject(err);
    };

    const ok = () => {
      cleanupGmf(gmf);
      resolve();
    };

    const gmf = () => {
      if (typeof previousGmf === "function") previousGmf();
      fail(new Error("Authentification Google Maps refusee (cle API, domaine ou billing)"));
    };

    if (win) {
      win.gm_authFailure = gmf;
    }

    const finishBootstrap = () =>
      waitForGoogleMapsNamespace()
        .then(() => bootstrapGoogleMapsLibraries(libraryList))
        .then(ok)
        .catch((error) => fail(error instanceof Error ? error : new Error(String(error))));

    if (getBrowserWindow()?.google?.maps) {
      finishBootstrap();
      return;
    }

    const existing = document.getElementById(GOOGLE_MAPS_SCRIPT_ID);
    if (existing) {
      finishBootstrap();
      return;
    }

    const libs = buildLibrariesQuery(libraryList);
    const script = document.createElement("script");
    script.id = GOOGLE_MAPS_SCRIPT_ID;
    script.setAttribute("data-atmr-maps-loader", "1");
    script.async = true;
    script.defer = true;
    script.src = `https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(
      apiKey
    )}&v=weekly&libraries=${libs}&loading=async`;

    script.onload = () => finishBootstrap();
    script.onerror = () => fail(new Error("Echec chargement Google Maps SDK"));

    document.head.appendChild(script);
  });

  return inFlight;
}
