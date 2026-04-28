import * as Linking from "expo-linking";
import { useRouter } from "expo-router";
import { AppState } from "react-native";
import { useEffect, useRef, useState } from "react";
import { appendSessionJournalEvent } from "../observability/sessionJournal";
import { useSession } from "../sessionProvider";
import {
  consumeExpiredExternalIntentMarker,
  clearPendingExternalIntent,
  hydratePendingExternalIntent,
  parseExternalIntent,
  peekPendingExternalIntent,
  upsertPendingExternalIntent,
} from "../navigation/externalIntent";
import { resolveExternalIntent } from "../navigation/externalIntentResolver";
import type { PropsWithChildren } from "../reactCompat";

function normalizeAppState(raw: string): "cold" | "background" | "foreground" {
  if (raw === "active") return "foreground";
  if (raw === "background") return "background";
  return "cold";
}

export function ExternalIntentProvider({ children }: PropsWithChildren) {
  const router = useRouter();
  const { status, bootstrap } = useSession();
  const [revision, setRevision] = useState(0);
  const processingIntentIdRef = useRef<string | null>(null);

  useEffect(() => {
    void hydratePendingExternalIntent().then(() => {
      setRevision((v) => v + 1);
    });

    const ingestUrl = (url: string | null | undefined) => {
      void appendSessionJournalEvent("auth_deeplink_received", {
        has_url: Boolean(url),
        appState: normalizeAppState(AppState.currentState),
      });
      const parsedIntent = parseExternalIntent(url);
      void appendSessionJournalEvent("auth_deeplink_parsed", {
        success: Boolean(parsedIntent),
        appState: normalizeAppState(AppState.currentState),
      });
      if (!parsedIntent) return;
      const replaceInfo = upsertPendingExternalIntent(parsedIntent);
      void appendSessionJournalEvent("auth_external_intent_detected", {
        source: parsedIntent.source,
        intentType: parsedIntent.type,
        intentId: parsedIntent.intentId,
        replaced_intent_id: replaceInfo.replacedIntentId,
        appState: normalizeAppState(AppState.currentState),
      });
      setRevision((v) => v + 1);
    };

    void Linking.getInitialURL()
      .then((url) => ingestUrl(url))
      .catch(() => undefined);

    const subscription = Linking.addEventListener("url", ({ url }) => {
      ingestUrl(url);
    });
    return () => {
      subscription.remove();
    };
  }, []);

  useEffect(() => {
    if (status !== "ready") return;
    const intent = peekPendingExternalIntent();
    if (!intent) {
      if (consumeExpiredExternalIntentMarker()) {
        router.replace("/(public)/fallback/expired-link" as any);
      }
      return;
    }
    if (processingIntentIdRef.current === intent.intentId) return;

    processingIntentIdRef.current = intent.intentId;
    try {
      const resolved = resolveExternalIntent(intent, bootstrap);
      if (resolved.route) {
        router.replace(resolved.route as any);
      }
      if (resolved.terminal || !resolved.retryable) {
        clearPendingExternalIntent();
      }
      void appendSessionJournalEvent("auth_external_intent_consumed", {
        source: intent.source,
        intentType: intent.type,
        intentId: intent.intentId,
        success: Boolean(resolved.route),
        latencyMs: Date.now() - intent.receivedAt,
        appState: normalizeAppState(AppState.currentState),
      });
    } finally {
      processingIntentIdRef.current = null;
    }
  }, [bootstrap, revision, router, status]);

  return children;
}
