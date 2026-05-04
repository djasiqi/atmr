import { useEffect, useMemo, useState } from "react";
import { View } from "react-native";
import { useRouter } from "expo-router";
import { DriverContextGuard } from "../../../src/core/guards";
import type { AuthContext } from "../../../src/core/contracts/auth";
import { isCompanyDriverSwitchAllowedForRequest } from "../../../src/core/contextSwitchPolicy";
import { useSession } from "../../../src/core/sessionProvider";
import { authenticateDriverBiometric } from "../../../src/features/driver/biometricAuth";
import { openBatteryOptimizationSettings } from "../../../src/features/driver/batteryOptimization";
import {
  getDriverProfile,
  getDriverRoute,
  triggerDriverTestPush,
  updateDriverAvailability,
  updateDriverPhoto,
} from "../../../src/features/driver/api";
import {
  AppButton,
  AppInput,
  AppText,
  brandPrimary,
  brandSurfaceSoft,
  brandTextMuted,
  ResponsiveContainer,
  Screen,
  useResponsiveTokens,
} from "../../../src/design/responsive";

export default function DriverProfileScreen() {
  const router = useRouter();
  const { bootstrap, activeContext, changeContext, error: sessionError } = useSession();
  const user = bootstrap?.user ?? null;
  const [securityMessage, setSecurityMessage] = useState<string | null>(null);
  const [isAvailable, setIsAvailable] = useState(true);
  const [availabilityPending, setAvailabilityPending] = useState(false);
  const [profileLoading, setProfileLoading] = useState(false);
  const [profileName, setProfileName] = useState<string | null>(null);
  const [photoUrl, setPhotoUrl] = useState<string>("");
  const [photoPending, setPhotoPending] = useState(false);
  const [pushTestPending, setPushTestPending] = useState(false);
  const [routePoints, setRoutePoints] = useState<number | null>(null);
  const [routePending, setRoutePending] = useState(false);
  const [returnCompanyPending, setReturnCompanyPending] = useState(false);
  const [returnCompanyMessage, setReturnCompanyMessage] = useState<string | null>(null);
  const t = useResponsiveTokens();

  const companyContexts = useMemo(
    () => (bootstrap?.available_contexts ?? []).filter((c: AuthContext) => c.context_type === "company"),
    [bootstrap?.available_contexts]
  );
  const activeCompanyContext = useMemo(
    () =>
      companyContexts.find(
        (ctx: AuthContext) => ctx.context_id === activeContext?.context_id
      ) ??
      companyContexts[0] ??
      null,
    [activeContext?.context_id, companyContexts]
  );
  const canReturnToCompanyProfile =
    activeContext?.context_type === "driver" &&
    activeCompanyContext != null &&
    isCompanyDriverSwitchAllowedForRequest(
      activeContext,
      activeCompanyContext,
      bootstrap?.user?.role
    );

  const contextDriverId = useMemo(() => {
    const rawContextId = activeContext?.context_id ?? "";
    if (!rawContextId.startsWith("driver:")) return null;
    const parsed = Number.parseInt(rawContextId.slice("driver:".length), 10);
    return Number.isFinite(parsed) ? parsed : null;
  }, [activeContext?.context_id]);

  useEffect(() => {
    let cancelled = false;
    setProfileLoading(true);
    void getDriverProfile()
      .then((profile) => {
        if (cancelled) return;
        const fullName =
          typeof profile.full_name === "string" && profile.full_name.length > 0
            ? profile.full_name
            : [profile.first_name, profile.last_name]
                .filter((value) => typeof value === "string" && value.length > 0)
                .join(" ");
        setProfileName(fullName.length > 0 ? fullName : null);
        setPhotoUrl(typeof profile.photo_url === "string" ? profile.photo_url : "");
      })
      .catch(() => {
        if (!cancelled) setProfileName(null);
      })
      .finally(() => {
        if (!cancelled) setProfileLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <DriverContextGuard>
      <Screen
        scroll
        backgroundColor={brandSurfaceSoft}
        contentContainerStyle={{
          paddingTop: t.spacingSm,
          paddingBottom: t.spacingLg,
          flexGrow: 1,
        }}
      >
        <ResponsiveContainer>
          <View style={{ width: "100%", gap: t.pageGap }}>
            <AppText variant="screenTitle">Profil chauffeur</AppText>
            <AppText variant="body">
              Nom : {profileName ?? user?.full_name ?? user?.username ?? "N/A"}
            </AppText>
            <AppText variant="body">Email : {user?.email ?? "N/A"}</AppText>
            <AppText variant="bodyMuted">Driver ID contexte : {contextDriverId ?? "N/A"}</AppText>
            <AppText variant="bodyMuted">Contexte actif : {activeContext?.label ?? "N/A"}</AppText>
            <AppText variant="bodyMuted">
              Permissions : {(activeContext?.permissions ?? []).join(", ") || "N/A"}
            </AppText>
            <AppText variant="body">Disponibilité : {isAvailable ? "Disponible" : "Indisponible"}</AppText>
            {sessionError ? (
              <AppText variant="error">Session : {sessionError}</AppText>
            ) : null}
            <AppText variant="bodyMuted">Profil sync : {profileLoading ? "chargement…" : "ok"}</AppText>
            <AppText variant="bodyMuted">Points route active : {routePoints ?? "n/a"}</AppText>
            <AppInput
              label="Photo URL"
              value={photoUrl}
              onChangeText={setPhotoUrl}
              autoCapitalize="none"
              autoCorrect={false}
              placeholder="https://cdn.example.com/photo.jpg"
              helperText={photoUrl ? undefined : "Collez une URL publique vers votre photo."}
            />
            <AppButton
              title={photoPending ? "Mise a jour photo..." : "Mettre a jour photo (URL)"}
              variant="secondary"
              loading={photoPending}
              disabled={photoPending || photoUrl.trim().length === 0}
              onPress={async () => {
                setPhotoPending(true);
                try {
                  const updated = await updateDriverPhoto(photoUrl.trim());
                  const nextPhoto =
                    typeof updated.photo_url === "string" && updated.photo_url.length > 0
                      ? updated.photo_url
                      : photoUrl.trim();
                  setPhotoUrl(nextPhoto);
                  setSecurityMessage("Photo profil mise a jour.");
                } catch (error) {
                  setSecurityMessage(
                    error instanceof Error ? error.message : "Echec mise a jour photo."
                  );
                } finally {
                  setPhotoPending(false);
                }
              }}
            />
            <AppButton
              title={
                availabilityPending ? "Mise a jour..." : isAvailable ? "Passer indisponible" : "Passer disponible"
              }
              variant="primary"
              loading={availabilityPending}
              disabled={availabilityPending}
              onPress={async () => {
                setAvailabilityPending(true);
                try {
                  const next = !isAvailable;
                  await updateDriverAvailability(next);
                  setIsAvailable(next);
                  setSecurityMessage(next ? "Disponibilite activee." : "Disponibilite desactivee.");
                } catch (error) {
                  setSecurityMessage(
                    error instanceof Error ? error.message : "Impossible de mettre a jour la disponibilite."
                  );
                } finally {
                  setAvailabilityPending(false);
                }
              }}
            />
            {canReturnToCompanyProfile && activeCompanyContext ? (
              <AppButton
                title={returnCompanyPending ? "Bascule entreprise..." : "Retour profil / espace entreprise"}
                variant="primary"
                loading={returnCompanyPending}
                disabled={returnCompanyPending}
                onPress={async () => {
                  setReturnCompanyMessage(null);
                  setReturnCompanyPending(true);
                  try {
                    await changeContext(activeCompanyContext.context_id);
                    router.replace("/(app)/(company)/dashboard" as any);
                  } catch (e) {
                    setReturnCompanyMessage(
                      e instanceof Error ? e.message : "Impossible de revenir a l’espace entreprise."
                    );
                  } finally {
                    setReturnCompanyPending(false);
                  }
                }}
              />
            ) : !canReturnToCompanyProfile && (bootstrap?.user?.role ?? "").toString().toUpperCase() === "COMPANY" ? (
              <AppText variant="caption" style={{ color: brandTextMuted }}>
                Bascule entreprise indisponible (contexte dispatch, ou session non reconnue). Paramètres ailleurs :
                espace entreprise.
              </AppText>
            ) : null}
            {returnCompanyMessage ? <AppText variant="body">{returnCompanyMessage}</AppText> : null}
            <AppButton
              title="Verifier biometrie"
              variant="secondary"
              onPress={async () => {
                const ok = await authenticateDriverBiometric();
                setSecurityMessage(ok ? "Biometrie validee." : "Biometrie indisponible ou echouee.");
              }}
            />
            <AppButton
              title="Ouvrir optimisation batterie"
              variant="secondary"
              onPress={async () => {
                await openBatteryOptimizationSettings();
                setSecurityMessage("Ecran optimisation batterie ouvert (si supporte).");
              }}
            />
            <AppButton
              title={routePending ? "Chargement route..." : "Verifier route courante"}
              variant="secondary"
              loading={routePending}
              disabled={routePending}
              onPress={async () => {
                setRoutePending(true);
                try {
                  const routePayload = await getDriverRoute();
                  const pointsCandidate =
                    Array.isArray(routePayload.points) ? routePayload.points.length : null;
                  setRoutePoints(pointsCandidate);
                  setSecurityMessage("Route chargee avec succes.");
                } catch (error) {
                  setSecurityMessage(
                    error instanceof Error ? error.message : "Impossible de charger la route."
                  );
                } finally {
                  setRoutePending(false);
                }
              }}
            />
            <AppButton
              title={pushTestPending ? "Envoi test push..." : "Declencher test push"}
              variant="secondary"
              loading={pushTestPending}
              disabled={pushTestPending}
              onPress={async () => {
                setPushTestPending(true);
                try {
                  await triggerDriverTestPush();
                  setSecurityMessage("Test push declenche avec succes.");
                } catch (error) {
                  setSecurityMessage(
                    error instanceof Error ? error.message : "Impossible de declencher test push."
                  );
                } finally {
                  setPushTestPending(false);
                }
              }}
            />
            {securityMessage ? (
              <AppText variant="body" style={{ color: brandPrimary, marginTop: t.spacingXs }}>
                {securityMessage}
              </AppText>
            ) : null}
          </View>
        </ResponsiveContainer>
      </Screen>
    </DriverContextGuard>
  );
}
