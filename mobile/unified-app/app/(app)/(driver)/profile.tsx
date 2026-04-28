import { useEffect, useMemo, useState } from "react";
import { Text, TextInput, View } from "react-native";
import { useRouter } from "expo-router";
import { DriverContextGuard } from "../../../src/core/guards";
import type { AuthContext } from "../../../src/core/contracts/auth";
import { isCompanyDriverSwitchAllowedForRequest } from "../../../src/core/contextSwitchPolicy";
import { useSession } from "../../../src/core/sessionProvider";
import { Button } from "../../../src/components/ui";
import { authenticateDriverBiometric } from "../../../src/features/driver/biometricAuth";
import { openBatteryOptimizationSettings } from "../../../src/features/driver/batteryOptimization";
import {
  getDriverProfile,
  getDriverRoute,
  triggerDriverTestPush,
  updateDriverAvailability,
  updateDriverPhoto,
} from "../../../src/features/driver/api";

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
      <View style={{ flex: 1, padding: 24, gap: 10 }}>
        <Text style={{ fontSize: 22, fontWeight: "700" }}>Profil chauffeur</Text>
        <Text>Nom: {profileName ?? user?.full_name ?? user?.username ?? "N/A"}</Text>
        <Text>Email: {user?.email ?? "N/A"}</Text>
        <Text>Driver ID contexte: {contextDriverId ?? "N/A"}</Text>
        <Text>Contexte actif: {activeContext?.label ?? "N/A"}</Text>
        <Text>Permissions: {(activeContext?.permissions ?? []).join(", ") || "N/A"}</Text>
        <Text>Disponibilite: {isAvailable ? "Disponible" : "Indisponible"}</Text>
        {sessionError ? <Text style={{ color: "#B00020" }}>Session: {sessionError}</Text> : null}
        <Text>Profil operations sync: {profileLoading ? "chargement..." : "ok"}</Text>
        <Text>Points route active: {routePoints ?? "n/a"}</Text>
        <Text>Photo URL: {photoUrl || "N/A"}</Text>
        <TextInput
          value={photoUrl}
          onChangeText={setPhotoUrl}
          autoCapitalize="none"
          autoCorrect={false}
          placeholder="https://cdn.example.com/photo.jpg"
          style={{
            borderWidth: 1,
            borderColor: "#DDD",
            borderRadius: 8,
            paddingHorizontal: 10,
            paddingVertical: 8,
          }}
        />
        <Button
          label={photoPending ? "Mise a jour photo..." : "Mettre a jour photo (URL)"}
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
        <Button
          label={availabilityPending ? "Mise a jour..." : isAvailable ? "Passer indisponible" : "Passer disponible"}
          variant="primary"
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
          <Button
            label={returnCompanyPending ? "Bascule entreprise..." : "Retour profil / espace entreprise"}
            variant="primary"
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
          <Text style={{ color: "#666" }}>
            Bascule entreprise indisponible (contexte dispatch, ou session non reconnue). Parametres ailleurs: espace
            entreprise.
          </Text>
        ) : null}
        {returnCompanyMessage ? <Text>{returnCompanyMessage}</Text> : null}
        <Button
          label="Verifier biometrie"
          onPress={async () => {
            const ok = await authenticateDriverBiometric();
            setSecurityMessage(ok ? "Biometrie validee." : "Biometrie indisponible ou echouee.");
          }}
        />
        <Button
          label="Ouvrir optimisation batterie"
          onPress={async () => {
            await openBatteryOptimizationSettings();
            setSecurityMessage("Ecran optimisation batterie ouvert (si supporte).");
          }}
        />
        <Button
          label={routePending ? "Chargement route..." : "Verifier route courante"}
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
        <Button
          label={pushTestPending ? "Envoi test push..." : "Declencher test push"}
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
        {securityMessage ? <Text>{securityMessage}</Text> : null}
      </View>
    </DriverContextGuard>
  );
}
