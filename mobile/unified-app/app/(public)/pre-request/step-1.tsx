import { useEffect, useMemo, useState } from "react";
import {
  ActivityIndicator,
  ImageBackground,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { ResponsiveContainer, Screen, useAppViewport } from "../../../src/design/responsive";
import { checkServiceArea, upsertPublicPreRequestDraft } from "../../../src/core/api/client";
import { queueExternalIntentResume } from "../../../src/core/navigation/externalIntent";
import {
  PublicPreRequestDraft,
  createDraftId,
  loadPublicPreRequestDraft,
  savePublicPreRequestDraft,
} from "../../../src/core/public/preRequestDraft";
import { ADDRESS_INPUT_PLACEHOLDER_VISUAL } from "../../../src/features/public/addressInputPlaceholder";

const LANDING_BACKGROUND = require("../../../assets/images/landing-background.png");

function firstSearchParam(value: string | string[] | undefined): string {
  if (value == null) return "";
  const raw = Array.isArray(value) ? value[0] : value;
  return typeof raw === "string" ? raw.trim() : "";
}

function todayYyyyMmDd(): string {
  const d = new Date();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${d.getFullYear()}-${m}-${day}`;
}

function nowHhMm(): string {
  const d = new Date();
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

export default function PublicPreRequestStepOneScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{
    departure?: string;
    destination?: string;
    source?: string;
    schedule?: string;
  }>();
  const { topInset } = useAppViewport();
  const depParamLive = firstSearchParam(params.departure);
  const destParamLive = firstSearchParam(params.destination);
  const sourceParamLive = firstSearchParam(params.source);
  const scheduleParamLive = firstSearchParam(params.schedule);
  const expressFromHome =
    sourceParamLive === "home" && depParamLive.length > 0 && destParamLive.length > 0;
  const scheduleImmediate = scheduleParamLive === "immediate";
  const [draftId, setDraftId] = useState("");
  const [departure, setDeparture] = useState("");
  const [destination, setDestination] = useState("");
  const [date, setDate] = useState("");
  const [pickupTime, setPickupTime] = useState("");
  const [tripType, setTripType] = useState<"one_way" | "round_trip">("one_way");
  const [passengers, setPassengers] = useState("1");
  const [transportType, setTransportType] = useState("assis");
  const [specialRequirements, setSpecialRequirements] = useState("");
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [email, setEmail] = useState("");
  const [phone, setPhone] = useState("");
  const [zoneCheckPending, setZoneCheckPending] = useState(false);
  const [submitPending, setSubmitPending] = useState(false);
  const [serviceAreaStatus, setServiceAreaStatus] = useState<
    "available" | "conditional" | "unavailable" | null
  >(null);
  const [serviceAreaMessage, setServiceAreaMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showTripOptions, setShowTripOptions] = useState(!expressFromHome);
  const [showScheduleEditor, setShowScheduleEditor] = useState(false);
  /** Si false + parcours immédiat, l'heure affichée = maintenant (rafraîchie à l'ouverture). */
  const [pickupScheduleExact, setPickupScheduleExact] = useState(false);

  useEffect(() => {
    setShowTripOptions(!expressFromHome);
  }, [expressFromHome]);

  useEffect(() => {
    void (async () => {
      const depParam = firstSearchParam(params.departure);
      const destParam = firstSearchParam(params.destination);
      const source = firstSearchParam(params.source);
      const scheduleRaw = firstSearchParam(params.schedule);
      const isImmediate = scheduleRaw === "immediate";
      const fromHomeExpress = source === "home" && depParam.length > 0 && destParam.length > 0;

      const existing = await loadPublicPreRequestDraft();

      if (!existing) {
        setDraftId(createDraftId());
        setDeparture(depParam || "");
        setDestination(destParam || "");
        // Parcours classique : l'utilisateur saisit une heure précise ; "immédiat" = heure courante, modifiable.
        setPickupScheduleExact(!isImmediate);
        if (isImmediate) {
          setDate(todayYyyyMmDd());
          setPickupTime(nowHhMm());
        }
        return;
      }

      setDraftId(existing.draft_id);
      const nextDep = fromHomeExpress
        ? depParam || existing.departure?.trim() || ""
        : existing.departure?.trim() || depParam || "";
      const nextDest = fromHomeExpress
        ? destParam || existing.destination?.trim() || ""
        : existing.destination?.trim() || destParam || "";
      setDeparture(nextDep);
      setDestination(nextDest);

      if (isImmediate) {
        const exact = existing.pickup_schedule_exact === true;
        setPickupScheduleExact(exact);
        if (exact) {
          setDate(existing.date ?? "");
          setPickupTime(existing.pickup_time ?? "");
        } else {
          // Dès que possible : on aligne sur la date/heure du jour à chaque retour sur l'écran
          // (un brouillon 10:42 le matin ne doit pas rester affiché à 16:00 l'après-midi).
          setDate(todayYyyyMmDd());
          setPickupTime(nowHhMm());
        }
      } else {
        setPickupScheduleExact(true);
        setDate(existing.date ?? "");
        setPickupTime(existing.pickup_time ?? "");
      }

      setTripType(existing.trip_type === "round_trip" ? "round_trip" : "one_way");
      setPassengers(existing.passengers ? String(existing.passengers) : "1");
      setTransportType(existing.transport_type ?? "assis");
      setSpecialRequirements(existing.special_requirements ?? "");
      setServiceAreaStatus(existing.service_area_status ?? null);
      setFirstName(existing.contact_first_name ?? "");
      setLastName(existing.contact_last_name ?? "");
      setEmail(existing.contact_email ?? "");
      setPhone(existing.contact_phone ?? "");
    })();
  }, [params.departure, params.destination, params.source, params.schedule]);

  const canCheckServiceArea = useMemo(
    () =>
      departure.trim().length > 0 &&
      destination.trim().length > 0 &&
      date.trim().length > 0 &&
      pickupTime.trim().length > 0,
    [date, departure, destination, pickupTime]
  );

  const runServiceAreaCheck = async () => {
    if (!canCheckServiceArea) return;
    setZoneCheckPending(true);
    setError(null);
    try {
      const response = await checkServiceArea({
        departure: departure.trim(),
        destination: destination.trim(),
        date: date.trim(),
        transport_type: transportType.trim() || "assis",
      });
      setServiceAreaStatus(response.status);
      setServiceAreaMessage(response.message);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Impossible de verifier la zone.");
      setServiceAreaStatus(null);
      setServiceAreaMessage(null);
    } finally {
      setZoneCheckPending(false);
    }
  };

  const submitPreRequest = async () => {
    const nextPassengers = Number(passengers);
    if (!departure.trim() || !destination.trim() || !date.trim() || !pickupTime.trim()) {
      setError("Depart, destination, date et heure sont requis.");
      return;
    }
    if (!Number.isInteger(nextPassengers) || nextPassengers < 1 || nextPassengers > 9) {
      setError("Le nombre de passagers doit etre compris entre 1 et 9.");
      return;
    }
    const phoneDigits = phone.replace(/\D/g, "");
    if (phoneDigits.length > 0 && phoneDigits.length < 8) {
      setError("Numero de telephone incomplet (minimum 8 chiffres) ou laissez le champ vide.");
      return;
    }
    const nextDraftId = draftId || createDraftId();
    const draft: Omit<PublicPreRequestDraft, "updated_at"> = {
      draft_id: nextDraftId,
      departure: departure.trim(),
      destination: destination.trim(),
      date: date.trim(),
      pickup_time: pickupTime.trim(),
      pickup_schedule_exact: pickupScheduleExact,
      reservation_urgency: scheduleImmediate ? "immediate" : "planned",
      trip_type: tripType,
      passengers: nextPassengers,
      transport_type: transportType.trim() || "assis",
      special_requirements: specialRequirements.trim() || null,
      service_area_status: serviceAreaStatus,
      contact_first_name: firstName.trim() || null,
      contact_last_name: lastName.trim() || null,
      contact_email: email.trim() || null,
      contact_phone: phone.trim() || null,
    };
    setSubmitPending(true);
    setError(null);
    try {
      const updated = await savePublicPreRequestDraft(draft);
      await upsertPublicPreRequestDraft({
        draft_id: updated.draft_id,
        departure: updated.departure,
        destination: updated.destination,
        date: updated.date,
        transport_type: updated.transport_type,
        pickup_time: updated.pickup_time ?? null,
        trip_type: updated.trip_type ?? null,
        passengers: updated.passengers ?? null,
        special_requirements: updated.special_requirements ?? null,
        contact_first_name: updated.contact_first_name ?? null,
        contact_last_name: updated.contact_last_name ?? null,
        contact_email: updated.contact_email ?? null,
        contact_phone: updated.contact_phone ?? null,
        service_area_status: updated.service_area_status ?? null,
      });
      await queueExternalIntentResume({
        type: "pre-request-resume",
        draftId: updated.draft_id,
      });
      router.push({
        pathname: "/(public)/pre-request/guest-checkout",
        params: { draftId: updated.draft_id },
      } as any);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Impossible d'enregistrer la pre-demande.");
    } finally {
      setSubmitPending(false);
    }
  };

  const serviceBannerStyle =
    serviceAreaStatus === "unavailable"
      ? styles.statusBannerError
      : serviceAreaStatus === "conditional"
        ? styles.statusBannerWarn
        : styles.statusBannerOk;

  return (
    <View style={styles.screen}>
      <ImageBackground
        source={LANDING_BACKGROUND}
        style={StyleSheet.absoluteFillObject}
        resizeMode="cover"
        imageStyle={styles.backgroundImage}
      />
      <View style={styles.overlay} />

      <Screen
        scroll
        withHorizontalPadding={false}
        backgroundColor="transparent"
        keyboardVerticalOffset={Platform.OS === "ios" ? topInset : 0}
        contentContainerStyle={styles.scrollContent}
      >
        <ResponsiveContainer>
          <View style={styles.card}>
          <Pressable
            onPress={() => {
              if (router.canGoBack()) {
                router.back();
                return;
              }
              router.replace("/(public)" as any);
            }}
            style={styles.backButton}
            accessibilityRole="button"
            accessibilityLabel="Retour"
          >
            <Ionicons name="arrow-back" size={22} color="#0A8F7A" />
          </Pressable>

          <Text style={styles.stepPill}>Sans creer de compte</Text>
          <Text style={styles.title}>Votre trajet</Text>
          <Text style={styles.subtitle}>
            {expressFromHome
              ? "Verifiez le trajet ci-dessous. Un numero de telephone suffit pour la confirmation ; le paiement vient ensuite."
              : "Renseignez le trajet et un telephone pour la confirmation. Pas de profil a creer ici."}
          </Text>

          {expressFromHome ? (
            <>
              <View style={styles.routeSummary}>
                <View style={styles.routeSummaryBlock}>
                  <Text style={styles.routeSummaryLabel}>Prise en charge</Text>
                  <Text style={styles.routeSummaryValue}>{departure}</Text>
                </View>
                <View style={styles.routeSummaryBlock}>
                  <Text style={styles.routeSummaryLabel}>Depot / arrivee</Text>
                  <Text style={styles.routeSummaryValue}>{destination}</Text>
                </View>
                <Pressable
                  onPress={() => router.replace("/(public)" as any)}
                  style={styles.editRouteLink}
                  accessibilityRole="button"
                  accessibilityLabel="Modifier les adresses sur l accueil"
                >
                  <Text style={styles.editRouteLinkText}>Modifier les adresses</Text>
                </Pressable>
              </View>

              <View style={styles.fieldBlock}>
                <Text style={styles.label}>Depart prevu</Text>
                {scheduleImmediate && !showScheduleEditor ? (
                  <View style={styles.immediateBox}>
                    <Text style={styles.immediateTitle}>Des que possible</Text>
                    <Text style={styles.immediateSub}>
                      Date et heure du jour utilisees pour la verification de zone (modifiable).
                    </Text>
                    <Pressable
                      onPress={() => {
                        setPickupScheduleExact(true);
                        setShowScheduleEditor(true);
                      }}
                      style={styles.editRouteLink}
                      accessibilityRole="button"
                    >
                      <Text style={styles.editRouteLinkText}>Choisir date et heure precises</Text>
                    </Pressable>
                  </View>
                ) : (
                  <View style={styles.dateTimeRow}>
                    <View style={styles.dateTimeCol}>
                      <Text style={styles.label}>Date</Text>
                      <TextInput
                        value={date}
                        onChangeText={setDate}
                        placeholder="AAAA-MM-JJ"
                        placeholderTextColor="#91A59D"
                        style={styles.fieldInput}
                      />
                    </View>
                    <View style={styles.dateTimeCol}>
                      <Text style={styles.label}>Heure</Text>
                      <TextInput
                        value={pickupTime}
                        onChangeText={setPickupTime}
                        placeholder="HH:MM"
                        placeholderTextColor="#91A59D"
                        style={styles.fieldInput}
                      />
                    </View>
                  </View>
                )}
              </View>
            </>
          ) : (
            <>
              <View style={styles.fieldBlock}>
                <Text style={styles.label}>Depart</Text>
                <TextInput
                  value={departure}
                  onChangeText={setDeparture}
                  placeholder={ADDRESS_INPUT_PLACEHOLDER_VISUAL}
                  accessibilityLabel="Adresse ou lieu de prise en charge"
                  placeholderTextColor="#91A59D"
                  style={[
                    styles.fieldInput,
                    departure.trim().length === 0 ? styles.fieldInputEmpty : null,
                  ]}
                />
              </View>

              <View style={styles.fieldBlock}>
                <Text style={styles.label}>Destination</Text>
                <TextInput
                  value={destination}
                  onChangeText={setDestination}
                  placeholder={ADDRESS_INPUT_PLACEHOLDER_VISUAL}
                  accessibilityLabel="Adresse ou lieu d'arrivée"
                  placeholderTextColor="#91A59D"
                  style={[
                    styles.fieldInput,
                    destination.trim().length === 0 ? styles.fieldInputEmpty : null,
                  ]}
                />
              </View>

              <View style={styles.dateTimeRow}>
                <View style={styles.dateTimeCol}>
                  <Text style={styles.label}>Date</Text>
                  <TextInput
                    value={date}
                    onChangeText={setDate}
                    placeholder="AAAA-MM-JJ"
                    placeholderTextColor="#91A59D"
                    style={styles.fieldInput}
                  />
                </View>
                <View style={styles.dateTimeCol}>
                  <Text style={styles.label}>Heure</Text>
                  <TextInput
                    value={pickupTime}
                    onChangeText={setPickupTime}
                    placeholder="HH:MM"
                    placeholderTextColor="#91A59D"
                    style={styles.fieldInput}
                  />
                </View>
              </View>
            </>
          )}

          <Pressable
            onPress={() => setShowTripOptions((o) => !o)}
            style={({ pressed }) => [styles.optionsToggle, pressed && styles.optionsTogglePressed]}
            accessibilityRole="button"
            accessibilityLabel={
              showTripOptions ? "Masquer les options du trajet" : "Afficher les options du trajet"
            }
          >
            <Text style={styles.optionsToggleText}>
              {showTripOptions ? "Masquer les options" : "Options du trajet (facultatif)"}
            </Text>
            <Ionicons
              name={showTripOptions ? "chevron-up" : "chevron-down"}
              size={20}
              color="#0A8F7A"
            />
          </Pressable>

          {showTripOptions ? (
            <>
              <Text style={[styles.label, styles.segmentLabel]}>Type de trajet</Text>
              <View style={styles.segmentRow}>
                <Pressable
                  onPress={() => setTripType("one_way")}
                  style={({ pressed }) => [
                    styles.segmentOption,
                    tripType === "one_way" ? styles.segmentOptionActive : null,
                    pressed && styles.segmentOptionPressed,
                  ]}
                  accessibilityRole="button"
                  accessibilityState={{ selected: tripType === "one_way" }}
                >
                  <Ionicons
                    name="arrow-forward"
                    size={18}
                    color={tripType === "one_way" ? "#0A8F7A" : "#5F7369"}
                  />
                  <Text
                    style={[
                      styles.segmentText,
                      tripType === "one_way" ? styles.segmentTextActive : null,
                    ]}
                  >
                    Aller simple
                  </Text>
                </Pressable>
                <Pressable
                  onPress={() => setTripType("round_trip")}
                  style={({ pressed }) => [
                    styles.segmentOption,
                    tripType === "round_trip" ? styles.segmentOptionActive : null,
                    pressed && styles.segmentOptionPressed,
                  ]}
                  accessibilityRole="button"
                  accessibilityState={{ selected: tripType === "round_trip" }}
                >
                  <Ionicons
                    name="swap-horizontal"
                    size={18}
                    color={tripType === "round_trip" ? "#0A8F7A" : "#5F7369"}
                  />
                  <Text
                    style={[
                      styles.segmentText,
                      tripType === "round_trip" ? styles.segmentTextActive : null,
                    ]}
                  >
                    Aller-retour
                  </Text>
                </Pressable>
              </View>

              <View style={styles.fieldBlock}>
                <Text style={styles.label}>Passagers (1-9)</Text>
                <TextInput
                  value={passengers}
                  onChangeText={(value) => setPassengers(value.replace(/[^\d]/g, "").slice(0, 1))}
                  placeholder="1"
                  placeholderTextColor="#91A59D"
                  keyboardType="number-pad"
                  style={styles.fieldInput}
                />
              </View>

              <View style={styles.fieldBlock}>
                <Text style={styles.label}>Type de transport</Text>
                <TextInput
                  value={transportType}
                  onChangeText={setTransportType}
                  placeholder="ex. assis, fauteuil roulant..."
                  placeholderTextColor="#91A59D"
                  style={styles.fieldInput}
                />
              </View>

              <View style={styles.fieldBlock}>
                <Text style={styles.label}>Besoins specifiques (optionnel)</Text>
                <TextInput
                  value={specialRequirements}
                  onChangeText={setSpecialRequirements}
                  placeholder="Informations utiles pour le chauffeur"
                  placeholderTextColor="#91A59D"
                  multiline
                  textAlignVertical="top"
                  style={[styles.fieldInput, styles.fieldInputMultiline]}
                />
              </View>

              <Pressable
                onPress={() => void runServiceAreaCheck()}
                disabled={!canCheckServiceArea || zoneCheckPending}
                style={({ pressed }) => [
                  styles.outlineButton,
                  (!canCheckServiceArea || zoneCheckPending) && styles.outlineButtonDisabled,
                  pressed && canCheckServiceArea && !zoneCheckPending && styles.outlineButtonPressed,
                ]}
                accessibilityRole="button"
                accessibilityLabel="Verifier la zone de service"
              >
                {zoneCheckPending ? (
                  <ActivityIndicator color="#0A8F7A" />
                ) : (
                  <Text
                    style={[
                      styles.outlineButtonText,
                      !canCheckServiceArea && styles.outlineButtonTextDisabled,
                    ]}
                  >
                    Verifier la zone
                  </Text>
                )}
              </Pressable>
            </>
          ) : null}

          {serviceAreaStatus ? (
            <View style={[styles.statusBanner, serviceBannerStyle]}>
              <Ionicons
                name={
                  serviceAreaStatus === "unavailable"
                    ? "close-circle"
                    : serviceAreaStatus === "conditional"
                      ? "warning"
                      : "checkmark-circle"
                }
                size={20}
                color={
                  serviceAreaStatus === "unavailable"
                    ? "#B42318"
                    : serviceAreaStatus === "conditional"
                      ? "#B45309"
                      : "#2E7D32"
                }
              />
              <View style={styles.statusBannerTextWrap}>
                <Text style={styles.statusBannerTitle}>
                  Zone : {serviceAreaStatus.toUpperCase()}
                </Text>
                <Text style={styles.statusBannerMessage}>
                  {serviceAreaMessage ?? "Resultat disponible."}
                </Text>
              </View>
            </View>
          ) : null}

          <Text style={styles.sectionHeading}>Contact pour le transport</Text>
          <Text style={styles.sectionHint}>
            Nous utilisons ces informations pour confirmer la course et vous joindre si besoin. Aucun mot de
            passe ni profil n&apos;est cree a cette etape.
          </Text>

          <View style={styles.fieldBlock}>
            <Text style={styles.label}>Telephone (obligatoire)</Text>
            <TextInput
              value={phone}
              onChangeText={setPhone}
              placeholder="+41 79 123 45 67"
              placeholderTextColor="#91A59D"
              keyboardType="phone-pad"
              autoComplete="tel"
              textContentType="telephoneNumber"
              style={styles.fieldInput}
            />
          </View>

          <View style={styles.fieldBlock}>
            <Text style={styles.label}>
              Email <Text style={styles.labelMuted}>(facultatif)</Text>
            </Text>
            <TextInput
              value={email}
              onChangeText={setEmail}
              placeholder="vous@exemple.ch"
              placeholderTextColor="#91A59D"
              autoCapitalize="none"
              keyboardType="email-address"
              autoComplete="email"
              textContentType="emailAddress"
              style={styles.fieldInput}
            />
          </View>

          <Text style={styles.subSectionHeading}>Personne de contact ou passager</Text>
          <Text style={styles.subSectionHint}>
            Utile pour l&apos;equipe terrain ; laissez vide si vous preferez ne fournir que le telephone.
          </Text>

          <View style={styles.nameRow}>
            <View style={styles.nameCol}>
              <Text style={styles.label}>
                Prenom <Text style={styles.labelMuted}>(facultatif)</Text>
              </Text>
              <TextInput
                value={firstName}
                onChangeText={setFirstName}
                placeholder="Prenom"
                placeholderTextColor="#91A59D"
                autoComplete="given-name"
                textContentType="givenName"
                style={styles.fieldInput}
              />
            </View>
            <View style={styles.nameCol}>
              <Text style={styles.label}>
                Nom <Text style={styles.labelMuted}>(facultatif)</Text>
              </Text>
              <TextInput
                value={lastName}
                onChangeText={setLastName}
                placeholder="Nom"
                placeholderTextColor="#91A59D"
                autoComplete="family-name"
                textContentType="familyName"
                style={styles.fieldInput}
              />
            </View>
          </View>

          {error ? <Text style={styles.errorText}>{error}</Text> : null}

          <Pressable
            onPress={() => void submitPreRequest()}
            disabled={submitPending}
            style={({ pressed }) => [
              styles.primaryButton,
              submitPending && styles.primaryButtonDisabled,
              pressed && !submitPending && styles.primaryButtonPressed,
            ]}
            accessibilityRole="button"
            accessibilityLabel="Continuer vers le paiement"
          >
            {submitPending ? (
              <ActivityIndicator color="#FFFFFF" />
            ) : (
              <>
                <Text style={styles.primaryButtonText}>Poursuivre vers le paiement</Text>
                <Ionicons name="arrow-forward" size={20} color="#FFFFFF" style={styles.primaryIcon} />
              </>
            )}
          </Pressable>
        </View>
        </ResponsiveContainer>
      </Screen>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: "#EAF3F1",
  },
  backgroundImage: {
    opacity: 0.08,
  },
  overlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "rgba(234,243,241,0.88)",
  },
  scrollContent: {
    flexGrow: 1,
    paddingVertical: 16,
  },
  card: {
    width: "100%",
    maxWidth: 440,
    alignSelf: "center",
    borderRadius: 26,
    padding: 24,
    borderWidth: 1,
    borderColor: "rgba(145,165,157,0.45)",
    backgroundColor: "#FFFFFF",
    ...Platform.select({
      web: { boxShadow: "0 20px 48px rgba(22,58,52,0.12)" },
      default: {
        shadowColor: "#163A34",
        shadowOpacity: 0.12,
        shadowRadius: 18,
        shadowOffset: { width: 0, height: 8 },
        elevation: 4,
      },
    }),
  },
  backButton: {
    alignSelf: "flex-start",
    paddingVertical: 6,
    paddingHorizontal: 2,
    marginBottom: 10,
  },
  stepPill: {
    alignSelf: "flex-start",
    fontSize: 12,
    fontWeight: "700",
    letterSpacing: 0.4,
    textTransform: "uppercase",
    color: "#0A8F7A",
    backgroundColor: "rgba(10,143,122,0.12)",
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 999,
    overflow: "hidden",
    marginBottom: 14,
  },
  title: {
    color: "#163A34",
    fontSize: 28,
    lineHeight: 32,
    fontWeight: "700",
  },
  subtitle: {
    color: "#5F7369",
    fontSize: 15,
    lineHeight: 22,
    marginTop: 10,
  },
  routeSummary: {
    marginTop: 8,
    padding: 14,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "rgba(145,165,157,0.45)",
    backgroundColor: "#F3F8F6",
    gap: 12,
  },
  routeSummaryBlock: {
    gap: 4,
  },
  routeSummaryLabel: {
    fontSize: 12,
    fontWeight: "700",
    color: "#5F7369",
    textTransform: "uppercase",
    letterSpacing: 0.4,
  },
  routeSummaryValue: {
    fontSize: 16,
    fontWeight: "600",
    color: "#163A34",
    lineHeight: 22,
  },
  editRouteLink: {
    alignSelf: "flex-start",
    paddingVertical: 4,
  },
  editRouteLinkText: {
    fontSize: 14,
    fontWeight: "700",
    color: "#0A8F7A",
  },
  immediateBox: {
    marginTop: 4,
    padding: 12,
    borderRadius: 12,
    backgroundColor: "rgba(10,143,122,0.08)",
    borderWidth: 1,
    borderColor: "rgba(10,143,122,0.25)",
    gap: 8,
  },
  immediateTitle: {
    fontSize: 17,
    fontWeight: "700",
    color: "#163A34",
  },
  immediateSub: {
    fontSize: 14,
    lineHeight: 20,
    color: "#5F7369",
  },
  optionsToggle: {
    marginTop: 14,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingVertical: 12,
    paddingHorizontal: 4,
    borderBottomWidth: 1,
    borderBottomColor: "rgba(145,165,157,0.35)",
  },
  optionsTogglePressed: {
    opacity: 0.85,
  },
  optionsToggleText: {
    fontSize: 15,
    fontWeight: "600",
    color: "#163A34",
  },
  sectionHeading: {
    marginTop: 22,
    fontSize: 17,
    fontWeight: "700",
    color: "#163A34",
  },
  sectionHint: {
    marginTop: 6,
    fontSize: 14,
    lineHeight: 20,
    color: "#5F7369",
  },
  subSectionHeading: {
    marginTop: 18,
    fontSize: 15,
    fontWeight: "700",
    color: "#163A34",
  },
  subSectionHint: {
    marginTop: 6,
    fontSize: 13,
    lineHeight: 18,
    color: "#5F7369",
  },
  labelMuted: {
    fontWeight: "500",
    color: "#5F7369",
  },
  nameRow: {
    flexDirection: "row",
    gap: 12,
    marginTop: 14,
  },
  nameCol: {
    flex: 1,
    minWidth: 0,
  },
  fieldBlock: {
    marginTop: 16,
  },
  label: {
    fontSize: 13,
    fontWeight: "600",
    color: "#163A34",
    marginBottom: 8,
  },
  fieldInput: {
    minHeight: 50,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "#91A59D",
    backgroundColor: "#FAFCFB",
    paddingHorizontal: 14,
    paddingVertical: Platform.OS === "web" ? 12 : 10,
    color: "#163A34",
    fontSize: 16,
  },
  fieldInputEmpty: {
    borderWidth: 2,
    borderColor: "rgba(91,115,107,0.55)",
    backgroundColor: "#FFFFFF",
  },
  fieldInputMultiline: {
    minHeight: 100,
    paddingTop: 12,
  },
  dateTimeRow: {
    flexDirection: "row",
    gap: 12,
    marginTop: 16,
  },
  dateTimeCol: {
    flex: 1,
    minWidth: 0,
  },
  segmentLabel: {
    marginTop: 18,
  },
  segmentRow: {
    flexDirection: "row",
    gap: 10,
    marginTop: 8,
  },
  segmentOption: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    minHeight: 52,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "#C5D4CE",
    backgroundColor: "#FAFCFB",
    paddingHorizontal: 10,
  },
  segmentOptionActive: {
    borderColor: "#0A8F7A",
    backgroundColor: "rgba(10,143,122,0.1)",
  },
  segmentOptionPressed: {
    opacity: 0.92,
  },
  segmentText: {
    fontSize: 14,
    fontWeight: "600",
    color: "#5F7369",
  },
  segmentTextActive: {
    color: "#163A34",
  },
  outlineButton: {
    marginTop: 20,
    minHeight: 52,
    borderRadius: 14,
    borderWidth: 1.5,
    borderColor: "#0A8F7A",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#FFFFFF",
  },
  outlineButtonPressed: {
    backgroundColor: "rgba(10,143,122,0.06)",
  },
  outlineButtonDisabled: {
    borderColor: "#B8C9C3",
    backgroundColor: "#F0F5F3",
  },
  outlineButtonText: {
    color: "#0A8F7A",
    fontSize: 16,
    fontWeight: "700",
  },
  outlineButtonTextDisabled: {
    color: "#91A59D",
  },
  statusBanner: {
    marginTop: 14,
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 10,
    padding: 14,
    borderRadius: 14,
    borderWidth: 1,
  },
  statusBannerOk: {
    backgroundColor: "rgba(46,125,50,0.08)",
    borderColor: "rgba(46,125,50,0.35)",
  },
  statusBannerWarn: {
    backgroundColor: "rgba(180,83,9,0.08)",
    borderColor: "rgba(180,83,9,0.35)",
  },
  statusBannerError: {
    backgroundColor: "rgba(180,35,24,0.08)",
    borderColor: "rgba(180,35,24,0.35)",
  },
  statusBannerTextWrap: {
    flex: 1,
  },
  statusBannerTitle: {
    fontSize: 13,
    fontWeight: "700",
    color: "#163A34",
    marginBottom: 4,
  },
  statusBannerMessage: {
    fontSize: 14,
    lineHeight: 20,
    color: "#5F7369",
  },
  errorText: {
    marginTop: 12,
    color: "#B42318",
    fontWeight: "600",
    fontSize: 14,
  },
  primaryButton: {
    marginTop: 18,
    minHeight: 54,
    borderRadius: 14,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#0A8F7A",
    paddingHorizontal: 20,
  },
  primaryButtonDisabled: {
    backgroundColor: "#84B7AE",
  },
  primaryButtonPressed: {
    opacity: 0.92,
  },
  primaryButtonText: {
    color: "#FFFFFF",
    fontSize: 16,
    fontWeight: "700",
    letterSpacing: 0.2,
  },
  primaryIcon: {
    marginLeft: 8,
  },
});
