import { useEffect, useRef, useState } from "react";
import { ScrollView, StyleSheet, View } from "react-native";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { LirieButton } from "../../../design/foundations/LirieButton";
import { LirieText } from "../../../design/foundations/LirieText";
import { TextField } from "../../../design/ui/formControls";
import { apiClient } from "../../../core/api/client";
import {
  representativeComplete,
  SIGNATURE_STEP_LABELS,
  signatureOffer,
  signatureRequest,
  traceAcceptable,
  type SignatureStep,
} from "./mandateSignatureFlow";
import { SpecimenTrace, type TracePoint } from "./SpecimenTrace";

const DISCLAIMER = "DOCUMENT DE DÉMONSTRATION — SANS VALEUR CONTRACTUELLE";
const STEPS: SignatureStep[] = ["identity", "tariffs", "conditions", "signature", "document", "transmission"];

type SignatureMeta = {
  signer?: string;
  representative_name?: string | null;
  admin_review_required?: boolean;
  billing_prepared?: boolean;
  document_version?: string;
  tariff_version_presented?: string;
};

type Convention = {
  status?: string;
  status_label?: string;
  company_name?: string;
  debtor_label?: string;
  tariff_version?: string | null;
  service_scope?: string;
  billing_conditions?: string;
  duration_label?: string;
  document_version?: string | null;
  version_number?: number | null;
  convention_id?: number | null;
  amount?: string | null;
  applicability?: string;
  label?: string;
  amendment_label?: string | null;
  signature_offer?: string;
  legally_effective?: boolean;
  real_signature_collected?: boolean;
  invoice_issued?: boolean;
  billing_prepared?: boolean;
  disclaimer?: string;
  suggested_email?: string | null;
  suggested_phone?: string | null;
  demonstration_signature?: SignatureMeta | null;
  admin_review_required?: boolean;
  replay?: boolean;
  error?: string;
};

function httpMessage(error: unknown, fallback: string): string {
  const data = (error as { response?: { data?: { error?: string } } })?.response?.data;
  return data?.error || fallback;
}

/**
 * Parcours guidé de signature de démonstration.
 * Aucune facture réelle, aucune signature juridiquement active.
 */
export function DriverInvoiceSimulation({
  missionId,
  onBack,
  writesEnabled = true,
}: {
  missionId: number;
  onBack: () => void;
  writesEnabled?: boolean;
}) {
  const client = useQueryClient();
  const path = `/driver/me/bookings/${missionId}/convention-demo`;
  const [step, setStep] = useState<SignatureStep>("identity");
  const [signer, setSigner] = useState<"patient" | "representative">("patient");
  const [name, setName] = useState("");
  const [quality, setQuality] = useState("");
  const [relation, setRelation] = useState("");
  const [acknowledged, setAcknowledged] = useState(false);
  const [trace, setTrace] = useState<TracePoint[]>([]);
  const [destination, setDestination] = useState("");
  const [reason, setReason] = useState("");
  const [notice, setNotice] = useState<string | null>(null);
  const signKey = useRef(`sign-${missionId}-${Date.now()}`);
  const placed = useRef(false);
  const signing = useRef(false);

  const convention = useQuery({
    queryKey: ["driver-convention-demo", missionId],
    queryFn: async () => {
      const { data } = await apiClient.get<Convention>(path);
      return data;
    },
    retry: false,
  });

  useEffect(() => {
    const current = convention.data;
    if (!current || placed.current) return;
    placed.current = true;
    if (signatureOffer(current.applicability, current.status) === "document") {
      setStep("document");
    }
    if (current.suggested_email) setDestination(current.suggested_email);
  }, [convention.data]);

  function store(data: Convention) {
    client.setQueryData(["driver-convention-demo", missionId], data);
  }

  const prepare = useMutation({
    mutationFn: async () => (await apiClient.post<Convention>(path, {})).data,
    retry: false,
    onSuccess: (data) => {
      store(data);
      setNotice(null);
      setStep("tariffs");
    },
    onError: (error) => setNotice(httpMessage(error, "Préparation refusée. Aucune facture n'est émise.")),
  });

  const sign = useMutation({
    mutationFn: async () => {
      const current = convention.data;
      const { data } = await apiClient.post<Convention>(
        `${path}/specimen`,
        signatureRequest({
          trace,
          signer,
          representativeName: name,
          representativeQuality: quality,
          representativeRelation: relation,
          tariffVersion: current?.tariff_version,
          versionNumber: current?.version_number,
          idempotencyKey: signKey.current,
        }),
      );
      return data;
    },
    retry: false,
    onSuccess: (data) => {
      store(data);
      setNotice("Mandat enregistré — démonstration");
      setStep("document");
    },
    onError: (error) => setNotice(httpMessage(error, "Signature refusée. Aucune signature réelle n'est collectée.")),
  });

  const refuse = useMutation({
    mutationFn: async (category: "refus" | "incapable") =>
      (await apiClient.post<Convention>(`${path}/refusal`, { category, reason: reason.trim() })).data,
    retry: false,
    onSuccess: (data) => {
      store(data);
      setNotice("Motif transmis à l'entreprise. Aucun mandat n'est activé. La mission reste terminable.");
    },
    onError: (error) => setNotice(httpMessage(error, "Le motif n'a pas été enregistré.")),
  });

  const signal = useMutation({
    mutationFn: async () => (await apiClient.post<Convention>(`${path}/signal`, {})).data,
    retry: false,
    onSuccess: () => setNotice("L'entreprise est informée. Aucune signature n'est enregistrée."),
    onError: (error) => setNotice(httpMessage(error, "Le signalement n'a pas abouti.")),
  });

  const transmit = useMutation({
    mutationFn: async (input: { channel: "email" | "sms"; destination: string }) =>
      (
        await apiClient.post<Convention>(`${path}/dispatch`, {
          channel: input.channel,
          destination: input.destination.trim(),
          idempotency_key: `phase6-${input.channel}-${missionId}-${input.destination.trim()}`,
        })
      ).data,
    retry: false,
    onSuccess: (data) => {
      store(data);
      setNotice("Transmission fictive enregistrée. Aucun message réel n'est parti.");
    },
    onError: (error) => setNotice(httpMessage(error, "Transmission non effectuée. Le mandat n'est pas recréé.")),
  });

  const current = convention.data;
  const offer = signatureOffer(current?.applicability, current?.status);
  const visible = !placed.current && offer === "document" ? "document" : step;
  const index = STEPS.indexOf(visible);

  function goBack() {
    setNotice(null);
    if (visible === "transmission") {
      setStep("document");
      return;
    }
    if (visible === "document" || index <= 0) {
      onBack();
      return;
    }
    setStep(STEPS[index - 1]);
  }

  function continueIdentity() {
    if (signer === "representative" && !representativeComplete(name, quality, relation)) {
      setNotice("Indiquez le nom, la qualité et le pouvoir déclaré du représentant.");
      return;
    }
    setNotice(null);
    if (writesEnabled && current?.status === "demo_absent") {
      prepare.mutate();
      return;
    }
    setStep("tariffs");
  }

  if (convention.isPending) {
    return (
      <View accessibilityLabel="Facture patient simulée">
        <LirieText variant="message">Chargement du mandat de démonstration.</LirieText>
      </View>
    );
  }

  if (convention.isError || !current) {
    return (
      <View accessibilityLabel="Facture patient simulée" style={styles.stack}>
        <LirieText variant="message">Consultation impossible. Aucune facture n{"'"}est émise.</LirieText>
        <LirieButton variant="quiet" accessibilityLabel="Retour aux modalités" label="Retour" onPress={onBack} />
      </View>
    );
  }

  const readOnly = !writesEnabled || offer === "already" || offer === "blocked";

  return (
    <ScrollView
      accessibilityLabel="Facture patient simulée"
      keyboardShouldPersistTaps="handled"
      contentContainerStyle={styles.stack}
    >
      <LirieText variant="title">Facture patient</LirieText>
      <LirieText variant="description">{current.disclaimer || DISCLAIMER}</LirieText>
      {!readOnly ? (
        <LirieText variant="message" accessibilityLabel={`Étape ${index + 1} sur 6`}>
          {`Étape ${index + 1} sur 6 — ${SIGNATURE_STEP_LABELS[index]}`}
        </LirieText>
      ) : null}
      {notice ? (
        <LirieText variant="message" accessibilityLiveRegion="polite">
          {notice}
        </LirieText>
      ) : null}

      {offer === "already" ? (
        <LirieText variant="message">
          Un mandat de démonstration couvre déjà cette réservation. Aucune nouvelle signature.
        </LirieText>
      ) : null}

      {offer === "blocked" ? (
        <View style={styles.stack}>
          <LirieText variant="message">
            {current.amendment_label || current.label || "Cette couverture ne permet pas de signature."}
          </LirieText>
          <LirieText variant="description">
            Aucune signature applicable n{"'"}est enregistrée. L{"'"}entreprise doit traiter le dossier.
          </LirieText>
          {writesEnabled ? (
            <LirieButton
              variant="secondary"
              accessibilityLabel="Signaler à l'entreprise"
              label="Signaler à l'entreprise"
              busy={signal.isPending}
              onPress={() => signal.mutate()}
            />
          ) : null}
        </View>
      ) : null}

      {!readOnly && visible === "identity" ? (
        <View style={styles.stack}>
          <LirieText variant="message">Entreprise : {current.company_name || "Entreprise de la réservation"}</LirieText>
          <LirieText variant="message">Patient : {current.debtor_label || "Patient de la réservation"}</LirieText>
          <LirieText variant="description">
            {`Convention : ${current.convention_id || "brouillon de cette entreprise"}`}
          </LirieText>
          <LirieButton
            variant={signer === "patient" ? "primary" : "secondary"}
            accessibilityLabel="Le patient signe"
            label="Le patient signe"
            onPress={() => setSigner("patient")}
          />
          <LirieButton
            variant={signer === "representative" ? "primary" : "secondary"}
            accessibilityLabel="Un représentant signe"
            label="Un représentant signe"
            onPress={() => setSigner("representative")}
          />
          {signer === "patient" ? (
            <LirieText variant="description">
              Identité affichée à partir de la réservation. Elle n{"'"}est pas choisie par le chauffeur.
            </LirieText>
          ) : (
            <View style={styles.stack}>
              <TextField label="Prénom et nom" value={name} onChangeText={setName} />
              <TextField label="Qualité du représentant" value={quality} onChangeText={setQuality} />
              <TextField label="Pouvoir déclaré" value={relation} onChangeText={setRelation} />
              <LirieText variant="description">
                Patient représenté : {current.debtor_label || "Patient de la réservation"}. Un accompagnant n{"'"}est pas habilité automatiquement.
              </LirieText>
            </View>
          )}
          <LirieButton
            variant="primary"
            accessibilityLabel="Continuer vers les tarifs"
            label="Continuer"
            busy={prepare.isPending}
            onPress={continueIdentity}
          />
          {writesEnabled ? <ExitChoices reason={reason} onReason={setReason} busy={refuse.isPending} onRefuse={(category) => refuse.mutate(category)} /> : null}
        </View>
      ) : null}

      {!readOnly && visible === "tariffs" ? (
        <View style={styles.stack}>
          <LirieText variant="message">Créancier : {current.company_name}</LirieText>
          <LirieText variant="message">Montant de la réservation : {current.amount || "non indiqué"} CHF</LirieText>
          <LirieText variant="description">Ce montant n{"'"}est pas modifié par le mandat.</LirieText>
          <LirieText variant="message">Version : {current.tariff_version || "aucune grille tarifaire configurée"}</LirieText>
          <LirieText variant="description">{current.service_scope || "Périmètre de cette entreprise uniquement."}</LirieText>
          <LirieText variant="description">{current.billing_conditions || "Facturation simulée. Aucune facture réelle."}</LirieText>
          <LirieText variant="description">Suppléments configurés : aucun.</LirieText>
          <LirieText variant="description">Annexe tarifaire : version {current.tariff_version || "non configurée"}.</LirieText>
          <LirieButton variant="primary" accessibilityLabel="Continuer vers les conditions" label="Continuer" onPress={() => setStep("conditions")} />
        </View>
      ) : null}

      {!readOnly && visible === "conditions" ? (
        <View style={styles.stack}>
          <LirieText variant="message">Parties : {current.company_name} et {current.debtor_label}</LirieText>
          <LirieText variant="description">Objet : autorisation de facturation différée, en démonstration.</LirieText>
          <LirieText variant="description">{current.service_scope}</LirieText>
          <LirieText variant="description">{current.billing_conditions}</LirieText>
          <LirieText variant="description">Durée : {current.duration_label || "Spécimen sans durée juridique."}</LirieText>
          <LirieText variant="description">
            {`Référence : convention ${current.convention_id || "préparée"}, document ${current.document_version || "specimen-1"}.`}
          </LirieText>
          <LirieText variant="description">
            La signature tactile est une démonstration. Elle ne rend pas le mandat juridiquement actif.
          </LirieText>
          <LirieButton
            variant="primary"
            accessibilityLabel="J'ai pris connaissance des conditions"
            label="J'ai pris connaissance des conditions"
            onPress={() => {
              setAcknowledged(true);
              setStep("signature");
            }}
          />
        </View>
      ) : null}

      {!readOnly && visible === "signature" ? (
        <View style={styles.stack}>
          <SpecimenTrace initialPoints={trace} onChange={setTrace} />
          <LirieButton
            variant="quiet"
            accessibilityLabel="Retour aux conditions"
            label="Retour aux conditions"
            onPress={() => setStep("conditions")}
          />
          <LirieButton
            variant="primary"
            accessibilityLabel="Valider la signature de démonstration"
            label="Valider la démonstration"
            busy={sign.isPending}
            disabled={!traceAcceptable(trace) || !acknowledged}
            onPress={() => {
              if (signing.current || sign.isPending || !traceAcceptable(trace) || !acknowledged) return;
              signing.current = true;
              sign.mutate(undefined, { onSettled: () => { signing.current = false; } });
            }}
          />
          <ExitChoices reason={reason} onReason={setReason} busy={refuse.isPending} onRefuse={(category) => refuse.mutate(category)} />
        </View>
      ) : null}

      {!readOnly && (visible === "document" || visible === "transmission") ? (
        <View style={styles.stack}>
          <LirieText variant="title">Mandat enregistré — démonstration</LirieText>
          <LirieText variant="description">{DISCLAIMER}</LirieText>
          <LirieText variant="message">
            {`Document ${current.document_version || current.demonstration_signature?.document_version || "specimen-1"}`}
          </LirieText>
          <LirieText variant="description">
            Version présentée : {current.demonstration_signature?.tariff_version_presented || current.tariff_version || "non indiquée"}
          </LirieText>
          <LirieText variant="description">Aucune facture n{"'"}est émise. Le dossier de démonstration est préparé.</LirieText>
          {current.admin_review_required ? (
            <LirieText variant="message">Vérification administrative à effectuer par l{"'"}entreprise.</LirieText>
          ) : null}
          {visible === "document" ? (
            <LirieButton
              variant="primary"
              accessibilityLabel="Choisir la transmission"
              label="Transmettre le document"
              onPress={() => setStep("transmission")}
            />
          ) : (
            <View style={styles.stack}>
              <TextField
                label="Destinataire de cet envoi"
                value={destination}
                onChangeText={setDestination}
                autoCapitalize="none"
              />
              <LirieText variant="description">
                La correction ne change pas le débiteur contractuel. Aucun envoi réel.
              </LirieText>
              <LirieButton
                variant="secondary"
                accessibilityLabel="Envoyer par e-mail"
                label="Envoyer par e-mail"
                busy={transmit.isPending}
                onPress={() => transmit.mutate({ channel: "email", destination })}
              />
              <LirieButton
                variant="secondary"
                accessibilityLabel="Envoyer par SMS"
                label="Envoyer par SMS"
                busy={transmit.isPending}
                onPress={() => {
                  const phone = current.suggested_phone && destination.includes("@")
                    ? current.suggested_phone
                    : destination;
                  setDestination(phone);
                  transmit.mutate({ channel: "sms", destination: phone });
                }}
              />
            </View>
          )}
        </View>
      ) : null}

      {(!readOnly && visible !== "signature" && visible !== "identity") || readOnly ? (
        <LirieButton
          variant="quiet"
          accessibilityLabel={visible === "identity" || readOnly ? "Retour aux modalités" : "Étape précédente"}
          label={readOnly || visible === "document" || index === 0 ? "Retour" : "Étape précédente"}
          onPress={readOnly ? onBack : goBack}
        />
      ) : (
        <LirieButton variant="quiet" accessibilityLabel="Retour aux modalités" label="Retour au tableau de bord" onPress={onBack} />
      )}
      {(visible === "document" || visible === "transmission") && !readOnly ? (
        <LirieButton variant="primary" accessibilityLabel="Retour au tableau de bord" label="Retour au tableau de bord" onPress={onBack} />
      ) : null}
    </ScrollView>
  );
}

function ExitChoices({
  reason,
  onReason,
  busy,
  onRefuse,
}: {
  reason: string;
  onReason: (value: string) => void;
  busy: boolean;
  onRefuse: (category: "refus" | "incapable") => void;
}) {
  return (
    <View style={styles.stack}>
      <TextField label="Motif si la signature n'a pas lieu" value={reason} onChangeText={onReason} />
      <LirieButton
        variant="secondary"
        accessibilityLabel="Le patient ne peut pas signer"
        label="Le patient ne peut pas signer"
        busy={busy}
        onPress={() => onRefuse("incapable")}
      />
      <LirieButton
        variant="danger"
        accessibilityLabel="Refus de signature"
        label="Refus de signature"
        busy={busy}
        onPress={() => onRefuse("refus")}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  stack: { gap: 8, paddingBottom: 24 },
});
