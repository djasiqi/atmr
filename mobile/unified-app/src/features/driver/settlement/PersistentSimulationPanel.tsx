import { useRef, useState } from "react";
import { Pressable, StyleSheet, Text, View } from "react-native";

export type SimulationDossier = {
  banner: string;
  amount_chf: string;
  currency: string;
  state: string;
  mode: string | null;
  modes: string[];
  debtor_label: string;
  company_id: number;
  company_name?: string | null;
  booking_id?: number;
  contractual_basis: string;
  document_version: string;
  signature_recorded: boolean;
  history: { action: string; to_state: string }[];
};

type PersistentSimulationPanelProps = {
  dossier: SimulationDossier;
  onAction: (
    action: string,
    body: Record<string, unknown>
  ) => Promise<void>;
  onBack?: () => void;
};

const STATE_LABEL: Record<string, string> = {
  preparing: "Préparation",
  operation_in_progress: "Opération fictive en cours",
  simulated_confirmed: "Confirmation simulée",
  simulated_refused: "Refus simulé",
  uncertain: "Résultat incertain",
  administrative_simulated: "Traitement administratif simulé",
  closed: "Dossier fictif clos",
};

/**
 * Parcours relié au dossier serveur. Les boutons n'inventent pas le résultat :
 * ils envoient une transition et affichent l'état renvoyé.
 */
export function PersistentSimulationPanel({
  dossier,
  onAction,
  onBack,
}: PersistentSimulationPanelProps) {
  const [busy, setBusy] = useState(false);
  const busyRef = useRef(false);
  const [error, setError] = useState<string | null>(null);
  const run = (action: string, body: Record<string, unknown> = {}) => {
    if (busyRef.current) return;
    busyRef.current = true;
    setBusy(true);
    setError(null);
    const key = `${action}-${dossier.state}-${dossier.history.length}`;
    void onAction(action, { ...body, idempotency_key: key })
      .catch(() => {
        setError("Le serveur n'a pas confirmé la transition. Aucun second débit.");
      })
      .finally(() => {
        busyRef.current = false;
        setBusy(false);
      });
  };
  const modes = dossier.modes;
  const preparing = dossier.state === "preparing" || dossier.state === "simulated_refused";
  const companyLabel = dossier.company_name || `Entreprise ${dossier.company_id}`;
  const uncertain = dossier.state === "uncertain";

  return (
    <View accessibilityLabel="Simulation de règlement">
      <Text>{dossier.banner}</Text>
      <Text>{`Entreprise bénéficiaire : ${companyLabel}`}</Text>
      <Text accessibilityLabel="Montant à régler">{`Montant demandé : ${dossier.amount_chf} ${dossier.currency}`}</Text>
      <Text>{`État : ${STATE_LABEL[dossier.state] ?? dossier.state}`}</Text>
      <Text>{`Débiteur désigné : ${dossier.debtor_label}`}</Text>
      <Text>{`Référence du transport : ${dossier.booking_id ?? "—"}`}</Text>
      {error ? <Text>{error}</Text> : null}
      {uncertain ? (
        <Text>Aucun nouveau paiement n'est proposé.</Text>
      ) : null}
      {preparing && modes.includes("tap_to_pay") ? (
        <View>
          <Text>Paiement par carte simulé. Préparation.</Text>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Lancer Tap to Pay simulé"
            disabled={busy}
            onPress={() => run("start_tap")}
            style={styles.button}
          >
            <Text>Lancer Tap to Pay simulé</Text>
          </Pressable>
        </View>
      ) : null}
      {dossier.state === "operation_in_progress" ? (
        <View>
          <Text>Opération fictive en cours.</Text>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Résultat simulé accepté"
            disabled={busy}
            onPress={() => run("resolve_tap", { simulated_result: "confirmed" })}
            style={styles.button}
          >
            <Text>Résultat simulé : accepté</Text>
          </Pressable>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Résultat simulé refusé"
            disabled={busy}
            onPress={() => run("resolve_tap", { simulated_result: "refused" })}
            style={styles.button}
          >
            <Text>Résultat simulé : refusé</Text>
          </Pressable>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Résultat simulé incertain"
            disabled={busy}
            onPress={() => run("resolve_tap", { simulated_result: "uncertain" })}
            style={styles.button}
          >
            <Text>Résultat simulé : incertain</Text>
          </Pressable>
        </View>
      ) : null}
      {dossier.state === "uncertain" || dossier.state === "operation_in_progress" ? (
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Relire le résultat serveur"
          disabled={busy}
          onPress={() => run("recover_tap")}
          style={styles.button}
        >
          <Text>Relire le résultat serveur</Text>
        </Pressable>
      ) : null}
      {preparing && modes.includes("cash") ? (
        <View>
          <Text>Espèces présentées : 50.00 CHF. Monnaie à rendre : 5.00 CHF.</Text>
          <Text>La réussite fictive attend la confirmation que 5.00 CHF ont été rendus.</Text>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Espèces 50 francs monnaie 5 confirmée"
            disabled={busy}
            onPress={() =>
              run("declare_cash", {
                tendered_chf: "50.00",
                change_chf: "5.00",
                change_confirmed: true,
              })
            }
            style={styles.button}
          >
            <Text>50 CHF reçus, 5 CHF rendus</Text>
          </Pressable>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Espèces insuffisantes restituées"
            disabled={busy}
            onPress={() =>
              run("cash_shortfall", {
                tendered_chf: "30.00",
                disposition: "returned",
              })
            }
            style={styles.button}
          >
            <Text>30 CHF proposés puis restitués. Aucune créance.</Text>
          </Pressable>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Somme conservée à titre exceptionnel"
            disabled={busy}
            onPress={() =>
              run("cash_shortfall", { tendered_chf: "30.00", disposition: "held" })
            }
            style={styles.button}
          >
            <Text>30 CHF conservés, à examiner. Ce n'est pas une créance définitive.</Text>
          </Pressable>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Somme contestée"
            disabled={busy}
            onPress={() =>
              run("cash_shortfall", { tendered_chf: "30.00", disposition: "contested" })
            }
            style={styles.button}
          >
            <Text>30 CHF contestés</Text>
          </Pressable>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Transmission administrative des espèces"
            disabled={busy}
            onPress={() =>
              run("cash_shortfall", {
                tendered_chf: "30.00",
                disposition: "administrative",
              })
            }
            style={styles.button}
          >
            <Text>30 CHF transmis à l'administration</Text>
          </Pressable>
        </View>
      ) : null}
      {preparing && modes.includes("invoice") ? (
        <View>
          <Text>{`Entreprise créancière : ${companyLabel}`}</Text>
          <Text>{`Base contractuelle : ${dossier.contractual_basis}`}</Text>
          <Text>{`Document non approuvé : ${dossier.document_version}`}</Text>
          <Text>Traitement administratif nécessaire.</Text>
          <Text>Aucune signature n'est enregistrée. Elle n'a aucune valeur juridique.</Text>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Information préalable absente"
            disabled={busy}
            onPress={() =>
              run("invoice_event", { kind: "prior_information_missing" })
            }
            style={styles.button}
          >
            <Text>Information préalable absente</Text>
          </Pressable>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Acceptation simulée"
            disabled={busy}
            onPress={() => run("invoice_event", { kind: "simulated_acceptance" })}
            style={styles.button}
          >
            <Text>Acceptation simulée</Text>
          </Pressable>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Refus de signer"
            disabled={busy}
            onPress={() => run("invoice_event", { kind: "refused" })}
            style={styles.button}
          >
            <Text>Le patient refuse de signer</Text>
          </Pressable>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Incapacité physique"
            disabled={busy}
            onPress={() => run("invoice_event", { kind: "unable_to_sign" })}
            style={styles.button}
          >
            <Text>Incapacité physique de signer</Text>
          </Pressable>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Difficulté de compréhension"
            disabled={busy}
            onPress={() => run("invoice_event", { kind: "comprehension_difficulty" })}
            style={styles.button}
          >
            <Text>Difficulté de compréhension</Text>
          </Pressable>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Représentant non habilité"
            disabled={busy}
            onPress={() =>
              run("invoice_event", { kind: "unverified_representative" })
            }
            style={styles.button}
          >
            <Text>Représentant dont les pouvoirs ne sont pas vérifiés</Text>
          </Pressable>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Payeur contesté"
            disabled={busy}
            onPress={() => run("invoice_event", { kind: "contested_payer" })}
            style={styles.button}
          >
            <Text>Payeur contesté</Text>
          </Pressable>
        </View>
      ) : null}
      {onBack ? (
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Retour à la mission"
          onPress={onBack}
          style={styles.button}
        >
          <Text>Retour à la mission</Text>
        </Pressable>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  button: { minHeight: 48, paddingVertical: 12 },
});
