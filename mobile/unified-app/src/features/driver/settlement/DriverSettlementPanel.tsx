import { useRef, useState } from "react";
import { Pressable, StyleSheet, Text, View } from "react-native";
import {
  contractAllowsMock,
  discreetMessage,
  settlementSurface,
  visibleModes,
  type DriverSettlementContract,
  type SettlementMode,
} from "./driverSettlementContract";

type Screen =
  | "summary"
  | "confirm"
  | "tap-prepare"
  | "tap-progress"
  | "tap-accepted"
  | "tap-refused"
  | "tap-uncertain"
  | "tap-interrupted"
  | "tap-already"
  | "cash"
  | "cash-difference"
  | "invoice"
  | "invoice-unable"
  | "invoice-refused"
  | "invoice-representative"
  | "invoice-technical";

type DriverSettlementPanelProps = {
  contract: DriverSettlementContract | null;
  loading?: boolean;
  error?: string | null;
  offline?: boolean;
  onBack?: () => void;
  showBack?: boolean;
};

const MODE_LABEL: Record<SettlementMode, string> = {
  tap_to_pay: "Tap to Pay",
  cash: "Espèces",
  invoice: "Facture",
};

const MODE_HINT: Record<SettlementMode, string> = {
  tap_to_pay: "Carte sur le téléphone",
  cash: "Montant reçu en main",
  invoice: "Document à signer",
};

const PREVIEW_SCREENS = new Set<Screen>([
  "summary",
  "tap-prepare",
  "tap-progress",
  "tap-accepted",
  "tap-refused",
  "tap-uncertain",
  "tap-interrupted",
  "tap-already",
  "cash",
  "cash-difference",
  "invoice",
  "invoice-unable",
  "invoice-refused",
  "invoice-representative",
  "invoice-technical",
]);

function openingScreen(contract: DriverSettlementContract | null): Screen {
  const preview = contract?.preview_screen;
  if (contract?.simulation && preview && PREVIEW_SCREENS.has(preview as Screen)) {
    return preview as Screen;
  }
  return "summary";
}

/**
 * Panneau facultatif. Il n'est pas monté sur la clôture de mission :
 * le transport terminé ne dépend pas d'un encaissement.
 */
export function DriverSettlementPanel({
  contract,
  loading = false,
  error = null,
  offline = false,
  onBack,
  showBack = true,
}: DriverSettlementPanelProps) {
  const surface = settlementSurface(contract, {
    loading,
    error: Boolean(error),
    offline,
  });
  const [screen, setScreen] = useState<Screen>(openingScreen(contract));
  const [busy, setBusy] = useState(false);
  const [tapClosed, setTapClosed] = useState(false);
  const [pendingMode, setPendingMode] = useState<SettlementMode | null>(null);

  if (surface === "hidden") return null;

  const back = () => {
    if (screen !== "summary") {
      setScreen("summary");
      setBusy(false);
      return;
    }
    onBack?.();
  };

  return (
    <View
      accessibilityLabel="Règlement de la mission"
      style={styles.panel}
    >
      <Text style={styles.title}>Règlement</Text>
      {surface === "loading" ? <Text>Lecture du règlement…</Text> : null}
      {surface === "error" ? (
        <Text>
          {offline
            ? "Connexion indisponible. Le transport reste terminé. Aucun paiement n'est tenté."
            : "Réponse du serveur indisponible. Aucune autorisation n'est conservée."}
        </Text>
      ) : null}
      {surface === "discreet" && contract ? <Text>{discreetMessage(contract)}</Text> : null}
      {surface === "review" ? (
        <Text>{"Règlement à vérifier par l'entreprise."}</Text>
      ) : null}
      {surface === "simulation" && contract ? (
          <Simulation
            contract={contract}
            screen={screen}
            busy={busy}
            tapClosed={tapClosed}
            pendingMode={pendingMode}
            onScreen={setScreen}
            onBusy={setBusy}
            onTapClosed={setTapClosed}
            onPendingMode={setPendingMode}
          />
      ) : null}
      {showBack ? (
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Retour à la mission"
          onPress={back}
          style={styles.button}
        >
          <Text style={styles.buttonText}>Retour à la mission</Text>
        </Pressable>
      ) : null}
    </View>
  );
}

function Simulation({
  contract,
  screen,
  busy,
  tapClosed,
  pendingMode,
  onScreen,
  onBusy,
  onTapClosed,
  onPendingMode,
}: {
  contract: DriverSettlementContract;
  screen: Screen;
  busy: boolean;
  tapClosed: boolean;
  pendingMode: SettlementMode | null;
  onScreen: (screen: Screen) => void;
  onBusy: (busy: boolean) => void;
  onTapClosed: (closed: boolean) => void;
  onPendingMode: (mode: SettlementMode | null) => void;
}) {
  const modes = visibleModes(contract);
  const amount = contract.amount_chf ? `${contract.amount_chf} ${contract.currency}` : "";
  const allow = contractAllowsMock(contract);
  const tapChosen = useRef(false);
  const chooseTap = (next: Screen) => {
    if (tapChosen.current) return;
    tapChosen.current = true;
    onBusy(false);
    if (next === "tap-accepted" || next === "tap-uncertain" || next === "tap-interrupted") {
      onTapClosed(true);
    }
    onScreen(next);
  };

  return (
    <View>
      <Text>{"Simulation uniquement. Aucun paiement n'est enregistré."}</Text>
      {screen === "summary" ? (
        <View>
          <Text accessibilityLabel="Montant à régler" style={styles.amount}>{`Montant fixé : ${amount}`}</Text>
          {modes.map((mode) => (
            <Pressable
              key={mode}
              accessibilityRole="button"
              accessibilityLabel={MODE_LABEL[mode]}
              disabled={!allow || busy}
              onPress={() => {
                onPendingMode(mode);
                onScreen("confirm");
              }}
              style={[styles.button, modeStyle(mode)]}
            >
              <Text style={styles.buttonText}>{MODE_LABEL[mode]}</Text>
              <Text style={styles.hint}>{MODE_HINT[mode]}</Text>
            </Pressable>
          ))}
        </View>
      ) : null}
      {screen === "confirm" && pendingMode ? (
        <View>
          <Text>{"Confirmer cette simulation ? Aucun paiement réel ne sera enregistré."}</Text>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Confirmer la simulation"
            onPress={() => {
              if (pendingMode === "tap_to_pay") {
                onScreen(tapClosed ? "tap-already" : "tap-prepare");
              }
              if (pendingMode === "cash") onScreen("cash");
              if (pendingMode === "invoice") onScreen("invoice");
            }}
            style={styles.button}
          >
            <Text style={styles.buttonText}>Confirmer la simulation</Text>
          </Pressable>
        </View>
      ) : null}
      {screen === "tap-prepare" ? (
        <View>
          <Text>{`Montant à régler : ${amount}`}</Text>
          <Text>Préparation du paiement simulé.</Text>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Lancer la simulation"
            disabled={busy}
            onPress={() => {
              onBusy(true);
              onScreen("tap-progress");
            }}
            style={styles.button}
          >
            <Text style={styles.buttonText}>Lancer la simulation</Text>
          </Pressable>
        </View>
      ) : null}
      {screen === "tap-progress" ? (
        <View>
          <Text>Paiement simulé en cours.</Text>
          <Outcome label="Résultat simulé : accepté" onPress={() => chooseTap("tap-accepted")} />
          <Outcome label="Résultat simulé : refusé" onPress={() => chooseTap("tap-refused")} />
          <Outcome label="Résultat simulé : incertain" onPress={() => chooseTap("tap-uncertain")} />
          <Outcome label="Résultat simulé : interrompu" onPress={() => chooseTap("tap-interrupted")} />
        </View>
      ) : null}
      {screen === "tap-accepted" ? (
        <Text>{"Paiement simulé accepté. Aucun paiement COMPLETED n'est enregistré."}</Text>
      ) : null}
      {screen === "tap-refused" ? (
        <Text>{"Paiement simulé refusé. Aucun débit n'a été fait."}</Text>
      ) : null}
      {screen === "tap-uncertain" ? (
        <Text>
          {"Réponse incertaine. Une vérification serait nécessaire. Aucune nouvelle tentative automatique."}
        </Text>
      ) : null}
      {screen === "tap-interrupted" ? (
        <Text>{"Paiement simulé interrompu. Aucun débit n'a été fait. Aucune nouvelle tentative automatique."}</Text>
      ) : null}
      {screen === "tap-already" ? (
        <Text>Un paiement simulé existe déjà. Aucune nouvelle tentative.</Text>
      ) : null}
      {screen === "cash" || screen === "cash-difference" ? (
        <CashScreen
          amount={amount}
          companyId={contract.company_id}
          difference={screen === "cash-difference"}
          onDifference={() => onScreen("cash-difference")}
          onExact={() => onScreen("summary")}
        />
      ) : null}
      {screen.startsWith("invoice") ? (
        <InvoiceScreen
          contract={contract}
          amount={amount}
          screen={screen}
          onScreen={onScreen}
        />
      ) : null}
    </View>
  );
}

function Outcome({ label, onPress }: { label: string; onPress: () => void }) {
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={label}
      onPress={onPress}
      style={styles.button}
    >
      <Text style={styles.buttonText}>{label}</Text>
    </Pressable>
  );
}

function CashScreen({
  amount,
  companyId,
  difference,
  onDifference,
  onExact,
}: {
  amount: string;
  companyId: number | null;
  difference: boolean;
  onDifference: () => void;
  onExact: () => void;
}) {
  return (
    <View>
      <Text>{`Montant attendu : ${amount}`}</Text>
      <Text>{`Entreprise bénéficiaire : ${companyId ?? "non indiquée"}`}</Text>
      <Text>Chauffeur concerné : chauffeur affecté à la mission.</Text>
      <Text>Le reçu client et le rapprochement de caisse restent à prévoir.</Text>
      {difference ? (
        <Text>
          {
            "Montant différent. Aucune correction arbitraire. " +
            "Vérification par l'entreprise. Aucune réception réelle n'est enregistrée."
          }
        </Text>
      ) : (
        <View>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Confirmer le montant attendu"
            onPress={onExact}
            style={styles.button}
          >
            <Text style={styles.buttonText}>Confirmer le montant attendu</Text>
          </Pressable>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Le montant reçu est différent"
            onPress={onDifference}
            style={styles.button}
          >
            <Text style={styles.buttonText}>Le montant reçu est différent</Text>
          </Pressable>
        </View>
      )}
      <Text>{"Aucune réception d'espèces n'est enregistrée."}</Text>
    </View>
  );
}

function InvoiceScreen({
  contract,
  amount,
  screen,
  onScreen,
}: {
  contract: DriverSettlementContract;
  amount: string;
  screen: Screen;
  onScreen: (screen: Screen) => void;
}) {
  const unapproved = !contract.document_approved || !contract.document_version;
  return (
    <View>
      <Text>{`Débiteur : ${contract.debtor_label || "non indiqué"}`}</Text>
      <Text>Transport concerné : mission affichée.</Text>
      <Text>{`Montant : ${amount}`}</Text>
      <Text>{"La facture n'est pas un paiement."}</Text>
      <Text>
        {"Cette signature est distincte de l'attestation de réalisation du transport."}
      </Text>
      {contract.attestation_recorded ? (
        <Text>{"L'attestation déjà enregistrée reste un document séparé."}</Text>
      ) : null}
      {unapproved ? (
        <Text>
          {"Aucun document approuvé. Aucune signature financière n'est enregistrée."}
        </Text>
      ) : (
        <Text>{`Document fictif, version ${contract.document_version}.`}</Text>
      )}
      {screen === "invoice-unable" ? <Text>Le patient ne peut pas signer.</Text> : null}
      {screen === "invoice-refused" ? <Text>Le patient refuse de signer.</Text> : null}
      {screen === "invoice-representative" ? (
        <Text>Un représentant exige une validation administrative.</Text>
      ) : null}
      {screen === "invoice-technical" ? <Text>Problème technique. Aucune signature.</Text> : null}
      <Pressable
        accessibilityRole="button"
        accessibilityLabel="Patient incapable de signer"
        onPress={() => onScreen("invoice-unable")}
        style={styles.button}
      >
        <Text style={styles.buttonText}>Patient incapable de signer</Text>
      </Pressable>
      <Pressable
        accessibilityRole="button"
        accessibilityLabel="Patient refusant de signer"
        onPress={() => onScreen("invoice-refused")}
        style={styles.button}
      >
        <Text style={styles.buttonText}>Patient refusant de signer</Text>
      </Pressable>
      <Pressable
        accessibilityRole="button"
        accessibilityLabel="Représentant à valider"
        onPress={() => onScreen("invoice-representative")}
        style={styles.button}
      >
        <Text style={styles.buttonText}>Représentant à valider</Text>
      </Pressable>
      <Pressable
        accessibilityRole="button"
        accessibilityLabel="Problème technique"
        onPress={() => onScreen("invoice-technical")}
        style={styles.button}
      >
        <Text style={styles.buttonText}>Problème technique</Text>
      </Pressable>
    </View>
  );
}

function modeStyle(mode: SettlementMode) {
  if (mode === "tap_to_pay") return styles.cardMode;
  if (mode === "cash") return styles.cashMode;
  return styles.invoiceMode;
}

const styles = StyleSheet.create({
  panel: {
    width: "100%",
    padding: 16,
    gap: 12,
  },
  title: {
    fontSize: 20,
    lineHeight: 28,
    fontWeight: "700",
  },
  button: {
    minHeight: 48,
    marginTop: 8,
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 12,
    backgroundColor: "#E8F6EF",
    justifyContent: "center",
  },
  buttonText: {
    fontSize: 17,
    lineHeight: 24,
  },
  hint: {
    fontSize: 15,
    lineHeight: 22,
  },
  amount: {
    fontSize: 28,
    lineHeight: 36,
    fontWeight: "700",
  },
  cardMode: {
    backgroundColor: "#E8F1FB",
  },
  cashMode: {
    backgroundColor: "#EAF6EA",
  },
  invoiceMode: {
    backgroundColor: "#F8F1E7",
  },
});
