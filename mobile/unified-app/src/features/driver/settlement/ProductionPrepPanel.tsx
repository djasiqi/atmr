import { useRef } from "react";
import { Pressable, Text, View } from "react-native";

export type CashPrep = {
  banner: string;
  amount_due_chf: string;
  currency: string;
  debtor_label: string;
  company_name?: string | null;
  state: string;
  accounted?: boolean;
  production_activation?: boolean;
  net_chf?: string | null;
  change_chf?: string | null;
  tendered_chf?: string | null;
};

export type PatientPrep = {
  banner: string;
  amount_chf: string;
  debtor_label: string;
  contractual_basis: string;
  document_approved?: boolean;
  legally_binding?: boolean;
  real_signature_collected?: boolean;
  invoice_issued?: boolean;
  production_activation?: boolean;
};

type Action = (body: Record<string, unknown>) => Promise<void>;

/**
 * Écrans de préparation. Ils ne présentent jamais un encaissement activé.
 * Le montant dû n'est pas un champ.
 */
export function ProductionPrepPanel({
  cash,
  patient,
  onCash,
  onPatient,
}: {
  cash: CashPrep;
  patient: PatientPrep;
  onCash: Action;
  onPatient: Action;
}) {
  const busy = useRef(false);

  async function run(action: Action, body: Record<string, unknown>) {
    if (busy.current || cash.production_activation || patient.production_activation) {
      return;
    }
    busy.current = true;
    try {
      await action(body);
    } finally {
      busy.current = false;
    }
  }

  return (
    <View>
      <Text>{cash.banner}</Text>
      <Text>{patient.banner}</Text>
      <Text>{`Entreprise : ${cash.company_name || "Entreprise"}`}</Text>
      <Text>{`Débiteur : ${cash.debtor_label}`}</Text>
      <Text>{`Montant à payer : ${cash.amount_due_chf} ${cash.currency}`}</Text>
      <Text>Activation : non</Text>
      {cash.state === "received" ? (
        <Text>
          Enregistrement confirmé. Net {cash.net_chf} CHF. Monnaie rendue{" "}
          {cash.change_chf} CHF. {"Ce n'est pas une activation de caisse."}
        </Text>
      ) : null}
      {cash.state === "returned" ? (
        <Text>{"Les espèces présentées doivent être restituées. Rien n'est encaissé."}</Text>
      ) : null}
      {cash.state === "administrative" ? (
        <Text>
          {
            "Situation administrative. Ce n'est pas une créance définitive et ce n'est pas une quittance."
          }
        </Text>
      ) : null}
      {cash.state === "uncertain" ? (
        <Text>{"Le serveur n'a pas confirmé l'enregistrement. Aucun second encaissement."}</Text>
      ) : null}
      {cash.state === "prepared" ? (
        <View>
          <Pressable
            onPress={() =>
              run(onCash, {
                action: "declare_cash",
                idempotency_key: "ui-exact",
                tendered_chf: cash.amount_due_chf,
                change_chf: "0.00",
                change_confirmed: false,
              })
            }
          >
            <Text>Encaisser {cash.amount_due_chf} CHF</Text>
          </Pressable>
          <Pressable
            onPress={() =>
              run(onCash, {
                action: "declare_cash",
                idempotency_key: "ui-change",
                tendered_chf: "50.00",
                change_chf: "5.00",
                change_confirmed: true,
              })
            }
          >
            <Text>Reçu 50.00 CHF, rendre 5.00 CHF, confirmer la monnaie</Text>
          </Pressable>
          <Pressable
            onPress={() =>
              run(onCash, {
                action: "declare_cash",
                idempotency_key: "ui-return",
                tendered_chf: "30.00",
                change_chf: "0.00",
                change_confirmed: false,
                disposition: "returned",
              })
            }
          >
            <Text>30.00 CHF présentés, restituer</Text>
          </Pressable>
        </View>
      ) : null}
      <Text>{`Facture destinée à : ${patient.debtor_label}`}</Text>
      <Text>{`Montant facture : ${patient.amount_chf} CHF`}</Text>
      <Text>{`Information préalable : ${patient.contractual_basis}`}</Text>
      <Text>Facture préparée</Text>
      <Text>Mandat à signer</Text>
      <Pressable
        onPress={() =>
          run(onPatient, {
            action: "prior_information",
            idempotency_key: "ui-prior",
            signatory_role: "patient",
          })
        }
      >
        <Text>{"Présenter l'information préalable"}</Text>
      </Pressable>
      <Pressable
        onPress={() =>
          run(onPatient, {
            action: "refusal",
            idempotency_key: "ui-refusal",
            signatory_role: "patient",
          })
        }
      >
        <Text>Refus de signature</Text>
      </Pressable>
    </View>
  );
}
