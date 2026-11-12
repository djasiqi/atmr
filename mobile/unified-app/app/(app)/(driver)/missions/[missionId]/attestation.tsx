import { View } from "react-native";
import { AppText, Screen } from "../../../../../src/design/responsive";

/**
 * Stub d'attestation pour FIN-CAISSE : hors périmètre règlement / caisse.
 * Pas de panneau d'encaissement ni d'entrée mission de règlement.
 */
export default function DriverAttestationScreen() {
  return (
    <Screen>
      <View>
        <AppText variant="body">Attestation hors lot FIN-CAISSE.</AppText>
      </View>
    </Screen>
  );
}
