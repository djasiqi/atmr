import { useRef, useState } from "react";
import { PanResponder, StyleSheet, View } from "react-native";
import { LirieButton } from "../../../design/foundations/LirieButton";
import { LirieText } from "../../../design/foundations/LirieText";
import { lirieColor, lirieRadius } from "../../../design/foundations/tokens";

export type TracePoint = { x: number; y: number };

/**
 * Tracé tactile de démonstration. Il n'est pas une signature juridique.
 */
export function SpecimenTrace({
  onChange,
  initialPoints = [],
}: {
  onChange: (points: TracePoint[]) => void;
  initialPoints?: TracePoint[];
}) {
  const [points, setPoints] = useState<TracePoint[]>(initialPoints);
  const pointsRef = useRef(initialPoints);
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;

  function push(x: number, y: number) {
    const next = [...pointsRef.current, { x: Math.round(x), y: Math.round(y) }].slice(-400);
    pointsRef.current = next;
    setPoints(next);
    onChangeRef.current(next);
  }

  const responder = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => true,
      onMoveShouldSetPanResponder: () => true,
      onPanResponderGrant: (event) => {
        push(event.nativeEvent.locationX, event.nativeEvent.locationY);
      },
      onPanResponderMove: (event) => {
        push(event.nativeEvent.locationX, event.nativeEvent.locationY);
      },
    })
  ).current;

  function erase() {
    pointsRef.current = [];
    setPoints([]);
    onChangeRef.current([]);
  }

  return (
    <View style={styles.stack}>
      <LirieText variant="message">Signature de démonstration</LirieText>
      <View
        accessibilityLabel="Zone de signature de démonstration"
        style={styles.pad}
        {...responder.panHandlers}
      >
        {points.map((point, index) => (
          <View
            key={`${point.x}-${point.y}-${index}`}
            style={[styles.dot, { left: point.x, top: point.y }]}
          />
        ))}
      </View>
      <LirieButton
        variant="secondary"
        accessibilityLabel="Effacer le spécimen"
        label="Effacer"
        onPress={erase}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  stack: { gap: 8 },
  pad: {
    minHeight: 140,
    borderWidth: 1,
    borderColor: lirieColor.border,
    borderRadius: lirieRadius.card,
    backgroundColor: lirieColor.white,
  },
  dot: {
    position: "absolute",
    width: 4,
    height: 4,
    borderRadius: 2,
    backgroundColor: lirieColor.ink,
  },
});
