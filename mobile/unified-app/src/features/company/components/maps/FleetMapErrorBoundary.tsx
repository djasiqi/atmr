import React, { Component, type ErrorInfo, type ReactNode } from "react";
import { StyleSheet, Text, View } from "react-native";
import * as Sentry from "@sentry/react-native";

type Props = {
  children: ReactNode;
  fallbackMessage?: string;
};

type State = {
  hasError: boolean;
};

export class FleetMapErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false };

  static getDerivedStateFromError(): State {
    return { hasError: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    Sentry.captureException(error, {
      tags: { component: "fleet_map" },
      extra: { componentStack: info.componentStack },
    });
  }

  render(): ReactNode {
    if (this.state.hasError) {
      return (
        <View style={styles.fallback} accessibilityRole="alert">
          <Text style={styles.title}>Carte temporairement indisponible</Text>
          <Text style={styles.body}>
            {this.props.fallbackMessage ??
              "Les positions chauffeurs restent accessibles via la liste. Réessayez dans un instant."}
          </Text>
        </View>
      );
    }
    return this.props.children;
  }
}

const styles = StyleSheet.create({
  fallback: {
    flex: 1,
    minHeight: 120,
    justifyContent: "center",
    alignItems: "center",
    padding: 20,
    backgroundColor: "#EAF3F1",
  },
  title: {
    fontSize: 16,
    fontWeight: "700",
    color: "#0F172A",
    marginBottom: 8,
    textAlign: "center",
  },
  body: {
    fontSize: 14,
    color: "#64748B",
    textAlign: "center",
  },
});
