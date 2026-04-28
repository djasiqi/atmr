import { PropsWithChildren } from "react";
import { View, ViewProps } from "react-native";

export function Card({ children, style, ...props }: PropsWithChildren<ViewProps>) {
  return (
    <View
      {...props}
      style={[
        {
          borderWidth: 1,
          borderColor: "#ddd",
          borderRadius: 10,
          padding: 12,
          gap: 6,
          backgroundColor: "#fff",
        },
        style,
      ]}
    >
      {children}
    </View>
  );
}

