import { Pressable, PressableProps, Text } from "react-native";

type ButtonProps = PressableProps & {
  label: string;
  variant?: "primary" | "secondary";
};

export function Button({ label, variant = "secondary", style, ...props }: ButtonProps) {
  const isPrimary = variant === "primary";
  return (
    <Pressable
      {...props}
      style={(state) => [
        {
          borderWidth: 1,
          borderColor: isPrimary ? "#0A8F7A" : "rgba(145, 165, 157, 0.5)",
          backgroundColor: isPrimary ? "#0A8F7A" : "#fff",
          borderRadius: 10,
          paddingHorizontal: 11,
          paddingVertical: 8,
        },
        typeof style === "function" ? style(state) : style,
      ]}
    >
      <Text style={{ color: isPrimary ? "#fff" : "#222", fontWeight: "600" }}>{label}</Text>
    </Pressable>
  );
}

