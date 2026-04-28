import { TextInput, TextInputProps } from "react-native";

export function InputField(props: TextInputProps) {
  return (
    <TextInput
      {...props}
      style={[
        {
          borderWidth: 1,
          borderColor: "#ddd",
          borderRadius: 10,
          paddingHorizontal: 10,
          paddingVertical: 8,
        },
        props.style,
      ]}
    />
  );
}

