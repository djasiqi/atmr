// components/ui/InputField.tsx
import React, { useState, useRef } from 'react';
import {
  View,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  TextInputProps,
  LayoutChangeEvent,
  StyleProp,
  ViewStyle,
  TextStyle,
  Text, // Ajout de l'import Text
} from 'react-native';
import Ionicons from 'react-native-vector-icons/Ionicons';

export interface InputFieldProps extends Omit<TextInputProps, 'style'> {
  /** Affiche ou non le bouton œil */
  showToggle?: boolean;
  /** Label affiché au-dessus du champ */
  label?: string;
  /** Styles appliqués au conteneur (View) */
  containerStyle?: StyleProp<ViewStyle>;
  /** Styles appliqués au TextInput */
  inputStyle?: StyleProp<TextStyle>;
}

export function InputField({
  secureTextEntry = false,
  showToggle = true,
  label,
  containerStyle,
  inputStyle,
  ...props
}: InputFieldProps) {
  const [secure, setSecure] = useState(secureTextEntry);
  const [inputHeight, setInputHeight] = useState(0);

  const onLayout = (e: LayoutChangeEvent) => {
    setInputHeight(e.nativeEvent.layout.height);
  };

  return (
    <View style={[styles.container, containerStyle]}>
      {label && (
        <Text style={styles.label}>{label}</Text>
      )}
      <TextInput
        {...props}
        secureTextEntry={secure}
        placeholderTextColor="#999"
        onLayout={onLayout}
        style={[
          styles.input,
          { paddingRight: showToggle ? 40 : 12, height: Math.max(44, inputHeight) },
          inputStyle,
        ]}
      />
      {showToggle && (
        <TouchableOpacity
          style={[styles.iconButton, { height: Math.max(44, inputHeight) }]}
          onPress={() => setSecure((s) => !s)}
        >
          <Ionicons name={secure ? 'eye-off' : 'eye'} size={24} color="#000" />
        </TouchableOpacity>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    position: 'relative',
    marginVertical: 8,
  },
  label: {
    fontSize: 14,
    fontWeight: '500',
    color: '#374151',
    marginBottom: 4,
  },
  input: {
    borderWidth: 1,
    borderColor: '#CCC',
    borderRadius: 8,
    paddingLeft: 12,
    fontSize: 16,
    color: '#000',
    backgroundColor: '#FFF',
  },
  iconButton: {
    position: 'absolute',
    right: 8,
    width: 32,
    justifyContent: 'center',
    alignItems: 'center',
  },
});
