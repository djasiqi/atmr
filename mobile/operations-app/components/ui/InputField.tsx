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
  Text,
} from 'react-native';
import Ionicons from 'react-native-vector-icons/Ionicons';

export interface InputFieldProps extends Omit<TextInputProps, 'style'> {
  showToggle?: boolean;
  label?: string;
  containerStyle?: StyleProp<ViewStyle>;
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
      {label && <Text style={styles.label}>{label}</Text>}
      <TextInput
        {...props}
        secureTextEntry={secure}
        placeholderTextColor="#94A3B8"
        onLayout={onLayout}
        style={[
          styles.input,
          { paddingRight: showToggle ? 40 : 12, height: Math.max(42, inputHeight) },
          inputStyle,
        ]}
      />
      {showToggle && (
        <TouchableOpacity
          style={[styles.iconButton, { height: Math.max(42, inputHeight) }]}
          onPress={() => setSecure((s) => !s)}
        >
          <Ionicons name={secure ? 'eye-off' : 'eye'} size={20} color="#64748B" />
        </TouchableOpacity>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    position: 'relative',
    marginVertical: 6,
  },
  label: {
    fontSize: 12,
    fontWeight: '600',
    color: '#64748B',
    marginBottom: 4,
    letterSpacing: 0.2,
    textTransform: 'uppercase',
  },
  input: {
    borderWidth: 1,
    borderColor: 'rgba(0,121,107,0.12)',
    borderRadius: 10,
    paddingLeft: 12,
    fontSize: 14,
    color: '#1E293B',
    backgroundColor: '#f8fafc',
  },
  iconButton: {
    position: 'absolute',
    right: 8,
    width: 32,
    justifyContent: 'center',
    alignItems: 'center',
    bottom: 0,
  },
});
