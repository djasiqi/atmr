import { Stack } from "expo-router";

export default function PublicLayout() {
  return (
    <Stack screenOptions={{ headerShown: false }}>
      <Stack.Screen name="index" />
      <Stack.Screen name="onboarding-step-1" />
      <Stack.Screen name="onboarding-step-2" />
      <Stack.Screen name="onboarding-step-3" />
      <Stack.Screen name="choice-guest-signup" />
      <Stack.Screen name="how-it-works" />
      <Stack.Screen name="why-create-account" />
      <Stack.Screen name="booking-status" />
      <Stack.Screen name="service-area" />
      <Stack.Screen name="pre-request/step-1" />
      <Stack.Screen name="pre-request/step-2" />
      <Stack.Screen name="pre-request/auth-gate" />
      <Stack.Screen name="pre-request/guest-checkout" />
      <Stack.Screen name="fallback/expired-link" />
      <Stack.Screen name="fallback/invalid-link" />
      <Stack.Screen name="fallback/auth-required" />
      <Stack.Screen name="fallback/resume-later" />
      <Stack.Screen name="fallback/install-app" />
      <Stack.Screen name="login" />
      <Stack.Screen name="passwordless-otp" />
      <Stack.Screen name="signup" />
      <Stack.Screen name="activate" />
      <Stack.Screen name="forgot-password" />
      <Stack.Screen name="reset-password" />
      <Stack.Screen name="mfa" />
      <Stack.Screen name="contact" />
      <Stack.Screen name="help" />
    </Stack>
  );
}
