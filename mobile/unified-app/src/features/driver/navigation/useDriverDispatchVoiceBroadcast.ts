import { useCallback, useEffect, useRef, useState } from "react";
import { Linking, Platform } from "react-native";
import { useQueryClient } from "@tanstack/react-query";
import { useChatVoiceRecorder } from "../../chat/services/audioAdapter";
import { invalidateDriverHubScope, useDriverCompanyId } from "../messages/hooks";
import { MESSAGE_HUB_THREAD_DISPATCH } from "../messages/contracts";
import { sendBottomBarDispatchVoiceMessage } from "./bottomBarDispatchVoice";

const MIN_VOICE_MS = 450;

export type DriverDispatchVoiceFeedback = {
  message: string;
  tone: "success" | "error";
  openSettings?: boolean;
  onRetry?: () => void;
};

export function useDriverDispatchVoiceBroadcast(options?: {
  onFeedback?: (feedback: DriverDispatchVoiceFeedback | null) => void;
}) {
  const companyId = useDriverCompanyId();
  const queryClient = useQueryClient();
  const { startRecording, stopRecording, abortRecording } = useChatVoiceRecorder();
  const abortRecordingRef = useRef(abortRecording);
  abortRecordingRef.current = abortRecording;
  const [isRecording, setIsRecording] = useState(false);
  const [voiceBusy, setVoiceBusy] = useState(false);
  const voiceInteractionEpochRef = useRef(0);
  const pressStartMsRef = useRef(0);
  const retryUriRef = useRef<string | null>(null);

  const disabled = Platform.OS === "web";

  const onFeedbackRef = useRef(options?.onFeedback);
  onFeedbackRef.current = options?.onFeedback;

  const clearFeedback = useCallback(() => {
    onFeedbackRef.current?.(null);
  }, []);

  const reportError = useCallback((message: string, extras?: { openSettings?: boolean }) => {
    onFeedbackRef.current?.({
      message,
      tone: "error",
      openSettings: extras?.openSettings,
    });
  }, []);

  const publishDispatchVoice = useCallback(
    async (localUri: string) => {
      if (!companyId) {
        reportError("Connexion indisponible. Réessayez.");
        return;
      }
      try {
        await sendBottomBarDispatchVoiceMessage(localUri, { companyId });
        retryUriRef.current = null;
        invalidateDriverHubScope(queryClient, companyId, {
          threadId: MESSAGE_HUB_THREAD_DISPATCH,
          includeMessages: true,
        });
        onFeedbackRef.current?.({
          message: "✓ Audio envoyé dans Dispatch",
          tone: "success",
        });
      } catch {
        retryUriRef.current = localUri;
        onFeedbackRef.current?.({
          message: "Audio non envoyé. Réessayer",
          tone: "error",
          onRetry: () => {
            const uri = retryUriRef.current;
            if (!uri || voiceBusy) return;
            void (async () => {
              setVoiceBusy(true);
              try {
                await publishDispatchVoice(uri);
              } finally {
                setVoiceBusy(false);
              }
            })();
          },
        });
      }
    },
    [companyId, queryClient, reportError, voiceBusy]
  );

  const finalizeRecording = useCallback(async () => {
    if (voiceBusy || !companyId) return;
    setVoiceBusy(true);
    setIsRecording(false);
    const elapsed = Date.now() - pressStartMsRef.current;
    const result = await stopRecording();
    try {
      if (!result.ok) {
        if (result.reason === "no_active_recording" || result.reason === "aborted") {
          reportError(
            "Enregistrement non démarré. Appuyez sur le micro, parlez, puis appuyez à nouveau pour envoyer."
          );
          return;
        }
        reportError("Impossible d'enregistrer le message vocal.");
        return;
      }
      if (!result.data || elapsed < MIN_VOICE_MS) {
        reportError("Enregistrement trop court. Appuyez sur le micro et parlez.");
        return;
      }
      await publishDispatchVoice(result.data);
    } finally {
      setVoiceBusy(false);
    }
  }, [companyId, publishDispatchVoice, reportError, stopRecording, voiceBusy]);

  const startRecordingSession = useCallback(
    async (attemptEpoch: number) => {
      if (!companyId) {
        reportError("Connexion indisponible. Réessayez.");
        return;
      }
      pressStartMsRef.current = Date.now();
      try {
        const started = await startRecording({
          isAborted: () => voiceInteractionEpochRef.current !== attemptEpoch,
        });
        if (!started.ok) {
          setIsRecording(false);
          if (started.reason === "permission_denied") {
            void Linking.openSettings();
            reportError(
              "Activez le micro dans les réglages du téléphone pour envoyer des messages vocaux.",
              { openSettings: true }
            );
          } else if (started.reason !== "aborted") {
            reportError("Impossible de démarrer l'enregistrement vocal.");
          }
          return;
        }
        clearFeedback();
        setIsRecording(true);
      } catch {
        await abortRecording();
        setIsRecording(false);
      }
    },
    [abortRecording, clearFeedback, companyId, reportError, startRecording]
  );

  const handlePress = useCallback(() => {
    if (disabled || voiceBusy) return;
    if (isRecording) {
      void finalizeRecording();
      return;
    }
    voiceInteractionEpochRef.current += 1;
    const attemptEpoch = voiceInteractionEpochRef.current;
    void startRecordingSession(attemptEpoch);
  }, [disabled, finalizeRecording, isRecording, startRecordingSession, voiceBusy]);

  useEffect(() => {
    return () => {
      voiceInteractionEpochRef.current += 1;
      void abortRecordingRef.current();
    };
  }, []);

  return {
    disabled,
    isRecording,
    voiceBusy,
    handlePress,
  };
}
