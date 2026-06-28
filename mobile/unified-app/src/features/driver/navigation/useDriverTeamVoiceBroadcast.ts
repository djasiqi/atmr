import { useCallback, useEffect, useRef, useState } from "react";
import { Linking, Platform } from "react-native";
import { useQueryClient } from "@tanstack/react-query";
import { useChatVoiceRecorder } from "../../chat/services/audioAdapter";
import {
  invalidateDriverHubScope,
  useDriverCompanyId,
} from "../messages/hooks";
import { MESSAGE_HUB_THREAD_TEAM } from "../messages/contracts";
import { sendDriverHubVoiceMessage } from "../messages/sendDriverHubVoiceMessage";

const MIN_VOICE_MS = 450;

export type DriverTeamVoiceFeedback = {
  message: string;
  openSettings?: boolean;
};

export function useDriverTeamVoiceBroadcast(options?: {
  onFeedback?: (feedback: DriverTeamVoiceFeedback | null) => void;
}) {
  const companyId = useDriverCompanyId();
  const queryClient = useQueryClient();
  const { startRecording, stopRecording, abortRecording } = useChatVoiceRecorder();
  const [isRecording, setIsRecording] = useState(false);
  const [voiceBusy, setVoiceBusy] = useState(false);
  const voiceInteractionEpochRef = useRef(0);
  const pressStartMsRef = useRef(0);

  const disabled = Platform.OS === "web";

  const onFeedbackRef = useRef(options?.onFeedback);
  onFeedbackRef.current = options?.onFeedback;

  const clearFeedback = useCallback(() => {
    onFeedbackRef.current?.(null);
  }, []);

  const reportError = useCallback((message: string, openSettings = false) => {
    onFeedbackRef.current?.({ message, openSettings });
  }, []);

  const finalizeRecording = useCallback(async () => {
    if (voiceBusy || !companyId) return;
    setVoiceBusy(true);
    setIsRecording(false);
    const elapsed = Date.now() - pressStartMsRef.current;
    const result = await stopRecording();
    try {
      if (!result.ok) {
        if (result.reason !== "no_active_recording" && result.reason !== "aborted") {
          reportError("Impossible d'enregistrer le message vocal.");
        }
        return;
      }
      if (!result.data || elapsed < MIN_VOICE_MS) {
        reportError("Enregistrement trop court. Appuyez sur le micro et parlez.");
        return;
      }
      await sendDriverHubVoiceMessage(result.data, { companyId });
      invalidateDriverHubScope(queryClient, companyId, {
        threadId: MESSAGE_HUB_THREAD_TEAM,
        includeMessages: true,
      });
      clearFeedback();
    } catch {
      reportError("Impossible d'envoyer le message vocal au canal équipe.");
    } finally {
      setVoiceBusy(false);
    }
  }, [clearFeedback, companyId, queryClient, reportError, stopRecording, voiceBusy]);

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
              true
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
      void abortRecording();
    };
  }, [abortRecording]);

  return {
    disabled,
    isRecording,
    voiceBusy,
    handlePress,
  };
}
