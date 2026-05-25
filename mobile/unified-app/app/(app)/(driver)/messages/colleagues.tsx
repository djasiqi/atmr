import { useMemo } from "react";

import { ActivityIndicator, Pressable, ScrollView, StyleSheet, View } from "react-native";

import { useRouter } from "expo-router";

import { useQuery } from "@tanstack/react-query";

import { DriverContextGuard, PermissionGuard } from "../../../../src/core/guards";

import { Screen, AppText, useAppViewport } from "../../../../src/design/responsive";

import { D } from "../../../../src/features/driver/theme/driverDashboardTheme";

import { InboxThreadRow } from "../../../../src/features/driver/messages/components/InboxThreadRow";

import {

  useDriverCompanyId,

  useMessageHubThreads,

} from "../../../../src/features/driver/messages/hooks";

import { directThreadId } from "../../../../src/features/driver/messages/contracts";

import { fetchHubColleagues } from "../../../../src/features/driver/messages/api";

import type { MessageHubThread } from "../../../../src/features/driver/messages/types";



function rosterToThreads(

  roster: { peer_user_id: number; title: string; thread_id: string }[]

): MessageHubThread[] {

  return roster.map((row) => ({

    thread_id: row.thread_id,

    section: "colleagues",

    title: row.title,

    subtitle: "Message direct",

    peer_user_id: row.peer_user_id,

    booking_id: null,

    status: null,

    unread_count: 0,

    priority: "normal",

    last_message_preview: "Démarrer une conversation",

    last_message_at: null,

  }));

}



export default function DriverColleaguesPickerScreen() {

  const router = useRouter();

  const { horizontalPadding } = useAppViewport();

  const companyId = useDriverCompanyId();

  const threadsQuery = useMessageHubThreads(companyId);



  const rosterQuery = useQuery({

    queryKey: ["driver", "message-hub", "colleagues-roster", companyId ?? "none"],

    enabled: Boolean(companyId),

    queryFn: async () => (await fetchHubColleagues(companyId as number)).colleagues,

    staleTime: 60_000,

  });



  const colleagues = useMemo(() => {

    const fromHub = (threadsQuery.data?.threads ?? []).filter((t) => t.section === "colleagues");

    if (fromHub.length > 0) return fromHub;



    const roster = rosterQuery.data ?? [];

    if (roster.length > 0) return rosterToThreads(roster);



    return [];

  }, [rosterQuery.data, threadsQuery.data?.threads]);



  const openColleague = (thread: MessageHubThread) => {

    const peerId = thread.peer_user_id;

    router.push({

      pathname: "/(app)/(driver)/messages/[threadId]",

      params: {

        threadId: peerId != null ? directThreadId(peerId) : thread.thread_id,

      },

    });

  };



  const loading = threadsQuery.isLoading || rosterQuery.isLoading;



  return (

    <DriverContextGuard>

      <PermissionGuard permission="chat:read">

        <Screen scroll={false} backgroundColor={D.pageBg}>

          <View style={{ paddingHorizontal: horizontalPadding, paddingTop: 12, gap: 8, flex: 1 }}>

            <AppText variant="bodyMuted">

              Choisissez un collègue pour démarrer ou reprendre une conversation directe.

            </AppText>

            <Pressable

              onPress={() =>

                router.push({

                  pathname: "/(app)/(driver)/messages/[threadId]",

                  params: { threadId: "team" },

                })

              }

              style={styles.teamLink}

            >

              <AppText variant="caption" style={styles.teamLinkText}>

                Ouvrir le canal Équipe chauffeurs

              </AppText>

            </Pressable>

            {loading ? (

              <ActivityIndicator color="#0A8F7A" style={{ marginTop: 24 }} />

            ) : (

              <ScrollView contentContainerStyle={{ paddingBottom: 32 }}>

                {colleagues.length === 0 ? (

                  <AppText variant="bodyMuted" style={styles.empty}>

                    Aucun autre chauffeur actif dans votre entreprise pour le moment.

                  </AppText>

                ) : (

                  colleagues.map((thread) => (

                    <InboxThreadRow

                      key={thread.thread_id}

                      thread={thread}

                      onPress={() => openColleague(thread)}

                    />

                  ))

                )}

              </ScrollView>

            )}

          </View>

        </Screen>

      </PermissionGuard>

    </DriverContextGuard>

  );

}



const styles = StyleSheet.create({

  empty: { textAlign: "center", marginTop: 32 },

  teamLink: {

    alignSelf: "flex-start",

    paddingVertical: 6,

    paddingHorizontal: 10,

    backgroundColor: "#ecfdf5",

    borderRadius: 8,

  },

  teamLinkText: { color: "#047857", fontWeight: "600" },

});

