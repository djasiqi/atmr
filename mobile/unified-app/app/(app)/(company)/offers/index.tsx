import { useMemo, useState } from "react";
import { ActivityIndicator, RefreshControl, StyleSheet, View } from "react-native";
import { useRouter } from "expo-router";
import { isCompanyRealtimeSocketExpected } from "../../../../src/core/featureFlags/registry";
import { Screen, AppText } from "../../../../src/design/responsive";
import { FONT_SIZE } from "../../../../src/design/responsive/typographyTokens";
import { DayPickerSheet } from "../../../../src/features/company/components/DayPickerSheet";
import { EnterpriseHeader } from "../../../../src/features/company/components/EnterpriseHeader";
import { InstitutionOfferListCard } from "../../../../src/features/company/components/offers/InstitutionOfferListCard";
import { LiveSystemStatusPill } from "../../../../src/features/company/components/dashboard/LiveSystemStatusPill";
import { resolveCockpitLiveStatus } from "../../../../src/features/company/dashboard/cockpit/cockpitLiveStatus";
import { useCompanyRealtimeStatus, useInstitutionOffersQuery } from "../../../../src/features/company/hooks";
import {
  filterVisibleInstitutionOffers,
  segmentInstitutionOffer,
  type InstitutionOfferSegment,
} from "../../../../src/features/company/utils/institutionOfferResponse";
import { buildInstitutionOfferListPreview } from "../../../../src/features/company/utils/institutionOfferDisplay";
import { getTodayIsoDateInZurich } from "../../../../src/features/company/utils/companyDateUtils";
import { E } from "../../../../src/features/company/theme/enterpriseOpsTheme";

const SEGMENT_ORDER: InstitutionOfferSegment[] = ["urgent", "today", "upcoming"];

const SEGMENT_LABELS: Record<InstitutionOfferSegment, string> = {
  urgent: "Urgentes",
  today: "Aujourd'hui",
  upcoming: "À venir",
};

export default function InstitutionOffersListScreen() {
  const router = useRouter();
  const [selectedDate, setSelectedDate] = useState(() => getTodayIsoDateInZurich());
  const [dateSheetOpen, setDateSheetOpen] = useState(false);
  const realtime = useCompanyRealtimeStatus();
  const realtimeSocketExpected = isCompanyRealtimeSocketExpected();
  const liveStatus = useMemo(
    () => resolveCockpitLiveStatus(realtime.transportStatus, { realtimeSocketExpected }),
    [realtime.transportStatus, realtimeSocketExpected]
  );
  const { data, isLoading, refetch, isFetching } = useInstitutionOffersQuery("PENDING");

  const grouped = useMemo(() => {
    const visible = filterVisibleInstitutionOffers(data?.offers ?? []);
    const buckets: Record<InstitutionOfferSegment, typeof visible> = {
      urgent: [],
      today: [],
      upcoming: [],
    };
    for (const offer of visible) {
      buckets[segmentInstitutionOffer(offer)].push(offer);
    }
    return buckets;
  }, [data?.offers]);

  const totalVisible = useMemo(
    () => Object.values(grouped).reduce((sum, items) => sum + items.length, 0),
    [grouped]
  );

  const openOffer = (offerId: number, requestId?: number) => {
    router.push({
      pathname: "/(app)/(company)/offers/[offerId]",
      params: {
        offerId: String(offerId),
        ...(requestId != null ? { request: String(requestId) } : {}),
      },
    });
  };

  return (
    <>
      <Screen
        scroll
        backgroundColor={E.BG}
        contentContainerStyle={s.scrollContent}
        stickyHeader={
          <EnterpriseHeader
            metaDetail="networkOnly"
            date={selectedDate}
            mode={null}
            realtimeStatus={realtime.transportStatus}
            showModeChip={false}
            onOpenDatePicker={() => setDateSheetOpen(true)}
            liveStatusPill={
              <LiveSystemStatusPill
                variant="header"
                status={liveStatus}
                realtimeSocketExpected={realtimeSocketExpected}
                dataFreshness={
                  liveStatus === "connected" &&
                  (realtime.dataFreshness === "fresh" ||
                    realtime.dataFreshness === "idle" ||
                    realtime.dataFreshness === "stale")
                    ? realtime.dataFreshness
                    : undefined
                }
              />
            }
          />
        }
        refreshControl={
          <RefreshControl refreshing={isFetching} onRefresh={() => void refetch()} tintColor={E.BRAND} />
        }
      >
        <AppText style={s.pageTitle}>Demandes institution</AppText>

        {isLoading ? <ActivityIndicator color={E.BRAND} style={s.loader} /> : null}

        {SEGMENT_ORDER.map((segment) => {
          const items = grouped[segment];
          if (items.length === 0) return null;
          return (
            <View key={segment} style={s.section}>
              <View style={s.sectionHead}>
                <AppText style={[s.sectionTitle, segment === "urgent" && s.sectionTitleUrgent]}>
                  {SEGMENT_LABELS[segment]}
                </AppText>
                <View style={[s.sectionCount, segment === "urgent" && s.sectionCountUrgent]}>
                  <AppText
                    style={[s.sectionCountText, segment === "urgent" && s.sectionCountTextUrgent]}
                  >
                    {items.length}
                  </AppText>
                </View>
              </View>
              {items.map((offer) => (
                <InstitutionOfferListCard
                  key={offer.id}
                  segment={segment}
                  preview={buildInstitutionOfferListPreview(offer.transport_request)}
                  onPress={() => openOffer(offer.id, offer.transport_request?.id ?? undefined)}
                />
              ))}
            </View>
          );
        })}

        {!isLoading && totalVisible === 0 ? (
          <AppText variant="bodyMuted" style={s.empty}>
            Aucune demande en attente.
          </AppText>
        ) : null}
      </Screen>

      <DayPickerSheet
        visible={dateSheetOpen}
        selectedDate={selectedDate}
        onClose={() => setDateSheetOpen(false)}
        onSelectDate={(iso) => {
          setSelectedDate(iso);
          setDateSheetOpen(false);
        }}
      />
    </>
  );
}

const s = StyleSheet.create({
  scrollContent: {
    paddingTop: 12,
  },
  pageTitle: {
    color: E.TEXT,
    fontSize: FONT_SIZE.px18,
    fontWeight: "700",
    lineHeight: 22,
    marginBottom: 14,
  },
  loader: { marginTop: 8, marginBottom: 12 },
  section: { marginBottom: 18 },
  sectionHead: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginBottom: 10,
  },
  sectionTitle: {
    color: E.TEXT_SEC,
    fontSize: FONT_SIZE.px12,
    fontWeight: "700",
    letterSpacing: 0.6,
    textTransform: "uppercase",
  },
  sectionTitleUrgent: {
    color: E.URGENT,
  },
  sectionCount: {
    minWidth: 22,
    height: 22,
    paddingHorizontal: 7,
    borderRadius: 11,
    backgroundColor: "rgba(148, 163, 184, 0.14)",
    alignItems: "center",
    justifyContent: "center",
  },
  sectionCountUrgent: {
    backgroundColor: "rgba(239, 68, 68, 0.12)",
  },
  sectionCountText: {
    color: E.TEXT_SEC,
    fontSize: FONT_SIZE.px11,
    fontWeight: "700",
  },
  sectionCountTextUrgent: {
    color: E.URGENT,
  },
  empty: { marginTop: 24, textAlign: "center" },
});
