import { useMemo, useState } from "react";
import { ActivityIndicator, Pressable, RefreshControl, StyleSheet, View } from "react-native";
import { useRouter } from "expo-router";
import { Screen, AppText } from "../../../../src/design/responsive";
import { useInstitutionOffersQuery } from "../../../../src/features/company/hooks";
import {
  filterVisibleInstitutionOffers,
  segmentInstitutionOffer,
  type InstitutionOfferSegment,
} from "../../../../src/features/company/utils/institutionOfferResponse";
import type { InstitutionRequestOffer } from "../../../../src/features/company/api/institutionOffersApi";
import { E } from "../../../../src/features/company/theme/enterpriseOpsTheme";

const SEGMENT_LABELS: Record<InstitutionOfferSegment, string> = {
  urgent: "Urgentes",
  today: "Aujourd'hui",
  upcoming: "À venir",
};

function OfferRow({
  offer,
  onPress,
}: {
  offer: InstitutionRequestOffer;
  onPress: () => void;
}) {
  const req = offer.transport_request;
  const title = req?.institution_name ?? `Offre #${offer.id}`;
  const subtitle = req?.scheduled_time ?? req?.pickup_location ?? "Horaire à confirmer";
  return (
    <Pressable onPress={onPress} style={({ pressed }) => [s.row, pressed && s.pressed]}>
      <AppText variant="label" style={s.rowTitle}>
        {title}
      </AppText>
      <AppText variant="bodyMuted" numberOfLines={2}>
        {subtitle}
      </AppText>
    </Pressable>
  );
}

export default function InstitutionOffersListScreen() {
  const router = useRouter();
  const { data, isLoading, refetch, isFetching } = useInstitutionOffersQuery("PENDING");

  const grouped = useMemo(() => {
    const visible = filterVisibleInstitutionOffers(data?.offers ?? []);
    const buckets: Record<InstitutionOfferSegment, InstitutionRequestOffer[]> = {
      urgent: [],
      today: [],
      upcoming: [],
    };
    for (const offer of visible) {
      buckets[segmentInstitutionOffer(offer)].push(offer);
    }
    return buckets;
  }, [data?.offers]);

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
    <Screen
      scroll
      backgroundColor={E.BG}
      refreshControl={
        <RefreshControl refreshing={isFetching} onRefresh={() => void refetch()} tintColor={E.BRAND} />
      }
    >
      <AppText variant="sectionTitle" style={s.pageTitle}>
        Demandes institution
      </AppText>
      {isLoading ? <ActivityIndicator color={E.BRAND} style={{ marginTop: 16 }} /> : null}
      {(Object.keys(grouped) as InstitutionOfferSegment[]).map((segment) => {
        const items = grouped[segment];
        if (items.length === 0) return null;
        return (
          <View key={segment} style={s.section}>
            <AppText variant="sectionTitle" style={s.sectionTitle}>
              {SEGMENT_LABELS[segment]}
            </AppText>
            {items.map((offer) => (
              <OfferRow
                key={offer.id}
                offer={offer}
                onPress={() =>
                  openOffer(offer.id, offer.transport_request?.id ?? undefined)
                }
              />
            ))}
          </View>
        );
      })}
      {!isLoading && (data?.offers?.length ?? 0) === 0 ? (
        <AppText variant="bodyMuted" style={s.empty}>
          Aucune demande en attente.
        </AppText>
      ) : null}
    </Screen>
  );
}

const s = StyleSheet.create({
  section: { marginBottom: 16 },
  sectionTitle: { marginBottom: 8, color: E.TEXT },
  row: {
    borderWidth: 1,
    borderColor: E.BORDER,
    borderRadius: 10,
    padding: 12,
    marginBottom: 8,
    backgroundColor: E.CARD,
  },
  pressed: { opacity: 0.85 },
  rowTitle: { marginBottom: 4, color: E.TEXT },
  empty: { marginTop: 16, textAlign: "center" },
  pageTitle: { marginBottom: 12, color: E.TEXT },
});
