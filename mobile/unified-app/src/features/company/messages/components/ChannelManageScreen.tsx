import { useMemo, useRef, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Image,
  Linking,
  Modal,
  Pressable,
  ScrollView,
  StyleSheet,
  TextInput,
  View,
} from "react-native";
import { useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { resolveMediaUrl } from "../../../../core/api/mediaUrl";
import { AppText } from "../../../../design/ui/AppText";
import { M } from "../../../messaging/messagingTheme";
import { ConfirmDialogModal } from "../../../messaging/components/ConfirmDialogModal";
import { useDispatchChannelManage } from "../useDispatchChannelManage";
import type {
  AvailableDriverRow,
  ChannelHistoryEntry,
  ChannelManagePayload,
  ChannelManagePermissions,
  ConversationAttachmentRow,
  ConversationParticipantRow,
} from "../conversationManageApi";
import { FONT_SIZE } from "../../../../design/responsive/typographyTokens";

function formatFrenchDate(iso?: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleDateString("fr-FR", { day: "numeric", month: "long", year: "numeric" });
}

function formatRelativeActivity(iso?: string | null): string {
  if (!iso) return "Aucune activité récente";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "Activité inconnue";
  const diffMs = Date.now() - d.getTime();
  const mins = Math.floor(diffMs / 60_000);
  if (mins < 1) return "Actif à l'instant";
  if (mins < 60) return `Dernière activité il y a ${mins} min`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `Dernière activité il y a ${hours} h`;
  return `Dernière activité le ${d.toLocaleDateString("fr-FR")}`;
}

function PermissionRow({ label, enabled }: { label: string; enabled: boolean }) {
  return (
    <View style={styles.permissionRow}>
      <Ionicons
        name={enabled ? "checkbox" : "square-outline"}
        size={20}
        color={enabled ? M.BRAND : M.TEXT_MUTED}
      />
      <AppText variant="body" style={styles.permissionLabel}>
        {label}
      </AppText>
    </View>
  );
}

type Props = {
  conversationId: number;
  threadId: string;
  topInset: number;
  horizontalPadding: number;
};

export function ChannelManageScreen({
  conversationId,
  threadId,
  topInset,
  horizontalPadding,
}: Props) {
  const router = useRouter();
  const scrollRef = useRef<ScrollView>(null);
  const [participantsOpen, setParticipantsOpen] = useState(true);
  const [showAllMedia, setShowAllMedia] = useState(false);
  const [editOpen, setEditOpen] = useState(false);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [editTitle, setEditTitle] = useState("");
  const [editDescription, setEditDescription] = useState("");
  const [removeTarget, setRemoveTarget] = useState<ConversationParticipantRow | null>(null);
  const [clearHistoryOpen, setClearHistoryOpen] = useState(false);

  const { detailQuery, updateChannel, addParticipant, removeParticipant, clearHistory } =
    useDispatchChannelManage(conversationId);

  const data: ChannelManagePayload | undefined = detailQuery.data;
  const channel = data?.channel;
  const canManage = data?.can_manage ?? false;

  const previewAttachments = useMemo(() => {
    const all = data?.attachments_all ?? data?.attachments_preview ?? [];
    return all.slice(0, 4);
  }, [data?.attachments_all, data?.attachments_preview]);

  const openEdit = () => {
    if (!channel) return;
    setEditTitle(channel.title);
    setEditDescription(channel.description);
    setEditOpen(true);
  };

  const saveEdit = () => {
    void updateChannel
      .mutateAsync({
        title: editTitle.trim(),
        description: editDescription.trim(),
      })
      .then(() => setEditOpen(false))
      .catch(() => Alert.alert("Erreur", "Impossible de mettre à jour le canal."));
  };

  const handleAdd = (driver: AvailableDriverRow) => {
    setPickerOpen(false);
    void addParticipant.mutateAsync(driver.driver_id).catch(() => {
      Alert.alert("Erreur", "Impossible d'ajouter ce participant.");
    });
  };

  const confirmRemove = (row: ConversationParticipantRow) => {
    setRemoveTarget(row);
  };

  const handleConfirmRemove = () => {
    if (!removeTarget) return;
    void removeParticipant
      .mutateAsync(removeTarget.user_id)
      .then(() => setRemoveTarget(null))
      .catch(() => {
        Alert.alert("Erreur", "Impossible de retirer ce participant.");
      });
  };

  const confirmClearHistory = () => {
    setClearHistoryOpen(true);
  };

  const handleConfirmClearHistory = () => {
    void clearHistory
      .mutateAsync()
      .then(() => {
        setClearHistoryOpen(false);
        Alert.alert("Historique vidé", "Les messages du canal ont été supprimés.");
      })
      .catch(() => {
        Alert.alert("Erreur", "Impossible de vider l'historique du canal.");
      });
  };

  if (detailQuery.isLoading && !data) {
    return (
      <View style={styles.loadingWrap}>
        <ActivityIndicator size="large" color={M.BRAND} />
      </View>
    );
  }

  const errorDetail =
    detailQuery.error instanceof Error ? detailQuery.error.message : null;

  if (detailQuery.isError || !channel?.id) {
    return (
      <View style={[styles.loadingWrap, { paddingTop: topInset }]}>
        <Ionicons name="cloud-offline-outline" size={40} color={M.TEXT_MUTED} />
        <AppText variant="body" style={styles.errorTitle}>
          Impossible de charger le canal
        </AppText>
        <AppText variant="bodyMuted" style={styles.errorHint}>
          {errorDetail ?? "Vérifiez la connexion puis réessayez."}
        </AppText>
        <Pressable style={styles.retryBtn} onPress={() => void detailQuery.refetch()}>
          <AppText variant="label" style={styles.retryBtnText}>
            Réessayer
          </AppText>
        </Pressable>
        <Pressable onPress={() => router.back()} style={{ marginTop: 12 }}>
          <AppText variant="label" style={styles.linkText}>
            Retour au fil
          </AppText>
        </Pressable>
      </View>
    );
  }

  return (
    <View style={styles.screenRoot}>
      <View style={[styles.topBar, { paddingTop: topInset + 4, paddingHorizontal: horizontalPadding }]}>
        <Pressable onPress={() => router.back()} style={styles.backBtn} accessibilityLabel="Retour">
          <Ionicons name="chevron-back" size={26} color={M.BRAND} />
        </Pressable>
        <View style={{ flex: 1 }} />
        <Pressable
          style={styles.moreBtn}
          onPress={() =>
            Alert.alert("Actions", undefined, [
              canManage ? { text: "Modifier le canal", onPress: openEdit } : undefined,
              { text: "Retour au fil", onPress: () => router.back() },
              { text: "Annuler", style: "cancel" },
            ].filter(Boolean) as { text: string; onPress?: () => void; style?: "cancel" }[])
          }
        >
          <Ionicons name="ellipsis-vertical" size={22} color={M.TEXT} />
        </Pressable>
      </View>

      <ScrollView
        ref={scrollRef}
        style={styles.scroll}
        contentContainerStyle={{ paddingBottom: 48 }}
        showsVerticalScrollIndicator={false}
      >
        {/* Header type WhatsApp — fond blanc pleine largeur */}
        <View style={styles.waHero}>
          <View style={styles.avatarRing}>
            <View style={styles.avatar}>
              <Ionicons name="business" size={40} color="#fff" />
            </View>
            {canManage ? (
              <Pressable style={styles.avatarEdit} accessibilityLabel="Changer l'icône">
                <Ionicons name="camera" size={14} color="#fff" />
              </Pressable>
            ) : null}
          </View>
          <AppText variant="sectionTitle" style={styles.heroTitle}>
            {channel.title}
          </AppText>
          <AppText variant="bodyMuted" style={styles.heroSubtitle}>
            {channel.description}
          </AppText>
          <AppText variant="caption" style={styles.heroMeta}>
            Canal · {channel.participant_count} participant{channel.participant_count > 1 ? "s" : ""} ·{" "}
            {channel.channel_type_label}
          </AppText>
        </View>

        {/* Barre d'actions horizontale (structure WhatsApp) */}
        <View style={[styles.waQuickBar, { paddingHorizontal: horizontalPadding }]}>
          <WaQuickPill icon="create-outline" label="Modifier" onPress={openEdit} disabled={!canManage} />
          <WaQuickPill icon="person-add-outline" label="Ajouter" onPress={() => setPickerOpen(true)} disabled={!canManage} />
          <WaQuickPill icon="attach-outline" label="Fichiers" onPress={() => setShowAllMedia(true)} />
          <WaQuickPill
            icon="search-outline"
            label="Rechercher"
            onPress={() =>
              router.push({
                pathname: "/(app)/(company)/messages/[threadId]",
                params: { threadId },
              })
            }
          />
        </View>

        <View style={styles.waSectionGap} />

        {/* Médias — ligne liste + aperçu */}
        <Pressable style={styles.waListRow} onPress={() => setShowAllMedia(true)}>
          <View style={styles.waListIcon}>
            <Ionicons name="images-outline" size={20} color={M.BRAND} />
          </View>
          <AppText variant="body" style={styles.waListLabel}>
            Médias, liens et documents
          </AppText>
          <AppText variant="caption" style={styles.waListMeta}>
            {data.attachment_counts.all} ›
          </AppText>
          <Ionicons name="chevron-forward" size={18} color={M.TEXT_MUTED} />
        </Pressable>
        {previewAttachments.length > 0 ? (
          <ScrollView
            horizontal
            showsHorizontalScrollIndicator={false}
            contentContainerStyle={[styles.mediaRow, { paddingHorizontal: horizontalPadding }]}
          >
            {previewAttachments.map((item) => (
              <MediaTile key={item.id} item={item} />
            ))}
          </ScrollView>
        ) : null}

        <View style={styles.waSectionGap} />

        {/* Participants — section repliable type WhatsApp */}
        <View style={styles.waGroup}>
          <View style={[styles.waParticipantsHeader, { paddingHorizontal: 16 }]}>
            <AppText variant="label" style={styles.waGroupTitle}>
              {channel.participant_count} participant{channel.participant_count > 1 ? "s" : ""}
            </AppText>
            <Pressable onPress={() => setParticipantsOpen((v) => !v)} accessibilityLabel="Afficher les participants">
              <Ionicons
                name={participantsOpen ? "chevron-up" : "chevron-down"}
                size={20}
                color={M.TEXT_SEC}
              />
            </Pressable>
          </View>
          {canManage ? (
            <Pressable style={styles.waListRow} onPress={() => setPickerOpen(true)}>
              <View style={[styles.waListIcon, styles.waListIconAdd]}>
                <Ionicons name="person-add" size={20} color={M.BRAND} />
              </View>
              <AppText variant="body" style={[styles.waListLabel, { color: M.BRAND }]}>
                Ajouter des participants
              </AppText>
            </Pressable>
          ) : null}
          {participantsOpen
            ? data.participants.map((row, idx) => (
                <ParticipantRow
                  key={row.user_id}
                  row={row}
                  canManage={canManage}
                  onRemove={() => confirmRemove(row)}
                  last={idx === data.participants.length - 1}
                />
              ))
            : (
              <Pressable style={styles.waCollapsedHint} onPress={() => setParticipantsOpen(true)}>
                <AppText variant="bodyMuted" numberOfLines={2}>
                  {data.participants.map((p) => p.display_name).join(", ")}
                </AppText>
              </Pressable>
            )}
        </View>

        <View style={styles.waSectionGap} />

        {/* Informations */}
        <View style={styles.waGroup}>
          <WaListNavRow icon="information-circle-outline" label="Nom du canal" value={channel.title} onPress={canManage ? openEdit : undefined} />
          <WaListNavRow icon="document-text-outline" label="Description" value={channel.description} onPress={canManage ? openEdit : undefined} />
          <WaListNavRow icon="person-outline" label="Créé par" value={channel.created_by_name} />
          <WaListNavRow icon="calendar-outline" label="Créé le" value={formatFrenchDate(channel.created_at)} />
          <WaListNavRow icon="lock-closed-outline" label="Type" value={channel.channel_type_label} last />
        </View>

        <View style={styles.waSectionGap} />

        {/* Réglages type WhatsApp */}
        <View style={styles.waGroup}>
          <WaListNavRow
            icon="notifications-outline"
            label="Notifications"
            value="Par défaut"
            onPress={() => notifySoon("Notifications")}
          />
          <WaListNavRow
            icon="lock-closed-outline"
            label="Chiffrement"
            value="Les messages du canal sont transmis de façon sécurisée."
            last
          />
        </View>

        <View style={styles.waSectionGap} />

        {/* Permissions */}
        <View style={styles.waGroup}>
          <View style={styles.waGroupHeader}>
            <AppText variant="label" style={styles.waGroupTitle}>
              Permissions
            </AppText>
          </View>
          <PermissionsBlock permissions={data.permissions} />
        </View>

        <View style={styles.waSectionGap} />

        {/* Historique */}
        <View style={styles.waGroup}>
          <View style={styles.waGroupHeader}>
            <AppText variant="label" style={styles.waGroupTitle}>
              Historique
            </AppText>
          </View>
          {data.history.slice(0, 6).map((entry, idx) => (
            <HistoryRow key={`${entry.type}-${idx}`} entry={entry} />
          ))}
        </View>

        {canManage ? (
          <>
            <View style={styles.waSectionGap} />
            <View style={styles.waGroup}>
              <DangerAction label="Archiver le canal" onPress={() => notifySoon("Archivage")} />
              <DangerAction
                label="Vider l'historique"
                onPress={confirmClearHistory}
                disabled={clearHistory.isPending}
              />
              <DangerAction label="Supprimer le canal" destructive last onPress={() => notifySoon("Suppression")} />
            </View>
          </>
        ) : null}
      </ScrollView>

      {/* Modals */}
      <EditChannelModal
        visible={editOpen}
        title={editTitle}
        description={editDescription}
        pending={updateChannel.isPending}
        onChangeTitle={setEditTitle}
        onChangeDescription={setEditDescription}
        onClose={() => setEditOpen(false)}
        onSave={saveEdit}
      />

      <AddParticipantModal
        visible={pickerOpen}
        drivers={data.available_drivers}
        onClose={() => setPickerOpen(false)}
        onSelect={handleAdd}
      />

      <AllMediaModal
        visible={showAllMedia}
        attachments={data.attachments_all ?? data.attachments_preview}
        onClose={() => setShowAllMedia(false)}
      />

      <ConfirmDialogModal
        visible={removeTarget != null}
        title="Retirer du canal"
        icon="person-remove-outline"
        confirmLabel="Retirer"
        destructive
        pending={removeParticipant.isPending}
        onClose={() => setRemoveTarget(null)}
        onConfirm={handleConfirmRemove}
      >
        {removeTarget ? (
          <>
            <View style={styles.confirmParticipant}>
              <View style={styles.confirmParticipantAvatar}>
                <Ionicons name="person" size={22} color={M.BRAND} />
              </View>
              <View style={styles.confirmParticipantBody}>
                <AppText variant="body" style={styles.confirmParticipantName}>
                  {removeTarget.display_name}
                </AppText>
                <AppText variant="caption" style={styles.confirmParticipantRole}>
                  {removeTarget.role_label ??
                    (removeTarget.is_admin ? "Exploitation" : "Chauffeur")}
                </AppText>
              </View>
            </View>
            <AppText variant="bodyMuted" style={styles.confirmMessage}>
              Cette personne ne pourra plus lire ni envoyer de messages dans{" "}
              <AppText variant="body" style={styles.confirmChannelName}>
                {channel.title}
              </AppText>
              .
            </AppText>
          </>
        ) : null}
      </ConfirmDialogModal>

      <ConfirmDialogModal
        visible={clearHistoryOpen}
        title="Vider l'historique"
        icon="trash-outline"
        message="Cette action supprimera définitivement tous les messages du canal pour tous les participants."
        confirmLabel="Vider"
        destructive
        pending={clearHistory.isPending}
        onClose={() => setClearHistoryOpen(false)}
        onConfirm={handleConfirmClearHistory}
      />
    </View>
  );
}

function WaQuickPill({
  icon,
  label,
  onPress,
  disabled,
}: {
  icon: keyof typeof Ionicons.glyphMap;
  label: string;
  onPress: () => void;
  disabled?: boolean;
}) {
  return (
    <Pressable
      style={[styles.waPill, disabled && styles.waPillDisabled]}
      onPress={onPress}
      disabled={disabled}
    >
      <View style={styles.waPillIcon}>
        <Ionicons name={icon} size={20} color={M.BRAND} />
      </View>
      <AppText variant="caption" style={styles.waPillLabel}>
        {label}
      </AppText>
    </Pressable>
  );
}

function WaListNavRow({
  icon,
  label,
  value,
  onPress,
  last,
}: {
  icon: keyof typeof Ionicons.glyphMap;
  label: string;
  value: string;
  onPress?: () => void;
  last?: boolean;
}) {
  const content = (
    <>
      <View style={styles.waListIcon}>
        <Ionicons name={icon} size={20} color={M.TEXT_SEC} />
      </View>
      <View style={styles.waListNavBody}>
        <AppText variant="caption" style={styles.waListNavLabel}>
          {label}
        </AppText>
        <AppText variant="body" numberOfLines={2}>
          {value}
        </AppText>
      </View>
      {onPress ? <Ionicons name="chevron-forward" size={18} color={M.TEXT_MUTED} /> : null}
    </>
  );
  if (onPress) {
    return (
      <Pressable style={[styles.waListRow, last && styles.waListRowLast]} onPress={onPress}>
        {content}
      </Pressable>
    );
  }
  return <View style={[styles.waListRow, last && styles.waListRowLast]}>{content}</View>;
}

function notifySoon(feature: string) {
  Alert.alert(feature, "Cette action sera disponible dans une prochaine version.");
}

function MediaTile({ item }: { item: ConversationAttachmentRow }) {
  return (
    <Pressable
      style={styles.mediaTile}
      onPress={() => void Linking.openURL(resolveMediaUrl(item.url) ?? item.url)}
    >
      {item.kind === "photo" ? (
        <Image source={{ uri: resolveMediaUrl(item.url) ?? item.url }} style={styles.mediaThumb} />
      ) : (
        <View style={[styles.mediaDoc, item.kind === "audio" && styles.mediaAudio]}>
          <AppText variant="caption" style={styles.mediaDocLabel}>
            {item.kind === "audio" ? "AUDIO" : "PDF"}
          </AppText>
        </View>
      )}
    </Pressable>
  );
}

function ParticipantRow({
  row,
  canManage,
  onRemove,
  last,
}: {
  row: ConversationParticipantRow;
  canManage: boolean;
  onRemove: () => void;
  last?: boolean;
}) {
  return (
    <View style={[styles.participantRow, last && styles.waListRowLast]}>
      <View style={styles.participantAvatar}>
        <Ionicons name="person" size={18} color={M.BRAND} />
      </View>
      <View style={styles.participantBody}>
        <View style={styles.participantTitleRow}>
          <AppText variant="body" style={styles.participantName}>
            {row.display_name}
          </AppText>
          {row.is_admin ? (
            <View style={styles.adminBadge}>
              <AppText variant="caption" style={styles.adminBadgeText}>
                Admin. du groupe
              </AppText>
            </View>
          ) : null}
        </View>
        <AppText variant="caption" style={styles.participantRole}>
          {row.role_label ?? (row.is_admin ? "Exploitation" : "Chauffeur")}
        </AppText>
        <AppText variant="caption" style={styles.participantActivity}>
          {formatRelativeActivity(row.last_activity_at)}
        </AppText>
      </View>
      {canManage && row.can_remove ? (
        <Pressable onPress={onRemove} style={styles.participantAction} accessibilityLabel="Retirer">
          <Ionicons name="remove-circle-outline" size={22} color={M.DANGER} />
        </Pressable>
      ) : null}
    </View>
  );
}

function PermissionsBlock({ permissions }: { permissions: ChannelManagePermissions }) {
  return (
    <>
      <PermissionRow label="Ajouter participants" enabled={permissions.add_participants} />
      <PermissionRow label="Envoyer fichiers" enabled={permissions.send_files} />
      <PermissionRow label="Répondre" enabled={permissions.reply} />
      <PermissionRow label="Modifier canal" enabled={permissions.edit_channel} />
      <PermissionRow label="Supprimer messages" enabled={permissions.delete_messages} />
    </>
  );
}

function HistoryRow({ entry }: { entry: ChannelHistoryEntry }) {
  return (
    <View style={styles.historyRow}>
      <View style={styles.historyDot} />
      <View style={styles.historyBody}>
        <AppText variant="body">{entry.label}</AppText>
        {entry.at ? (
          <AppText variant="caption" style={styles.historyDate}>
            {formatFrenchDate(entry.at)}
          </AppText>
        ) : null}
      </View>
    </View>
  );
}

function DangerAction({
  label,
  onPress,
  destructive,
  last,
  disabled,
}: {
  label: string;
  onPress: () => void;
  destructive?: boolean;
  last?: boolean;
  disabled?: boolean;
}) {
  return (
    <Pressable
      style={[styles.dangerAction, last && styles.waListRowLast, disabled && styles.waPillDisabled]}
      onPress={onPress}
      disabled={disabled}
    >
      <AppText variant="body" style={destructive ? styles.dangerActionDestructive : styles.dangerActionText}>
        {label}
      </AppText>
    </Pressable>
  );
}

function EditChannelModal({
  visible,
  title,
  description,
  pending,
  onChangeTitle,
  onChangeDescription,
  onClose,
  onSave,
}: {
  visible: boolean;
  title: string;
  description: string;
  pending: boolean;
  onChangeTitle: (v: string) => void;
  onChangeDescription: (v: string) => void;
  onClose: () => void;
  onSave: () => void;
}) {
  return (
    <Modal visible={visible} transparent animationType="slide" onRequestClose={onClose}>
      <View style={styles.modalBackdrop}>
        <View style={styles.modalSheet}>
          <AppText variant="sectionTitle">Modifier le canal</AppText>
          <AppText variant="caption">Nom</AppText>
          <TextInput value={title} onChangeText={onChangeTitle} style={styles.input} />
          <AppText variant="caption">Description</AppText>
          <TextInput
            value={description}
            onChangeText={onChangeDescription}
            style={[styles.input, styles.inputMultiline]}
            multiline
          />
          <View style={styles.modalActions}>
            <Pressable style={styles.modalBtnSecondary} onPress={onClose}>
              <AppText variant="label">Annuler</AppText>
            </Pressable>
            <Pressable style={styles.modalBtnPrimary} onPress={onSave} disabled={pending}>
              {pending ? (
                <ActivityIndicator color="#fff" />
              ) : (
                <AppText variant="label" style={styles.modalBtnPrimaryText}>
                  Enregistrer
                </AppText>
              )}
            </Pressable>
          </View>
        </View>
      </View>
    </Modal>
  );
}

function AddParticipantModal({
  visible,
  drivers,
  onClose,
  onSelect,
}: {
  visible: boolean;
  drivers: AvailableDriverRow[];
  onClose: () => void;
  onSelect: (d: AvailableDriverRow) => void;
}) {
  return (
    <Modal visible={visible} transparent animationType="fade" onRequestClose={onClose}>
      <Pressable style={styles.modalBackdrop} onPress={onClose}>
        <View style={styles.modalSheet}>
          <AppText variant="sectionTitle">Ajouter un chauffeur</AppText>
          {drivers.length === 0 ? (
            <AppText variant="bodyMuted">Tous les chauffeurs actifs sont déjà dans le canal.</AppText>
          ) : (
            drivers.map((driver) => (
              <Pressable key={driver.driver_id} style={styles.option} onPress={() => onSelect(driver)}>
                <AppText variant="body">{driver.display_name}</AppText>
              </Pressable>
            ))
          )}
        </View>
      </Pressable>
    </Modal>
  );
}

function AllMediaModal({
  visible,
  attachments,
  onClose,
}: {
  visible: boolean;
  attachments: ConversationAttachmentRow[];
  onClose: () => void;
}) {
  return (
    <Modal visible={visible} animationType="slide" onRequestClose={onClose}>
      <View style={styles.allMediaWrap}>
        <View style={styles.allMediaHeader}>
          <Pressable onPress={onClose}>
            <Ionicons name="close" size={26} color={M.TEXT} />
          </Pressable>
          <AppText variant="sectionTitle">Toutes les pièces jointes</AppText>
          <View style={{ width: 26 }} />
        </View>
        <ScrollView contentContainerStyle={styles.allMediaGrid}>
          {attachments.map((item) => (
            <Pressable
              key={item.id}
              style={styles.allMediaTile}
              onPress={() => void Linking.openURL(resolveMediaUrl(item.url) ?? item.url)}
            >
              {item.kind === "photo" ? (
                <Image source={{ uri: resolveMediaUrl(item.url) ?? item.url }} style={styles.allMediaImage} />
              ) : (
                <View style={styles.allMediaDoc}>
                  <AppText variant="caption">{item.kind === "audio" ? "AUDIO" : "PDF"}</AppText>
                  <AppText variant="caption" numberOfLines={2}>
                    {item.label}
                  </AppText>
                </View>
              )}
            </Pressable>
          ))}
        </ScrollView>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  screenRoot: { flex: 1, backgroundColor: M.PAGE_BG },
  scroll: { flex: 1 },
  loadingWrap: { flex: 1, alignItems: "center", justifyContent: "center", padding: 24, gap: 8 },
  errorTitle: { fontWeight: "600", color: M.TEXT, marginTop: 8 },
  errorHint: { textAlign: "center" },
  retryBtn: {
    marginTop: 16,
    backgroundColor: M.BRAND,
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 10,
  },
  retryBtnText: { color: "#fff", fontWeight: "700" },
  topBar: {
    flexDirection: "row",
    alignItems: "center",
    paddingBottom: 8,
    backgroundColor: M.CARD,
  },
  backBtn: { padding: 8, paddingHorizontal: 12 },
  moreBtn: { padding: 8, paddingHorizontal: 12 },
  waHero: {
    alignItems: "center",
    backgroundColor: M.CARD,
    paddingVertical: 28,
    paddingHorizontal: 20,
    gap: 6,
  },
  avatarRing: { position: "relative", marginBottom: 8 },
  avatar: {
    width: 96,
    height: 96,
    borderRadius: 48,
    backgroundColor: M.BRAND,
    alignItems: "center",
    justifyContent: "center",
  },
  avatarEdit: {
    position: "absolute",
    right: 0,
    bottom: 4,
    width: 30,
    height: 30,
    borderRadius: 15,
    backgroundColor: "#334155",
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 2,
    borderColor: M.CARD,
  },
  heroTitle: { fontSize: FONT_SIZE.px22, fontWeight: "700", color: M.TEXT, textAlign: "center" },
  heroSubtitle: { textAlign: "center", color: M.TEXT_SEC, paddingHorizontal: 12 },
  heroMeta: { color: M.TEXT_MUTED, marginTop: 4 },
  waQuickBar: {
    flexDirection: "row",
    justifyContent: "space-between",
    backgroundColor: M.CARD,
    paddingVertical: 14,
    gap: 8,
  },
  waPill: {
    flex: 1,
    alignItems: "center",
    gap: 8,
    paddingVertical: 4,
  },
  waPillDisabled: { opacity: 0.45 },
  waPillIcon: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: "rgba(0,121,107,0.08)",
    alignItems: "center",
    justifyContent: "center",
  },
  waPillLabel: { color: M.TEXT_SEC, fontWeight: "600", fontSize: FONT_SIZE.px11, textAlign: "center" },
  waSectionGap: { height: 10, backgroundColor: M.PAGE_BG },
  waGroup: { backgroundColor: M.CARD },
  waGroupHeader: { paddingHorizontal: 16, paddingTop: 14, paddingBottom: 6 },
  waGroupTitle: { color: M.BRAND, fontWeight: "700", fontSize: FONT_SIZE.px13 },
  waListRow: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 14,
    paddingHorizontal: 16,
    gap: 14,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: "#eef2f7",
    backgroundColor: M.CARD,
  },
  waListRowLast: { borderBottomWidth: 0 },
  waListIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: "#f1f5f9",
    alignItems: "center",
    justifyContent: "center",
  },
  waListIconAdd: { backgroundColor: "rgba(0,121,107,0.08)" },
  waListLabel: { flex: 1, color: M.TEXT, fontWeight: "500" },
  waListMeta: { color: M.TEXT_MUTED, marginRight: 4 },
  waListNavBody: { flex: 1, gap: 2 },
  waListNavLabel: { color: M.TEXT_MUTED },
  waParticipantsHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingTop: 14,
    paddingBottom: 6,
  },
  waCollapsedHint: { paddingHorizontal: 16, paddingVertical: 12 },
  linkText: { color: M.BRAND, fontWeight: "600" },
  mediaRow: { gap: 8, paddingVertical: 12, backgroundColor: M.CARD },
  mediaTile: { width: 72, height: 72, borderRadius: 8, overflow: "hidden" },
  mediaThumb: { width: "100%", height: "100%" },
  mediaDoc: {
    flex: 1,
    backgroundColor: "#fef3c7",
    alignItems: "center",
    justifyContent: "center",
    height: "100%",
  },
  mediaAudio: { backgroundColor: "#dbeafe" },
  mediaDocLabel: { fontWeight: "700", color: M.TEXT_SEC, fontSize: FONT_SIZE.px10 },
  participantRow: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 12,
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: "#eef2f7",
  },
  participantAvatar: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: "rgba(0,121,107,0.08)",
    alignItems: "center",
    justifyContent: "center",
  },
  participantBody: { flex: 1, gap: 2 },
  participantTitleRow: { flexDirection: "row", alignItems: "center", gap: 8, flexWrap: "wrap" },
  participantName: { fontWeight: "600", color: M.TEXT },
  adminBadge: {
    backgroundColor: "rgba(0,121,107,0.12)",
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 6,
  },
  adminBadgeText: { color: M.BRAND_DARK, fontWeight: "700", fontSize: FONT_SIZE.px10 },
  participantRole: { color: M.TEXT_SEC },
  participantActivity: { color: M.TEXT_MUTED, fontSize: FONT_SIZE.px11 },
  participantAction: { padding: 4, marginTop: 4 },
  permissionRow: { flexDirection: "row", alignItems: "center", gap: 10, paddingVertical: 10, paddingHorizontal: 16 },
  permissionLabel: { color: M.TEXT },
  historyRow: { flexDirection: "row", gap: 10, paddingVertical: 10, paddingHorizontal: 16 },
  historyDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: M.BRAND,
    marginTop: 6,
  },
  historyBody: { flex: 1, gap: 2 },
  historyDate: { color: M.TEXT_MUTED },
  dangerAction: {
    paddingVertical: 16,
    paddingHorizontal: 16,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: "#fee2e2",
  },
  dangerActionText: { color: M.TEXT_SEC },
  dangerActionDestructive: { color: M.DANGER, fontWeight: "600" },
  confirmParticipant: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    backgroundColor: "#f8fafc",
    borderRadius: 12,
    padding: 12,
  },
  confirmParticipantAvatar: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: "rgba(0,121,107,0.08)",
    alignItems: "center",
    justifyContent: "center",
  },
  confirmParticipantBody: { flex: 1, gap: 2 },
  confirmParticipantName: { fontWeight: "700", color: M.TEXT },
  confirmParticipantRole: { color: M.TEXT_SEC },
  confirmMessage: { textAlign: "center", lineHeight: 21 },
  confirmChannelName: { fontWeight: "700", color: M.TEXT },
  modalBackdrop: {
    flex: 1,
    backgroundColor: "rgba(15,23,42,0.45)",
    justifyContent: "flex-end",
  },
  modalSheet: {
    backgroundColor: M.CARD,
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    padding: 20,
    gap: 10,
    maxHeight: "75%",
  },
  input: {
    borderWidth: 1,
    borderColor: M.SHELL_BORDER,
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 10,
    color: M.TEXT,
    backgroundColor: "#fafafa",
  },
  inputMultiline: { minHeight: 80, textAlignVertical: "top" },
  modalActions: { flexDirection: "row", justifyContent: "flex-end", gap: 10, marginTop: 8 },
  modalBtnSecondary: {
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 10,
    backgroundColor: "#f1f5f9",
  },
  modalBtnPrimary: {
    paddingHorizontal: 18,
    paddingVertical: 12,
    borderRadius: 10,
    backgroundColor: M.BRAND,
    minWidth: 110,
    alignItems: "center",
  },
  modalBtnPrimaryText: { color: "#fff", fontWeight: "700" },
  option: {
    paddingVertical: 14,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: "#e5e7eb",
  },
  allMediaWrap: { flex: 1, backgroundColor: M.PAGE_BG },
  allMediaHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    padding: 16,
    backgroundColor: M.CARD,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: M.SHELL_BORDER,
  },
  allMediaGrid: { flexDirection: "row", flexWrap: "wrap", gap: 10, padding: 16 },
  allMediaTile: { width: "47%", aspectRatio: 1, borderRadius: 12, overflow: "hidden" },
  allMediaImage: { width: "100%", height: "100%" },
  allMediaDoc: {
    flex: 1,
    backgroundColor: "#fef3c7",
    alignItems: "center",
    justifyContent: "center",
    padding: 8,
    gap: 4,
  },
});
