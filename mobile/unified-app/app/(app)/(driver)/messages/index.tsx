import { DriverContextGuard, PermissionGuard } from "../../../../src/core/guards";
import { DriverMessagesInboxView } from "../../../../src/features/driver/messages/components/DriverMessagesInboxView";

/** Inbox Messages chauffeur — UI maquette (onglets + liste plate). */
export default function DriverMessagesInboxScreen() {
  return (
    <DriverContextGuard>
      <PermissionGuard permission="chat:read">
        <DriverMessagesInboxView />
      </PermissionGuard>
    </DriverContextGuard>
  );
}
