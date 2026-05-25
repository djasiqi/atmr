import { CompanyContextGuard, PermissionGuard } from "../../../../src/core/guards";
import { CompanyMessagesInboxView } from "../../../../src/features/company/messages/components/CompanyMessagesInboxView";

export default function CompanyMessagesInboxScreen() {
  return (
    <CompanyContextGuard>
      <PermissionGuard permission="company:dashboard:read">
        <CompanyMessagesInboxView />
      </PermissionGuard>
    </CompanyContextGuard>
  );
}
