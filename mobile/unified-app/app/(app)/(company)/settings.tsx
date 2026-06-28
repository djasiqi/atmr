import { PermissionGuard } from "../../../src/core/guards";
import { CompanySettingsScreenContent } from "../../../src/features/company/components/CompanySettingsScreenContent";

export default function CompanySettingsScreen() {
  return (
    <PermissionGuard permission="company:dashboard:read">
      <CompanySettingsScreenContent />
    </PermissionGuard>
  );
}
