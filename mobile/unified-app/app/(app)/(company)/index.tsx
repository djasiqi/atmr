import { Redirect } from "expo-router";
import { PermissionGuard } from "../../../src/core/guards";

export default function CompanyHomeScreen() {
  return (
    <PermissionGuard permission="company:dashboard:read">
      <Redirect href="/(app)/(company)/dashboard" />
    </PermissionGuard>
  );
}
