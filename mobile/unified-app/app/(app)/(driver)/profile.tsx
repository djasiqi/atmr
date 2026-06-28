import { DriverContextGuard } from "../../../src/core/guards";
import { DriverSettingsScreenContent } from "../../../src/features/driver/components/DriverSettingsScreenContent";

export default function DriverProfileScreen() {
  return (
    <DriverContextGuard>
      <DriverSettingsScreenContent />
    </DriverContextGuard>
  );
}
