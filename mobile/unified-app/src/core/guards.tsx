import { Redirect } from "expo-router";
// eslint-disable-next-line @typescript-eslint/no-require-imports
const ReactRuntime: any = require("react");
import { useSession } from "./sessionProvider";
import {
  resolveAuthGuardRedirect,
  resolveCompanyContextGuardRedirect,
  resolveContextGuardRedirect,
  resolveDriverUnifiedGate,
  resolveOnboardingGuardRedirect,
  resolvePermissionGuardRedirect,
  resolveVersionGuardRedirect,
} from "./guardDecisions";
import { isFeatureEnabled } from "./featureFlags/registry";
import { emitDriverTelemetry } from "./observability/driverTelemetry";
type PropsWithChildren<P = object> = P & { children?: any };

export function AuthGuard({ children }: PropsWithChildren) {
  const { bootstrap } = useSession();
  const redirectTo = resolveAuthGuardRedirect(bootstrap);
  if (redirectTo) return <Redirect href={redirectTo as any} />;
  return <>{children}</>;
}

export function ContextGuard({ children }: PropsWithChildren) {
  const { activeContext } = useSession();
  const redirectTo = resolveContextGuardRedirect(activeContext);
  if (redirectTo) return <Redirect href={redirectTo as any} />;
  return <>{children}</>;
}

export function PermissionGuard({
  permission,
  children,
}: PropsWithChildren<{ permission: string }>) {
  const { can } = useSession();
  const redirectTo = resolvePermissionGuardRedirect(can(permission));
  if (redirectTo) return <Redirect href={redirectTo as any} />;
  return <>{children}</>;
}

export function DriverContextGuard({ children }: PropsWithChildren) {
  const { activeContext } = useSession();
  if (!activeContext) {
    return <Redirect href={"/(app)/context-selector" as any} />;
  }
  if (activeContext.context_type === "company") {
    return <Redirect href={"/(app)/(company)/dashboard" as any} />;
  }
  if (activeContext.context_type !== "driver") {
    return <Redirect href={"/(app)/unauthorized" as any} />;
  }
  return <>{children}</>;
}

export function CompanyContextGuard({ children }: PropsWithChildren) {
  const { activeContext } = useSession();
  const redirectTo = resolveCompanyContextGuardRedirect(activeContext);
  if (redirectTo) return <Redirect href={redirectTo as any} />;
  return <>{children}</>;
}

export function OnboardingGuard({ children }: PropsWithChildren) {
  const { bootstrap } = useSession();
  const redirectTo = resolveOnboardingGuardRedirect(bootstrap);
  if (redirectTo) return <Redirect href={redirectTo as any} />;
  return <>{children}</>;
}

export function AppVersionGuard({ children }: PropsWithChildren) {
  const { bootstrap } = useSession();
  const redirectTo = resolveVersionGuardRedirect(bootstrap);
  if (redirectTo) return <Redirect href={redirectTo as any} />;
  return <>{children}</>;
}

export function DriverUnifiedGateGuard({ children }: PropsWithChildren) {
  const evaluatedRef = ReactRuntime.useRef(false);
  const enabled = isFeatureEnabled("driver_unified_enabled");
  const result = resolveDriverUnifiedGate(enabled);

  ReactRuntime.useEffect(() => {
    if (evaluatedRef.current) return;
    evaluatedRef.current = true;
    emitDriverTelemetry("driver.gate.unified_evaluated", {
      source: "DriverUnifiedGateGuard",
      enabled,
      allowed: result.allowed,
      option: result.option,
    });
  }, [enabled, result.allowed, result.option]);

  if (!result.allowed) {
    return <Redirect href={result.redirectTo as any} />;
  }
  return <>{children}</>;
}
