import { Redirect } from "expo-router";

export default function CompanyInvoicesScreen() {
  return <Redirect href={"/(app)/(company)/clients?section=invoices" as any} />;
}
