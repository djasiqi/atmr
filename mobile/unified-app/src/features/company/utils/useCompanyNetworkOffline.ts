import { useEffect, useState } from "react";
import { getNetworkSnapshot, subscribeNetworkState } from "../../../core/network/networkState";
import { isCompanyNetworkOffline } from "./companyOfflinePolicy";

export function useCompanyNetworkOffline(): boolean {
  const [offline, setOffline] = useState(() => isCompanyNetworkOffline(getNetworkSnapshot()));
  useEffect(() => {
    return subscribeNetworkState((snapshot) => {
      setOffline(isCompanyNetworkOffline(snapshot));
    });
  }, []);
  return offline;
}
