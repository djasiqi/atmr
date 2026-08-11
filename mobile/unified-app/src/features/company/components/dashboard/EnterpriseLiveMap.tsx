import { StyleSheet, View } from "react-native";

import type { CompanyDispatchMission, CompanyDriverLiveLocation } from "../../api/contracts";

import { OperationalFleetMap } from "../maps/OperationalFleetMap";

import { D, dashboardCardShadowWeb } from "../../theme/companyDashboardTokens";



type Props = {

  drivers: CompanyDriverLiveLocation[];

  missions?: CompanyDispatchMission[];

  rosterResolved?: boolean;

  mapHeight?: number;

  onViewMission?: (missionId: number) => void;

  onMessage?: () => void;

};



export function EnterpriseLiveMap({

  drivers,

  missions = [],

  rosterResolved = false,

  mapHeight = 248,

  onViewMission,

  onMessage,

}: Props) {

  return (

    <View style={s.shell} accessibilityLabel="Carte flotte en direct">

      <OperationalFleetMap

        drivers={drivers}

        missions={missions}

        rosterResolved={rosterResolved}

        mapHeight={mapHeight}

        containerStyle={s.mapFill}

        onViewMission={onViewMission}

        onMessage={onMessage}

      />

    </View>

  );

}



const s = StyleSheet.create({

  shell: {

    height: D.mapHeroMaxHeight,

    maxHeight: D.mapHeroMaxHeight,

    borderRadius: D.radiusCard,

    overflow: "hidden",

    borderWidth: 1,

    borderColor: D.border,

    backgroundColor: D.cardBg,

    ...dashboardCardShadowWeb,

  },

  mapFill: {

    flex: 1,

    height: D.mapHeroMaxHeight,

    maxHeight: D.mapHeroMaxHeight,

    borderRadius: 0,

  },

});


