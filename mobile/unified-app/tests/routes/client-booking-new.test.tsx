import React from "react";

import { act, create, type ReactTestInstance } from "react-test-renderer";

import { beforeEach, describe, expect, it, jest } from "@jest/globals";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

jest.mock("@expo/vector-icons", () => {

  // eslint-disable-next-line @typescript-eslint/no-require-imports, @typescript-eslint/consistent-type-imports

  const { View } = require("react-native");

  return {

    // eslint-disable-next-line @typescript-eslint/consistent-type-imports

    Ionicons: (props: { name?: string; size?: number }) => (

      <View style={{ width: props.size ?? 16, height: props.size ?? 16 }} />

    ),

  };

});



import ClientBookingCreateScreen from "../../app/(app)/(client)/booking/new";

import {
  autocompleteAddress,
  createClientBooking,
  getGeocodePlaceDetails,
  previewClientBooking,
} from "../../src/features/client/api";



jest.mock("expo-router", () => ({

  useLocalSearchParams: () => ({}),

  useRouter: () => ({

    back: jest.fn(),

    replace: jest.fn(),

  }),

}));



const mockSession = {

  activeContext: { context_type: "client" as const, context_id: "c1" },

  bootstrap: { preview_contract_version: "1" },

  bootstrapSession: jest.fn().mockResolvedValue(undefined),

};



jest.mock("../../src/core/sessionProvider", () => ({

  useSession: () => mockSession,

}));



const mockUseClientProfileQuery = jest.fn(() => ({ data: null, isLoading: false }));



jest.mock("../../src/features/client/hooks", () => ({

  useActiveClientContextId: () => "c1",

  useClientProfileQuery: () => mockUseClientProfileQuery(),

}));



jest.mock("../../src/core/guards", () => ({

  PermissionGuard: ({ children }: { children: React.ReactNode }) => <>{children}</>,

}));



jest.mock("../../src/features/client/navigation/ClientFloatingAppBar", () => ({

  useClientBottomContentPadding: () => 0,

}));



jest.mock("../../src/features/client/statusEvents", () => ({

  trackClientKpiEvent: jest.fn(),

}));



jest.mock("../../src/features/client/queryKeys", () => ({

  invalidateClientQueries: jest.fn(),

}));



jest.mock("../../src/core/api/client", () => ({

  fetchPublicPreRequestDraft: jest.fn().mockResolvedValue(null),

  consumePublicPreRequestDraft: jest.fn().mockResolvedValue(undefined),

}));



jest.mock("../../src/core/public/preRequestDraft", () => ({

  clearPublicPreRequestDraft: jest.fn().mockResolvedValue(undefined),

  loadPublicPreRequestDraft: jest.fn().mockResolvedValue(null),

}));



jest.mock("@react-native-community/datetimepicker", () => "DateTimePicker");



jest.mock("../../src/features/client/api", () => ({

  autocompleteAddress: jest.fn().mockResolvedValue([]),

  getGeocodePlaceDetails: jest.fn().mockResolvedValue(null),

  previewClientBooking: jest.fn(),

  createClientBooking: jest.fn(),

  getBookingDetail: jest.fn().mockResolvedValue({

    payment_required: true,

    payment_status: "required",

  }),

  postIndicativeFareEstimate: jest.fn().mockResolvedValue({ success: false }),

  reverseGeocodeFromCoordinates: jest.fn().mockResolvedValue(null),

}));



const preview = previewClientBooking as unknown as jest.MockedFunction<typeof previewClientBooking>;

const createB = createClientBooking as unknown as jest.MockedFunction<typeof createClientBooking>;

const autocompleteMock = autocompleteAddress as unknown as jest.MockedFunction<typeof autocompleteAddress>;

const getPlaceDetailsMock = getGeocodePlaceDetails as unknown as jest.MockedFunction<
  typeof getGeocodePlaceDetails
>;



function createTestQueryClient() {

  return new QueryClient({

    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },

  });

}



jest.mock("react-native-safe-area-context", () => ({

  useSafeAreaInsets: () => ({ top: 12, bottom: 0, left: 0, right: 0 }),

  SafeAreaProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,

}));



let queryClient: QueryClient = createTestQueryClient();



function TestRoot({ children }: { children: React.ReactNode }): React.ReactElement {

  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;

}



function jsonHas(tree: { toJSON: (() => unknown) | null }, s: string): boolean {

  const t = tree.toJSON?.() ?? null;

  return t != null && JSON.stringify(t).includes(s);

}



function findByTestId(

  node: ReactTestInstance,

  testID: string

): { props: { onPress?: () => void } } {

  const a = node.findAll((x) => (x as { props?: { testID?: string } })?.props?.testID === testID);

  const el = a[0] as { props: { onPress?: () => void; testID?: string } } | undefined;

  if (!el) {

    throw new Error(`testID not found: ${testID}`);

  }

  return el;

}



describe("ClientBookingCreateScreen (2 étapes)", () => {

  beforeEach(() => {

    queryClient = createTestQueryClient();

    preview.mockClear();

    createB.mockClear();

    autocompleteMock.mockReset();

    autocompleteMock.mockResolvedValue([]);

    getPlaceDetailsMock.mockReset();

    getPlaceDetailsMock.mockResolvedValue(null);

    mockUseClientProfileQuery.mockReturnValue({ data: null, isLoading: false });

  });



  it("affiche « Continuer vers le récapitulatif » (étape 1) et le libellé 1/2 à l’état initial", () => {

    let tree!: ReturnType<typeof create>;

    act(() => {

      tree = create(

        <TestRoot>

          <ClientBookingCreateScreen />

        </TestRoot>

      );

    });

    expect(jsonHas(tree, "Continuer vers le récapitulatif")).toBe(true);

    expect(jsonHas(tree, "1/2 · Demande")).toBe(true);

  });



  it("le clic sur le bouton étape 1 sans adresses n’appelle ni preview ni create", async () => {

    let tree!: ReturnType<typeof create>;

    act(() => {

      tree = create(

        <TestRoot>

          <ClientBookingCreateScreen />

        </TestRoot>

      );

    });

    const cta = findByTestId(tree.root, "booking-cta-go-summary");

    await act(async () => {

      cta.props.onPress?.();

    });

    expect(preview).not.toHaveBeenCalled();

    expect(createB).not.toHaveBeenCalled();

    expect(jsonHas(tree, "Continuer vers le paiement")).toBe(false);

  });



  it("détails facultatifs (repliable) : le toggle révèle les champs, sans paiement", async () => {

    let tree!: ReturnType<typeof create>;

    act(() => {

      tree = create(

        <TestRoot>

          <ClientBookingCreateScreen />

        </TestRoot>

      );

    });

    expect(jsonHas(tree, "Établissement (hôpital, clinique, cabinet")).toBe(false);

    const toggle = findByTestId(tree.root, "booking-optional-details-toggle");

    await act(async () => {

      toggle.props.onPress?.();

    });

    expect(jsonHas(tree, "Établissement (hôpital, clinique, cabinet")).toBe(true);

    expect(jsonHas(tree, "2/2 · Récapitulatif")).toBe(false);

    expect(jsonHas(tree, "Continuer vers le paiement")).toBe(false);

  });



  it("l’écran récapitulatif (_testFormStep) affiche « Continuer vers le paiement » et 2/2", () => {

    let tree!: ReturnType<typeof create>;

    act(() => {

      tree = create(

        <TestRoot>

          <ClientBookingCreateScreen _testFormStep="summary" />

        </TestRoot>

      );

    });

    expect(jsonHas(tree, "2/2 · Récapitulatif")).toBe(true);

    expect(jsonHas(tree, "Continuer vers le paiement")).toBe(true);

    expect(

      jsonHas(

        tree,

        "Votre demande sera transmise après le paiement"

      )

    ).toBe(true);

  });



  it("« Modifier ma demande » revient à l’écran de demande (1/2)", async () => {

    let tree!: ReturnType<typeof create>;

    act(() => {

      tree = create(

        <TestRoot>

          <ClientBookingCreateScreen _testFormStep="summary" />

        </TestRoot>

      );

    });

    expect(jsonHas(tree, "Continuer vers le paiement")).toBe(true);

    const back = findByTestId(tree.root, "booking-cta-back-details");

    await act(async () => {

      back.props.onPress?.();

    });

    expect(jsonHas(tree, "1/2 · Demande")).toBe(true);

    expect(jsonHas(tree, "Continuer vers le récapitulatif")).toBe(true);

    expect(jsonHas(tree, "Continuer vers le paiement")).toBe(false);

  });



  it("Domicile sans lat/lon profil : message à confirmer après clic Domicile", async () => {

    mockUseClientProfileQuery.mockReturnValue({

      data: {

        domicile: {

          address: "Rue du Test 1",

          zip: "1200",

          city: "Genève",

          lat: null,

          lon: null,

        },

      },

      isLoading: false,

    });

    let tree!: ReturnType<typeof create>;

    act(() => {

      tree = create(

        <TestRoot>

          <ClientBookingCreateScreen />

        </TestRoot>

      );

    });

    const home = findByTestId(tree.root, "booking-pickup-home");

    await act(async () => {

      home.props.onPress?.();

    });

    expect(jsonHas(tree, "domicile à confirmer")).toBe(true);

    expect(preview).not.toHaveBeenCalled();

    expect(createB).not.toHaveBeenCalled();

  });



  it("texte d’adresse saisi sans sélection : affiche l’aide liste (point exact)", async () => {

    let tree!: ReturnType<typeof create>;

    act(() => {

      tree = create(

        <TestRoot>

          <ClientBookingCreateScreen />

        </TestRoot>

      );

    });

    const inputs = tree.root.findAll(

      (n) => (n as { props?: { accessibilityLabel?: string } }).props?.accessibilityLabel === "Adresse de prise en charge"

    );

    const input = inputs[0] as { props: { onChangeText?: (t: string) => void } };

    await act(async () => {

      input.props.onChangeText?.("12 rue inventée, Genève");

      await Promise.resolve();

    });

    expect(jsonHas(tree, "Sélectionnez une adresse dans la liste")).toBe(true);

  });



  it("seule suggestion géoloc : reconnaissance automatique si le texte saisi correspond (sans tap)", async () => {

    autocompleteMock.mockResolvedValue([

      { label: "Avenue Uniq 1", address: "Avenue Uniq 1, Genève", lat: 46.2, lon: 6.14, place_id: "p1" },

    ]);

    let tree!: ReturnType<typeof create>;

    act(() => {

      tree = create(

        <TestRoot>

          <ClientBookingCreateScreen />

        </TestRoot>

      );

    });

    const input = tree.root.findAll(

      (n) => (n as { props?: { accessibilityLabel?: string } }).props?.accessibilityLabel === "Adresse de prise en charge"

    )[0] as { props: { onChangeText?: (t: string) => void } };

    const exact = "Avenue Uniq 1, Genève";

    await act(async () => {

      input.props.onChangeText?.(exact);

      await Promise.resolve();

    });

    expect(jsonHas(tree, "Adresse reconnue")).toBe(true);

    expect(jsonHas(tree, "Sélectionnez une adresse dans la liste")).toBe(false);

    act(() => {

      tree.unmount();

    });

  });



  it("suggestion géolocalisée choisie : affiche Adresse reconnue", async () => {

    autocompleteMock.mockResolvedValue([

      { label: "Point A", address: "Point A", lat: 46.2, lon: 6.14 },

    ]);

    let tree!: ReturnType<typeof create>;

    act(() => {

      tree = create(

        <TestRoot>

          <ClientBookingCreateScreen />

        </TestRoot>

      );

    });

    const inputs = tree.root.findAll(

      (n) => (n as { props?: { accessibilityLabel?: string } }).props?.accessibilityLabel === "Adresse de prise en charge"

    );

    const input = inputs[0] as { props: { onChangeText?: (t: string) => void } };

    await act(async () => {

      input.props.onChangeText?.("Point");

      await Promise.resolve();

    });

    const sug = tree.root.findAll(

      (n) =>

        (n as { props?: { testID?: string; onPress?: unknown } }).props?.testID ===

          "booking-suggestion-pickup-0" &&

        typeof (n as { props: { onPress?: unknown } }).props.onPress === "function"

    );

    expect(sug.length).toBeGreaterThanOrEqual(1);

    await act(async () => {

      (sug[0] as { props: { onPress?: () => void } }).props.onPress?.();

    });

    expect(jsonHas(tree, "Adresse reconnue")).toBe(true);

    act(() => {

      tree.unmount();

    });

  });



  it("demande : adresses univoques mènent au récap 2/2 sans preview ni create", async () => {

    autocompleteMock.mockImplementation(async (q: string) => {

      const t = q.trim();

      if (t.includes("DepartU")) {

        return [{ label: "DepartU X", address: "DepartU X, Genève", lat: 46.2, lon: 6.1 }];

      }

      if (t.includes("ArriveeU")) {

        return [{ label: "ArriveeU Y", address: "ArriveeU Y, Nyon", lat: 46.21, lon: 6.11 }];

      }

      return [];

    });

    let tree!: ReturnType<typeof create>;

    act(() => {

      tree = create(

        <TestRoot>

          <ClientBookingCreateScreen />

        </TestRoot>

      );

    });

    const pickupIn = tree.root.findAll(

      (n) => (n as { props?: { accessibilityLabel?: string } }).props?.accessibilityLabel === "Adresse de prise en charge"

    )[0] as { props: { onChangeText?: (t: string) => void } };

    const dropIn = tree.root.findAll(

      (n) => (n as { props?: { accessibilityLabel?: string } }).props?.accessibilityLabel === "Adresse de destination"

    )[0] as { props: { onChangeText?: (t: string) => void } };

    await act(async () => {

      pickupIn.props.onChangeText?.("DepartU X, Genève");

      dropIn.props.onChangeText?.("ArriveeU Y, Nyon");

      await Promise.resolve();

    });

    await act(async () => {

      findByTestId(tree.root, "booking-cta-go-summary").props.onPress?.();

      await Promise.resolve();

    });

    expect(preview).not.toHaveBeenCalled();

    expect(createB).not.toHaveBeenCalled();

    expect(jsonHas(tree, "2/2 · Récapitulatif")).toBe(true);

    act(() => {

      tree.unmount();

    });

  });



  it("domicile avec coords : variante de texte (normAddrKey) autorise le récapitulatif", async () => {

    mockUseClientProfileQuery.mockReturnValue({

      data: {

        domicile: {

          address: "Avenue TestNorm 1",

          zip: "1203",

          city: "Genève",

          lat: 46.2,

          lon: 6.14,

        },

      },

      isLoading: false,

    });



    autocompleteMock.mockImplementation(async (q: string) => {

      const t = q.trim();

      if (t.includes("DestNorm")) {

        return [{ label: "DestNorm Y", address: "DestNorm Y, Nyon", lat: 46.21, lon: 6.11 }];

      }

      return [];

    });



    let tree!: ReturnType<typeof create>;

    act(() => {

      tree = create(

        <TestRoot>

          <ClientBookingCreateScreen />

        </TestRoot>

      );

    });



    const pickupIn = tree.root.findAll(

      (n) => (n as { props?: { accessibilityLabel?: string } }).props?.accessibilityLabel === "Adresse de prise en charge"

    )[0] as { props: { onChangeText?: (t: string) => void } };



    const dropIn = tree.root.findAll(

      (n) => (n as { props?: { accessibilityLabel?: string } }).props?.accessibilityLabel === "Adresse de destination"

    )[0] as { props: { onChangeText?: (t: string) => void } };



    await act(async () => {

      pickupIn.props.onChangeText?.(

        "Avenue TestNorm 1, 1203, Genève, Suisse"

      );

      dropIn.props.onChangeText?.("DestNorm Y, Nyon");

      await Promise.resolve();

    });



    await act(async () => {

      findByTestId(tree.root, "booking-cta-go-summary").props.onPress?.();

      await Promise.resolve();

    });



    expect(jsonHas(tree, "2/2 · Récapitulatif")).toBe(true);

  });



  it("suggestion Google (place_id seulement) + place-details OK : adresse reconnue", async () => {

    getPlaceDetailsMock.mockResolvedValue({

      address: "Ligne Après Détails, Genève",

      place_id: "g1",

      label: "Ligne Après Détails, Genève",

      lat: 46.204,

      lon: 6.143,

    });



    autocompleteMock.mockResolvedValue([

      {

        label: "Prédiction G",

        address: "Prédiction G",

        place_id: "g1",

        lat: null,

        lon: null,

      },

      {

        label: "Autre G",

        address: "Autre G",

        place_id: "g2",

        lat: null,

        lon: null,

      },

    ]);



    let tree!: ReturnType<typeof create>;

    act(() => {

      tree = create(

        <TestRoot>

          <ClientBookingCreateScreen />

        </TestRoot>

      );

    });



    const input = tree.root.findAll(

      (n) => (n as { props?: { accessibilityLabel?: string } }).props?.accessibilityLabel === "Adresse de prise en charge"

    )[0] as { props: { onChangeText?: (t: string) => void } };



    await act(async () => {

      input.props.onChangeText?.("Pré");

      await Promise.resolve();

    });



    const sug = tree.root.findAll(

      (n) =>

        (n as { props?: { testID?: string; onPress?: unknown } }).props?.testID ===

          "booking-suggestion-pickup-0" &&

        typeof (n as { props: { onPress?: unknown } }).props.onPress === "function"

    );



    expect(sug.length).toBeGreaterThanOrEqual(1);



    await act(async () => {

      (sug[0] as { props: { onPress?: () => void } }).props.onPress?.();

      await Promise.resolve();

    });



    expect(getPlaceDetailsMock).toHaveBeenCalledWith("g1");

    expect(jsonHas(tree, "Adresse reconnue")).toBe(true);

  });



  it("suggestion place_id seulement + place-details KO : Adresse à préciser", async () => {

    getPlaceDetailsMock.mockResolvedValue(null);



    autocompleteMock.mockResolvedValue([

      {

        label: "SansCoords",

        address: "SansCoords",

        place_id: "n1",

        lat: null,

        lon: null,

      },

    ]);



    let tree!: ReturnType<typeof create>;

    act(() => {

      tree = create(

        <TestRoot>

          <ClientBookingCreateScreen />

        </TestRoot>

      );

    });



    const input = tree.root.findAll(

      (n) => (n as { props?: { accessibilityLabel?: string } }).props?.accessibilityLabel === "Adresse de prise en charge"

    )[0] as { props: { onChangeText?: (t: string) => void } };



    await act(async () => {

      input.props.onChangeText?.("San");

      await Promise.resolve();

    });



    const sug = tree.root.findAll(

      (n) =>

        (n as { props?: { testID?: string; onPress?: unknown } }).props?.testID ===

          "booking-suggestion-pickup-0" &&

        typeof (n as { props: { onPress?: unknown } }).props.onPress === "function"

    );



    await act(async () => {

      (sug[0] as { props: { onPress?: () => void } }).props.onPress?.();

      await Promise.resolve();

    });



    expect(jsonHas(tree, "Adresse à préciser")).toBe(true);

  });

});


