import React, { useState, useEffect } from "react";
import { View, ScrollView, Alert, TouchableOpacity, Image } from "react-native";
import { useAuth } from "@/hooks/useAuth";
import { updateDriverProfile, updateDriverPhoto } from "@/services/api";
import { InputField } from "@/components/ui/InputField";
import { Button } from "@/components/ui/Button";
import { Loader } from "@/components/ui/Loader";
import { ThemedText } from "@/components/ThemedText";
import { ThemedView } from "@/components/ThemedView";
import * as ImagePicker from "expo-image-picker";

// Import direct de l'image par défaut
import DefaultDriver from "../../assets/images/default-driver.png";

export default function ProfileScreen() {
  const { driver, refreshProfile } = useAuth();
  const [form, setForm] = useState({
    vehicle_assigned: "",
    brand: "",
    license_plate: "",
    phone: "",
    photo: "",
  });
  const [profileLoading, setProfileLoading] = useState(false);
  const [photoLoading, setPhotoLoading] = useState(false);

  useEffect(() => {
    if (driver) {
      setForm({
        vehicle_assigned: driver.vehicle_assigned,
        brand: driver.brand,
        license_plate: driver.license_plate,
        phone: driver.phone,
        photo: driver.photo,
      });
    }
  }, [driver]);

  const handleSaveProfile = async () => {
    setProfileLoading(true);
    try {
      await updateDriverProfile({
        vehicle_assigned: form.vehicle_assigned,
        brand: form.brand,
        license_plate: form.license_plate,
        phone: form.phone,
      });
      await refreshProfile();
      Alert.alert("Succès", "Votre profil a été mis à jour.");
    } catch (error) {
      Alert.alert("Erreur", "Échec de la mise à jour du profil.");
    } finally {
      setProfileLoading(false);
    }
  };

  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.7,
      base64: true,
    });

    if (!result.canceled && result.assets.length > 0) {
      const base64Img = `data:image/jpeg;base64,${result.assets[0].base64}`;
      setForm((prev) => ({ ...prev, photo: base64Img }));
    }
  };

  const handleSavePhoto = async () => {
    if (!form.photo) {
      Alert.alert("Erreur", "Aucune photo sélectionnée.");
      return;
    }
    setPhotoLoading(true);
    try {
      await updateDriverPhoto(form.photo);
      await refreshProfile();
      Alert.alert("Succès", "Photo mise à jour.");
    } catch (error) {
      Alert.alert("Erreur", "Impossible de mettre à jour la photo.");
    } finally {
      setPhotoLoading(false);
    }
  };

  if (!driver) {
    return (
      <ThemedView className="flex-1 justify-center items-center">
        <Loader />
      </ThemedView>
    );
  }

  return (
    <ScrollView className="flex-1 bg-gray-50 dark:bg-black px-4 pt-6">
      <ThemedText className="text-xl font-semibold mb-4">Mon Profil</ThemedText>

      <TouchableOpacity className="items-center mb-4" onPress={pickImage}>
        <Image
          source={form.photo ? { uri: form.photo } : DefaultDriver}
          className="w-32 h-32 rounded-full bg-gray-200"
        />
        <ThemedText className="mt-2 text-blue-500">Changer la photo</ThemedText>
      </TouchableOpacity>

      <Button
        onPress={handleSavePhoto}
        disabled={photoLoading}
        className="mb-6"
      >
        {photoLoading ? <Loader /> : "Enregistrer la photo"}
      </Button>

      <InputField
        label="Téléphone"
        value={form.phone}
        keyboardType="phone-pad"
        onChangeText={(phone) => setForm({ ...form, phone })}
      />

      <InputField
        label="Véhicule assigné"
        value={form.vehicle_assigned}
        onChangeText={(vehicle_assigned) =>
          setForm({ ...form, vehicle_assigned })
        }
      />

      <InputField
        label="Marque du véhicule"
        value={form.brand}
        onChangeText={(brand) => setForm({ ...form, brand })}
      />

      <InputField
        label="Plaque d'immatriculation"
        value={form.license_plate}
        onChangeText={(license_plate) => setForm({ ...form, license_plate })}
      />

      <Button
        onPress={handleSaveProfile}
        disabled={profileLoading}
        className="mt-6"
      >
        {profileLoading ? <Loader /> : "Enregistrer les modifications"}
      </Button>

      <View className="mb-8" />
    </ScrollView>
  );
}
