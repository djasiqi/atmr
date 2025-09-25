// src/pages/company/Dashboard/components/NewClientForm.jsx
import React, { useState } from "react";
import { Button } from "./ui/Button";
import { Input } from "./ui/Input";
import { Label } from "./ui/Label";
import AddressAutocomplete from "../../../../components/common/AddressAutocomplete";

export default function NewClientForm({
  initialName = "",
  onCancel,
  onSubmit,
}) {
  const [form, setForm] = useState({
    client_type: "SELF_SERVICE", // valeur par défaut
    first_name: initialName,
    last_name: "",
    email: "",
    phone: "",
    address: "",
    birth_date: "",             // ← Nouveau champ ajouté
  });
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((f) => ({ ...f, [name]: value }));
  };

  const handleSubmitForm = async (e) => {
    e.preventDefault();

    // Prépare le payload sans muter le state directement
    const payload = {
      ...form,
      client_type: form.email.trim() ? form.client_type : "PRIVATE",
      birth_date: form.birth_date || undefined, // ← On inclut la date de naissance si renseignée
    };

    setLoading(true);
    try {
      await onSubmit(payload);
      // onCancel() // si tu veux fermer la modal automatiquement
    } catch (err) {
      console.error("Erreur createClient:", err.response?.data || err);
      alert(err.response?.data?.error || "Erreur lors de la création du client");
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmitForm} className="space-y-4 p-4">
      <h3 className="text-lg font-semibold">Nouveau client</h3>

      {/* Prénom */}
      <Label>Prénom</Label>
      <Input
        name="first_name"
        value={form.first_name}
        onChange={handleChange}
        required
      />

      {/* Nom */}
      <Label>Nom</Label>
      <Input
        name="last_name"
        value={form.last_name}
        onChange={handleChange}
        required
      />

      {/* Email */}
      <Label>Email</Label>
      <Input
        name="email"
        type="email"
        value={form.email}
        onChange={handleChange}
        placeholder="laisser vide pour facturation interne"
      />

      {/* Téléphone */}
      <Label>Téléphone</Label>
      <Input
        name="phone"
        value={form.phone}
        onChange={handleChange}
      />

      {/* Adresse */}
      <Label>Adresse</Label>
      <AddressAutocomplete
        name="address"
        value={form.address}
        onChange={handleChange}
        placeholder="Adresse du client"
        required
        className="w-full px-3 py-2 border rounded"
      />

      {/* Date de naissance */}
      <Label>Date de naissance</Label>
      <Input
        name="birth_date"
        type="date"
        value={form.birth_date}
        onChange={handleChange}
        placeholder="YYYY-MM-DD"
      />

      {/* Actions */}
      <div className="flex justify-end gap-2 mt-4">
        <Button type="button" onClick={onCancel} disabled={loading}>
          Annuler
        </Button>
        <Button type="submit" disabled={loading}>
          {loading ? "Création…" : "Créer client"}
        </Button>
      </div>
    </form>
  );
}
