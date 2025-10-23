import React, { useState, useRef, useEffect } from 'react';
import AsyncSelect from 'react-select/async';
import { listServicesByEstab } from '../../services/companyService';

export default function ServiceSelect({
  establishmentId,
  value,
  onChange,
  placeholder = 'Choisir le service…',
  clearOnEstablishmentChange = true,
}) {
  const cacheRef = useRef(new Map());
  const [key, setKey] = useState(0); // Force re-render

  // Efface le cache et force le reload quand l'établissement change
  useEffect(() => {
    if (clearOnEstablishmentChange && establishmentId !== undefined) {
      onChange?.(null); // Efface la sélection
      cacheRef.current.clear(); // Efface le cache
      setKey((prev) => prev + 1); // Force re-render du AsyncSelect
    }
  }, [establishmentId, clearOnEstablishmentChange, onChange]);

  // Charge les options depuis l'API
  const loadOptions = async (inputValue = '') => {
    const id = Number(establishmentId);
    if (!Number.isFinite(id) || id <= 0) {
      return [];
    }

    // Cache par établissement
    const cacheKey = `estab_${establishmentId}`;

    if (!cacheRef.current.has(cacheKey)) {
      try {
        // Appel avec la nouvelle signature
        const services = await listServicesByEstab(id, inputValue);
        cacheRef.current.set(cacheKey, services || []);
      } catch (error) {
        cacheRef.current.set(cacheKey, []);
      }
    }

    const allServices = cacheRef.current.get(cacheKey) || [];

    // Filtrage côté client
    const query = (inputValue || '').toLowerCase();
    const filtered = allServices.filter((srv) => {
      const name = (srv.name || '').toLowerCase();
      const category = (srv.category || '').toLowerCase();
      return !query || name.includes(query) || category.includes(query);
    });

    // Format pour react-select
    const options = filtered.map((srv) => ({
      value: srv.id,
      label: srv.category ? `${srv.category} — ${srv.name}` : srv.name,
      data: srv,
    }));

    return options;
  };

  // Gestion du changement de sélection
  const handleChange = (selectedOption) => {
    if (selectedOption) {
      const serviceObj = {
        id: selectedOption.value,
        name: selectedOption.data?.name || selectedOption.label,
      };

      onChange?.(serviceObj);
    } else {
      onChange?.(null);
    }
  };

  // Valeur pour react-select
  const selectValue = value
    ? {
        value: value.id,
        label: value.name,
      }
    : null;

  return (
    <AsyncSelect
      key={key} // Force re-render quand établissement change
      cacheOptions
      loadOptions={loadOptions}
      defaultOptions={!!establishmentId} // Ne charge pas si pas d'établissement
      value={selectValue}
      onChange={handleChange}
      placeholder={establishmentId ? placeholder : "Sélectionnez d'abord un établissement"}
      isDisabled={!establishmentId}
      isClearable
      noOptionsMessage={({ inputValue }) =>
        !establishmentId
          ? "Sélectionnez d'abord un établissement"
          : inputValue
          ? `Aucun service trouvé pour "${inputValue}"`
          : 'Tapez pour rechercher un service'
      }
      loadingMessage={() => 'Chargement des services...'}
      styles={{
        container: (base) => ({ ...base, zIndex: 9999 }),
        menu: (base) => ({ ...base, zIndex: 9999 }),
      }}
    />
  );
}
