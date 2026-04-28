import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  createAdminPricingZoneSet,
  fetchServiceAreaZones,
  fetchAdminPricingZoneSetByKey,
  fetchAdminPricingZoneSets,
  updateAdminPricingZoneSet,
} from '../../../services/settingsService';
import IndicativeFareAdminSection from '../../../components/admin/IndicativeFareAdminSection';
import styles from './AdminSettings.module.css';
import shell from '../adminShell.module.css';

const CANTON_OPTIONS = [
  { value: 'GE', label: 'Genève (GE)' },
  { value: 'VD', label: 'Vaud (VD)' },
  { value: 'VS', label: 'Valais (VS)' },
  { value: 'NE', label: 'Neuchâtel (NE)' },
  { value: 'FR', label: 'Fribourg (FR)' },
  { value: 'JU', label: 'Jura (JU)' },
];

const slugify = (value) =>
  String(value || '')
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 40);

const buildZoneSetKey = ({ scope, label, existingKeys }) => {
  const scopePart = String(scope || '').trim().toLowerCase() || 'xx';
  const labelPart = slugify(label) || 'zonage';
  const base = `zoneset_${scopePart}_${labelPart}_v1`;
  if (!existingKeys.has(base)) return base;
  let idx = 2;
  let candidate = `${base}_${idx}`;
  while (existingKeys.has(candidate)) {
    idx += 1;
    candidate = `${base}_${idx}`;
  }
  return candidate;
};

const AdminSettings = () => {
  const toUiError = (err, fallback) => {
    const apiError = err?.response?.data?.error;
    const apiMessage = err?.response?.data?.message;
    const status = Number(err?.response?.status || 0);
    if (apiError === 'not_found' || status === 404) {
      return "Endpoint introuvable. Redémarre le backend et vérifie que les routes /pricing/admin/zone-sets sont chargées.";
    }
    if (status === 403) {
      return "Accès refusé: connecte-toi avec un compte ADMIN plateforme.";
    }
    if (apiError === 'server_error' || status >= 500) {
      if (apiMessage) return `Erreur serveur: ${apiMessage}`;
      return "Erreur serveur interne. Vérifie les logs backend pour voir l'exception exacte.";
    }
    if (apiMessage) {
      return apiMessage;
    }
    return apiError || err?.message || fallback;
  };
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const [zoneSets, setZoneSets] = useState([]);
  const [selectedKey, setSelectedKey] = useState('');

  const [createForm, setCreateForm] = useState({
    label: '',
    scope: 'GE',
  });

  const [editForm, setEditForm] = useState({
    label: '',
    scope: '',
    version: 1,
    is_active: true,
  });
  const [editZones, setEditZones] = useState([]);
  const [tokenAssignments, setTokenAssignments] = useState({});
  const [tokenMeta, setTokenMeta] = useState({});
  const [activeZoneCode, setActiveZoneCode] = useState('');
  const [communeQuery, setCommuneQuery] = useState('');
  const [communeResults, setCommuneResults] = useState([]);
  const [loadingCommunes, setLoadingCommunes] = useState(false);
  const [showAdvancedFields, setShowAdvancedFields] = useState(false);
  const [listScopeFilter, setListScopeFilter] = useState('ALL');
  const [listStatusFilter, setListStatusFilter] = useState('ALL');

  const selected = useMemo(
    () => zoneSets.find((item) => String(item.key) === String(selectedKey)) || null,
    [zoneSets, selectedKey]
  );

  const generatedKeyPreview = useMemo(() => {
    const existingKeys = new Set(zoneSets.map((item) => String(item?.key || '')));
    return buildZoneSetKey({
      scope: createForm.scope,
      label: createForm.label,
      existingKeys,
    });
  }, [createForm.scope, createForm.label, zoneSets]);

  const hydrateList = useCallback(async () => {
    const items = await fetchAdminPricingZoneSets({ active: null, limit: 300 });
    setZoneSets(items);
    if (!selectedKey && items.length > 0) {
      setSelectedKey(items[0].key);
    }
  }, [selectedKey]);

  const loadSelected = useCallback(async (key) => {
    if (!key) return;
    const detail = await fetchAdminPricingZoneSetByKey(key);
    if (!detail) return;
    const zones = Array.isArray(detail.zones) ? detail.zones : [];
    const memberships = Array.isArray(detail.memberships) ? detail.memberships : [];
    const zoneById = new Map(zones.map((zone) => [zone.id, zone.code]));

    const nextZones = zones.map((zone) => ({
      code: String(zone.code || '').trim(),
      label: String(zone.label || zone.code || '').trim(),
    }));
    const assignments = {};
    memberships.forEach((membership) => {
      const code = zoneById.get(membership.zone_id);
      const token = String(membership.commune_token || '').trim();
      if (code && token.startsWith('commune:')) {
        assignments[token] = code;
      }
    });
    setEditForm({
      label: detail.label || '',
      scope: detail.scope || '',
      version: Number(detail.version || 1),
      is_active: Boolean(detail.active),
    });
    const normalizedZones =
      nextZones.length > 0 ? nextZones : [{ code: 'Z1', label: 'Zone principale' }];
    setEditZones(normalizedZones);
    setTokenAssignments(assignments);
    setActiveZoneCode(normalizedZones[0]?.code || 'Z1');

    const tokens = Object.keys(assignments);
    if (tokens.length > 0) {
      const zoneItems = await fetchServiceAreaZones({
        tokens,
        types: 'commune',
        limit: 500,
      });
      const nextMeta = {};
      zoneItems.forEach((item) => {
        const token = String(item?.token || '').trim();
        if (!token) return;
        nextMeta[token] = {
          name: String(item?.name || token),
          canton_code: String(item?.canton_code || ''),
        };
      });
      setTokenMeta(nextMeta);
    } else {
      setTokenMeta({});
    }
  }, []);

  useEffect(() => {
    const load = async () => {
      try {
        setLoading(true);
        setError('');
        await hydrateList();
      } catch (err) {
        setError(toUiError(err, 'Erreur de chargement des zone sets.'));
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [hydrateList]);

  useEffect(() => {
    if (!selectedKey) return;
    loadSelected(selectedKey).catch((err) => {
      setError(toUiError(err, 'Erreur de chargement du zone set.'));
    });
  }, [selectedKey, loadSelected]);

  const hasDuplicateZoneCode = useMemo(() => {
    const seen = new Set();
    for (const zone of editZones) {
      const code = String(zone.code || '').trim();
      if (!code) continue;
      if (seen.has(code)) return true;
      seen.add(code);
    }
    return false;
  }, [editZones]);

  const membershipRowsByZone = useMemo(() => {
    const grouped = {};
    Object.entries(tokenAssignments).forEach(([token, zoneCode]) => {
      if (!grouped[zoneCode]) grouped[zoneCode] = [];
      grouped[zoneCode].push(token);
    });
    return grouped;
  }, [tokenAssignments]);

  const communeResultsUnassigned = useMemo(
    () =>
      communeResults.filter((item) => {
        const token = String(item?.token || '').trim();
        if (!token.startsWith('commune:')) return false;
        return !tokenAssignments[token];
      }),
    [communeResults, tokenAssignments]
  );
  const defaultZoneCode = useMemo(() => String(editZones[0]?.code || 'Z1').trim(), [editZones]);
  const zoneCodeForAssignment = showAdvancedFields ? activeZoneCode : defaultZoneCode;
  const flatAssignedTokens = useMemo(() => {
    const tokens = Object.keys(tokenAssignments);
    return tokens.sort((a, b) => {
      const nameA = String(tokenMeta[a]?.name || a);
      const nameB = String(tokenMeta[b]?.name || b);
      return nameA.localeCompare(nameB);
    });
  }, [tokenAssignments, tokenMeta]);

  const filteredSortedZoneSets = useMemo(() => {
    const byScope = zoneSets.filter((item) => {
      if (listScopeFilter === 'ALL') return true;
      return String(item?.scope || '').toUpperCase() === listScopeFilter;
    });
    const byStatus = byScope.filter((item) => {
      if (listStatusFilter === 'ALL') return true;
      if (listStatusFilter === 'ACTIVE') return Boolean(item?.active);
      if (listStatusFilter === 'INACTIVE') return !item?.active;
      return true;
    });
    return [...byStatus].sort((a, b) => {
      const activeA = a?.active ? 1 : 0;
      const activeB = b?.active ? 1 : 0;
      if (activeA !== activeB) return activeB - activeA;
      const scopeCmp = String(a?.scope || '').localeCompare(String(b?.scope || ''));
      if (scopeCmp !== 0) return scopeCmp;
      return String(a?.label || '').localeCompare(String(b?.label || ''));
    });
  }, [zoneSets, listScopeFilter, listStatusFilter]);

  const zoneSetStats = useMemo(() => {
    const total = zoneSets.length;
    const active = zoneSets.filter((item) => Boolean(item?.active)).length;
    const inactive = Math.max(total - active, 0);
    const communesCovered = zoneSets.reduce(
      (sum, item) => sum + Number(item?.communes_count || 0),
      0
    );
    const selectedCommunes = Object.keys(tokenAssignments).length;
    return { total, active, inactive, communesCovered, selectedCommunes };
  }, [zoneSets, tokenAssignments]);

  const formatCommuneLabel = (token) => {
    const name = String(tokenMeta[token]?.name || token);
    const canton = String(tokenMeta[token]?.canton_code || '');
    if (name === token && token.startsWith('commune:')) {
      const code = token.split(':')[1] || '';
      return code ? `Commune ${code}` : token;
    }
    if (!canton) return name;
    if (name.includes(`(${canton})`)) return name;
    return `${name} (${canton})`;
  };

  useEffect(() => {
    const assignedTokens = Object.keys(tokenAssignments).filter((token) => token.startsWith('commune:'));
    if (assignedTokens.length === 0) return;
    const missingTokens = assignedTokens.filter((token) => {
      const name = String(tokenMeta[token]?.name || '').trim();
      if (!name) return true;
      if (name.toLowerCase() === token.toLowerCase()) return true;
      if (/^commune:\d+$/i.test(name)) return true;
      if (/^commune\s+\d+$/i.test(name)) return true;
      return false;
    });
    if (missingTokens.length === 0) return;
    let cancelled = false;
    const hydrateMissingTokenMeta = async () => {
      try {
        const items = await fetchServiceAreaZones({
          tokens: missingTokens,
          types: 'commune',
          limit: 500,
        });
        if (cancelled || !Array.isArray(items) || items.length === 0) return;
        setTokenMeta((prev) => {
          const next = { ...prev };
          items.forEach((item) => {
            const token = String(item?.token || '').trim();
            if (!token || !token.startsWith('commune:')) return;
            next[token] = {
              name: String(item?.name || token),
              canton_code: String(item?.canton_code || ''),
            };
          });
          return next;
        });
      } catch (_err) {
        // keep silent: display fallback label if lookup fails
      }
    };
    hydrateMissingTokenMeta();
    return () => {
      cancelled = true;
    };
  }, [tokenAssignments, tokenMeta]);

  const handleCreate = async () => {
    try {
      setSaving(true);
      setError('');
      setSuccess('');
      if (!createForm.scope.trim() || !createForm.label.trim()) {
        setError('Le canton et le nom du zone set sont obligatoires.');
        return;
      }
      const existingKeys = new Set(zoneSets.map((item) => String(item?.key || '')));
      const safeScope = createForm.scope.trim().toUpperCase();
      const generatedKey = buildZoneSetKey({
        scope: safeScope,
        label: createForm.label.trim(),
        existingKeys,
      });
      const created = await createAdminPricingZoneSet({
        key: generatedKey,
        label: createForm.label.trim(),
        scope: safeScope || null,
        version: 1,
        is_active: true,
      });
      await hydrateList();
      if (created?.key) {
        setSelectedKey(created.key);
      }
      setSuccess(`Zone set créé (${generatedKey}).`);
      setCreateForm({ label: '', scope: createForm.scope || 'GE' });
    } catch (err) {
      setError(toUiError(err, 'Erreur lors de la création.'));
    } finally {
      setSaving(false);
    }
  };

  const handleSaveSelected = async () => {
    if (!selectedKey) return;
    try {
      setSaving(true);
      setError('');
      setSuccess('');
      await updateAdminPricingZoneSet(selectedKey, {
        label: editForm.label.trim(),
        scope: editForm.scope.trim().toUpperCase() || null,
        version: Number(editForm.version || 1),
        is_active: Boolean(editForm.is_active),
        zones: editZones
          .map((zone) => ({
            code: String(zone.code || '').trim(),
            label: String(zone.label || zone.code || '').trim(),
          }))
          .filter((zone) => zone.code),
        memberships: Object.entries(tokenAssignments)
          .map(([commune_token, zone_code]) => ({ zone_code, commune_token }))
          .filter((item) => item.zone_code && item.commune_token.startsWith('commune:')),
      });
      await hydrateList();
      await loadSelected(selectedKey);
      setSuccess('Zone set mis à jour.');
    } catch (err) {
      setError(toUiError(err, 'Erreur lors de la mise à jour.'));
    } finally {
      setSaving(false);
    }
  };

  useEffect(() => {
    const q = String(communeQuery || '').trim();
    if (q.length < 2 || !selected) {
      setCommuneResults([]);
      return;
    }
    const timeout = window.setTimeout(async () => {
      try {
        setLoadingCommunes(true);
        const items = await fetchServiceAreaZones({
          q,
          types: 'commune',
          limit: 30,
          cantonCode: editForm.scope || undefined,
        });
        setCommuneResults(Array.isArray(items) ? items : []);
      } catch (_err) {
        setCommuneResults([]);
      } finally {
        setLoadingCommunes(false);
      }
    }, 250);
    return () => window.clearTimeout(timeout);
  }, [communeQuery, editForm.scope, selected]);

  const addZone = () => {
    const nextCode = `Z${editZones.length + 1}`;
    const next = [...editZones, { code: nextCode, label: `Zone ${nextCode}` }];
    setEditZones(next);
    if (!activeZoneCode) setActiveZoneCode(nextCode);
  };

  const updateZone = (index, patch) => {
    setEditZones((prev) => prev.map((zone, i) => (i === index ? { ...zone, ...patch } : zone)));
  };

  const removeZone = (index) => {
    const zoneCode = editZones[index]?.code;
    const nextZones = editZones.filter((_, i) => i !== index);
    setEditZones(nextZones);
    if (zoneCode) {
      setTokenAssignments((prev) => {
        const next = { ...prev };
        Object.keys(next).forEach((token) => {
          if (next[token] === zoneCode) delete next[token];
        });
        return next;
      });
    }
    if (activeZoneCode === zoneCode) {
      setActiveZoneCode(nextZones[0]?.code || '');
    }
  };

  const assignCommuneToActiveZone = (item) => {
    const token = String(item?.token || '').trim();
    if (!zoneCodeForAssignment || !token.startsWith('commune:')) return;
    setTokenAssignments((prev) => ({ ...prev, [token]: zoneCodeForAssignment }));
    setTokenMeta((prev) => ({
      ...prev,
      [token]: {
        name: String(item?.name || token),
        canton_code: String(item?.canton_code || ''),
      },
    }));
  };

  const removeAssignment = (token) => {
    setTokenAssignments((prev) => {
      const next = { ...prev };
      delete next[token];
      return next;
    });
  };

  return (
    <main className={shell.content}>
          <IndicativeFareAdminSection />

          <section className={styles.hero}>
            <h1>Paramètres administrateur</h1>
            <p>
              Configuration centrale des zone sets tarifaires plateforme utilisés par les sociétés.
            </p>
            <p>
              Une zone = un groupe de communes. Exemple: zone <strong>A</strong> = centre-ville
              (plusieurs `commune:xxxx`), zone <strong>B</strong> = rive droite, etc.
            </p>
            <div className={styles.workflowHint}>
              <span>1. Créer un zone set</span>
              <span>2. Sélectionner dans la liste</span>
              <span>3. Editer et enregistrer</span>
            </div>
          </section>

          {error && <div className={styles.error}>{error}</div>}
          {success && <div className={styles.success}>{success}</div>}

          <section className={styles.summaryGrid} aria-label="Synthese zone sets">
            <article className={styles.summaryCard}>
              <span>Total zone sets</span>
              <strong>{zoneSetStats.total}</strong>
            </article>
            <article className={styles.summaryCard}>
              <span>Actifs</span>
              <strong>{zoneSetStats.active}</strong>
            </article>
            <article className={styles.summaryCard}>
              <span>Inactifs</span>
              <strong>{zoneSetStats.inactive}</strong>
            </article>
            <article className={styles.summaryCard}>
              <span>Communes couvertes (cumule)</span>
              <strong>{zoneSetStats.communesCovered}</strong>
            </article>
            <article className={styles.summaryCard}>
              <span>Communes du zone set selectionne</span>
              <strong>{zoneSetStats.selectedCommunes}</strong>
            </article>
          </section>

          {loading ? (
            <section className={styles.placeholder}>
              <h2>Chargement…</h2>
            </section>
          ) : (
            <section className={styles.zoneGrid}>
              <div className={styles.card}>
                <h2>Créer un zone set</h2>
                <p className={styles.helperText}>
                  Étape 1: choisis le canton. Étape 2: donne un nom métier au zonage. La clé technique
                  est générée automatiquement (sans saisie manuelle).
                </p>
                <div className={styles.previewBox}>
                  <small>Clé technique prévisionnelle</small>
                  <code>{generatedKeyPreview}</code>
                </div>
                <div className={styles.formGroup}>
                  <label htmlFor="zoneset_scope">Canton</label>
                  <select
                    id="zoneset_scope"
                    value={createForm.scope}
                    onChange={(event) =>
                      setCreateForm((prev) => ({ ...prev, scope: event.target.value.toUpperCase() }))
                    }
                    disabled={saving}
                  >
                    {CANTON_OPTIONS.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </div>
                <div className={styles.formGroup}>
                  <label htmlFor="zoneset_label">Nom du zone set</label>
                  <input
                    id="zoneset_label"
                    value={createForm.label}
                    onChange={(event) => setCreateForm((prev) => ({ ...prev, label: event.target.value }))}
                    placeholder="Zonage Genève V1"
                    disabled={saving}
                  />
                </div>
                <button type="button" className={styles.primaryButton} onClick={handleCreate} disabled={saving}>
                  Créer
                </button>
              </div>

              <div className={styles.card}>
                <h2>Zone sets existants</h2>
                <div className={styles.listFilters}>
                  <div className={styles.formGroup}>
                    <label htmlFor="zoneset_filter_scope">Canton</label>
                    <select
                      id="zoneset_filter_scope"
                      value={listScopeFilter}
                      onChange={(event) => setListScopeFilter(event.target.value)}
                    >
                      <option value="ALL">Tous</option>
                      {CANTON_OPTIONS.map((option) => (
                        <option key={option.value} value={option.value}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className={styles.formGroup}>
                    <label htmlFor="zoneset_filter_status">Statut</label>
                    <select
                      id="zoneset_filter_status"
                      value={listStatusFilter}
                      onChange={(event) => setListStatusFilter(event.target.value)}
                    >
                      <option value="ALL">Tous</option>
                      <option value="ACTIVE">Actifs</option>
                      <option value="INACTIVE">Inactifs</option>
                    </select>
                  </div>
                </div>
                <p className={styles.helperText}>
                  {filteredSortedZoneSets.length} résultat(s) affiché(s)
                </p>
                <div className={styles.list}>
                  {filteredSortedZoneSets.map((item) => (
                    <button
                      type="button"
                      key={item.key || item.id}
                      className={`${styles.listItem} ${selectedKey === item.key ? styles.listItemActive : ''}`}
                      onClick={() => setSelectedKey(item.key)}
                    >
                      <strong>{item.label}</strong>
                      <span>
                        {item.scope || 'N/A'} · {item.active ? 'Actif' : 'Inactif'} ·{' '}
                        {Number(item.communes_count || 0)} communes
                      </span>
                      <small className={styles.secondaryMeta}>
                        {item.active ? 'Production' : 'Archive'}
                      </small>
                      {showAdvancedFields && (
                        <small className={styles.secondaryMeta}>{item.key}</small>
                      )}
                    </button>
                  ))}
                  {zoneSets.length === 0 && <p>Aucun zone set.</p>}
                  {zoneSets.length > 0 && filteredSortedZoneSets.length === 0 && (
                    <p>Aucun zone set ne correspond au filtre.</p>
                  )}
                </div>
              </div>

              <div className={styles.cardWide}>
                <h2>Éditer le zone set sélectionné</h2>
                {!selected ? (
                  <p>Sélectionne un zone set dans la liste.</p>
                ) : (
                  <>
                    <div className={styles.selectedMetaRow}>
                      <span className={styles.metaChip}>Clé: {selected.key}</span>
                      <span className={styles.metaChip}>
                        Scope: {String(editForm.scope || selected.scope || '').toUpperCase() || 'N/A'}
                      </span>
                      <span className={styles.metaChip}>
                        Statut: {editForm.is_active ? 'Actif' : 'Inactif'}
                      </span>
                    </div>
                    <div className={styles.advancedActions}>
                      <button
                        type="button"
                        className={styles.inlineButton}
                        onClick={() => setShowAdvancedFields((prev) => !prev)}
                        disabled={saving}
                      >
                        {showAdvancedFields ? 'Masquer mode avancé' : 'Afficher mode avancé'}
                      </button>
                    </div>
                    <div className={styles.formRow}>
                      <div className={styles.formGroup}>
                        <label>Nom du zonage</label>
                        <input value={editForm.label} onChange={(event) => setEditForm((prev) => ({ ...prev, label: event.target.value }))} disabled={saving} />
                      </div>
                      <div className={styles.formGroup}>
                        <label>Canton</label>
                        <select
                          value={editForm.scope}
                          onChange={(event) =>
                            setEditForm((prev) => ({ ...prev, scope: event.target.value.toUpperCase() }))
                          }
                          disabled={saving}
                        >
                          {CANTON_OPTIONS.map((option) => (
                            <option key={option.value} value={option.value}>
                              {option.label}
                            </option>
                          ))}
                        </select>
                      </div>
                      <label className={styles.toggleRow}>
                        <input type="checkbox" checked={editForm.is_active} onChange={(event) => setEditForm((prev) => ({ ...prev, is_active: event.target.checked }))} disabled={saving} />
                        Actif
                      </label>
                      {showAdvancedFields && (
                        <>
                          <div className={styles.formGroup}>
                            <label>Clé technique</label>
                            <input value={selected.key} disabled />
                          </div>
                          <div className={styles.formGroup}>
                            <label>Version</label>
                            <input type="number" min="1" value={editForm.version} onChange={(event) => setEditForm((prev) => ({ ...prev, version: Number(event.target.value || 1) }))} disabled={saving} />
                          </div>
                        </>
                      )}
                    </div>

                    {showAdvancedFields ? (
                      <>
                        <div className={styles.formGroup}>
                          <label>Zones de tarification (codes stables)</label>
                          <div className={styles.zoneEditorGrid}>
                            {editZones.map((zone, index) => (
                              <div key={`${zone.code || 'zone'}-${index}`} className={styles.zoneRow}>
                                <input
                                  value={zone.code}
                                  onChange={(event) => updateZone(index, { code: event.target.value.toUpperCase() })}
                                  placeholder="A"
                                  disabled={saving}
                                />
                                <input
                                  value={zone.label}
                                  onChange={(event) => updateZone(index, { label: event.target.value })}
                                  placeholder="Centre-ville"
                                  disabled={saving}
                                />
                                <button type="button" className={styles.inlineButton} onClick={() => removeZone(index)} disabled={saving}>
                                  Supprimer
                                </button>
                              </div>
                            ))}
                          </div>
                          <button type="button" className={styles.inlineButton} onClick={addZone} disabled={saving}>
                            + Ajouter zone
                          </button>
                          {hasDuplicateZoneCode && (
                            <small className={styles.helperText}>Chaque code zone doit être unique.</small>
                          )}
                        </div>

                        <div className={styles.formGroup}>
                          <label htmlFor="active_zone_code">Zone active pour l’assignation de communes</label>
                          <select
                            id="active_zone_code"
                            value={activeZoneCode}
                            onChange={(event) => setActiveZoneCode(event.target.value)}
                            disabled={saving || editZones.length === 0}
                          >
                            <option value="">Choisir une zone</option>
                            {editZones.map((zone) => (
                              <option key={zone.code} value={zone.code}>
                                {zone.code} - {zone.label}
                              </option>
                            ))}
                          </select>
                        </div>
                      </>
                    ) : (
                      <p className={styles.helperText}>
                        Mode simple: cherche des communes et ajoute-les directement au zonage.
                      </p>
                    )}

                    <div className={styles.formGroup}>
                      <label htmlFor="commune_search">Recherche commune</label>
                      <small className={styles.helperText}>
                        Les communes déjà assignées sont automatiquement exclues.
                      </small>
                      <input
                        id="commune_search"
                        value={communeQuery}
                        onChange={(event) => setCommuneQuery(event.target.value)}
                        placeholder="Ex: Genève, Carouge, Meyrin"
                        disabled={saving}
                      />
                      <div className={styles.resultsList}>
                        {loadingCommunes && <p>Recherche…</p>}
                        {!loadingCommunes && communeResultsUnassigned.map((item) => {
                          const token = String(item?.token || '');
                          return (
                            <button
                              type="button"
                              key={token}
                              className={styles.resultItem}
                              onClick={() => assignCommuneToActiveZone(item)}
                              disabled={!zoneCodeForAssignment || saving}
                            >
                              <span>{item?.name} ({item?.canton_code || '-'})</span>
                              <span className={styles.badge}>
                                Ajouter{showAdvancedFields ? ` à ${activeZoneCode || 'zone active'}` : ''}
                              </span>
                            </button>
                          );
                        })}
                        {!loadingCommunes && communeQuery.trim().length >= 2 && communeResultsUnassigned.length === 0 && (
                          <p>
                            Aucune commune disponible pour ce filtre (soit introuvable, soit déjà affectée
                            à une zone).
                          </p>
                        )}
                      </div>
                    </div>

                    <div className={styles.formGroup}>
                      <label>{showAdvancedFields ? 'Communes assignées par zone' : 'Communes du zonage'}</label>
                      {showAdvancedFields ? (
                        editZones.map((zone) => {
                          const tokens = membershipRowsByZone[zone.code] || [];
                          return (
                            <div key={`members-${zone.code}`} className={styles.groupBlock}>
                              <strong>{zone.code} - {zone.label}</strong>
                              {tokens.length === 0 ? (
                                <p>Aucune commune.</p>
                              ) : (
                                tokens.map((token) => (
                                  <div key={token} className={styles.tokenRow}>
                                    <span>{formatCommuneLabel(token)}</span>
                                    <button type="button" className={styles.inlineButton} onClick={() => removeAssignment(token)} disabled={saving}>
                                      Retirer
                                    </button>
                                  </div>
                                ))
                              )}
                            </div>
                          );
                        })
                      ) : (
                        <div className={styles.groupBlock}>
                          {flatAssignedTokens.length === 0 ? (
                            <p>Aucune commune.</p>
                          ) : (
                            flatAssignedTokens.map((token) => (
                              <div key={token} className={styles.tokenRow}>
                                <span>{formatCommuneLabel(token)}</span>
                                <button type="button" className={styles.inlineButton} onClick={() => removeAssignment(token)} disabled={saving}>
                                  Retirer
                                </button>
                              </div>
                            ))
                          )}
                        </div>
                      )}
                    </div>

                    <button type="button" className={styles.primaryButton} onClick={handleSaveSelected} disabled={saving}>
                      Enregistrer zone set
                    </button>
                  </>
                )}
              </div>
            </section>
          )}
    </main>
  );
};

export default AdminSettings;
