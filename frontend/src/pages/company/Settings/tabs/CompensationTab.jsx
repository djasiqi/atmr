import React, { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { FiChevronDown } from 'react-icons/fi';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import {
  createCompensationPolicy,
  fetchCompensationPolicies,
  fetchWorkTimeSettings,
  saveWorkTimeSettings,
} from '../../../../services/companyService';
import s from './CompensationTab.module.css';

const WORK_TYPES = [
  ['extra_transport', 'Transport supplémentaire'],
  ['delivery', 'Livraison'],
  ['accompaniment', 'Accompagnement'],
  ['waiting', 'Attente'],
  ['administrative', 'Administratif'],
  ['cleaning', 'Nettoyage'],
  ['training', 'Formation'],
  ['other', 'Autre'],
];

const TYPE_LABELS = Object.fromEntries(WORK_TYPES);

const MODE_OPTIONS = [
  { value: '', label: 'Non défini' },
  { value: 'flat', label: 'Forfait' },
  { value: 'real_time', label: 'Temps réel' },
];

function ChipMenu({ label, value, options, onChange }) {
  const [open, setOpen] = useState(false);
  const btnRef = useRef(null);
  const menuRef = useRef(null);
  const [pos, setPos] = useState({ top: 0, left: 0, width: 0 });
  const current = options.find((option) => option.value === value);

  useEffect(() => {
    if (!open) return undefined;
    const place = () => {
      if (!btnRef.current) return;
      const rect = btnRef.current.getBoundingClientRect();
      setPos({ top: rect.bottom + 4, left: rect.left, width: rect.width });
    };
    place();
    const selected = menuRef.current?.querySelector('[aria-selected="true"]');
    if (typeof selected?.scrollIntoView === 'function') {
      selected.scrollIntoView({ block: 'nearest' });
    }
    const onPointer = (event) => {
      if (btnRef.current?.contains(event.target) || menuRef.current?.contains(event.target)) return;
      setOpen(false);
    };
    const onKey = (event) => {
      if (event.key !== 'Escape') return;
      event.preventDefault();
      event.stopPropagation();
      setOpen(false);
    };
    window.addEventListener('scroll', place, true);
    window.addEventListener('resize', place);
    document.addEventListener('mousedown', onPointer);
    document.addEventListener('keydown', onKey, true);
    return () => {
      window.removeEventListener('scroll', place, true);
      window.removeEventListener('resize', place);
      document.removeEventListener('mousedown', onPointer);
      document.removeEventListener('keydown', onKey, true);
    };
  }, [open]);

  return (
    <div className={s.chipDrop}>
      <button
        ref={btnRef}
        type="button"
        className={`${s.chipBtn} ${open ? s.chipBtnOpen : ''}`}
        aria-label={label}
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => setOpen((currentOpen) => !currentOpen)}
      >
        <span className={s.chipText}>{current?.label || 'Choisir'}</span>
        <FiChevronDown size={11} className={`${s.chipArrow} ${open ? s.chipArrowOpen : ''}`} />
      </button>
      {open
        ? createPortal(
            <div
              ref={menuRef}
              className={s.chipMenu}
              role="listbox"
              aria-label={label}
              style={{ top: pos.top, left: pos.left, width: pos.width }}
            >
              {options.map((option) => (
                <button
                  key={option.label}
                  type="button"
                  role="option"
                  aria-selected={option.value === value}
                  className={`${s.chipOption} ${option.value === value ? s.chipOptionActive : ''}`}
                  onClick={() => {
                    onChange(option.value);
                    setOpen(false);
                  }}
                >
                  {option.label}
                </button>
              ))}
            </div>,
            document.body,
          )
        : null}
    </div>
  );
}

function effectDateOptions(selected) {
  const start = new Date();
  start.setHours(12, 0, 0, 0);
  const pad = (n) => String(n).padStart(2, '0');
  const options = [];
  for (let offset = 0; offset < 180; offset += 1) {
    const cursor = new Date(start);
    cursor.setDate(start.getDate() + offset);
    const iso = `${cursor.getFullYear()}-${pad(cursor.getMonth() + 1)}-${pad(cursor.getDate())}`;
    options.push({ value: iso, label: formatDay(iso) });
  }
  if (selected && !options.some((option) => option.value === selected)) {
    options.unshift({ value: selected, label: formatDay(selected) });
  }
  return options;
}

function todayIso() {
  const now = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
}

function formatDay(iso) {
  if (!iso) return '—';
  const [year, month, day] = String(iso).slice(0, 10).split('-').map(Number);
  if (!year || !month || !day) return String(iso);
  return new Intl.DateTimeFormat('fr-CH', {
    day: 'numeric',
    month: 'long',
    year: 'numeric',
  }).format(new Date(year, month - 1, day));
}

function ruleLabel(type, rule) {
  const name = TYPE_LABELS[type] || type;
  if (!rule?.mode) return null;
  if (rule.mode === 'real_time') return `${name} : temps réel`;
  return `${name} : forfait ${Number(rule.minutes) || 0} min`;
}

export default function CompensationTab() {
  const queryClient = useQueryClient();
  const policies = useQuery({
    queryKey: ['lirie', 'compensation-policies'],
    queryFn: fetchCompensationPolicies,
  });
  const settings = useQuery({
    queryKey: ['lirie', 'work-time-settings'],
    queryFn: fetchWorkTimeSettings,
  });
  const [form, setForm] = useState({
    effective_from: todayIso(),
    mode: 'flat_per_trip',
    transport_flat_minutes: '30',
    notes: '',
  });
  const [margin, setMargin] = useState('5');
  const [estimateEnabled, setEstimateEnabled] = useState(true);
  const [settingsSaved, setSettingsSaved] = useState(false);
  const [rules, setRules] = useState({});
  const [error, setError] = useState('');
  const [saved, setSaved] = useState(false);
  const [saving, setSaving] = useState(false);

  const setField = (key) => (event) => {
    setSaved(false);
    setForm((prev) => ({ ...prev, [key]: event.target.value }));
  };

  const setRule = (key, value) => {
    setSaved(false);
    setRules((prev) => ({ ...prev, [key]: value }));
  };

  const submit = async (event) => {
    event.preventDefault();
    setError('');
    setSaved(false);
    const workTypeRules = {};
    WORK_TYPES.forEach(([type]) => {
      const mode = rules[`${type}_mode`];
      if (!mode) return;
      workTypeRules[type] = {
        mode,
        minutes: Number(rules[`${type}_minutes`] || 0),
      };
    });
    setSaving(true);
    try {
      await createCompensationPolicy({
        effective_from: form.effective_from,
        transport_flat_minutes: Number(form.transport_flat_minutes),
        mode: form.mode,
        notes: form.notes,
        work_type_rules: workTypeRules,
      });
      queryClient.invalidateQueries({ queryKey: ['lirie', 'compensation-policies'] });
      setSaved(true);
    } catch (err) {
      setError(err?.response?.data?.error || err?.error || 'Enregistrement impossible.');
    } finally {
      setSaving(false);
    }
  };

  const flat = form.mode === 'flat_per_trip';
  const history = policies.data?.policies || [];

  const saveEstimate = async (event) => {
    event.preventDefault();
    setError('');
    setSettingsSaved(false);
    const minutes = Number(margin);
    if (!Number.isFinite(minutes) || minutes < 0) {
      setError('La marge doit être un nombre de minutes positif.');
      return;
    }
    try {
      await saveWorkTimeSettings({
        route_margin_minutes: minutes,
        route_estimate_enabled: estimateEnabled,
      });
      queryClient.invalidateQueries({ queryKey: ['lirie', 'work-time-settings'] });
      setSettingsSaved(true);
    } catch (err) {
      setError(err?.response?.data?.error || err?.error || 'Réglage impossible.');
    }
  };

  useEffect(() => {
    if (!settings.data) return;
    setMargin(String(settings.data.route_margin_minutes ?? 5));
    setEstimateEnabled(settings.data.route_estimate_enabled !== false);
  }, [settings.data]);

  return (
    <section className={s.page}>
      <header className={s.head}>
        <h2>Rémunération chauffeurs</h2>
        <p className={s.lead}>
          Chaque version s’applique à partir de sa date d’effet. Les courses déjà
          réalisées conservent la règle en vigueur à leur date de fin.
        </p>
      </header>

      <form className={s.card} onSubmit={saveEstimate}>
        <h3>Calcul automatique du temps de travail</h3>
        <p className={s.hint}>
          Le temps de trajet est la durée estimée de la réservation. Les minutes
          ci-dessous sont offertes au chauffeur pour chaque course. Elles ne
          corrigent pas le trajet, et ce n’est pas une règle de rémunération.
        </p>
        <div className={s.grid}>
          <label className={s.field}>
            Minutes offertes par course
            <input type="number" min="0" value={margin} onChange={(event) => { setSettingsSaved(false); setMargin(event.target.value); }} />
          </label>
          <div className={s.field}>
            <span>Estimation</span>
            <ChipMenu
              label="Estimation"
              value={estimateEnabled ? 'on' : 'off'}
              options={[
                { value: 'on', label: 'Activée' },
                { value: 'off', label: 'Désactivée' },
              ]}
              onChange={(value) => {
                setSettingsSaved(false);
                setEstimateEnabled(value === 'on');
              }}
            />
          </div>
        </div>
        {settingsSaved ? <p className={s.success}>Marge enregistrée.</p> : null}
        <div className={s.actions}>
          <button type="submit" className={s.primary}>Enregistrer la marge</button>
        </div>
      </form>

      <form className={s.card} onSubmit={submit}>
        <h3>Nouvelle version</h3>
        <fieldset className={s.modes}>
          <legend>Mode de rémunération des transports</legend>
          <label>
            <input
              type="radio"
              name="transport-mode"
              checked={flat}
              onChange={() => setField('mode')({ target: { value: 'flat_per_trip' } })}
            />
            Forfait par transport
          </label>
          <label>
            <input
              type="radio"
              name="transport-mode"
              checked={!flat}
              onChange={() => setField('mode')({ target: { value: 'validated_work_time' } })}
            />
            Temps réel
          </label>
        </fieldset>
        <p className={s.hint}>
          {flat
            ? 'Chaque course terminée compte une fois cette durée, y compris dans un aller-retour.'
            : 'La clôture retient le temps vérifié ou validé. La durée forfaitaire reste enregistrée pour l’affichage.'}
        </p>
        <div className={s.grid}>
          <div className={s.field}>
            <span>Date d’effet</span>
            <ChipMenu
              label="Date d’effet"
              value={form.effective_from}
              options={effectDateOptions(form.effective_from)}
              onChange={(value) => setField('effective_from')({ target: { value } })}
            />
          </div>
          <label className={s.field}>
            Durée forfaitaire par transport
            <input
              type="number"
              min="0"
              value={form.transport_flat_minutes}
              onChange={setField('transport_flat_minutes')}
              required
            />
          </label>
        </div>

        <h3>Temps ajoutés</h3>
        <p className={s.hint}>
          En temps réel, la durée saisie est toujours comptée. En forfait, une règle absente
          ou non définie reste à vérifier et n’entre pas dans le total.
        </p>
        <div className={s.rules}>
          {WORK_TYPES.map(([type, label]) => {
            const mode = rules[`${type}_mode`] || '';
            return (
              <div className={s.rule} key={type}>
                <span className={s.ruleName}>{label}</span>
                <div className={s.field}>
                  <span>Mode</span>
                  <ChipMenu
                    label={`Mode de ${label}`}
                    value={mode}
                    options={MODE_OPTIONS}
                    onChange={(value) => setRule(`${type}_mode`, value)}
                  />
                </div>
                <label className={s.field}>
                  Minutes
                  <input
                    className={s.minutes}
                    type="number"
                    min="0"
                    placeholder="0"
                    disabled={mode !== 'flat'}
                    value={rules[`${type}_minutes`] || ''}
                    onChange={(event) => setRule(`${type}_minutes`, event.target.value)}
                  />
                </label>
              </div>
            );
          })}
        </div>

        <label className={s.field}>
          Note
          <textarea value={form.notes} onChange={setField('notes')} placeholder="Facultatif" />
        </label>
        {error ? <p className={s.error}>{error}</p> : null}
        {saved ? <p className={s.success}>Version enregistrée.</p> : null}
        <div className={s.actions}>
          <button type="submit" className={s.primary} disabled={saving}>
            {saving ? 'Enregistrement…' : 'Enregistrer une version'}
          </button>
        </div>
      </form>

      <section className={s.card}>
        <h3>Historique</h3>
        {policies.isLoading ? <p className={s.empty}>Chargement…</p> : null}
        {!policies.isLoading && history.length === 0 ? (
          <p className={s.empty}>Aucune version enregistrée.</p>
        ) : null}
        <div className={s.history}>
          {history.map((policy) => {
            const extras = Object.entries(policy.work_type_rules || {})
              .map(([type, rule]) => ruleLabel(type, rule))
              .filter(Boolean);
            return (
              <article className={s.version} key={policy.id}>
                <div className={s.versionHead}>
                  <strong>
                    {formatDay(policy.effective_from)}
                    {policy.effective_until ? ` — ${formatDay(policy.effective_until)}` : ''}
                  </strong>
                  {!policy.effective_until ? <span className={s.badge}>En cours</span> : null}
                </div>
                <p className={s.metrics}>
                  <span>{policy.mode === 'validated_work_time' || policy.mode === 'real_time' ? 'Temps réel' : 'Forfait par transport'}</span>
                  <span>{policy.transport_flat_minutes} min par transport</span>
                </p>
                {extras.length > 0 ? <p className={s.note}>{extras.join(' · ')}</p> : null}
                {policy.notes ? <p className={s.note}>{policy.notes}</p> : null}
              </article>
            );
          })}
        </div>
      </section>
    </section>
  );
}
