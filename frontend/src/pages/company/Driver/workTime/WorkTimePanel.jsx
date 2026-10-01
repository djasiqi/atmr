import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { FiChevronDown } from 'react-icons/fi';
import { useLocation, useNavigate } from 'react-router-dom';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import {
  cancelManualWorkEntry,
  createManualWorkEntry,
  createWorkTimeAdjustment,
  createWorkTimeDurationDecision,
  fetchBookingWorkTimeExplain,
  fetchDriverWorkTime,
  fetchWorkTimeSummary,
  finalizeWorkTimePeriod,
  reopenWorkTimePeriod,
} from '../../../../services/companyService';
import { getBusinessCalendarDate } from '../../../../utils/businessTime';
import {
  adjustmentFormDefaults,
  dayMetrics,
  formatCivilDate,
  formatMinutes,
  periodHeading,
  routeProposalCopy,
} from './workTimeFormat';
import InlineDatePicker from '../../../../components/ui/InlineDatePicker';
import InlineTimePicker from '../../../../components/ui/InlineTimePicker';
import { PeriodChrome, FleetTable, DriverSummary, ReviewQueue } from './WorkTimeViews';
import { monthClosureState } from './monthClosure';
import { periodRange, shiftPeriod } from './periodRange';
import s from './workTime.module.css';

const PRESETS = [
  { id: 'today', label: "Aujourd'hui" },
  { id: 'week', label: 'Cette semaine' },
  { id: 'month', label: 'Ce mois' },
  { id: 'custom', label: 'Personnalisée' },
];

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

export default function WorkTimePanel({ onOpenBooking, companyDrivers = [] } = {}) {
  const navigate = useNavigate();
  const location = useLocation();
  const queryClient = useQueryClient();
  const [preset, setPreset] = useState('month');
  const [anchor, setAnchor] = useState('');
  const [customFrom, setCustomFrom] = useState('');
  const [customTo, setCustomTo] = useState('');
  const [driverId, setDriverId] = useState(null);
  const [queueOpen, setQueueOpen] = useState(false);
  const [openDays, setOpenDays] = useState({});
  const [revealReviews, setRevealReviews] = useState(false);
  const [displayMode, setDisplayMode] = useState('real');
  const [rulesOpen, setRulesOpen] = useState(false);
  const [explainId, setExplainId] = useState(null);
  const [adjustEntry, setAdjustEntry] = useState(null);
  const [durationEntry, setDurationEntry] = useState(null);
  const [manualOpen, setManualOpen] = useState(false);
  const [error, setError] = useState('');
  const [validatingId, setValidatingId] = useState(null);

  const range = useMemo(() => {
    const now = anchor ? new Date(`${anchor}T12:00:00Z`) : new Date();
    return periodRange(preset, customFrom, customTo, now);
  }, [preset, customFrom, customTo, anchor]);

  const summary = useQuery({
    queryKey: ['lirie', 'company-work-time', range.from, range.to],
    queryFn: () => fetchWorkTimeSummary(range),
  });

  const detail = useQuery({
    queryKey: ['lirie', 'company-work-time-driver', driverId, range.from, range.to],
    queryFn: async () => {
      const perPage = 100;
      const first = await fetchDriverWorkTime(driverId, { ...range, filter: 'all', perPage });
      const total = Number(first.total_days || 0);
      const days = [...(first.days || [])];
      const pages = Math.ceil(total / perPage);
      for (let page = 2; page <= pages; page += 1) {
        const next = await fetchDriverWorkTime(driverId, {
          ...range,
          filter: 'all',
          page,
          perPage,
        });
        days.push(...(next.days || []));
      }
      return { ...first, days };
    },
    enabled: driverId != null,
  });

  const explain = useQuery({
    queryKey: ['lirie', 'company-work-time-explain', explainId],
    queryFn: () => fetchBookingWorkTimeExplain(explainId),
    enabled: explainId != null,
  });

  const kpis = summary.data?.kpis || {};
  const refresh = () => {
    queryClient.invalidateQueries({ queryKey: ['lirie', 'company-work-time'] });
    queryClient.invalidateQueries({ queryKey: ['lirie', 'company-work-time-driver'] });
  };

  const validateProposal = async (entry) => {
    if (entry?.proposed_worked_minutes == null) return;
    setValidatingId(entry.booking_id);
    setError('');
    try {
      await createWorkTimeDurationDecision({
        booking_id: entry.booking_id,
        validated_worked_minutes: entry.proposed_worked_minutes,
      });
      refresh();
    } catch (err) {
      setError(err?.response?.data?.error || err?.error || 'Validation refusée.');
    } finally {
      setValidatingId(null);
    }
  };

  const openBooking = (id, entry) => {
    if (!id) return;
    if (typeof onOpenBooking === 'function') {
      onOpenBooking(id, entry);
      return;
    }
    const target = location.pathname.replace(/\/drivers\/?$/, '/reservations');
    navigate(`${target}?booking=${id}`);
  };

  const closePeriod = async () => {
    setError('');
    try {
      await finalizeWorkTimePeriod(range);
      refresh();
    } catch (err) {
      setError(err?.error || err?.message || 'Clôture impossible.');
    }
  };

  const reopenPeriod = async () => {
    const reason = window.prompt('Motif de réouverture de la période');
    if (!reason) return;
    setError('');
    try {
      await reopenWorkTimePeriod({ ...range, reason });
      refresh();
    } catch (err) {
      setError(err?.error || err?.message || 'Réouverture impossible.');
    }
  };

  useEffect(() => {
    setOpenDays({});
    setExplainId(null);
  }, [driverId, range.from, range.to]);

  useEffect(() => {
    if (!revealReviews || !detail.data?.days) return;
    const next = {};
    detail.data.days.forEach((day) => {
      if (dayMetrics(day, displayMode, summary.data?.period_finalized).review)
        next[day.date] = true;
    });
    setOpenDays(next);
    setRevealReviews(false);
  }, [revealReviews, detail.data, displayMode, summary.data?.period_finalized]);

  const openDriver = (id, options = {}) => {
    setQueueOpen(false);
    setDriverId(id);
    setRevealReviews(Boolean(options.focusReview));
  };

  const leaveDetail = () => {
    setDriverId(null);
    setQueueOpen(false);
  };

  const movePeriod = (direction) => {
    const next = shiftPeriod(preset, range, direction);
    if (preset === 'custom') {
      setCustomFrom(next.from);
      setCustomTo(next.to);
      return;
    }
    setAnchor(next.anchor);
  };

  const selectPreset = (id) => {
    setPreset(id);
    setAnchor('');
  };

  const periodLabel =
    range.from === range.to
      ? formatCivilDate(range.from)
      : `${formatCivilDate(range.from)} — ${formatCivilDate(range.to)}`;
  const roster = useMemo(() => manualRoster(companyDrivers, summary.data?.drivers), [
    companyDrivers,
    summary.data?.drivers,
  ]);
  const heading = periodHeading(preset, range);
  const closure = monthClosureState(preset, range);
  const selected = (summary.data?.drivers || []).find((driver) => driver.driver_id === driverId);
  const driverName = detail.data?.display_name || selected?.display_name || 'Chauffeur';
  const driverTotals = detail.data?.totals || selected || {};

  return (
    <section className={s.wrap}>
      <PeriodChrome
        presets={PRESETS}
        preset={preset}
        heading={heading}
        periodLabel={periodLabel}
        displayMode={displayMode}
        customFrom={customFrom}
        customTo={customTo}
        onPreset={selectPreset}
        onShift={movePeriod}
        onCustomFrom={setCustomFrom}
        onCustomTo={setCustomTo}
        onMode={setDisplayMode}
        showClosure={driverId == null}
        closure={closure}
        rules={summary.data?.contractual_rules}
        rulesOpen={rulesOpen}
        onToggleRules={() => setRulesOpen((value) => !value)}
        finalized={summary.data?.period_finalized}
        onClosePeriod={closePeriod}
        onReopenPeriod={reopenPeriod}
        source={driverId == null ? kpis : driverTotals}
        ready={driverId == null ? Boolean(summary.data) : Boolean(detail.data || selected)}
        onAdd={() => setManualOpen(true)}
        onReview={() => {
          setDriverId(null);
          setQueueOpen(true);
        }}
        showSummary={driverId == null && !queueOpen}
        showAdd={driverId == null}
        onBack={driverId != null || queueOpen ? leaveDetail : null}
      />
      {error ? <p className={s.error}>{error}</p> : null}
      {summary.isError ? (
        <p className={s.error}>Impossible de charger le temps de travail.</p>
      ) : null}
      {queueOpen && driverId == null ? (
        <ReviewQueue
          items={summary.data?.review_items || []}
          drivers={summary.data?.drivers || []}
          mode={displayMode}
          onExamine={(item) => openDriver(item.driver_id, { focusReview: true })}
        />
      ) : driverId == null ? (
        <FleetTable
          drivers={summary.data?.drivers || []}
          mode={displayMode}
          loading={summary.isLoading}
          onOpen={openDriver}
        />
      ) : (
        <DriverSummary
          name={driverName}
          mode={displayMode}
          days={detail.data?.days || []}
          finalized={summary.data?.period_finalized}
          loading={detail.isLoading}
          openDays={openDays}
          onToggleDay={(date) => setOpenDays((current) => ({ ...current, [date]: !current[date] }))}
          onOpenBooking={openBooking}
          onExplain={setExplainId}
          onAdjust={setAdjustEntry}
          onDuration={setDurationEntry}
          onValidate={validateProposal}
          validatingId={validatingId}
          onReplaceOpenDays={setOpenDays}
          onCancelManual={async (entry) => {
            const reason = window.prompt('Motif d’annulation');
            if (!reason) return;
            await cancelManualWorkEntry(entry.entry_id, reason);
            refresh();
          }}
          onAdd={() => {
            if (summary.data?.period_finalized) return;
            setManualOpen(true);
          }}
        />
      )}

      {explainId != null && (
        <div className={s.modalOverlay} onClick={() => setExplainId(null)} role="presentation">
          <div
            className={`${s.modal} ${s.manualModal}`}
            onClick={(event) => event.stopPropagation()}
            role="dialog"
            aria-modal="true"
          >
            <div className={s.modalHead}>
              <h3>Pourquoi cette durée ?</h3>
              <button type="button" className={s.ghost} onClick={() => setExplainId(null)}>
                Fermer
              </button>
            </div>
            {explain.data ? (
              <>
                <p className={s.note}>
                  Arrivé {explain.data.milestones?.arrived_at || '—'} · prise en charge{' '}
                  {explain.data.milestones?.boarded_at || '—'} · terminé{' '}
                  {explain.data.milestones?.completed_at || '—'}
                </p>
                {(explain.data.entries || [])
                  .filter((entry) => entry.routing_status)
                  .map((entry) => (
                    <RouteExplain key={`route-${entry.booking_id}`} entry={entry} />
                  ))}
                {(explain.data.compensation_lines || []).map((line) => (
                  <p key={line.journey_key || line.line_key} className={s.note}>
                    {line.journey_class || 'activité'} · {line.journey_status} ·{' '}
                    {line.compensation_status} · {formatMinutes(line.compensated_minutes)} · règle{' '}
                    {line.policy_id ?? 'aucune'} · source {line.classification_source}
                  </p>
                ))}
                {(explain.data.adjustments || []).map((item) => (
                  <p key={item.id} className={s.note}>
                    Correction #{item.id} : arrivée {item.corrected_arrived_at} · fin{' '}
                    {item.corrected_completed_at} · {item.reason}
                  </p>
                ))}
              </>
            ) : (
              <p className={s.note}>Chargement…</p>
            )}
          </div>
        </div>
      )}

      {adjustEntry && (
        <AdjustModal
          entry={adjustEntry}
          onClose={() => setAdjustEntry(null)}
          onSaved={() => {
            setAdjustEntry(null);
            refresh();
          }}
        />
      )}
      {durationEntry && (
        <DurationModal
          entry={durationEntry}
          onClose={() => setDurationEntry(null)}
          onSaved={() => {
            setDurationEntry(null);
            refresh();
          }}
        />
      )}
      {manualOpen && (
        <ManualModal
          drivers={roster}
          preferredDriverId={driverId}
          range={range}
          onClose={() => setManualOpen(false)}
          onSaved={() => {
            setManualOpen(false);
            refresh();
          }}
        />
      )}
    </section>
  );
}

function RouteExplain({ entry }) {
  const copy = routeProposalCopy(entry);
  if (!copy) {
    return <p className={s.note}>Temps à déterminer · aucun itinéraire routier OSRM</p>;
  }
  return (
    <dl className={`${s.detailMeta} ${s.detailMetaWide}`}>
      {copy.facts.map(([label, value]) => (
        <React.Fragment key={label}>
          <dt className={s.detailLabel}>{label}</dt>
          <dd className={s.detailValue}>{value}</dd>
        </React.Fragment>
      ))}
    </dl>
  );
}

function routeProposalCaption(entry, proposed) {
  const copy = routeProposalCopy(entry);
  if (!copy) return `Temps proposé : ${formatMinutes(proposed)}.`;
  return `${copy.proposedLabel}. ${copy.reference}. ${copy.margin}.`;
}

function DurationModal({ entry, onClose, onSaved }) {
  const proposed = Number(entry.proposed_worked_minutes) || 0;
  const [minutes, setMinutes] = useState(String(proposed));
  const [reason, setReason] = useState('');
  const [error, setError] = useState('');

  const submit = async (event) => {
    event.preventDefault();
    setError('');
    const retained = Number(minutes);
    if (!Number.isFinite(retained) || retained < 0) {
      setError('Indiquez une durée en minutes.');
      return;
    }
    if (retained !== proposed && !reason.trim()) {
      setError('Le motif est obligatoire si la durée change.');
      return;
    }
    try {
      await createWorkTimeDurationDecision({
        booking_id: entry.booking_id,
        validated_worked_minutes: retained,
        reason: reason.trim(),
      });
      onSaved();
    } catch (err) {
      setError(err?.response?.data?.error || err?.error || 'Rectification refusée.');
    }
  };

  return (
    <div className={s.modalOverlay} onClick={onClose} role="presentation">
      <form className={s.modal} onClick={(event) => event.stopPropagation()} onSubmit={submit}>
        <h3>Rectifier la durée</h3>
        <p className={s.caption}>
          {routeProposalCaption(entry, proposed)} La durée retenue ne crée pas d’heure d’arrivée.
        </p>
        <label>
          Minutes retenues
          <input
            type="number"
            min="0"
            value={minutes}
            onChange={(event) => setMinutes(event.target.value)}
            required
          />
        </label>
        <label>
          Motif
          <textarea value={reason} onChange={(event) => setReason(event.target.value)} />
        </label>
        {error ? <p className={s.error}>{error}</p> : null}
        <div className={s.modalActions}>
          <button type="button" className={s.ghost} onClick={onClose}>
            Annuler
          </button>
          <button type="submit" className={s.primary}>
            Enregistrer
          </button>
        </div>
      </form>
    </div>
  );
}

function AdjustModal({ entry, onClose, onSaved }) {
  const defaults = adjustmentFormDefaults(entry);
  const [arrived, setArrived] = useState(defaults.arrived);
  const [completed, setCompleted] = useState(defaults.completed);
  const [reason, setReason] = useState('');
  const [error, setError] = useState('');

  const submit = async (event) => {
    event.preventDefault();
    setError('');
    try {
      await createWorkTimeAdjustment({
        booking_id: entry.booking_id,
        corrected_arrived_at: arrived,
        corrected_completed_at: completed,
        reason,
      });
      onSaved();
    } catch (err) {
      setError(err?.error || 'Correction refusée.');
    }
  };

  return (
    <div className={s.modalOverlay}>
      <form className={s.modal} onSubmit={submit}>
        <h3>Rectifier l’horaire</h3>
        <p className={s.caption}>
          Les deux instants effectifs sont enregistrés, même si un seul change.
          {defaults.arrivalRecorded
            ? null
            : ' L’arrivée n’était pas enregistrée : la date de la course est préremplie.'}
        </p>
        <label>
          Arrivée
          <input
            type="datetime-local"
            value={arrived}
            onChange={(e) => setArrived(e.target.value)}
            required
          />
        </label>
        <label>
          Fin
          <input
            type="datetime-local"
            value={completed}
            onChange={(e) => setCompleted(e.target.value)}
            required
          />
        </label>
        <label>
          Motif
          <textarea value={reason} onChange={(e) => setReason(e.target.value)} required />
        </label>
        {error ? <p className={s.error}>{error}</p> : null}
        <div className={s.modalActions}>
          <button type="button" className={s.ghost} onClick={onClose}>
            Annuler
          </button>
          <button type="submit" className={s.primary}>
            Enregistrer
          </button>
        </div>
      </form>
    </div>
  );
}

function driverLabel(driver) {
  const name = [driver.first_name, driver.last_name].filter(Boolean).join(' ').trim();
  return name || driver.full_name || driver.display_name || `Chauffeur ${driver.id || driver.driver_id}`;
}

function manualRoster(companyDrivers, periodDrivers) {
  const rows = [];
  const seen = new Set();
  (companyDrivers || []).forEach((driver) => {
    if (driver.is_active === false || driver.id == null) return;
    seen.add(Number(driver.id));
    rows.push({ driver_id: driver.id, display_name: driverLabel(driver) });
  });
  (periodDrivers || []).forEach((driver) => {
    if (driver.driver_id == null || seen.has(Number(driver.driver_id))) return;
    rows.push({ driver_id: driver.driver_id, display_name: driver.display_name });
  });
  return rows.sort((a, b) => a.display_name.localeCompare(b.display_name, 'fr'));
}

function proposedDay(range) {
  const today = getBusinessCalendarDate(new Date());
  if (range?.from && range?.to && today >= range.from && today <= range.to) return today;
  return range?.to || today;
}

function atClock(day, hours, minutes) {
  const pad = (value) => String(value).padStart(2, '0');
  return `${day}T${pad(hours)}:${pad(minutes)}`;
}

function previewMinutes(started, ended) {
  if (!started || !ended) return null;
  const from = new Date(started);
  const to = new Date(ended);
  if (Number.isNaN(from.getTime()) || Number.isNaN(to.getTime()) || to <= from) return null;
  return Math.round((to.getTime() - from.getTime()) / 60000);
}

function ManualModal({ drivers, preferredDriverId, range, onClose, onSaved }) {
  const day = proposedDay(range);
  const [driverId, setDriverId] = useState(preferredDriverId ? String(preferredDriverId) : '');
  const [workType, setWorkType] = useState('');
  const [started, setStarted] = useState(() => atClock(proposedDay(range), 8, 0));
  const [ended, setEnded] = useState(() => atClock(proposedDay(range), 9, 0));
  const [description, setDescription] = useState('');
  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    const onKey = (event) => {
      if (event.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [onClose]);

  const submit = async (event) => {
    event.preventDefault();
    setError('');
    const minutes = previewMinutes(started, ended);
    if (!driverId || !workType || minutes == null) {
      setError('Choisissez un chauffeur, une activité, et une fin après le début.');
      return;
    }
    setSaving(true);
    try {
      await createManualWorkEntry({
        driver_id: Number(driverId),
        work_type: workType,
        started_at: started,
        ended_at: ended,
        description,
      });
      onSaved();
    } catch (err) {
      setError(err?.response?.data?.error || err?.error || 'Saisie refusée.');
      setSaving(false);
    }
  };

  const minutes = previewMinutes(started, ended);
  const ready = Boolean(driverId) && Boolean(workType) && minutes != null && !saving;
  const today = getBusinessCalendarDate(new Date());

  return (
    <div className={s.modalOverlay} onClick={onClose} role="presentation">
      <form
        className={`${s.modal} ${s.manualModal}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="manual-time-title"
        onClick={(event) => event.stopPropagation()}
        onSubmit={submit}
      >
        <header className={s.modalHead}>
          <div>
            <h3 id="manual-time-title">Ajouter un temps</h3>
            <p className={s.caption}>
              {day === today
                ? 'Temps hors course. La durée se calcule entre le début et la fin, sur aujourd’hui.'
                : `Temps hors course. La durée se calcule entre le début et la fin, le ${formatCivilDate(day)}.`}
            </p>
          </div>
          <button type="button" className={s.iconClose} onClick={onClose} aria-label="Fermer">
            ×
          </button>
        </header>
        <div className={s.modalGrid}>
          <div className={s.choiceField}>
            <span>Chauffeur</span>
            <ChipMenu
              label="Chauffeur"
              value={driverId}
              options={drivers.map((driver) => ({
                value: String(driver.driver_id),
                label: driver.display_name,
              }))}
              onChange={setDriverId}
            />
          </div>
          <div className={s.choiceField}>
            <span>Activité</span>
            <ChipMenu
              label="Activité"
              value={workType}
              options={WORK_TYPES.map(([id, name]) => ({ value: id, label: name }))}
              onChange={setWorkType}
            />
          </div>
        </div>
        <div className={s.timeRow}>
          <InstantFields
            label="Début"
            value={started}
            onChange={setStarted}
            minDate={range?.from}
            maxDate={range?.to}
          />
          <InstantFields
            label="Fin"
            value={ended}
            onChange={setEnded}
            minDate={range?.from}
            maxDate={range?.to}
          />
          <p
            className={minutes == null ? `${s.durationLive} ${s.durationInvalid}` : s.durationLive}
            aria-live="polite"
          >
            <span className={s.durationLabel}>Durée</span>
            <span className={s.durationValue}>
              {minutes == null ? 'Fin avant le début' : formatMinutes(minutes)}
            </span>
          </p>
        </div>
        <label className={s.fieldWide}>
          Note
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Facultatif"
            rows={2}
          />
        </label>
        {error ? <p className={s.error}>{error}</p> : null}
        <div className={s.modalActions}>
          <button type="button" className={s.ghost} onClick={onClose}>
            Annuler
          </button>
          <button type="submit" className={s.primary} disabled={!ready}>
            {saving ? 'Enregistrement…' : 'Enregistrer'}
          </button>
        </div>
      </form>
    </div>
  );
}

function splitLocal(value) {
  const [date, time] = String(value || '').split('T');
  return { date: date || '', time: (time || '').slice(0, 5) };
}

function joinLocal(date, time) {
  if (!date || !time) return '';
  return `${date}T${time}`;
}

function ChipMenu({ label, value, options, onChange }) {
  const [open, setOpen] = useState(false);
  const btnRef = useRef(null);
  const menuRef = useRef(null);
  const [pos, setPos] = useState({ top: 0, left: 0, width: 0 });
  const current = options.find((option) => option.value === value);

  const place = () => {
    if (!btnRef.current) return;
    const rect = btnRef.current.getBoundingClientRect();
    setPos({ top: rect.bottom + 4, left: rect.left, width: Math.max(rect.width, 148) });
  };

  useEffect(() => {
    if (!open) return undefined;
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
              style={{ top: pos.top, left: pos.left, minWidth: pos.width }}
            >
              {options.map((option) => (
                <button
                  key={option.value}
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

function InstantFields({ label, value, onChange, minDate, maxDate }) {
  const { date, time } = splitLocal(value);
  const lower = label.toLowerCase();
  return (
    <div className={s.instantField}>
      <span>{label}</span>
      <div className={s.instantMenus}>
        <InlineDatePicker
          className={s.dateField}
          value={date}
          onChange={(next) => onChange(joinLocal(next, time || '00:00'))}
          ariaLabel={`Date de ${lower}`}
          minDate={minDate || null}
          maxDate={maxDate || null}
        />
        <div className={s.timeField}>
          <InlineTimePicker
            value={time}
            onChange={(next) => onChange(next ? joinLocal(date, next) : '')}
            ariaLabel={`Heure de ${lower}`}
            required
          />
        </div>
      </div>
    </div>
  );
}
