import React, { useCallback, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import {
  aggregateDisplayedDays,
  closurePhrase,
  dayMetrics,
  entryDuration,
  entryNeedsReview,
  fleetSentence,
  formatCivilDate,
  formatClock,
  formatDayLabel,
  formatMinutes,
  narrativeSource,
  policyPhrase,
  reviewActivity,
  reviewCount,
  reviewItemsForMode,
  reviewReason,
  routeProposalCopy,
  rowBadge,
  totalMinutes,
  transportMinutes,
} from './workTimeFormat';
import s from './workTime.module.css';

const WORK_TYPE_LABELS = {
  extra_transport: 'Transport supplémentaire',
  delivery: 'Livraison',
  accompaniment: 'Accompagnement',
  waiting: 'Attente',
  administrative: 'Administratif',
  cleaning: 'Nettoyage',
  training: 'Formation',
  other: 'Autre',
};

function reviewCopy(count, rest = false) {
  if (count <= 0) return '';
  const noun = count > 1 ? 'éléments' : 'élément';
  return rest
    ? `${count} ${noun} reste${count > 1 ? 'nt' : ''} à vérifier`
    : `${count} ${noun} à vérifier`;
}

export function PeriodChrome({
  presets,
  preset,
  heading,
  periodLabel,
  displayMode,
  customFrom,
  customTo,
  onPreset,
  onShift,
  onCustomFrom,
  onCustomTo,
  onMode,
  showClosure,
  closure,
  rules,
  rulesOpen,
  onToggleRules,
  finalized,
  onClosePeriod,
  onReopenPeriod,
  source,
  ready,
  onAdd,
  onReview,
  showSummary = true,
  showAdd = true,
  onBack,
}) {
  const reviews = reviewCount(source, displayMode);
  const versions = rules?.versions || [];
  return (
    <header className={s.pageHead}>
      <div className={s.pageMain}>
        <div className={s.periodRow}>
          <div className={s.periodIdentity}>
            <p className={s.kicker}>Heures effectuées</p>
            <div className={s.periodNav}>
              <button
                type="button"
                className={s.navBtn}
                aria-label="Période précédente"
                onClick={() => onShift(-1)}
              >
                ‹
              </button>
              <h2 className={s.periodTitle}>{heading}</h2>
              <button
                type="button"
                className={s.navBtn}
                aria-label="Période suivante"
                onClick={() => onShift(1)}
              >
                ›
              </button>
            </div>
            <p className={s.note}>{periodLabel}</p>
          </div>
          {onBack ? (
            <button type="button" className={s.back} onClick={onBack}>
              ← Tous les chauffeurs
            </button>
          ) : null}
          {showClosure && closure?.visible ? (
            <aside
              className={`${s.closeBox} ${
                finalized ? s.closeClosed : closure.closable ? s.closeReady : s.closeWaiting
              }`}
            >
              <div className={s.closeHead}>
                <p className={s.modeLabel}>Règle de clôture</p>
                {finalized ? <span className={s.closedBadge}>Clôturé</span> : null}
              </div>
              <p className={s.rule}>{closurePhrase(rules)}</p>
              {!finalized && versions.length > 1 ? (
                <button type="button" className={s.linkish} onClick={onToggleRules}>
                  {rulesOpen ? 'Masquer les règles appliquées' : 'Voir les règles appliquées'}
                </button>
              ) : null}
              {rulesOpen && !finalized ? (
                <ul className={s.ruleList}>
                  {versions.map((version) => (
                    <li key={`${version.policy_id}-${version.effective_from}`}>
                      {policyPhrase(version)} · dès le {formatCivilDate(version.effective_from)}
                      {version.effective_until
                        ? ` jusqu’au ${formatCivilDate(version.effective_until)}`
                        : ''}
                    </li>
                  ))}
                </ul>
              ) : null}
              {finalized ? (
                <button type="button" className={s.reopen} onClick={onReopenPeriod}>
                  {`Réouvrir ${closure.monthName}`}
                </button>
              ) : closure.closable ? (
                <button type="button" className={`${s.primary} ${s.closeAction}`} onClick={onClosePeriod}>
                  Clôturer le mois
                </button>
              ) : (
                <p className={s.closeWhen}>
                  Clôture disponible à partir du <strong>{formatCivilDate(closure.availableOn)}</strong>
                </p>
              )}
            </aside>
          ) : null}
        </div>
        <div className={s.controls}>
          <div className={s.periodFilters}>
            <div className={s.segmented} role="tablist" aria-label="Période">
              {presets.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  className={`${s.chip} ${preset === item.id ? s.chipActive : ''}`}
                  onClick={() => onPreset(item.id)}
                >
                  {item.label}
                </button>
              ))}
            </div>
            {preset === 'custom' ? (
              <div className={s.customRange}>
                <input
                  type="date"
                  value={customFrom}
                  onChange={(event) => onCustomFrom(event.target.value)}
                  aria-label="Début"
                />
                <span>au</span>
                <input
                  type="date"
                  value={customTo}
                  onChange={(event) => onCustomTo(event.target.value)}
                  aria-label="Fin"
                />
              </div>
            ) : null}
          </div>
          <div className={s.modeBox}>
            <span className={s.modeLabel}>Affichage</span>
            <div className={s.segmented} role="tablist" aria-label="Mode d’affichage">
              <button
                type="button"
                className={`${s.chip} ${displayMode === 'real' ? s.chipActive : ''}`}
                onClick={() => onMode('real')}
              >
                Temps réel
              </button>
              <button
                type="button"
                className={`${s.chip} ${displayMode === 'flat' ? s.chipActive : ''}`}
                onClick={() => onMode('flat')}
              >
                Forfait entreprise
              </button>
            </div>
          </div>
        </div>
        {showSummary || showAdd ? (
          <div className={s.summaryRow}>
            <div className={s.summaryMain}>
              {showSummary && ready ? (
                <p className={s.sentence}>{fleetSentence(source, displayMode)}</p>
              ) : null}
              {showSummary && reviews > 0 ? (
                <button type="button" className={s.reviewAlert} onClick={onReview}>
                  {reviewCopy(reviews)}
                </button>
              ) : null}
            </div>
            {showAdd ? (
              <button
                type="button"
                className={s.addTime}
                onClick={onAdd}
                disabled={Boolean(finalized)}
                title={finalized ? 'Mois clôturé : réouvrez-le pour ajouter un temps.' : undefined}
              >
                Ajouter un temps
              </button>
            ) : null}
          </div>
        ) : null}
      </div>
    </header>
  );
}

export function FleetTable({ drivers, mode, loading, onOpen }) {
  return (
    <div className={s.tableCard} id="work-time-drivers">
      <table className={s.table}>
        <thead>
          <tr>
            <th>Chauffeur</th>
            <th className={s.num}>Transports</th>
            <th className={s.num}>Temps transport</th>
            <th className={s.num}>Temps ajouté</th>
            <th className={s.num}>Total</th>
            <th className={s.num}>À vérifier</th>
            <th className={s.num}>
              <span className={s.srOnly}>Ouvrir</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {drivers.map((driver) => {
            const reviews = reviewCount(driver, mode);
            return (
              <tr
                key={driver.driver_id}
                className={s.clickRow}
                onClick={() => onOpen(driver.driver_id)}
              >
                <td className={s.name}>{driver.display_name || 'Chauffeur'}</td>
                <td className={`${s.num} ${s.lead}`}>{Number(driver.transport_count || 0)}</td>
                <td className={`${s.num} ${s.quietNum}`}>
                  {transportMinutes(driver, mode) === 0 &&
                  reviews > 0 &&
                  Number(driver.transport_count || 0) > 0
                    ? 'À vérifier'
                    : formatMinutes(transportMinutes(driver, mode))}
                </td>
                <td
                  className={`${s.num} ${(mode === 'flat' ? driver.flat_added_minutes : driver.real_added_minutes) ? '' : s.muted}`}
                >
                  {(mode === 'flat'
                    ? Number(driver.flat_added_minutes || 0)
                    : Number(driver.real_added_minutes || 0)) > 0
                    ? formatMinutes(
                        mode === 'flat' ? driver.flat_added_minutes : driver.real_added_minutes
                      )
                    : '—'}
                </td>
                <td className={`${s.num} ${s.lead}`}>
                  {transportMinutes(driver, mode) === 0 &&
                  reviews > 0 &&
                  Number(driver.transport_count || 0) > 0
                    ? '—'
                    : formatMinutes(totalMinutes(driver, mode))}
                </td>
                <td className={reviews > 0 ? s.alertCount : `${s.num} ${s.muted}`}>
                  {reviews > 0 ? (
                    <button
                      type="button"
                      className={s.alertLink}
                      onClick={(event) => {
                        event.stopPropagation();
                        onOpen(driver.driver_id, { focusReview: true });
                      }}
                    >
                      {reviews}
                    </button>
                  ) : (
                    '—'
                  )}
                </td>
                <td className={s.actions}>
                  <button
                    type="button"
                    className={s.chevron}
                    aria-label={`Résumé des heures de ${driver.display_name || 'ce chauffeur'}`}
                    onClick={(event) => {
                      event.stopPropagation();
                      onOpen(driver.driver_id);
                    }}
                  >
                    ›
                  </button>
                </td>
              </tr>
            );
          })}
          {drivers.length === 0 && !loading ? (
            <tr>
              <td colSpan={7} className={s.empty}>
                Aucune activité sur cette période.
              </td>
            </tr>
          ) : null}
        </tbody>
      </table>
    </div>
  );
}

export function ReviewQueue({ items, drivers, mode, onExamine }) {
  const rows = reviewItemsForMode(items, mode);
  const names = new Map((drivers || []).map((driver) => [driver.driver_id, driver.display_name]));
  const groups = [];
  rows.forEach((item) => {
    const last = groups[groups.length - 1];
    if (last && last.driverId === item.driver_id) last.items.push(item);
    else groups.push({ driverId: item.driver_id, name: names.get(item.driver_id) || 'Chauffeur', items: [item] });
  });
  return (
    <section className={s.reviewQueue} aria-label="Éléments à vérifier">
      <h2 className={s.driverTitle}>À vérifier — {rows.length} élément{rows.length > 1 ? 's' : ''}</h2>
      {rows.length === 0 ? <p className={s.caption}>Aucun élément à vérifier sur cette période.</p> : null}
      {groups.map((group) => (
        <section key={group.driverId} className={s.reviewGroup}>
          <h3 className={s.reviewDriver}>{group.name}</h3>
          <ul className={s.reviewList}>
            {group.items.map((item) => (
              <li key={`${item.line_key || item.booking_id || item.entry_id}-${item.date}`} className={s.reviewItem}>
                <div>
                  <p className={s.reviewWhen}>{formatDayLabel(item.date)}</p>
                  <p className={s.reviewActivity}>{reviewActivity(item)}</p>
                  <p className={s.reviewReason}>{reviewReason(item)}</p>
                </div>
                <button type="button" className={s.primary} onClick={() => onExamine(item)}>
                  Examiner
                </button>
              </li>
            ))}
          </ul>
        </section>
      ))}
    </section>
  );
}

export function DriverSummary({
  name,
  mode,
  days,
  finalized,
  loading,
  openDays,
  onToggleDay,
  onOpenBooking,
  onExplain,
  onAdjust,
  onDuration,
  onValidate,
  validatingId,
  onCancelManual,
  onReplaceOpenDays,
  onAdd,
}) {
  const ordered = [...(days || [])].sort((a, b) => String(b.date).localeCompare(String(a.date)));
  const aggregate = aggregateDisplayedDays(ordered, mode, finalized);
  const sentenceSource = narrativeSource(aggregate, mode);
  if (mode === 'flat') sentenceSource.review_count_flat = aggregate.review_count || 0;
  else sentenceSource.review_count_real = aggregate.review_count || 0;
  const revealReviews = () => {
    const next = {};
    ordered.forEach((day) => {
      if (dayMetrics(day, mode, finalized).review) next[day.date] = true;
    });
    onReplaceOpenDays?.(next);
    window.requestAnimationFrame(() => {
      document.querySelector('[data-review-day="true"]')?.scrollIntoView({ block: 'center' });
    });
  };
  return (
    <div className={s.driverPage}>
      <div className={s.summaryRow}>
        <div className={s.summaryMain}>
          <h2 className={s.driverTitle}>Résumé des heures — {name}</h2>
          {!loading ? <p className={s.sentence}>{fleetSentence(sentenceSource, mode)}</p> : null}
          {aggregate.review_count > 0 ? (
            <button type="button" className={s.reviewAlert} onClick={revealReviews}>
              {reviewCopy(aggregate.review_count, true)}
            </button>
          ) : null}
        </div>
        <button
          type="button"
          className={s.addTime}
          onClick={onAdd}
          disabled={Boolean(finalized)}
          title={finalized ? 'Mois clôturé : réouvrez-le pour ajouter un temps.' : undefined}
        >
          Ajouter un temps
        </button>
      </div>
      {loading ? <p className={s.caption}>Chargement…</p> : null}
      {!loading && ordered.length === 0 ? (
        <p className={s.caption}>Aucune journée sur cette période.</p>
      ) : null}
      {ordered.length > 0 ? (
        <div className={s.tableCard}>
          <table className={s.table}>
            <thead>
              <tr>
                <th>Date</th>
                <th className={s.num}>Transports</th>
                <th className={s.num}>Temps transport</th>
                <th className={s.num}>Temps ajouté</th>
                <th className={s.num}>Total</th>
                <th className={s.num}>État</th>
              </tr>
            </thead>
            <tbody>
              {ordered.map((day) => {
                const metrics = dayMetrics(day, mode, finalized);
                const open = Boolean(openDays[day.date]);
                return (
                  <React.Fragment key={day.date}>
                    <tr
                      className={`${s.clickRow} ${metrics.review ? s.rowReview : ''}`}
                      data-review-day={metrics.review ? 'true' : 'false'}
                      onClick={() => onToggleDay(day.date)}
                    >
                      <td>
                        <button
                          type="button"
                          className={s.dayToggle}
                          aria-expanded={open}
                          onClick={(event) => {
                            event.stopPropagation();
                            onToggleDay(day.date);
                          }}
                        >
                          <span className={open ? s.caretOpen : s.caret} aria-hidden="true">
                            ›
                          </span>
                          {formatDayLabel(day.date)}
                        </button>
                      </td>
                      <td className={`${s.num} ${s.lead}`}>{metrics.transportCount}</td>
                      <td
                        className={
                          metrics.transportLabel === 'À vérifier'
                            ? s.alertCount
                            : `${s.num} ${s.quietNum}`
                        }
                      >
                        {metrics.transportLabel}
                      </td>
                      <td
                        className={
                          metrics.addedLabel === '—'
                            ? `${s.num} ${s.muted}`
                            : `${s.num} ${s.quietNum}`
                        }
                      >
                        {metrics.addedLabel}
                      </td>
                      <td className={`${s.num} ${s.lead}`}>
                        {metrics.totalLabel === '—' ? (
                          <span className={s.muted}>—</span>
                        ) : (
                          metrics.totalLabel
                        )}
                      </td>
                      <td className={metrics.review ? s.alertCount : `${s.num} ${s.stateOk}`}>
                        {metrics.statusLabel}
                      </td>
                    </tr>
                    {open ? (
                      <tr className={s.detailRow}>
                        <td colSpan={6}>
                          <DayEntries
                            day={day}
                            mode={mode}
                            finalized={finalized}
                            onOpenBooking={onOpenBooking}
                            onExplain={onExplain}
                            onAdjust={onAdjust}
                            onDuration={onDuration}
                            onValidate={onValidate}
                            validatingId={validatingId}
                            onCancelManual={onCancelManual}
                          />
                        </td>
                      </tr>
                    ) : null}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : null}
    </div>
  );
}

function DayEntries({
  day,
  mode,
  finalized,
  onOpenBooking,
  onExplain,
  onAdjust,
  onDuration,
  onValidate,
  validatingId,
  onCancelManual,
}) {
  const entries = [...(day.entries || [])].sort((a, b) =>
    String(a.effective_arrived_at || a.effective_completed_at || '').localeCompare(
      String(b.effective_arrived_at || b.effective_completed_at || '')
    )
  );
  return (
    <div className={s.dayDetail}>
      {entries.map((entry) => (
        <EntryCard
          key={`${entry.kind}-${entry.booking_id || entry.entry_id}`}
          entry={entry}
          mode={mode}
          finalized={finalized}
          onOpenBooking={onOpenBooking}
          onExplain={onExplain}
          onAdjust={onAdjust}
          onDuration={onDuration}
          onValidate={onValidate}
          validatingId={validatingId}
          onCancelManual={onCancelManual}
        />
      ))}
    </div>
  );
}

const MORE_MENU_WIDTH = 168;

function placeMoreMenu(trigger, menu) {
  const rect = trigger.getBoundingClientRect();
  const menuH = menu?.offsetHeight || 96;
  const gap = 6;
  const left = Math.min(
    Math.max(8, rect.right - MORE_MENU_WIDTH),
    window.innerWidth - MORE_MENU_WIDTH - 8
  );
  const below = rect.bottom + gap;
  const top = below + menuH <= window.innerHeight - 8 ? below : Math.max(8, rect.top - gap - menuH);
  return { top, left };
}

function EntryMoreMenu({ children }) {
  const items = React.Children.toArray(children).filter(Boolean);
  const [open, setOpen] = useState(false);
  const triggerRef = useRef(null);
  const menuRef = useRef(null);
  const [pos, setPos] = useState({ top: 0, left: 0 });

  const updatePosition = useCallback(() => {
    if (!triggerRef.current) return;
    setPos(placeMoreMenu(triggerRef.current, menuRef.current));
  }, []);

  useEffect(() => {
    if (!open) return;
    updatePosition();
    const onPointer = (event) => {
      if (triggerRef.current?.contains(event.target) || menuRef.current?.contains(event.target)) {
        return;
      }
      setOpen(false);
    };
    const onKey = (event) => {
      if (event.key === 'Escape') setOpen(false);
    };
    window.addEventListener('scroll', updatePosition, true);
    window.addEventListener('resize', updatePosition);
    document.addEventListener('mousedown', onPointer);
    document.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('scroll', updatePosition, true);
      window.removeEventListener('resize', updatePosition);
      document.removeEventListener('mousedown', onPointer);
      document.removeEventListener('keydown', onKey);
    };
  }, [open, updatePosition]);

  if (items.length === 0) return null;

  return (
    <div className={s.more}>
      <button
        ref={triggerRef}
        type="button"
        className={s.moreTrigger}
        aria-label="Autres actions"
        aria-expanded={open}
        aria-haspopup="menu"
        onClick={() => {
          if (open) {
            setOpen(false);
            return;
          }
          if (triggerRef.current) {
            setPos(placeMoreMenu(triggerRef.current, menuRef.current));
          }
          setOpen(true);
        }}
      >
        ···
      </button>
      {open
        ? createPortal(
            <div
              ref={menuRef}
              className={s.moreMenu}
              role="menu"
              aria-label="Autres actions"
              style={{ top: pos.top, left: pos.left }}
            >
              {items.map((item, index) =>
                React.isValidElement(item)
                  ? React.cloneElement(item, {
                      key: item.key ?? index,
                      role: 'menuitem',
                      onClick: (event) => {
                        item.props.onClick?.(event);
                        setOpen(false);
                      },
                    })
                  : item
              )}
            </div>,
            document.body
          )
        : null}
    </div>
  );
}

function EntryCard({
  entry,
  mode,
  finalized,
  onOpenBooking,
  onExplain,
  onAdjust,
  onDuration,
  onValidate,
  validatingId,
  onCancelManual,
}) {
  const [examine, setExamine] = useState(false);
  const manual = entry.kind === 'manual' || entry.is_manual;
  const shown = entryDuration(entry, mode, finalized);
  const review = entryNeedsReview(entry, mode);
  const pending = entry.work_time_status === 'pending_validation';
  const start = formatClock(entry.effective_arrived_at);
  const end = formatClock(entry.effective_completed_at);
  const clock = manual && start && end ? `${start} — ${end}` : start || end || '';
  const title = manual
    ? entry.description || WORK_TYPE_LABELS[entry.work_type] || 'Temps ajouté'
    : null;
  const badge = rowBadge(entry, mode);
  const status = review
    ? 'À vérifier'
    : badge?.label || (manual ? 'Temps ajouté' : 'Validé');
  const durationLabel = shown == null ? (pending ? 'À valider' : 'À vérifier') : formatMinutes(shown);
  const statusTone = review ? s.statusWarn : status === 'Validé' ? s.statusOk : '';
  return (
    <article className={s.entry}>
      <p className={s.clock}>{clock || '—'}</p>
      <div className={s.entryBody}>
        {manual ? (
          <p className={s.entryName}>{title}</p>
        ) : (
          <div className={s.place}>
            <p>{entry.pickup_label || 'Départ'}</p>
            {entry.dropoff_label ? <p className={s.dropoff}>{entry.dropoff_label}</p> : null}
          </div>
        )}
      </div>
      <div className={s.entryFacts}>
        <p className={s.entryDuration}>
          <span className={s.srOnly}>Durée</span>
          {durationLabel}
        </p>
        <p className={`${s.entryStatus} ${statusTone}`}>
          <span className={s.srOnly}>Statut</span>
          {status}
        </p>
      </div>
      {examine && pending ? (
        <div className={s.entryExamine}>
          <ProposalBreakdown entry={entry} />
        </div>
      ) : null}
      {examine && entry.kind === 'open' ? (
        <p className={`${s.caption} ${s.entryExamine}`}>
          Cette course n’est pas terminée. Elle ne compte pas dans les transports.
        </p>
      ) : null}
      <div className={s.entryActions}>
        {review && !manual ? (
          <button type="button" className={s.primary} onClick={() => setExamine((value) => !value)}>
            {examine ? 'Masquer' : 'Examiner'}
          </button>
        ) : null}
        {examine && pending ? (
          <>
            <button
              type="button"
              className={s.ghost}
              disabled={validatingId === entry.booking_id}
              onClick={() => onValidate(entry)}
            >
              Valider
            </button>
            <button type="button" className={s.ghost} onClick={() => onDuration(entry)}>
              Rectifier
            </button>
          </>
        ) : null}
        <EntryMoreMenu>
          {entry.booking_id ? (
            <button
              type="button"
              className={s.menuItem}
              onClick={() => onOpenBooking(entry.booking_id, entry)}
            >
              Ouvrir la course
            </button>
          ) : null}
          {!examine && pending ? (
            <button type="button" className={s.menuItem} onClick={() => onDuration(entry)}>
              Rectifier
            </button>
          ) : null}
          {!pending && entry.kind === 'transport' ? (
            <button type="button" className={s.menuItem} onClick={() => onAdjust(entry)}>
              Rectifier
            </button>
          ) : null}
          {entry.booking_id ? (
            <button
              type="button"
              className={s.menuItem}
              onClick={() => onExplain(entry.booking_id)}
            >
              Pourquoi
            </button>
          ) : null}
          {manual && entry.entry_id ? (
            <button type="button" className={s.menuItem} onClick={() => onCancelManual(entry)}>
              Annuler
            </button>
          ) : null}
        </EntryMoreMenu>
      </div>
    </article>
  );
}

function ProposalBreakdown({ entry }) {
  const copy = routeProposalCopy(entry);
  if (!copy)
    return <p className={s.caption}>{formatMinutes(entry.proposed_worked_minutes)} proposées</p>;
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
