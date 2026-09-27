import React, { useEffect, useRef, useState } from 'react';
import { FaGripVertical } from 'react-icons/fa';
import AddressAutocomplete from '../../../../components/common/AddressAutocomplete';
import InlineDatePicker from '../../../../components/ui/InlineDatePicker';
import InlineTimePicker from '../../../../components/ui/InlineTimePicker';
import EstablishmentSelect from '../../../../components/common/EstablishmentSelect';
import ServiceSelect from '../../../../components/common/ServiceSelect';
import { searchEstablishments } from '../../../../services/companyService';
import { parseAddressWithEstablishment } from '../../../../utils/addressParser';
import styles from './ManualBookingForm.module.css';

function normalizePlaceName(value) {
  return String(value || '').trim().toLowerCase();
}

function matchCatalogEstablishment(name, rows) {
  const target = normalizePlaceName(name);
  if (!target || !Array.isArray(rows)) return null;
  const labelsOf = (row) =>
    [row?.label, row?.name, ...(Array.isArray(row?.aliases) ? row.aliases : [])]
      .map(normalizePlaceName)
      .filter(Boolean);
  const exact = rows.find((row) => labelsOf(row).includes(target));
  if (exact) return exact;
  return (
    rows.find((row) =>
      labelsOf(row).some((label) => target.includes(label) || label.includes(target))
    ) || null
  );
}

async function resolveCatalogEstablishment(name) {
  const primary = await searchEstablishments(name, 8);
  const direct = matchCatalogEstablishment(name, primary);
  if (direct) return direct;
  const acronym = String(name).match(/\(([^)]+)\)/)?.[1]?.trim();
  if (!acronym || acronym.length < 2) return null;
  const secondary = await searchEstablishments(acronym, 8);
  return matchCatalogEstablishment(name, secondary) || matchCatalogEstablishment(acronym, secondary);
}

function MomentField({
  label,
  date,
  time,
  missionDate,
  onDate,
  onTime,
  testId,
  allowOtherDay = true,
  required = false,
}) {
  const differs = Boolean(date && missionDate && date !== missionDate);
  const [showDate, setShowDate] = useState(differs);
  const visibleDate = showDate || differs;

  return (
    <div className={styles.momentRow} data-testid={testId}>
      <div className={styles.momentLabelRow}>
        <span className={styles.momentLabel}>
          {label}
          {required ? ' *' : ''}
        </span>
        {!visibleDate && allowOtherDay ? (
          <button type="button" className={styles.otherDayBtn} onClick={() => setShowDate(true)}>
            Autre jour
          </button>
        ) : null}
      </div>
      <div className={styles.momentControls}>
        {visibleDate ? (
          <div className={styles.momentDate}>
            <InlineDatePicker
              value={date || missionDate || ''}
              onChange={onDate}
              ariaLabel={`${label}, date`}
              placeholder="Date"
            />
          </div>
        ) : null}
        <div className={styles.momentTime}>
          <InlineTimePicker
            value={time || ''}
            onChange={onTime}
            ariaLabel={label}
            placeholder="Heure"
            required={required}
          />
        </div>
      </div>
    </div>
  );
}

function RouteStepCard({
  point,
  index,
  destinationIndex,
  isLastDestination,
  isRoundTrip,
  missionDate,
  departureTime,
  selected,
  canRemove,
  isDropTarget,
  onSelect,
  onPatch,
  onRemove,
  onDepartureTime,
  onDragStart,
  onDragOver,
  onDrop,
  showConnector,
  durationMinutes,
}) {
  const isPickup = index === 0;
  const title = isPickup ? 'Départ' : `Destination ${destinationIndex}`;
  const placeSelectSeq = useRef(0);

  const reportEstablishment = (name) => {
    const seq = ++placeSelectSeq.current;
    const key = point.key;
    onPatch(key, { establishment: name, establishmentId: null });
    resolveCatalogEstablishment(name)
      .then((match) => {
        if (!match || placeSelectSeq.current !== seq) return;
        onPatch(key, {
          establishment: match.label || match.name || name,
          establishmentId: match.id ?? null,
        });
      })
      .catch(() => {});
  };

  useEffect(() => {
    if (isPickup) return;
    const location = point.location || '';
    if (!location || String(point.establishment || '').trim()) return;
    const establishmentName = parseAddressWithEstablishment(location, {}).establishment;
    if (establishmentName) reportEstablishment(establishmentName);
    // L'adresse déjà saisie est reportée une fois ; les choix suivants passent par onSelect.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const showDeparture = isPickup || !isLastDestination || isRoundTrip;
  const testId = isPickup ? 'route-step-pickup' : `route-step-destination-${destinationIndex}`;

  return (
    <article
      className={`${styles.routeStep} ${selected ? styles.routeStepSelected : ''} ${
        isDropTarget ? styles.routeStepDropTarget : ''
      }`}
      data-testid={testId}
      onDragOver={onDragOver}
      onDrop={onDrop}
    >
      <button
        type="button"
        className={styles.routeStepHandle}
        draggable
        aria-label={`Réorganiser ${title}`}
        onDragStart={onDragStart}
        onClick={() => onSelect(point.key)}
      >
        <FaGripVertical aria-hidden="true" />
      </button>
      <div className={styles.routeRail}>
        <span
          className={`${styles.routeDot} ${isPickup ? '' : styles.routeDotDestination}`}
          aria-hidden="true"
        />
        {showConnector ? (
          <span className={styles.routeConnector}>
            {durationMinutes ? (
              <span className={styles.routeDuration} title="Durée estimée">
                {durationMinutes} min
              </span>
            ) : null}
          </span>
        ) : null}
      </div>
      <div className={`${styles.routeStepBody} ${styles.routeStepInline}`}>
        <div className={styles.routeStepMain}>
        <div className={styles.routeStepHead}>
          <h3 className={styles.routeStepTitle}>{title}</h3>
          {!isPickup && canRemove ? (
            <button
              type="button"
              className={styles.routeStepRemove}
              onClick={() => onRemove(point.key)}
              aria-label={`Retirer ${title}`}
            >
              Retirer
            </button>
          ) : null}
        </div>
        <AddressAutocomplete
          name={isPickup ? 'pickup_location' : `destination_${destinationIndex}`}
          inputId={isPickup ? 'pickup_location' : `destination_${point.key}`}
          value={point.location || ''}
          onChange={(event) => onPatch(point.key, { location: event.target.value, lat: null, lon: null })}
          onSelect={(item) => {
            const location = item.label || item.address || '';
            onPatch(point.key, {
              location,
              lat: item.lat ?? null,
              lon: item.lon ?? null,
            });
            if (isPickup) return;
            const establishmentName = parseAddressWithEstablishment(location, item).establishment;
            if (establishmentName) reportEstablishment(establishmentName);
          }}
          placeholder={isPickup ? 'Adresse de départ' : 'Adresse de la destination'}
          required
        />
        </div>
        {isPickup && showDeparture ? (
          <MomentField
            label="Heure de départ"
            time={departureTime}
            missionDate={missionDate}
            onTime={onDepartureTime}
            testId="pickup-departure"
            allowOtherDay={false}
            required
          />
        ) : null}
        {!isPickup && showDeparture ? (
          <MomentField
            label={isLastDestination && isRoundTrip ? 'Heure' : 'Départ'}
            date={point.departureDate}
            time={point.departureTime}
            missionDate={missionDate}
            onDate={(value) => onPatch(point.key, { departureDate: value })}
            onTime={(value) => onPatch(point.key, { departureTime: value })}
            testId={`${testId}-departure`}
          />
        ) : null}
      </div>
    </article>
  );
}

export default function CompanyRouteBuilder({
  points,
  missionDate,
  departureTime,
  estimatedDuration,
  isRoundTrip,
  selectedKey,
  onSelect,
  onPatch,
  onReorder,
  onAddDestination,
  onRemoveDestination,
  onToggleRoundTrip,
  onDepartureTime,
}) {
  const dragIndexRef = useRef(null);
  const [dropIndex, setDropIndex] = useState(null);
  const destinations = points.slice(1);
  const pickupLocation = points[0]?.location || '';

  return (
    <section className={styles.routeSection} data-testid="company-route-builder" data-tour-id="booking-addresses">
      <h3 className={styles.routeHeading}>Parcours</h3>
      <div className={styles.routeList} onDragEnd={() => setDropIndex(null)}>
        {points.map((point, index) => {
          const destinationIndex = index;
          const isLastDestination = index === points.length - 1 && index > 0;
          return (
            <RouteStepCard
              key={point.key}
              point={point}
              index={index}
              destinationIndex={destinationIndex}
              isLastDestination={isLastDestination}
              isRoundTrip={isRoundTrip}
              missionDate={missionDate}
              departureTime={departureTime}
              selected={point.key === selectedKey}
              canRemove={destinations.length > 1}
              isDropTarget={dropIndex === index}
              showConnector={isRoundTrip || index < points.length - 1}
              durationMinutes={estimatedDuration}
              onSelect={onSelect}
              onPatch={onPatch}
              onRemove={onRemoveDestination}
              onDepartureTime={onDepartureTime}
              onDragStart={(event) => {
                dragIndexRef.current = index;
                event.dataTransfer.effectAllowed = 'move';
                event.dataTransfer.setData('text/plain', String(index));
              }}
              onDragOver={(event) => {
                event.preventDefault();
                setDropIndex(index);
              }}
              onDrop={(event) => {
                event.preventDefault();
                const from = dragIndexRef.current;
                dragIndexRef.current = null;
                setDropIndex(null);
                if (from == null) return;
                onReorder(from, index);
              }}
            />
          );
        })}
        {isRoundTrip ? (
          <article className={styles.routeStep} data-testid="route-step-return">
            <span className={styles.routeStepHandleSpacer} />
            <div className={styles.routeRail} aria-hidden="true">
              <span className={`${styles.routeDot} ${styles.routeDotReturn}`} />
            </div>
            <div className={`${styles.routeStepBody} ${styles.routeStepInline}`}>
              <div className={styles.routeStepMain}>
                <div className={styles.routeStepHead}>
                  <h3 className={styles.routeStepTitle}>Retour</h3>
                </div>
                <input
                  className={styles.routeReadonly}
                  value={pickupLocation}
                  readOnly
                  aria-label="Adresse de retour, identique au départ"
                />
              </div>
            </div>
          </article>
        ) : null}
      </div>
      <div className={styles.routeActions}>
        <button type="button" className={styles.addStepBtn} onClick={onAddDestination} data-testid="add-destination">
          + Ajouter une destination
        </button>
        <button
          type="button"
          className={`${styles.routeReturnBtn} ${isRoundTrip ? styles.isActive : ''}`}
          data-tour-id="booking-roundtrip-toggle"
          data-testid="route-round-trip"
          aria-pressed={isRoundTrip}
          onClick={onToggleRoundTrip}
        >
          ⇄ A/R
        </button>
      </div>
    </section>
  );
}

export function CompanyRouteDetails({
  points,
  isMaterialDelivery,
  onSelect,
  onPatch,
}) {
  const pickup = points[0];
  const destinations = points.slice(1);

  return (
    <div data-testid="route-details">
      {pickup ? (
        <section className={styles.detailSection} data-testid="pickup-access">
          <h3 className={styles.detailTitle}>Départ</h3>
          <div className={styles.detailAccessRow}>
            <label className={styles.detailLabel} htmlFor={`access-${pickup.key}`}>
              Accès
            </label>
            <input
              id={`access-${pickup.key}`}
              className={styles.detailInput}
              value={pickup.accessNotes || ''}
              onChange={(event) => onPatch(pickup.key, { accessNotes: event.target.value })}
              placeholder="Entrée, code, sonnette…"
            />
          </div>
        </section>
      ) : null}

      {destinations.map((point, index) => {
        const number = index + 1;
        const medical = !isMaterialDelivery && point.destinationKind === 'medical';
        const needsServiceOrDoctor =
          medical &&
          !String(point.service || '').trim() &&
          !String(point.doctor || '').trim();
        return (
          <section
            key={point.key}
            className={styles.detailSection}
            data-testid={`destination-details-${number}`}
            onClick={() => onSelect(point.key)}
          >
            <h3 className={styles.detailTitle}>Destination {number}</h3>
            {!isMaterialDelivery ? (
              <div className={styles.kindToggle} role="group" aria-label={`Type de lieu, destination ${number}`}>
                <button
                  type="button"
                  className={`${styles.kindBtn} ${point.destinationKind === 'medical' ? styles.isActive : ''}`}
                  aria-pressed={point.destinationKind === 'medical'}
                  onClick={() => onPatch(point.key, { destinationKind: 'medical' })}
                >
                  Médical
                </button>
                <button
                  type="button"
                  className={`${styles.kindBtn} ${point.destinationKind !== 'medical' ? styles.isActive : ''}`}
                  aria-pressed={point.destinationKind !== 'medical'}
                  onClick={() => onPatch(point.key, { destinationKind: 'other' })}
                >
                  Autre lieu
                </button>
              </div>
            ) : null}
            {medical ? (
              <>
                <div className={styles.detailAccessRow}>
                  <label className={styles.detailLabel} htmlFor={`establishment-${point.key}`}>
                    Établissement *
                  </label>
                  <EstablishmentSelect
                    inputId={`establishment-${point.key}`}
                    inputClassName={styles.detailInput}
                    required
                    value={point.establishment || ''}
                    onChange={(text) =>
                      onPatch(point.key, { establishment: text, establishmentId: null })
                    }
                    onPickEstablishment={(estab) =>
                      onPatch(point.key, {
                        establishment: estab?.label || estab?.display_name || '',
                        establishmentId: estab?.id ?? null,
                        ...(estab?.address
                          ? {
                              location: estab.address,
                              lat: estab.lat ?? point.lat,
                              lon: estab.lon ?? point.lon,
                            }
                          : {}),
                      })
                    }
                    placeholder="HUG, Clinique La Colline…"
                  />
                </div>
                <div
                  className={styles.serviceOrDoctor}
                  role="group"
                  aria-label="Service ou médecin, au moins un des deux"
                >
                <div className={styles.detailAccessRow}>
                  <label className={styles.detailLabel} htmlFor={`service-${point.key}`}>
                    Service{needsServiceOrDoctor ? ' *' : ''}
                  </label>
                  {point.establishmentId ? (
                    <ServiceSelect
                      key={point.establishmentId}
                      inputId={`service-${point.key}`}
                      establishmentId={point.establishmentId}
                      value={point.serviceId ? { id: point.serviceId, name: point.service } : null}
                      onChange={(service) =>
                        onPatch(point.key, {
                          service: service?.name || '',
                          serviceId: service?.id ?? null,
                        })
                      }
                      placeholder="Ex: Chirurgie, Urgences…"
                    />
                  ) : (
                    <input
                      id={`service-${point.key}`}
                      className={styles.detailInput}
                      value={point.service || ''}
                      onChange={(event) => onPatch(point.key, { service: event.target.value })}
                      placeholder="Ex: Chirurgie, Urgences…"
                    />
                  )}
                </div>
                <div className={styles.detailAccessRow}>
                  <label className={styles.detailLabel} htmlFor={`doctor-${point.key}`}>
                    Médecin{needsServiceOrDoctor ? ' *' : ''}
                  </label>
                  <input
                    id={`doctor-${point.key}`}
                    className={styles.detailInput}
                    value={point.doctor || ''}
                    onChange={(event) => onPatch(point.key, { doctor: event.target.value })}
                    placeholder="Ex : Dr Dupont"
                  />
                </div>
                </div>
              </>
            ) : null}
            <div className={styles.detailAccessRow}>
              <label className={styles.detailLabel} htmlFor={`access-dest-${point.key}`}>
                Accès
              </label>
              <input
                id={`access-dest-${point.key}`}
                className={styles.detailInput}
                value={point.accessNotes || ''}
                onChange={(event) => onPatch(point.key, { accessNotes: event.target.value })}
                placeholder="Entrée, étage, secrétariat…"
              />
            </div>
          </section>
        );
      })}
    </div>
  );
}
