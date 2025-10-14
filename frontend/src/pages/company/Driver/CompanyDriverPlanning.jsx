// CompanyDriverPlanning.jsx
import React, { useEffect, useMemo, useState } from "react";
import { useLocation, useParams } from "react-router-dom";
import {
  fetchCompanyDriver,
  fetchShifts,
  createShift,
  updateShift,
  deleteShift,
} from "../../../services/driverPlanningService";
import CompanyHeader from "../../../components/layout/Header/CompanyHeader";
import CompanySidebar from "../../../components/layout/Sidebar/CompanySidebar/CompanySidebar";
import styles from "./CompanyDriverPlanning.module.css";

function DriverSidePanel({ driver, onUpdate }) {
  const [form, setForm] = useState({
    contract_type: driver.contract_type || "CDI",
    weekly_hours: driver.weekly_hours || "",
    hourly_rate_cents: driver.hourly_rate_cents || "",
    employment_start_date: driver.employment_start_date || "",
    employment_end_date: driver.employment_end_date || "",
    license_categories: Array.isArray(driver.license_categories)
      ? driver.license_categories.join(",")
      : "",
    license_valid_until: driver.license_valid_until || "",
    medical_valid_until: driver.medical_valid_until || "",
  });

  useEffect(() => {
    setForm({
      contract_type: driver.contract_type || "CDI",
      weekly_hours: driver.weekly_hours || "",
      hourly_rate_cents: driver.hourly_rate_cents || "",
      employment_start_date: driver.employment_start_date || "",
      employment_end_date: driver.employment_end_date || "",
      license_categories: Array.isArray(driver.license_categories)
        ? driver.license_categories.join(",")
        : "",
      license_valid_until: driver.license_valid_until || "",
      medical_valid_until: driver.medical_valid_until || "",
    });
  }, [driver]);

  if (!driver) return null;

  const save = () => {
    const payload = {
      ...form,
      weekly_hours: form.weekly_hours === "" ? null : Number(form.weekly_hours),
      hourly_rate_cents:
        form.hourly_rate_cents === "" ? null : Number(form.hourly_rate_cents),
      license_categories: form.license_categories
        ? String(form.license_categories)
            .split(",")
            .map((s) => s.trim())
            .filter(Boolean)
        : [],
    };
    onUpdate?.(payload);
  };

  const Label = ({ children }) => (
    <label className={styles.label}>{children}</label>
  );

  return (
    <aside className={styles.sidePanel}>
      <div className={styles.sectionTitle}>Fiche chauffeur</div>
      <div className={styles.formGrid}>
        <div>
          <Label>Contrat</Label>
          <select
            value={form.contract_type}
            onChange={(e) =>
              setForm({ ...form, contract_type: e.target.value })
            }
          >
            <option value="CDI">CDI</option>
            <option value="CDD">CDD</option>
            <option value="HOURLY">Horaire</option>
          </select>
        </div>
        <div>
          <Label>Heures/sem.</Label>
          <input
            className={styles.input}
            value={form.weekly_hours}
            onChange={(e) => setForm({ ...form, weekly_hours: e.target.value })}
          />
        </div>
        <div>
          <Label>Taux horaire (cts)</Label>
          <input
            className={styles.input}
            value={form.hourly_rate_cents}
            onChange={(e) =>
              setForm({ ...form, hourly_rate_cents: e.target.value })
            }
          />
        </div>
        <div>
          <Label>Début</Label>
          <input
            className={styles.input}
            type="date"
            value={form.employment_start_date || ""}
            onChange={(e) =>
              setForm({ ...form, employment_start_date: e.target.value })
            }
          />
        </div>
        <div>
          <Label>Fin</Label>
          <input
            className={styles.input}
            type="date"
            value={form.employment_end_date || ""}
            onChange={(e) =>
              setForm({ ...form, employment_end_date: e.target.value })
            }
          />
        </div>
        <div style={{ gridColumn: "1 / span 2" }}>
          <Label>Permis (ex: B,C1)</Label>
          <input
            className={styles.input}
            value={form.license_categories}
            onChange={(e) =>
              setForm({ ...form, license_categories: e.target.value })
            }
          />
        </div>
        <div>
          <Label>Permis valable jusqu'</Label>
          <input
            className={styles.input}
            type="date"
            value={form.license_valid_until || ""}
            onChange={(e) =>
              setForm({ ...form, license_valid_until: e.target.value })
            }
          />
        </div>
        <div>
          <Label>Visite médicale</Label>
          <input
            className={styles.input}
            type="date"
            value={form.medical_valid_until || ""}
            onChange={(e) =>
              setForm({ ...form, medical_valid_until: e.target.value })
            }
          />
        </div>
      </div>
      <div className={styles.spacer}></div>
      <button
        className={`${styles.button} ${styles.buttonPrimary}`}
        onClick={save}
      >
        Enregistrer
      </button>
    </aside>
  );
}

function CompanyDriverPlanning() {
  const { public_id } = useParams();
  const location = useLocation();
  const params = new URLSearchParams(location.search);
  const preselectedDriverId = params.get("driver_id");

  const [drivers, setDrivers] = useState([]);
  const [selectedDriverId, setSelectedDriverId] = useState(
    preselectedDriverId ? Number(preselectedDriverId) : null
  );
  const [rangeFrom, setRangeFrom] = useState(() => new Date());
  const [rangeTo, setRangeTo] = useState(
    () => new Date(Date.now() + 7 * 86400000)
  );
  const [shifts, setShifts] = useState([]); // sélection chauffeur
  const [allShiftsByDriver, setAllShiftsByDriver] = useState({}); // vue tous les chauffeurs
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchCompanyDriver()
      .then(setDrivers)
      .catch(() => setDrivers([]));
  }, []);

  const selectedDriver = useMemo(
    () => drivers.find((d) => d.id === selectedDriverId) || null,
    [drivers, selectedDriverId]
  );

  async function loadShifts() {
    if (!selectedDriverId) return;
    setLoading(true);
    setError(null);
    try {
      const fromIso = new Date(rangeFrom).toISOString();
      const toIso = new Date(rangeTo).toISOString();
      const data = await fetchShifts({
        from: fromIso,
        to: toIso,
        driverId: selectedDriverId,
      });
      setShifts(Array.isArray(data?.items) ? data.items : []);
    } catch (e) {
      setError("Erreur de chargement des shifts");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadShifts();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedDriverId, rangeFrom, rangeTo]);

  async function loadAllDriversShifts() {
    if (!drivers || drivers.length === 0) return;
    setLoading(true);
    setError(null);
    try {
      const fromIso = new Date(rangeFrom).toISOString();
      const toIso = new Date(rangeTo).toISOString();
      const entries = await Promise.all(
        drivers.map(async (drv) => {
          try {
            const res = await fetchShifts({
              from: fromIso,
              to: toIso,
              driverId: drv.id,
            });
            const items = Array.isArray(res?.items) ? res.items : [];
            return [drv.id, items];
          } catch {
            return [drv.id, []];
          }
        })
      );
      setAllShiftsByDriver(Object.fromEntries(entries));
    } catch (e) {
      setError("Erreur de chargement multi-chauffeurs");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (selectedDriverId) return; // only when viewing all drivers
    loadAllDriversShifts();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [drivers, selectedDriverId, rangeFrom, rangeTo]);

  const handleCreate = async () => {
    if (!selectedDriverId) return;
    const start = new Date();
    const end = new Date(Date.now() + 2 * 3600000);
    try {
      await createShift(public_id, {
        driver_id: selectedDriverId,
        start_local: start.toISOString(),
        end_local: end.toISOString(),
        type: "regular",
        status: "planned",
      });
      await loadShifts();
    } catch (e) {
      alert("Création impossible: chevauchement ou données invalides");
    }
  };

  const handleUpdate = async (shift) => {
    try {
      const end = new Date(new Date(shift.end_local).getTime() + 3600000);
      await updateShift(public_id, shift.id, { end_local: end.toISOString() });
      await loadShifts();
    } catch (e) {
      alert("Mise à jour impossible");
    }
  };

  const handleDelete = async (shift) => {
    try {
      await deleteShift(public_id, shift.id);
      await loadShifts();
    } catch (e) {
      alert("Suppression impossible");
    }
  };

  const minutesBetween = (a, b) =>
    Math.max(0, Math.round((new Date(b) - new Date(a)) / 60000));
  const isWeekend = (iso) => {
    const d = new Date(iso);
    const day = d.getDay();
    return day === 0 || day === 6;
  };
  const dayPart = (startIso, endIso) => {
    const startH = new Date(startIso).getHours();
    const endH = new Date(endIso).getHours();
    if (isWeekend(startIso) || isWeekend(endIso)) return "week-end";
    if (startH < 6 || endH < 6 || startH >= 22 || endH >= 22) return "nuit";
    return "jour";
  };

  return (
    <div className={styles.pageContainer}>
      <CompanyHeader />
      <div className={styles.layoutWrapper}>
        <CompanySidebar />
        <div className={styles.mainContent}>
          <div className={styles.container}>
            <h1 className={styles.pageTitle}>Planning des chauffeurs</h1>
            <div className={styles.toolbar}>
        <div className={styles.controls}>
          <label>Chauffeur</label>
          <select
            className={styles.select}
            value={selectedDriverId || ""}
            onChange={(e) =>
              setSelectedDriverId(Number(e.target.value) || null)
            }
          >
            <option value="">Sélectionner un chauffeur</option>
            {drivers.map((drv) => (
              <option key={drv.id} value={drv.id}>
                {drv.full_name || drv.username}
              </option>
            ))}
          </select>
        </div>
        <div className={styles.controls} style={{ marginLeft: "auto" }}>
          <button
            className={styles.button}
            onClick={() => setSelectedDriverId(null)}
            title="Voir tous les chauffeurs"
          >
            Tous les chauffeurs
          </button>
          <label>Du</label>
          <input
            className={styles.input}
            type="date"
            value={new Date(rangeFrom).toISOString().slice(0, 10)}
            onChange={(e) => setRangeFrom(new Date(e.target.value))}
          />
          <label>Au</label>
          <input
            className={styles.input}
            type="date"
            value={new Date(rangeTo).toISOString().slice(0, 10)}
            onChange={(e) => setRangeTo(new Date(e.target.value))}
          />
          <button
            className={`${styles.button} ${styles.buttonPrimary}`}
            disabled={!selectedDriverId}
            onClick={handleCreate}
          >
            Créer shift 2h
          </button>
        </div>
      </div>

      {loading && <div>Chargement…</div>}
      {error && <div className={styles.error}>{error}</div>}

      {selectedDriver && (
        <div className={styles.grid} style={{ marginBottom: 12 }}>
          <div className={styles.card}>
            <div className={styles.cardHeader}>Identité</div>
            <div>
              Nom: {selectedDriver.full_name || selectedDriver.username}
            </div>
            <div>Rôle/Fonction: Chauffeur</div>
            <div>Équipe/Site: —</div>
          </div>
          <div className={styles.card}>
            <div className={styles.cardHeader}>KPI période</div>
            <div className={styles.kpis}>
              <div className={styles.kpi}>
                <div className={styles.kpiLabel}>Shifts</div>
                <div className={styles.kpiValue}>{shifts.length}</div>
              </div>
              <div className={styles.kpi}>
                <div className={styles.kpiLabel}>Durée totale</div>
                <div className={styles.kpiValue}>
                  {shifts.reduce(
                    (m, s) => m + minutesBetween(s.start_local, s.end_local),
                    0
                  )}{" "}
                  min
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {selectedDriver && (
        <div className={styles.tableWrap}>
          <h2>{selectedDriver.full_name || selectedDriver.username}</h2>
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Début</th>
                <th>Fin</th>
                <th>Durée</th>
                <th>Jour/Nuit/WE</th>
                <th>Type</th>
                <th>Statut</th>
                <th>Véhicule</th>
                <th>Notes</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {shifts.map((s) => (
                <tr key={s.id}>
                  <td>{s.start_local}</td>
                  <td>{s.end_local}</td>
                  <td>{minutesBetween(s.start_local, s.end_local)} min</td>
                  <td>{dayPart(s.start_local, s.end_local)}</td>
                  <td>{s.type}</td>
                  <td>{s.status}</td>
                  <td>{selectedDriver.vehicle_assigned || "—"}</td>
                  <td>{s.notes || ""}</td>
                  <td>
                    <div className={styles.tableActions}>
                      <button onClick={() => handleUpdate(s)}>+1h</button>
                      <button onClick={() => handleDelete(s)}>Supprimer</button>
                    </div>
                  </td>
                </tr>
              ))}
              {shifts.length === 0 && (
                <tr>
                  <td colSpan={8} style={{ textAlign: "center" }}>
                    Aucun shift
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      {!selectedDriver && (
        <div className={styles.tableWrap}>
          <h2>Planning — Tous les chauffeurs</h2>
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Chauffeur</th>
                <th>Début</th>
                <th>Fin</th>
                <th>Durée</th>
                <th>Jour/Nuit/WE</th>
                <th>Type</th>
                <th>Statut</th>
                <th>Véhicule</th>
              </tr>
            </thead>
            <tbody>
              {drivers.length === 0 && (
                <tr>
                  <td colSpan={8} style={{ textAlign: "center" }}>
                    Aucun chauffeur
                  </td>
                </tr>
              )}
              {drivers.flatMap((drv) => {
                const items = allShiftsByDriver[drv.id] || [];
                if (items.length === 0) {
                  return [
                    <tr key={`drv-${drv.id}-empty`}>
                      <td>{drv.username}</td>
                      <td colSpan={7} style={{ color: "#6b7280" }}>
                        Aucun shift
                      </td>
                    </tr>,
                  ];
                }
                return items.map((s) => (
                  <tr key={`drv-${drv.id}-${s.id}`}>
                    <td>{drv.full_name || drv.username}</td>
                    <td>{s.start_local}</td>
                    <td>{s.end_local}</td>
                    <td>{minutesBetween(s.start_local, s.end_local)} min</td>
                    <td>{dayPart(s.start_local, s.end_local)}</td>
                    <td>{s.type}</td>
                    <td>{s.status}</td>
                    <td>{drv.vehicle_assigned || "—"}</td>
                  </tr>
                ));
              })}
            </tbody>
          </table>
        </div>
      )}
      {/* Side panel RH */}
      {selectedDriver && (
        <DriverSidePanel
          driver={selectedDriver}
          onUpdate={async (payload) => {
            try {
              // Reuse driver profile endpoint
              // Uses /api/driver/me/profile if current user is driver; for company edit, call admin route if available
              const res = await fetch(
                "/api/driver/" + selectedDriver.id + "/update-profile",
                {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify(payload),
                }
              );
              if (!res.ok) throw new Error("bad status");
              // Soft refresh drivers list
              const list = await fetchCompanyDriver().catch(() => []);
              setDrivers(Array.isArray(list) ? list : []);
            } catch (_) {
              alert("Erreur lors de l'enregistrement");
            }
          }}
        />
      )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default CompanyDriverPlanning;
