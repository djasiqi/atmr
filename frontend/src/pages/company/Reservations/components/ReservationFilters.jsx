import React from 'react';
import styles from './ReservationFilters.module.css';

const ReservationFilters = ({
  selectedDay,
  setSelectedDay,
  searchTerm,
  setSearchTerm,
  sortOrder,
  setSortOrder,
  searchInputRef,
}) => {
  const [dateMode, setDateMode] = React.useState('all'); // "all", "single", "range"
  const [startDate, setStartDate] = React.useState('');
  const [endDate, setEndDate] = React.useState('');

  // Sync dateMode avec selectedDay quand le parent change (ex: reset)
  React.useEffect(() => {
    if (selectedDay === 'all') {
      setDateMode('all');
      setStartDate('');
      setEndDate('');
    } else if (selectedDay && selectedDay.includes(':')) {
      const [s, e] = selectedDay.split(':');
      setDateMode('range');
      setStartDate(s || '');
      setEndDate(e || '');
    } else if (selectedDay) {
      setDateMode('single');
    }
  }, [selectedDay]);

  // Gérer le changement de mode de date
  const handleDateModeChange = (mode) => {
    setDateMode(mode);
    if (mode === 'all') {
      setSelectedDay('all');
      setStartDate('');
      setEndDate('');
    } else if (mode === 'single') {
      setStartDate('');
      setEndDate('');
    } else if (mode === 'range') {
      setSelectedDay('');
    }
  };

  // Appliquer la plage de dates
  React.useEffect(() => {
    if (dateMode === 'range' && startDate && endDate) {
      // Créer une chaîne de plage pour le backend
      setSelectedDay(`${startDate}:${endDate}`);
    }
  }, [dateMode, startDate, endDate, setSelectedDay]);

  return (
    <>
      <div className={styles.filters}>
        <div className={styles.searchBox}>
          <label className={styles.searchLabel}>🔍 Recherche globale</label>
          <div className={styles.searchContainer}>
            <input
              type="text"
              placeholder="ID, prénom, nom, adresse (rue, restaurant...), clinique, HUG, docteur, date (13.01, 13/01, 13 janvier)..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className={styles.searchInput}
              ref={searchInputRef}
            />
            {searchTerm && (
              <button
                type="button"
                onClick={() => setSearchTerm('')}
                className={styles.clearSearchButton}
                title="Effacer la recherche"
              >
                ✖
              </button>
            )}
          </div>
        </div>

      </div>

      <div className={styles.filtersRow}>
        <div className={styles.filterGroup}>
          <label>📅 Période</label>
          <div className={styles.dateModeContainer}>
            <div className={styles.dateModeButtons}>
              <button
                type="button"
                onClick={() => handleDateModeChange('all')}
                className={`${styles.dateModeButton} ${dateMode === 'all' ? styles.active : ''}`}
              >
                Toutes
              </button>
              <button
                type="button"
                onClick={() => handleDateModeChange('single')}
                className={`${styles.dateModeButton} ${dateMode === 'single' ? styles.active : ''}`}
              >
                Une date
              </button>
              <button
                type="button"
                onClick={() => handleDateModeChange('range')}
                className={`${styles.dateModeButton} ${dateMode === 'range' ? styles.active : ''}`}
              >
                Période
              </button>
            </div>

            {dateMode === 'single' && (
              <div className={styles.singleDateContainer}>
                <input
                  type="date"
                  value={selectedDay === 'all' || selectedDay.includes(':') ? '' : selectedDay}
                  onChange={(e) => setSelectedDay(e.target.value)}
                  className={styles.dateInput}
                  placeholder="Sélectionner une date"
                />
              </div>
            )}

            {dateMode === 'range' && (
              <div className={styles.dateRangeContainer}>
                <input
                  type="date"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                  className={styles.dateInput}
                  placeholder="Du"
                />
                <span className={styles.dateRangeSeparator}>→</span>
                <input
                  type="date"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                  className={styles.dateInput}
                  placeholder="Au"
                  min={startDate}
                />
              </div>
            )}
          </div>
        </div>

        <div className={styles.filterGroup}>
          <label>🔄 Ordre de tri</label>
          <select
            value={sortOrder}
            onChange={(e) => setSortOrder(e.target.value)}
            className={styles.selectInput}
          >
            <option value="desc">⬇️ Plus récent d'abord</option>
            <option value="asc">⬆️ Plus ancien d'abord</option>
          </select>
        </div>
      </div>
    </>
  );
};

export default ReservationFilters;
