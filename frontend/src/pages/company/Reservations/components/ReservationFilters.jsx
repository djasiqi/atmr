import React from 'react';
import styles from './ReservationFilters.module.css';

const ReservationFilters = ({
  selectedDay,
  setSelectedDay,
  searchTerm,
  setSearchTerm,
  statusFilter,
  setStatusFilter,
  sortOrder,
  setSortOrder,
  searchInputRef,
}) => {
  const [dateMode, setDateMode] = React.useState('all'); // "all", "single", "range"
  const [startDate, setStartDate] = React.useState('');
  const [endDate, setEndDate] = React.useState('');

  // Fonction pour réinitialiser tous les filtres
  const handleResetFilters = () => {
    setSelectedDay('all');
    setDateMode('all');
    setStartDate('');
    setEndDate('');
    setSearchTerm('');
    setStatusFilter('all');
    setSortOrder('desc');
  };

  // Vérifier si des filtres sont actifs
  const hasActiveFilters =
    selectedDay !== 'all' ||
    dateMode !== 'all' ||
    startDate !== '' ||
    endDate !== '' ||
    searchTerm !== '' ||
    statusFilter !== 'all' ||
    sortOrder !== 'desc';

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
      <div className={styles.filtersHeader}>
        <h3>🔍 Filtres et Recherche</h3>
        {hasActiveFilters && (
          <button
            onClick={handleResetFilters}
            className={styles.resetButton}
            title="Réinitialiser tous les filtres"
          >
            ✖ Réinitialiser
          </button>
        )}
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
          <label>🔍 Recherche globale</label>
          <div className={styles.searchContainer}>
            <input
              type="text"
              placeholder="ID, client, email, téléphone, adresse, chauffeur..."
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

        <div className={styles.filterGroup}>
          <label>📊 Statut</label>
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className={styles.selectInput}
          >
            <option value="all">Tous les statuts</option>
            <option value="pending">⏳ En attente</option>
            <option value="accepted">✅ Acceptée</option>
            <option value="assigned">👤 Assignée</option>
            <option value="in_progress">🚗 En cours</option>
            <option value="completed">✔️ Terminée</option>
            <option value="canceled">❌ Annulée</option>
          </select>
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
