// frontend/src/components/ui/TabNavigation.jsx
import React from 'react';
import styles from './TabNavigation.module.css';

const TabNavigation = ({ tabs, activeTab, onTabChange }) => {
  return (
    <div className={styles.tabsContainer}>
      {tabs.map((tab) => (
        <button
          key={tab.id}
          className={activeTab === tab.id ? styles.tabActive : styles.tab}
          onClick={() => onTabChange(tab.id)}
          type="button"
        >
          {tab.icon && <span className={styles.tabIcon}>{tab.icon}</span>}
          <span className={styles.tabLabel}>{tab.label}</span>
        </button>
      ))}
    </div>
  );
};

export default TabNavigation;
