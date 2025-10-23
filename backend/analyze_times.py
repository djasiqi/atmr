#!/usr/bin/env python3
# ruff: noqa: T201, W293
"""
Script pour analyser les heures aller et retour dans les données de transport
"""

import json


def analyze_times():
    """Analyse les heures de départ aller et retour"""

    # Charger les données
    with open('/app/transport_analysis_complete.json', encoding='utf-8') as f:
        data = json.load(f)

    cleaned_data = data['cleaned_data']

    print("=== ANALYSE DES HEURES ALLER ET RETOUR ===")
    print("=" * 60)

    # Statistiques générales
    total_records = len(cleaned_data)
    records_with_depart_aller = len([r for r in cleaned_data if r['heure_depart']])
    records_with_depart_retour = len([r for r in cleaned_data if r['heure_arrivee']])
    records_with_both_times = len([r for r in cleaned_data if r['heure_depart'] and r['heure_arrivee']])

    print(f"Total d'enregistrements: {total_records}")
    print(f"Avec heure de départ aller: {records_with_depart_aller}")
    print(f"Avec heure de départ retour: {records_with_depart_retour}")
    print(f"Avec les deux heures (aller + retour): {records_with_both_times}")
    print()

    # Analyser les types de courses
    course_types = {}
    for record in cleaned_data:
        course_type = record['type_course']
        if course_type not in course_types:
            course_types[course_type] = 0
        course_types[course_type] += 1

    print("Types de courses:")
    for course_type, count in course_types.items():
        print(f"  - {course_type}: {count} courses")
    print()

    # Exemples détaillés
    print("=== EXEMPLES DETAILLES ===")
    print("=" * 60)

    for i, record in enumerate(cleaned_data[:15]):
        print(f"\nEnregistrement {i+1}:")
        print(f"  Client: {record['nom_prenom']}")
        print(f"  Date: {record['date']}")
        print(f"  Heure départ aller: {record['heure_depart']}")
        print(f"  Heure départ retour: {record['heure_arrivee']}")
        print(f"  Type: {record['type_course']}")
        print(f"  Départ: {record['adresse_depart'][:50]}...")
        print(f"  Arrivée: {record['adresse_arrivee'][:50]}...")

        # Analyser le pattern des heures
        if record['heure_depart'] and record['heure_arrivee']:
            print("  ✓ Course A/R complète avec heures aller et retour")
        elif record['heure_depart'] and not record['heure_arrivee']:
            if record['type_course'] == 'A/R':
                print("  ⚠ Course A/R avec seulement l'heure de départ aller")
            elif record['type_course'] == 'A':
                print("  ✓ Course aller simple")
            elif record['type_course'] == 'R':
                print("  ⚠ Course retour avec heure de départ aller (incohérent)")
        elif not record['heure_depart'] and record['heure_arrivee']:
            if record['type_course'] == 'R':
                print("  ✓ Course retour simple")
            else:
                print("  ⚠ Heure de retour sans heure d'aller")
        else:
            print("  ❌ Aucune heure spécifiée")

    # Analyser les patterns d'heures
    print("\n=== ANALYSE DES PATTERNS D'HEURES ===")
    print("=" * 60)

    # Heures de départ aller les plus fréquentes
    depart_aller_times = [r['heure_depart'] for r in cleaned_data if r['heure_depart']]
    depart_aller_freq = {}
    for time in depart_aller_times:
        depart_aller_freq[time] = depart_aller_freq.get(time, 0) + 1

    print("Heures de départ aller les plus fréquentes:")
    for time, freq in sorted(depart_aller_freq.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  - {time}: {freq} fois")

    # Heures de départ retour les plus fréquentes
    depart_retour_times = [r['heure_arrivee'] for r in cleaned_data if r['heure_arrivee']]
    depart_retour_freq = {}
    for time in depart_retour_times:
        depart_retour_freq[time] = depart_retour_freq.get(time, 0) + 1

    print("\nHeures de départ retour les plus fréquentes:")
    for time, freq in sorted(depart_retour_freq.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  - {time}: {freq} fois")

    # Analyser les courses A/R
    ar_courses = [r for r in cleaned_data if r['type_course'] == 'A/R']
    print(f"\nCourses A/R: {len(ar_courses)}")

    ar_with_both_times = [r for r in ar_courses if r['heure_depart'] and r['heure_arrivee']]
    print(f"Courses A/R avec heures aller et retour complètes: {len(ar_with_both_times)}")

    ar_with_only_aller = [r for r in ar_courses if r['heure_depart'] and not r['heure_arrivee']]
    print(f"Courses A/R avec seulement heure de départ aller: {len(ar_with_only_aller)}")

    # Analyser les courses simples
    aller_courses = [r for r in cleaned_data if r['type_course'] == 'A']
    retour_courses = [r for r in cleaned_data if r['type_course'] == 'R']

    print(f"\nCourses aller simples (A): {len(aller_courses)}")
    print(f"Courses retour simples (R): {len(retour_courses)}")

    print("\n=== CONCLUSION ===")
    print("=" * 60)

    if records_with_both_times > 0:
        print("✓ Certaines courses A/R ont des heures aller et retour complètes")
        print("  Format: Départ aller (domicile→destination) + Départ retour (destination→domicile)")

    if records_with_depart_aller > records_with_depart_retour:
        print("⚠ Plus d'heures de départ aller que de retour - pattern normal")
        print("  Pour les courses A/R incomplètes, l'heure de retour peut être calculée automatiquement")

    if len(ar_courses) > 0:
        print(f"✓ {len(ar_courses)} courses sont marquées comme A/R")
        print("  Ces courses nécessitent une planification aller et retour")

    print("\n📊 RÉSUMÉ:")
    print(f"  - Courses A/R complètes: {len(ar_with_both_times)}")
    print(f"  - Courses A/R à compléter: {len(ar_with_only_aller)}")
    print(f"  - Courses aller simples: {len(aller_courses)}")
    print(f"  - Courses retour simples: {len(retour_courses)}")

if __name__ == "__main__":
    analyze_times()
