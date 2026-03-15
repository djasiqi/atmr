import React, { useState, useRef, useEffect } from "react";
import {
    View,
    Text,
    TouchableOpacity,
    StyleSheet,
    TextInput,
    Modal,
    ScrollView,
    Dimensions,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import dayjs from "dayjs";
import customParseFormat from "dayjs/plugin/customParseFormat";
import { getLogger } from "@/utils/logger";

const log = getLogger("DatePicker");
dayjs.extend(customParseFormat);

const palette = {
    background: "#FFFFFF",
    border: "rgba(15,54,43,0.12)",
    text: "#15362B",
    textSecondary: "#5F7369",
    accent: "#0A7F59",
    buttonBg: "rgba(10,127,89,0.08)",
};

interface TimeDatePickerProps {
    label: string;
    value: Date | null;
    onChange: (date: Date | null) => void;
    mode?: "date" | "time" | "datetime";
    minimumDate?: Date;
    maximumDate?: Date;
    autoOpen?: boolean; // Si true, ouvre automatiquement le modal au montage
}

export const TimeDatePicker: React.FC<TimeDatePickerProps> = ({
    label,
    value,
    onChange,
    mode = "datetime",
    minimumDate,
    maximumDate,
    autoOpen = false,
}) => {
    const [showModal, setShowModal] = useState(false);
    const [dateInput, setDateInput] = useState(
        value ? dayjs(value).format("DD.MM.YYYY") : ""
    );
    const [timeInput, setTimeInput] = useState(
        value ? dayjs(value).format("HH:mm") : ""
    );
    const previousDateInputRef = useRef<string>(dateInput);
    const previousTimeInputRef = useRef<string>(timeInput);

    // États pour les roues iOS
    const [selectedDay, setSelectedDay] = useState<number>(
        value ? dayjs(value).date() : dayjs().date()
    );
    const [selectedMonth, setSelectedMonth] = useState<number>(
        value ? dayjs(value).month() + 1 : dayjs().month() + 1
    );
    const [selectedYear, setSelectedYear] = useState<number>(
        value ? dayjs(value).year() : dayjs().year()
    );
    const [selectedHour, setSelectedHour] = useState<number>(
        value ? dayjs(value).hour() : dayjs().hour()
    );
    const [selectedMinute, setSelectedMinute] = useState<number>(
        value ? dayjs(value).minute() : dayjs().minute()
    );

    // Refs pour les ScrollView
    const dayScrollRef = useRef<ScrollView>(null);
    const monthScrollRef = useRef<ScrollView>(null);
    const yearScrollRef = useRef<ScrollView>(null);
    const hourScrollRef = useRef<ScrollView>(null);
    const minuteScrollRef = useRef<ScrollView>(null);

    const ITEM_HEIGHT = 35;
    const WHEEL_HEIGHT = 3 * ITEM_HEIGHT; // 3 items visibles

    const formatDateInput = (text: string): string => {
        // Supprimer tout sauf les chiffres
        const digits = text.replace(/\D/g, "");

        if (digits.length === 0) {
            return "";
        }

        // Limiter à 8 chiffres max (DDMMYYYY)
        const limited = digits.slice(0, 8);

        // Détecter si on est en train de supprimer (en comparant avec la valeur précédente)
        const previousDigits = previousDateInputRef.current.replace(/\D/g, "");
        const isDeleting = limited.length < previousDigits.length;

        if (limited.length <= 2) {
            // Juste le jour (01-31)
            const day = parseInt(limited, 10);
            if (day > 31) {
                return "31";
            }
            // Ajouter le point seulement si on n'est pas en train de supprimer et qu'on a 2 chiffres
            if (limited.length === 2 && !isDeleting) {
                return `${limited}.`;
            }
            return limited;
        } else if (limited.length <= 4) {
            // Jour + mois
            const day = parseInt(limited.slice(0, 2), 10);

            // Valider le jour (01-31)
            const validDay = Math.min(day, 31).toString().padStart(2, "0");

            if (limited.length === 3) {
                // Si on a 3 chiffres, le 3ème est le premier chiffre du mois
                const firstMonthDigit = parseInt(limited[2], 10);
                if (firstMonthDigit > 1 && !isDeleting) {
                    return `${validDay}.12`;
                }
                return `${validDay}.${firstMonthDigit}`;
            } else {
                // 4 chiffres
                const month = parseInt(limited.slice(2, 4), 10);
                const validMonth = Math.min(month, 12).toString().padStart(2, "0");
                // Ajouter le deuxième point seulement si on n'est pas en train de supprimer
                if (!isDeleting) {
                    return `${validDay}.${validMonth}.`;
                }
                return `${validDay}.${validMonth}`;
            }
        } else {
            // Jour + mois + année
            const day = parseInt(limited.slice(0, 2), 10);
            const month = parseInt(limited.slice(2, 4), 10);

            // Valider le jour (01-31)
            const validDay = Math.min(day, 31).toString().padStart(2, "0");

            // Valider le mois (01-12)
            const validMonth = Math.min(month, 12).toString().padStart(2, "0");

            // Valider l'année
            const currentYear = dayjs().year();
            const currentMonth = dayjs().month() + 1; // 1-12
            
            // Déterminer l'année minimale et maximale
            // Si minimumDate/maximumDate sont fournis, les utiliser
            // Sinon, utiliser les valeurs par défaut (année en cours à +2/+3 ans)
            let minYear = currentYear; // Par défaut : année en cours (pour les rendez-vous)
            let maxYear = currentMonth >= 11 ? currentYear + 3 : currentYear + 2; // Par défaut : +2 ou +3 ans
            
            // ✅ Si minimumDate est fourni, utiliser son année (peut être dans le passé pour les dates de naissance)
            if (minimumDate) {
                const minYearFromDate = dayjs(minimumDate).year();
                minYear = minYearFromDate;
            }
            
            // ✅ Si maximumDate est fourni, utiliser son année (peut être aujourd'hui pour les dates de naissance)
            if (maximumDate) {
                const maxYearFromDate = dayjs(maximumDate).year();
                maxYear = maxYearFromDate;
            }

            if (limited.length === 5) {
                // Si on a 5 chiffres, on a juste le premier chiffre de l'année
                const firstYearDigit = parseInt(limited[4], 10);
                // Ne pas forcer, juste afficher le chiffre
                return `${validDay}.${validMonth}.${firstYearDigit}`;
            } else if (limited.length === 6) {
                // 2 chiffres d'année (ex: "20")
                const twoDigits = parseInt(limited.slice(4, 6), 10);
                // Laisser l'utilisateur continuer à écrire librement, sans complétion automatique
                return `${validDay}.${validMonth}.${twoDigits}`;
            } else if (limited.length === 7) {
                // 3 chiffres d'année
                const threeDigits = parseInt(limited.slice(4, 7), 10);
                // Laisser continuer sans forcer de complétion
                return `${validDay}.${validMonth}.${threeDigits}`;
            } else {
                // 4 chiffres d'année - validation finale
                const year = parseInt(limited.slice(4, 8), 10);
                let validYear = year;

                // Valider seulement si hors limites raisonnables
                if (year < minYear) {
                    validYear = minYear;
                } else if (year > maxYear) {
                    validYear = maxYear;
                }
                return `${validDay}.${validMonth}.${validYear}`;
            }
        }
    };

    const handleDateChange = (text: string) => {
        const formatted = formatDateInput(text);
        previousDateInputRef.current = dateInput;
        setDateInput(formatted);
    };

    const formatTimeInput = (text: string): string => {
        // Supprimer tout sauf les chiffres
        const digits = text.replace(/\D/g, "");

        if (digits.length === 0) {
            return "";
        }

        // Limiter à 4 chiffres max (HHMM)
        const limited = digits.slice(0, 4);

        // Détecter si on est en train de supprimer
        // Comparer la longueur totale du texte (avec ":") et la longueur des chiffres
        const previousText = previousTimeInputRef.current;
        const previousDigits = previousText.replace(/\D/g, "");
        const previousHasColon = previousText.includes(":");
        const currentHasColon = text.includes(":");

        // On est en train de supprimer si :
        // - Le nombre de chiffres a diminué, OU
        // - Le texte précédent avait un ":" et le texte actuel n'en a plus (suppression du ":")
        const isDeleting = limited.length < previousDigits.length || (previousHasColon && !currentHasColon);

        if (limited.length <= 2) {
            // Juste les heures (00-23)
            const hours = parseInt(limited, 10);
            if (hours > 23) {
                return "23";
            }
            // Ajouter le deux-points seulement si on n'est pas en train de supprimer et qu'on a 2 chiffres
            if (limited.length === 2 && !isDeleting) {
                return `${limited}:`;
            }
            return limited;
        } else {
            // Heures + minutes
            const hours = parseInt(limited.slice(0, 2), 10);
            const validHours = Math.min(hours, 23).toString().padStart(2, "0");

            if (limited.length === 3) {
                // Si on a 3 chiffres, le 3ème est le premier chiffre des minutes
                const firstMinuteDigit = parseInt(limited[2], 10);
                if (firstMinuteDigit > 5 && !isDeleting) {
                    return `${validHours}:59`;
                }
                return `${validHours}:${firstMinuteDigit}`;
            } else {
                // 4 chiffres
                const minutes = parseInt(limited.slice(2, 4), 10);
                const validMinutes = Math.min(minutes, 59).toString().padStart(2, "0");
                return `${validHours}:${validMinutes}`;
            }
        }
    };

    const handleTimeChange = (text: string) => {
        const formatted = formatTimeInput(text);
        previousTimeInputRef.current = timeInput;
        setTimeInput(formatted);

        // Synchroniser selectedHour et selectedMinute avec la valeur saisie
        if (formatted.includes(":")) {
            const [hours, minutes] = formatted.split(":").map(Number);
            if (!isNaN(hours) && hours >= 0 && hours <= 23) {
                setSelectedHour(hours);
            }
            if (!isNaN(minutes) && minutes >= 0 && minutes <= 59) {
                setSelectedMinute(minutes);
            }
        }
    };

    const displayValue = value
        ? mode === "date"
            ? dayjs(value).format("DD.MM.YYYY")
            : mode === "time"
                ? (dayjs(value).hour() === 0 && dayjs(value).minute() === 0)
                    ? "Non défini"
                    : dayjs(value).format("HH:mm")
                : dayjs(value).format("DD.MM.YYYY HH:mm")
        : "Non défini";

    const handlePress = () => {
        if (value) {
            const d = dayjs(value);
            setSelectedDay(d.date());
            setSelectedMonth(d.month() + 1);
            setSelectedYear(d.year());
            setSelectedHour(d.hour());
            setSelectedMinute(d.minute());
            setDateInput(d.format("DD.MM.YYYY"));
            // Si l'heure est à 00:00 en mode time, initialiser avec un champ vide
            if (mode === "time" && d.hour() === 0 && d.minute() === 0) {
                setTimeInput("");
            } else {
                setTimeInput(d.format("HH:mm"));
            }
        } else {
            const now = dayjs();
            setSelectedDay(now.date());
            setSelectedMonth(now.month() + 1);
            setSelectedYear(now.year());
            setSelectedHour(now.hour());
            setSelectedMinute(now.minute());
            setDateInput(now.format("DD.MM.YYYY"));
            setTimeInput(now.format("HH:mm"));
        }
        setShowModal(true);
    };

    // Générer les listes pour les roues
    const generateDays = (month: number, year: number): number[] => {
        const daysInMonth = dayjs(`${year}-${month}-01`).daysInMonth();
        return Array.from({ length: daysInMonth }, (_, i) => i + 1);
    };

    const months = Array.from({ length: 12 }, (_, i) => i + 1);
    const currentYear = dayjs().year();
    const years = Array.from({ length: 5 }, (_, i) => currentYear + i);
    const hours = Array.from({ length: 24 }, (_, i) => i);
    const minutes = Array.from({ length: 60 }, (_, i) => i);

    // Ouvrir automatiquement le modal si autoOpen est true
    useEffect(() => {
        if (autoOpen && !showModal) {
            handlePress();
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [autoOpen]);

    // Synchroniser timeInput avec selectedHour et selectedMinute quand ils changent
    useEffect(() => {
        if (mode !== "date" && showModal) {
            const newTimeInput = `${selectedHour.toString().padStart(2, "0")}:${selectedMinute.toString().padStart(2, "0")}`;
            // Ne mettre à jour que si timeInput est vide ou différent
            if (!timeInput || timeInput.trim() === "" || timeInput !== newTimeInput) {
                // Vérifier que timeInput n'est pas en cours de modification par l'utilisateur
                // (on ne synchronise que si l'utilisateur n'est pas en train de taper)
                const currentTimeParts = timeInput.split(":");
                if (currentTimeParts.length === 2) {
                    const [currentHours, currentMinutes] = currentTimeParts.map(Number);
                    // Si l'heure actuelle ne correspond pas à selectedHour/selectedMinute, synchroniser
                    if (currentHours !== selectedHour || currentMinutes !== selectedMinute) {
                        setTimeInput(newTimeInput);
                    }
                } else if (!timeInput || timeInput.trim() === "") {
                    setTimeInput(newTimeInput);
                }
            }
        }
    }, [selectedHour, selectedMinute, showModal, mode]);

    // Scroll vers la valeur sélectionnée au montage
    useEffect(() => {
        if (showModal) {
            setTimeout(() => {
                if (dayScrollRef.current) {
                    dayScrollRef.current.scrollTo({
                        y: (selectedDay - 1) * ITEM_HEIGHT,
                        animated: false,
                    });
                }
                if (monthScrollRef.current) {
                    monthScrollRef.current.scrollTo({
                        y: (selectedMonth - 1) * ITEM_HEIGHT,
                        animated: false,
                    });
                }
                if (yearScrollRef.current) {
                    const yearIndex = years.indexOf(selectedYear);
                    if (yearIndex >= 0) {
                        yearScrollRef.current.scrollTo({
                            y: yearIndex * ITEM_HEIGHT,
                            animated: false,
                        });
                    }
                }
                if (hourScrollRef.current) {
                    hourScrollRef.current.scrollTo({
                        y: selectedHour * ITEM_HEIGHT,
                        animated: false,
                    });
                }
                if (minuteScrollRef.current) {
                    minuteScrollRef.current.scrollTo({
                        y: selectedMinute * ITEM_HEIGHT,
                        animated: false,
                    });
                }
            }, 100);
        }
    }, [showModal]);

    const handleWheelScroll = (
        scrollView: ScrollView,
        setter: (value: number) => void,
        items: number[]
    ) => {
        return (event: any) => {
            const y = event.nativeEvent.contentOffset.y;
            const index = Math.round(y / ITEM_HEIGHT);
            const clampedIndex = Math.max(0, Math.min(index, items.length - 1));
            setter(items[clampedIndex]);
        };
    };

    const parseDateInput = (dateStr: string): dayjs.Dayjs | null => {
        // Accepter DD.MM.YYYY ou DD/MM/YYYY
        // Extraire manuellement les composants pour éviter toute confusion
        const normalized = dateStr.replace(/\./g, "/");
        const parts = normalized.split("/");

        if (parts.length === 3) {
            const day = parseInt(parts[0], 10);
            const month = parseInt(parts[1], 10);
            const year = parseInt(parts[2], 10);

            // Vérifier que les valeurs sont valides
            if (!isNaN(day) && !isNaN(month) && !isNaN(year) &&
                day >= 1 && day <= 31 &&
                month >= 1 && month <= 12 &&
                year >= 2000 && year <= 2100) {
                // Construire la date au format YYYY-MM-DD pour éviter toute ambiguïté
                const dateStrISO = `${year}-${month.toString().padStart(2, "0")}-${day.toString().padStart(2, "0")}`;
                const parsed = dayjs(dateStrISO, "YYYY-MM-DD", true);
                if (parsed.isValid()) {
                    log.info("date parsed dd/mm/yyyy", { dateStr, result: parsed.format("YYYY-MM-DD") });
                    return parsed;
                }
            }
        }

        // Fallback: essayer avec dayjs directement
        const parsed = dayjs(normalized, "DD/MM/YYYY", true);
        if (parsed.isValid()) {
            log.info("date parsed fallback", { dateStr, result: parsed.format("YYYY-MM-DD") });
            return parsed;
        }

        log.warn("invalid date", { dateStr });
        return null;
    };

    const handleSave = () => {
        if (mode === "date") {
            // Utiliser dateInput directement
            const parsed = parseDateInput(dateInput);
            if (parsed) {
                const date = parsed.toDate();
                if (minimumDate && date < minimumDate) {
                    onChange(minimumDate);
                } else if (maximumDate && date > maximumDate) {
                    onChange(maximumDate);
                } else {
                    onChange(date);
                }
            }
        } else if (mode === "time") {
            // Utiliser timeInput directement
            if (timeInput && timeInput.trim() && timeInput.includes(":")) {
                const [hours, minutes] = timeInput.split(":").map(Number);
                if (!isNaN(hours) && !isNaN(minutes)) {
                    // Si value existe, utiliser cette date comme base (même si c'est à minuit)
                    // Sinon utiliser aujourd'hui à minuit
                    const baseDate = value ? dayjs(value).startOf("day") : dayjs().startOf("day");
                    const newDate = baseDate.hour(hours).minute(minutes).toDate();
                    onChange(newDate);
                }
            } else if (!timeInput || timeInput.trim() === "") {
                // Si l'heure est supprimée, retourner une date à minuit (pour indiquer "heure à définir")
                const baseDate = value ? dayjs(value).startOf("day") : dayjs().startOf("day");
                onChange(baseDate.toDate());
            }
        } else {
            // datetime - utiliser dateInput et timeInput directement
            const parsed = parseDateInput(dateInput);
            log.info("handleSave datetime", { dateInput, parsed: parsed?.format("YYYY-MM-DD"), timeInput, selectedHour, selectedMinute });

            // Utiliser timeInput en priorité, sinon utiliser selectedHour/selectedMinute
            let finalHours = selectedHour;
            let finalMinutes = selectedMinute;

            if (timeInput && timeInput.trim() && timeInput.includes(":")) {
                const [hours, minutes] = timeInput.split(":").map(Number);
                log.info("parsing timeInput", { hours, minutes });
                if (!isNaN(hours) && !isNaN(minutes)) {
                    finalHours = hours;
                    finalMinutes = minutes;
                }
            }

            if (parsed && !isNaN(finalHours) && !isNaN(finalMinutes)) {
                const date = parsed.hour(finalHours).minute(finalMinutes).toDate();
                log.info("final datetime", { hours: finalHours, minutes: finalMinutes, date: dayjs(date).format("YYYY-MM-DD HH:mm") });
                if (minimumDate && date < minimumDate) {
                    onChange(minimumDate);
                } else if (maximumDate && date > maximumDate) {
                    onChange(maximumDate);
                } else {
                    onChange(date);
                }
            }
        }
        setShowModal(false);
    };

    // Scroll vers la valeur sélectionnée au montage
    useEffect(() => {
        if (showModal) {
            setTimeout(() => {
                if (dayScrollRef.current) {
                    dayScrollRef.current.scrollTo({
                        y: (selectedDay - 1) * ITEM_HEIGHT,
                        animated: false,
                    });
                }
                if (monthScrollRef.current) {
                    monthScrollRef.current.scrollTo({
                        y: (selectedMonth - 1) * ITEM_HEIGHT,
                        animated: false,
                    });
                }
                if (yearScrollRef.current) {
                    const yearIndex = years.indexOf(selectedYear);
                    if (yearIndex >= 0) {
                        yearScrollRef.current.scrollTo({
                            y: yearIndex * ITEM_HEIGHT,
                            animated: false,
                        });
                    }
                }
                if (hourScrollRef.current) {
                    hourScrollRef.current.scrollTo({
                        y: selectedHour * ITEM_HEIGHT,
                        animated: false,
                    });
                }
                if (minuteScrollRef.current) {
                    minuteScrollRef.current.scrollTo({
                        y: selectedMinute * ITEM_HEIGHT,
                        animated: false,
                    });
                }
            }, 100);
        }
    }, [showModal]);

    // Composant de roue iOS
    const WheelPicker = ({
        items,
        selectedValue,
        onValueChange,
        scrollRef,
        formatItem = (item: number) => item.toString().padStart(2, "0"),
    }: {
        items: number[];
        selectedValue: number;
        onValueChange: (value: number) => void;
        scrollRef: React.RefObject<ScrollView | null>;
        formatItem?: (item: number) => string;
    }) => {
        return (
            <View style={styles.wheelContainer}>
                <ScrollView
                    ref={scrollRef}
                    style={styles.wheelScroll}
                    contentContainerStyle={{
                        paddingTop: WHEEL_HEIGHT / 2 - ITEM_HEIGHT / 2,
                        paddingBottom: WHEEL_HEIGHT / 2 - ITEM_HEIGHT / 2,
                    }}
                    showsVerticalScrollIndicator={false}
                    snapToInterval={ITEM_HEIGHT}
                    decelerationRate="fast"
                    onMomentumScrollEnd={(e) => {
                        const y = e.nativeEvent.contentOffset.y;
                        const index = Math.round(y / ITEM_HEIGHT);
                        const clampedIndex = Math.max(0, Math.min(index, items.length - 1));
                        onValueChange(items[clampedIndex]);
                    }}
                >
                    {items.map((item, index) => {
                        const isSelected = selectedValue === item;
                        return (
                            <View key={index} style={styles.wheelItem}>
                                <Text
                                    style={[
                                        styles.wheelItemText,
                                        isSelected && styles.wheelItemTextSelected,
                                        !isSelected && styles.wheelItemTextUnselected,
                                    ]}
                                >
                                    {formatItem(item)}
                                </Text>
                            </View>
                        );
                    })}
                </ScrollView>
                <View style={styles.wheelSelector} />
            </View>
        );
    };

    return (
        <View style={styles.container}>
            <Text style={styles.label}>{label}</Text>
            <TouchableOpacity style={styles.button} onPress={handlePress}>
                <Ionicons
                    name={mode === "time" ? "time-outline" : "calendar-outline"}
                    size={18}
                    color={palette.accent}
                />
                <Text style={[styles.value, !value && styles.valuePlaceholder]}>
                    {displayValue}
                </Text>
                <Ionicons name="chevron-down" size={16} color={palette.textSecondary} />
            </TouchableOpacity>

            <Modal
                visible={showModal}
                transparent
                animationType="fade"
                onRequestClose={() => setShowModal(false)}
            >
                <View style={styles.modalOverlay}>
                    <View style={styles.modalContent}>
                        <Text style={styles.modalTitle}>
                            {mode === "date" ? "Sélectionner une date" : mode === "time" ? "Sélectionner une heure" : "Sélectionner date et heure"}
                        </Text>

                        <View style={styles.pickerContainer}>
                            {mode !== "time" && (
                                <View style={styles.inputGroup}>
                                    <Text style={styles.inputLabel}>Date (DD.MM.YYYY)</Text>
                                    <TextInput
                                        style={styles.input}
                                        value={dateInput}
                                        onChangeText={handleDateChange}
                                        placeholder="15.01.2025"
                                        keyboardType="numeric"
                                        maxLength={10}
                                    />
                                </View>
                            )}

                            {mode !== "date" && (
                                <View style={styles.inputGroup}>
                                    <Text style={styles.inputLabel}>Heure (HH:mm)</Text>
                                    <View style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
                                        <TextInput
                                            style={[styles.input, { flex: 1 }]}
                                            value={timeInput}
                                            onChangeText={handleTimeChange}
                                            placeholder="14:30"
                                            keyboardType="numeric"
                                            maxLength={5}
                                        />
                                        <TouchableOpacity
                                            style={styles.heureADefinirBtn}
                                            onPress={() => {
                                                setSelectedHour(0);
                                                setSelectedMinute(0);
                                                setTimeInput("00:00");
                                            }}
                                        >
                                            <Text style={styles.heureADefinirBtnText}>⏱️ À définir</Text>
                                        </TouchableOpacity>
                                    </View>
                                </View>
                            )}

                            <View style={styles.modalActions}>
                                <TouchableOpacity
                                    style={styles.modalCancel}
                                    onPress={() => setShowModal(false)}
                                >
                                    <Text style={styles.modalCancelText}>Annuler</Text>
                                </TouchableOpacity>
                                <TouchableOpacity style={styles.modalSave} onPress={handleSave}>
                                    <Text style={styles.modalSaveText}>Enregistrer</Text>
                                </TouchableOpacity>
                            </View>
                        </View>
                    </View>
                </View>
            </Modal>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        marginBottom: 0,
    },
    label: {
        color: palette.text,
        fontSize: 14,
        fontWeight: "600",
        marginBottom: 4,
    },
    button: {
        flexDirection: "row",
        alignItems: "center",
        backgroundColor: palette.background,
        borderRadius: 14,
        borderWidth: 1.5,
        borderColor: palette.border,
        paddingHorizontal: 14,
        paddingVertical: 12,
        gap: 10,
        marginBottom: 0,
    },
    value: {
        flex: 1,
        color: palette.text,
        fontSize: 15,
    },
    valuePlaceholder: {
        color: palette.textSecondary,
    },
    modalOverlay: {
        flex: 1,
        backgroundColor: "rgba(21,54,43,0.75)",
        justifyContent: "center",
        alignItems: "center",
        padding: 24,
    },
    modalContent: {
        width: "100%",
        maxWidth: 400,
        backgroundColor: palette.background,
        borderRadius: 20,
        padding: 16,
        maxHeight: "80%",
    },
    modalTitle: {
        color: palette.text,
        fontSize: 15,
        fontWeight: "700",
        marginBottom: 8,
    },
    pickerContainer: {
        marginVertical: 12,
    },
    datePickerRow: {
        flexDirection: "row",
        justifyContent: "center",
        alignItems: "center",
        height: 120,
        marginBottom: 16,
    },
    timePickerRow: {
        flexDirection: "row",
        justifyContent: "center",
        alignItems: "center",
        height: 120,
    },
    timeSeparator: {
        fontSize: 32,
        fontWeight: "600",
        color: palette.text,
        marginHorizontal: 8,
    },
    wheelContainer: {
        height: 105,
        width: 65,
        position: "relative",
        marginHorizontal: 3,
    },
    wheelScroll: {
        flex: 1,
    },
    wheelItem: {
        height: 35,
        justifyContent: "center",
        alignItems: "center",
    },
    wheelItemText: {
        fontSize: 18,
        color: palette.textSecondary,
        textAlign: "center",
    },
    wheelItemTextSelected: {
        fontSize: 20,
        fontWeight: "600",
        color: palette.text,
    },
    wheelItemTextUnselected: {
        opacity: 0.3,
    },
    wheelSelector: {
        position: "absolute",
        top: 35,
        left: 0,
        right: 0,
        height: 35,
        borderTopWidth: 1,
        borderBottomWidth: 1,
        borderColor: palette.border,
        backgroundColor: "rgba(10,127,89,0.05)",
        pointerEvents: "none",
    },
    inputGroup: {
        marginBottom: 16,
    },
    inputLabel: {
        color: palette.text,
        fontSize: 14,
        fontWeight: "600",
        marginBottom: 8,
    },
    input: {
        backgroundColor: palette.background,
        borderRadius: 14,
        borderWidth: 1.5,
        borderColor: palette.border,
        paddingHorizontal: 14,
        paddingVertical: 12,
        color: palette.text,
        fontSize: 15,
    },
    heureADefinirBtn: {
        backgroundColor: palette.buttonBg,
        paddingHorizontal: 12,
        paddingVertical: 10,
        borderRadius: 10,
        borderWidth: 1,
        borderColor: palette.border,
    },
    heureADefinirBtnText: {
        color: palette.accent,
        fontSize: 13,
        fontWeight: "600",
    },
    modalActions: {
        flexDirection: "row",
        justifyContent: "flex-end",
        gap: 12,
        marginTop: 8,
    },
    modalCancel: {
        paddingHorizontal: 16,
        paddingVertical: 10,
    },
    modalCancelText: {
        color: palette.textSecondary,
        fontSize: 15,
        fontWeight: "600",
    },
    modalSave: {
        backgroundColor: palette.accent,
        paddingHorizontal: 20,
        paddingVertical: 12,
        borderRadius: 14,
    },
    modalSaveText: {
        color: "#FFFFFF",
        fontSize: 15,
        fontWeight: "700",
    },
});

