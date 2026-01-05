import { useState, useCallback, useEffect, useRef } from "react";
import { AddressSuggestion } from "@/types/enterpriseDispatch";
import { searchAddresses } from "@/services/enterpriseDispatch";

export const useAddressSearch = (debounceMs: number = 300) => {
    const [query, setQuery] = useState("");
    const [suggestions, setSuggestions] = useState<AddressSuggestion[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    const performSearch = useCallback(
        async (searchQuery: string) => {
            if (!searchQuery || searchQuery.trim().length < 3) {
                setSuggestions([]);
                setLoading(false);
                return;
            }

            setLoading(true);
            setError(null);

            try {
                const results = await searchAddresses(searchQuery.trim());
                setSuggestions(results);
            } catch (err: any) {
                const message =
                    err?.response?.data?.error ??
                    err?.message ??
                    "Erreur lors de la recherche d'adresses.";
                setError(message);
                setSuggestions([]);
            } finally {
                setLoading(false);
            }
        },
        []
    );

    const search = useCallback(
        (searchQuery: string) => {
            setQuery(searchQuery);

            if (debounceTimerRef.current) {
                clearTimeout(debounceTimerRef.current);
            }

            debounceTimerRef.current = setTimeout(() => {
                performSearch(searchQuery);
            }, debounceMs);
        },
        [performSearch, debounceMs]
    );

    const clear = useCallback(() => {
        setQuery("");
        setSuggestions([]);
        setError(null);
        if (debounceTimerRef.current) {
            clearTimeout(debounceTimerRef.current);
        }
    }, []);

    useEffect(() => {
        return () => {
            if (debounceTimerRef.current) {
                clearTimeout(debounceTimerRef.current);
            }
        };
    }, []);

    return {
        query,
        suggestions,
        loading,
        error,
        search,
        clear,
    };
};

