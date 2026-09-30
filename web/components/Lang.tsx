"use client";

import React, { createContext, useContext, useEffect, useState } from "react";
import fr from "../messages/fr.json";
import en from "../messages/en.json";

export type Lang = "fr" | "en";
type Dict = typeof fr;

const dicts: Record<Lang, Dict> = { fr, en };

const LangCtx = createContext<{ lang: Lang; setLang: (l: Lang) => void; t: Dict }>({
  lang: "fr",
  setLang: () => {},
  t: fr,
});

export function LangProvider({ children }: { children: React.ReactNode }) {
  const [lang, setLangState] = useState<Lang>("fr");

  useEffect(() => {
    const saved = window.localStorage.getItem("sahel-lang");
    if (saved === "fr" || saved === "en") setLangState(saved);
    const theme = window.localStorage.getItem("sahel-theme") || "light";
    document.documentElement.setAttribute("data-theme", theme);
  }, []);

  const setLang = (l: Lang) => {
    setLangState(l);
    window.localStorage.setItem("sahel-lang", l);
  };

  return <LangCtx.Provider value={{ lang, setLang, t: dicts[lang] }}>{children}</LangCtx.Provider>;
}

export function useLang() {
  return useContext(LangCtx);
}

export function toggleTheme() {
  const cur = document.documentElement.getAttribute("data-theme") || "light";
  const next = cur === "light" ? "dark" : "light";
  document.documentElement.setAttribute("data-theme", next);
  window.localStorage.setItem("sahel-theme", next);
  return next;
}
