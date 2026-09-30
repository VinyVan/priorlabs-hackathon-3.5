"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, useEffect } from "react";
import { useLang, toggleTheme } from "./Lang";

export default function Nav() {
  const { lang, setLang, t } = useLang();
  const pathname = usePathname();
  const [theme, setTheme] = useState("light");

  useEffect(() => {
    setTheme(document.documentElement.getAttribute("data-theme") || "light");
  }, []);

  const link = (href: string, label: string) => (
    <Link
      key={href}
      href={href}
      className={`px-3 py-1.5 rounded-lg text-sm font-medium ${
        pathname === href ? "bg-green-700 text-white" : "hover:bg-green-100 dark:hover:bg-green-900"
      }`}
    >
      {label}
    </Link>
  );

  return (
    <header className="flex flex-wrap items-center gap-2 justify-between py-4">
      <Link href="/" className="font-bold text-lg">
        🌾 Sahel Agri
      </Link>
      <nav className="flex items-center gap-1">
        {link("/", t.nav.home)}
        {link("/predict", t.nav.predict)}
        {link("/compare", t.nav.compare)}
        {link("/about", t.nav.about)}
      </nav>
      <div className="flex items-center gap-2">
        <button
          onClick={() => setLang(lang === "fr" ? "en" : "fr")}
          className="px-2 py-1 text-sm border rounded-lg"
          aria-label="language"
        >
          {lang === "fr" ? "FR | EN" : "EN | FR"}
        </button>
        <button
          onClick={() => setTheme(toggleTheme())}
          className="px-2 py-1 text-sm border rounded-lg"
          aria-label="theme"
        >
          {theme === "light" ? "🌙" : "☀️"}
        </button>
      </div>
    </header>
  );
}
