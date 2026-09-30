import type { Metadata } from "next";
import "./globals.css";
import { LangProvider } from "../components/Lang";
import Nav from "../components/Nav";

export const metadata: Metadata = {
  title: "Sahel Agri Predictor",
  description: "TabPFN-3.5 cropland demo — predict, explain, recommend.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="fr" data-theme="light" suppressHydrationWarning>
      <body>
        <LangProvider>
          <div className="max-w-4xl mx-auto px-4 pb-16">
            <Nav />
            <main>{children}</main>
          </div>
        </LangProvider>
      </body>
    </html>
  );
}
