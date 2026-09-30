"use client";

import { useLang } from "../../components/Lang";

export default function AboutPage() {
  const { t } = useLang();
  return (
    <div className="flex flex-col gap-5 pt-6">
      <h1 className="text-3xl font-extrabold">{t.about.title}</h1>
      <section className="card p-5">
        <h2 className="font-bold mb-2">{t.about.methodTitle}</h2>
        <p className="text-sm leading-6">{t.about.method}</p>
      </section>
      <section className="card p-5">
        <h2 className="font-bold mb-2">{t.about.dataTitle}</h2>
        <p className="text-sm leading-6">{t.about.data}</p>
      </section>
      <section className="card p-5">
        <h2 className="font-bold mb-2">{t.about.linksTitle}</h2>
        <ul className="text-sm list-disc ml-5 flex flex-col gap-1">
          <li>
            <a className="underline" href="https://github.com/priorlabs">
              {t.about.repo}
            </a>
          </li>
          <li>
            <a className="underline" href="https://priorlabs.ai/hackathon">
              {t.about.hackathon}
            </a>
          </li>
        </ul>
      </section>
    </div>
  );
}
