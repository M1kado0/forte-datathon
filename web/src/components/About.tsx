// Team names and responsibilities are from the repository's AGENTS.md.
const team = [
  { name: "Тамерлан", role: "Fullstack", detail: "Интерфейс, API и очередь задач", github: "https://github.com/a100miles", handle: "a100miles" },
  { name: "Алдияр С.", role: "ML / Computer Vision", detail: "Детекторы, обучение и анализ видео", github: "https://github.com/M1kado0", handle: "M1kado0" },
  { name: "Ескендир", role: "Инфраструктура", detail: "Эвристики, видеопоток и развёртывание", github: "https://github.com/ytyourstruly", handle: "ytyourstruly" },
  { name: "Алдияр М.", role: "Данные и правила", detail: "Разметка, нормативные источники и документация", github: "https://github.com/trulyaldi", handle: "trulyaldi" },
];

export function About({ paused }: { paused: boolean }) {
  return <section id="about" className="about-section" data-paused={paused} aria-labelledby="about-title" tabIndex={-1}>
    <span className="about-particles" aria-hidden="true">{Array.from({ length: 12 }, (_, index) => <i className="about-particle" key={index} />)}</span>
    <div className="section-shell">
    <div className="about-intro">
      <h2 id="about-title" className="display-title about-title" data-reveal>За кадром —<br /><em>команда.</em></h2>
      <div className="about-copy"><p data-reveal data-reveal-delay="80">Forte Datathon ’26 — проект локальной видеоаналитики для проверки кассовой дисциплины.</p><p data-reveal data-reveal-delay="160">Наша задача — связать наблюдаемое действие, момент на видео и нормативный источник. Помочь проверяющему разобраться в ситуации, сохранив решение за человеком.</p><p className="about-note" data-reveal data-reveal-delay="240">Без распознавания лиц и анализа звука. Загруженные видео удаляются после обработки.</p></div>
    </div>
    <div className="team-list">{team.map(person => <div className="team-row" key={person.name}>
      <h3 data-reveal><a href={person.github} target="_blank" rel="noopener noreferrer" aria-label={`${person.name} — GitHub: ${person.handle}`}><span>{person.name}</span><svg className="github-mark" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2a10 10 0 0 0-3.16 19.49c.5.09.68-.22.68-.48v-1.87c-2.78.6-3.37-1.18-3.37-1.18-.45-1.16-1.11-1.47-1.11-1.47-.91-.62.07-.61.07-.61 1 .07 1.53 1.03 1.53 1.03.9 1.53 2.35 1.09 2.92.83.09-.65.35-1.09.64-1.34-2.22-.25-4.56-1.11-4.56-4.94 0-1.09.39-1.98 1.03-2.68-.1-.25-.45-1.27.1-2.64 0 0 .84-.27 2.75 1.02A9.6 9.6 0 0 1 12 6.82a9.6 9.6 0 0 1 2.5.34c1.91-1.29 2.75-1.02 2.75-1.02.55 1.37.2 2.39.1 2.64.64.7 1.03 1.59 1.03 2.68 0 3.84-2.34 4.68-4.57 4.93.36.31.68.92.68 1.86V21c0 .27.18.58.69.48A10 10 0 0 0 12 2Z" /></svg></a></h3>
      <span className="team-role" data-reveal data-reveal-delay="80">{person.role}</span><p data-reveal data-reveal-delay="160">{person.detail}</p>
    </div>)}</div>
    <div className="project-principle"><span data-reveal>Наблюдение → контекст → проверка</span><p data-reveal data-reveal-delay="80">Внимание к деталям.<br />Ответственность за выводы.</p></div>
    </div>
  </section>;
}
