# Forte Datathon '26 — Learning Materials v2 ("Tonight edition")

> For master prompt **v4**. Written 16 Oct 2026, the night before the event (check-in 09:00, case at 11:00).
> Budget: **≤ 75 min of learning per person tonight**, on top of building the templates.
> ✅ = page opened and checked on 16 Oct 2026 · 🔎 = from search results, not opened · ⚠️ = outdated, uncertain or conflicts with v4.
> Lang: EN / RU. Every tonight item is ≤ 20 min or comes with a "read only section X" note.

**Changelog**
- **v2 (16 Oct, Tonight edition):** pruned the v1 list against master prompt v4 (one common case, 90-s single-presenter pitch, Forte-branded React front-end, tunnel deploy). Long books, workshops, podcasts, 3-/7-min pitch material, process theory and practice drills were dropped. Old items still useful during the event moved to the KEEP-REFERENCE appendix. Added: front-end design with agents (Impeccable, shadcn/ui, Tailwind v4, Playwright MCP), Forte brand assets, 90-second pitch, tunnel deploy, citable niche techniques, a conditional HR/people-analytics pack, and the workshop speakers' track record.
- **v1 (16 Oct, earlier):** full 8-group library for master prompt v2/v3 (A agent harness … H banking/Kazakhstan), with 8-hour role paths, a team top 10 and 5 drills.

**Contents:** 1 Tonight lists · 2 Team item · 3 ADD tables A–G · 4 KEEP-REFERENCE appendix · 5 DROPPED list · 6 Gaps

---

## 1. TONIGHT — per role (≤ 75 min each, in this order)

Read in the gaps while builds run. Section notes keep each item short.

### P1 Engineer — app, design skills, deploy (≈ 73 min)
1. **Impeccable README** — Install, Commands, Design hook, Telemetry — [github](https://github.com/pbakaus/impeccable) — 10 min
2. **Impeccable "Install & first steps"** — skim; note Node ≥ 22.18 — [impeccable.style](https://impeccable.style/tutorials/getting-started) — 8 min
3. **Impeccable "Work with a design system"** — how to make it respect Forte (no "Brand Commitments" section exists) — [docs](https://impeccable.style/docs/design-system) — 6 min
4. **shadcn/ui Vite install** — [docs](https://ui.shadcn.com/docs/installation/vite) — 8 min
5. **shadcn theming** — map Forte tokens (`--primary`, `--chart-1…5`, `@theme inline`) — [docs](https://ui.shadcn.com/docs/theming) — 8 min
6. **shadcn skill + MCP** — incl. the manual Codex `config.toml` block — [skills](https://ui.shadcn.com/docs/skills) · [mcp](https://ui.shadcn.com/docs/mcp) — 6 min
7. **Playwright MCP** — install lines, `browser_take_screenshot`, `--viewport-size` — [README](https://github.com/microsoft/playwright-mcp) · [TIL](https://til.simonwillison.net/claude-code/playwright-mcp-claude-code) — 6 min
8. **Cloudflare quick tunnel + ngrok quickstart** — [Cloudflare](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/) · [ngrok](https://ngrok.com/docs/share-localhost/quickstart.md) — 6 min
9. **FastAPI `StaticFiles(html=True)`** — one-port serving of `web/dist` — [reference](https://fastapi.tiangolo.com/reference/staticfiles/) — 3 min
10. **mattpocock `to-tickets` + `grill-with-docs` SKILL.md** (skim `handoff`) — [repo](https://github.com/mattpocock/skills) — 10 min
11. **gh ≥ 2.94 changelog** — `--parent`, `--blocked-by` — [github.blog](https://github.blog/changelog/2026-06-10-manage-sub-issues-types-and-dependencies-from-github-cli/) — 2 min

*If time:* Tailwind `@theme` overview (5) · shadcn charts page (5) · Bakaus interview takeaways (8).

### P2 ML — baseline that returns `{result, reasons[], impact_kzt}` (≈ 73 min)
1. **scikit-learn common pitfalls — §10.2 only** (leakage) — [sklearn](https://scikit-learn.org/1.6/common_pitfalls.html) — 8 min
2. **Probability calibration — §1.16.1–1.16.3** — [sklearn](https://scikit-learn.org/1.5/modules/calibration.html) — 10 min
3. **Cost-sensitive threshold — German-credit half** (`TunedThresholdClassifierCV` + gain matrix) — [example](https://scikit-learn.org/1.7/auto_examples/model_selection/plot_cost_sensitive_learning.html) — 12 min
4. **SHAP intro — waterfall + beeswarm sections** — [SHAP docs](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html) — 15 min
5. **SHAP → grouped plain-language reason codes** — Zest white paper, reason-code + validation pages — [PDF](https://www.zest.ai/wp-content/uploads/2025/08/Modelexplainability_reexplained.pdf) — 10 min
6. **optbinning scorecard — "scorecard table" part** (credit-case challenger) — [tutorial](https://gnpalencia.org/optbinning/tutorials/tutorial_scorecard_binary_target.html) — 10 min
7. **Give Me Some Credit via empulse** — practice data with cost arrays — [docs](https://empulse.readthedocs.io/en/stable/guide/datasets/give_me_some_credit.html) — 3 min
8. **Section 3E table** (niche techniques) — pick at most one to keep in your pocket; monotonic constraints is the cheapest — 5 min

*If time:* MAPIE quick start (10) · Venn-Abers README (5) · TabPFN PyPI page — only if you'll actually test it tonight (5).

### P3 Analyst — EDA skill, data pack, robustness gate (≈ 74 min)
1. **skrub Getting Started** — `TableReport` for `/eda` — [tutorial](https://skrub-data.org/stable/auto_tutorials/0000_getting_started.html) — 12 min
2. **fg-data-profiling migration section** (ydata was renamed) — [README](https://github.com/Data-Centric-AI-Community/ydata-profiling) — 3 min
3. **`scipy.stats.bootstrap` — "difference in means" example** — [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) — 10 min
4. **`multipletests(method="fdr_bh")`** — [statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html) — 3 min
5. **Simpson's paradox** (gate item 3) — [Navarro](https://stats.libretexts.org/Bookshelves/Applied_Statistics/Learning_Statistics_with_R_-_A_tutorial_for_Psychology_Students_and_other_Beginners_(Navarro)/01%3A_Why_Do_We_Learn_Statistics/1.02%3A_The_Cautionary_Tale_of_Simpsons_Paradox) — 8 min
6. **NBK + stat.gov.kz pages for the data pack** — [NBK](https://www.nationalbank.kz/en) · [FX](https://nationalbank.kz/en/exchangerates/ezhednevnye-oficialnye-rynochnye-kursy-valyut) · [wages](https://stat.gov.kz/en/industries/labor-and-income/stat-wags/dynamic-tables/) — 10 min (download by hand)
7. **Adversarial validation + fairlearn worst-group** — rows in section 3E; open the [fairlearn assessment guide](https://fairlearn.org/main/user_guide/assessment) — 8 min
8. **NN/g dashboards** — which chart types to use/avoid — [nngroup](https://www.nngroup.com/articles/dashboards-preattentive/) — 10 min
9. **lifelines Quickstart** (HR contingency: time-to-attrition) — [docs](https://lifelines.readthedocs.io/en/stable/Quickstart.html) — 10 min

*If time:* attrition + SHAP tutorial (7) · skrub AggJoiner example (reference).

### P4 Pitch — deck, brand, jury, delivery (≈ 72 min)
1. **Forte brand book** — colours, gradients, Open Sans, logo rules — [forte.kz/logo](https://forte.kz/logo) — 8 min
2. **Open both PPTX templates** and note their layouts in `deck/README.md` — [Forte](https://forte.kz/docs/forte-presentation.pptx) · [Business](https://forte.kz/docs/forte-business-presentation.pptx) — 5 min
3. **Forte CDAIO interview** (for `JURY.md`) — [DigitalBusiness.kz](https://digitalbusiness.kz/2026-07-02/ainezamenyaet-lyudey-nomenyaet-protsessi-kak-forte-stroit-bank-novogo-tipa/) RU — 10 min
4. **inDrive judging criteria at Decentrathon 5.0** (the speakers' last jury) — [DigitalBusiness.kz](https://digitalbusiness.kz/2026-03-26/kak-s-pomoshchyu-ai-nahodit-liderov-i-raspredelyat-resursi-keysi-indrive-dlya-uchastnikov-decentrathon-5-0/) RU — 6 min
5. **Kazakhstan AI Law (EY)** — the fairness / "human decides" answer — [ey.com](https://www.ey.com/en_kz/technical/tax-alerts/2025/12/law-on-artificial-intelligence-kazakhstan) — 5 min
6. **Michael Seibel (YC) on clarity** — [SaaStr](https://www.saastr.com/the-y-combinator-guide-to-perfectly-pitching-your-seed-stage-startup-with-michael-seibel) — 8 min
7. **think-cell: takeaway titles + one accent colour** — [thinkcell.com](https://www.thinkcell.com/en/blog/how-to-tell-a-better-data-story-one-slide-at-a-time) — read the headline and colour sections — 8 min
8. **Slide master / layouts** — build the 6 v4 slides as layouts — [Microsoft](https://support.microsoft.com/en-US/PowerPoint/edit-and-re-apply-a-slide-layout) — 5 min
9. **Ubuntu built-in screencast** — test a 90-s take now — [Rocky docs](https://docs.rockylinux.org/desktop/gnome/screenshot/) — 3 min
10. **Metrics → money sentence pattern** — [r-statistics.co](https://r-statistics.co/From-Metrics-to-Money.html) — 8 min
11. **Q&A under pressure** — [Launchrock](https://www.launchrock.com/library/expert-advice/pitch-q-and-a) — 6 min

*If time:* "AI slop" pattern list for design QA (prg.sh, skim 5) · Xero playbook (5) · Halyk rubric (3).

---

## 2. Team item — all four, together (6 min)

**[How to win a hackathon: notes from the judging table — JetBrains (Jun 2026)](https://blog.jetbrains.com/ai/2026/06/how-to-win-a-hackathon-notes-from-the-judging-table/)** ✅ — 6 min. Judges in their own words: something working on screen within ~90 s, one flow, the judge in the user's shoes, pre-filled data and no stall points, candour about gaps. It is v4's "what wins here" from an outside source — read it out loud together, then agree on the one flow you'll demo.

---


### Why these v1 items stayed in tonight's lists (KEEP-TONIGHT, with the reason)

| Item | Link | Reason (v4 dependency) | Who |
|---|---|---|---|
| mattpocock/skills — read `to-tickets`, `grill-with-docs`, `handoff` SKILL.md only | [github.com/mattpocock/skills](https://github.com/mattpocock/skills) ✅ | Vendored tonight; used at 11:30–12:30 tomorrow | P1 |
| gh ≥ 2.94 changelog (sub-issues, `--blocked-by`) | [github.blog changelog](https://github.blog/changelog/2026-06-10-manage-sub-issues-types-and-dependencies-from-github-cli/) ✅ | Everyone upgrades gh tonight | P1 (all) |
| skrub Getting Started | [skrub tutorial](https://skrub-data.org/stable/auto_tutorials/0000_getting_started.html) 🔎 | `/eda` skill test tonight (TableReport) | P3 |
| fg-data-profiling (ex-ydata) README — migration section | [GitHub](https://github.com/Data-Centric-AI-Community/ydata-profiling) ✅ | Package renamed; `/eda` must import `data_profiling` | P3 |
| `scipy.stats.bootstrap` — examples "difference in means" | [SciPy docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) ✅ | Robustness gate item 4 | P3 |
| `multipletests(method="fdr_bh")` | [statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html) ✅ | Robustness gate item 4 (FDR across hypotheses) | P3 |
| Simpson's paradox (Navarro) | [LibreTexts](https://stats.libretexts.org/Bookshelves/Applied_Statistics/Learning_Statistics_with_R_-_A_tutorial_for_Psychology_Students_and_other_Beginners_(Navarro)/01%3A_Why_Do_We_Learn_Statistics/1.02%3A_The_Cautionary_Tale_of_Simpsons_Paradox) ✅ | Robustness gate item 3 (confounding by mix) | P3 (team) |
| NBK homepage, FX and inflation pages | [nationalbank.kz](https://www.nationalbank.kz/en) ✅ · [FX](https://nationalbank.kz/en/exchangerates/ezhednevnye-oficialnye-rynochnye-kursy-valyut) 🔎 · [inflation](https://nationalbank.kz/en/news/inflyacionnye-tendencii) 🔎 | External data pack tonight (base rate 16.25%; next decision 23 Oct) | P3 |
| stat.gov.kz wages dynamic tables | [stat.gov.kz](https://stat.gov.kz/en/industries/labor-and-income/stat-wags/dynamic-tables/) ✅ | Regional wages for the data pack (download by hand) | P3 |
| scikit-learn common pitfalls — section 10.2 only | [sklearn](https://scikit-learn.org/1.6/common_pitfalls.html) ✅ | Leakage tests in tonight's baseline pipeline | P2 |
| Probability calibration — 1.16.1–1.16.3 only | [sklearn](https://scikit-learn.org/1.5/modules/calibration.html) ✅ | Baseline is "calibrated" by tonight's definition of done | P2 |
| Cost-sensitive threshold — German-credit half only | [sklearn example](https://scikit-learn.org/1.7/auto_examples/model_selection/plot_cost_sensitive_learning.html) ✅ | "Threshold on cost" → `impact_kzt` field | P2 |
| SHAP intro — waterfall + beeswarm sections | [SHAP docs](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html) ✅ | `reasons[]` in the `/predict` contract | P2 |
| optbinning scorecard — "scorecard table" part | [optbinning tutorial](https://gnpalencia.org/optbinning/tutorials/tutorial_scorecard_binary_target.html) ✅ | Interpretable challenger if credit | P2 |
| Give Me Some Credit via empulse | [empulse docs](https://empulse.readthedocs.io/en/stable/guide/datasets/give_me_some_credit.html) ✅ | Tonight's practice dataset (with cost arrays) | P2 |
| Forte CDAIO interview (Smailov, Jul 2026) | [DigitalBusiness.kz](https://digitalbusiness.kz/2026-07-02/ainezamenyaet-lyudey-nomenyaet-protsessi-kak-forte-stroit-bank-novogo-tipa/) ✅ RU | `JURY.md`: what Forte's AI team cares about | P4 |
| Kazakhstan AI Law summary (EY) | [ey.com](https://www.ey.com/en_kz/technical/tax-alerts/2025/12/law-on-artificial-intelligence-kazakhstan) ✅ | Q&A on fairness/personal data; "human decides" line | P4 |
| think-cell: takeaway titles + one accent colour | [thinkcell.com](https://www.thinkcell.com/en/blog/how-to-tell-a-better-data-story-one-slide-at-a-time) ✅ | Deck template chart rules | P4 |


---

## 3. ADD — new material for v4 (tables A–G)

### A. Front-end design with agents (P1, P4 for design QA)

| Concept | Resource | Format | Lang | Time | Why / quality signal | Who |
|---|---|---|---|---|---|---|
| Impeccable: what it is + install | [pbakaus/impeccable README](https://github.com/pbakaus/impeccable) ✅ | ⚡ repo README | EN | 10 min — read "Install", "Commands", "Design hook", "Telemetry" | 81.8k stars, Apache-2.0, current release **v4.6.0 (15 Oct 2026)** per [changelog](https://impeccable.style/changelog) ✅. `npx impeccable install --providers=claude,codex --scope=project`; 59-rule deterministic detector (`npx impeccable detect`, no LLM). Opt out of telemetry: `npx impeccable telemetry off` or `DO_NOT_TRACK=1` | P1 |
| Impeccable: first run | [Install & first steps](https://impeccable.style/tutorials/getting-started) ✅ | 🛠 tutorial | EN | 10 min | Official. init → `PRODUCT.md`; `document` → `DESIGN.md`; build with `/impeccable <plain request>`; `critique` → `polish` → `audit` before shipping. ⚠️ Needs **Node ≥ 22.18** (check all 4 laptops) | P1 |
| Impeccable: keep our brand (replaces "Brand Commitments") | [Work with a design system](https://impeccable.style/docs/design-system) ✅ + [init](https://impeccable.style/docs/init) ✅ | ⚡ docs | EN | 8 min | ⚠️ **There is no "Brand Commitments" section** in current docs. The documented way: put Forte tokens in CSS first → `/impeccable document` (writes `DESIGN.md` + `.impeccable/design.json`) → state "Open Sans + Forte palette + never recolor logo" as constraints during `init` (→ `PRODUCT.md`) → name a reference screen when building → commit both files | P1, P4 |
| Impeccable: command reference | [Docs index](https://impeccable.style/docs) ✅ → [critique](https://impeccable.style/docs/critique) · [audit](https://impeccable.style/docs/audit) · [polish](https://impeccable.style/docs/polish) · [distill](https://impeccable.style/docs/distill) · [harden](https://impeccable.style/docs/harden) · [document](https://impeccable.style/docs/document) · [hooks](https://impeccable.style/docs/hooks) 🔎 | 📘 reference | EN | lookup | Official. ⚠️ **`craft` is not in the docs' command list** (README still lists it): if `/impeccable craft` doesn't exist tonight, use `/impeccable <request>` or `shape`. Codex uses `$impeccable` | P1 |
| Impeccable: demo | [Neo Mirai before/after case study](https://impeccable.style/cases/neo-mirai) 🔎 · [Review with a visual overlay](https://impeccable.style/tutorials/critique-with-overlay) 🔎 | ⚡ case study | EN | 5 min | Official demos. No official walkthrough video found (README points to @pbakaus on X for clips) | P1, P4 |
| Paul Bakaus interview (Latent Space) | [Skill engineering and the case against one-shot AI design](https://www.latent.space/p/skill-engineering-design) ✅ | ⚡ article (interview write-up) | EN | 8 min | Jul 2026. **Takeaways:** steer with vocabulary ("bolder", "quieter", "denser") instead of one-shot redesigns; "bold" = hierarchy, scale, typography — not gradients; AI does the first ~80%, humans own the last 20% of taste; no "auto" mode by design; Live mode for in-place alternatives | P1, P4 |
| shadcn/ui + Vite | [Installation — Vite](https://ui.shadcn.com/docs/installation/vite) ✅ | 🛠 docs | EN | 10 min | Official. Fastest: `pnpm dlx shadcn@latest init -t vite`. Manual path = Tailwind v4 (`@tailwindcss/vite`, `@import "tailwindcss";`) + `@/*` alias | P1 |
| Map Forte tokens into shadcn | [Theming](https://ui.shadcn.com/docs/theming) ✅ | ⚡ docs | EN | 8 min | Semantic pairs `--primary`/`--primary-foreground`, `--chart-1…5`, `--radius`; expose via `@theme inline` (`--color-primary: var(--primary)`). Plan: `--primary: #A31551`, `--chart-1: #A31551`, `--chart-2..5`: greys (v4 rule "grey + one accent"), custom `--warning` etc. per the doc's 3-step recipe | P1 |
| Charts | [Chart component](https://ui.shadcn.com/docs/components/base/chart) ✅ + [charts gallery](https://ui.shadcn.com/charts) 🔎 | 🛠 docs | EN | 10 min | Recharts **v3**: use `var(--chart-1)` (not `hsl(var(...))`); `ChartContainer` needs a `min-h-*`; `ChartConfig` holds labels/colours | P1 |
| Dashboard block | [Dashboard blocks](https://ui.shadcn.com/blocks/dashboard) ✅ | ⚡ gallery | EN | 3 min | `npx shadcn add dashboard-01` (sidebar + charts + table). ⚠️ Use for parts only — v4 wants one primary action per screen, not a full admin dashboard | P1 |
| shadcn skill | [Skills](https://ui.shadcn.com/docs/skills) ✅ | ⚡ docs | EN | 5 min | Official. `pnpm dlx skills add shadcn/ui` (npx equivalent). Reads `components.json` via `shadcn info --json`; knows Tailwind v4 theming | P1 |
| shadcn MCP | [MCP server](https://ui.shadcn.com/docs/mcp) ✅ | ⚡ docs | EN | 5 min | `pnpm dlx shadcn@latest mcp init --client claude`, then `/mcp`. **Codex: add `[mcp_servers.shadcn] command="npx" args=["shadcn@latest","mcp"]` to `~/.codex/config.toml` by hand** | P1 |
| Tailwind v4 `@theme` (fast path) | [Theme variables](https://tailwindcss.com/docs/theme) ✅ — read "Overview" + "Referencing other variables" | ⚡ docs | EN | 8 min | Official. `@theme` makes utilities (`bg-forte-magenta`); `@theme inline` for tokens that reference other vars; `:root` for non-utility vars | P1 |
| Vercel web-design-guidelines | [vercel-labs/agent-skills](https://github.com/vercel-labs/agent-skills) ✅ | ⚡ README | EN | 3 min | 32.5k stars. "100+ rules": accessibility & focus states, forms & animation, typography & images, performance, navigation/state, dark mode & theming, touch, locale/i18n. Run as "review my UI". ⚠️ Install command in the README is for the whole collection; v4's `--skill web-design-guidelines` flag is from the skills CLI — verify tonight | P1, P4 |
| Playwright MCP screenshots | [microsoft/playwright-mcp](https://github.com/microsoft/playwright-mcp) ✅ + [Simon Willison TIL](https://til.simonwillison.net/claude-code/playwright-mcp-claude-code) ✅ | 🛠 README + TIL | EN | 8 min | 38k stars. `claude mcp add playwright npx @playwright/mcp@latest`; Codex: `codex mcp add playwright npx "@playwright/mcp@latest"`. Tool `browser_take_screenshot`; `--viewport-size "1440x900"` / `"390x844"`; `--headless`. Say "playwright mcp" explicitly the first time | P1 |
| "AI slop" patterns to avoid | [Why your AI keeps building the same purple gradient website](https://prg.sh/ramblings/Why-Your-AI-Keeps-Building-the-Same-Purple-Gradient-Website) ✅ | ⚡ article | EN | 12 min (skim the pattern list + fixes) | Oct 2025, specific: Inter/Roboto defaults, indigo gradients (traces to Tailwind's `bg-indigo-500`), centered hero + 3 icon boxes, rounded-everything, forms without validation/error states. Second opinion: [Noqta, Jun 2026](https://noqta.tn/en/blog/ai-design-slop-overused-ui-patterns-fix-2026) ✅ (oversized radii, glowing gradients, wide tracking, fake "Live" badges; partly marketing) | P1, P4 |
| Dashboard UX | [Dashboards: making charts and graphs easier to understand — NN/g](https://www.nngroup.com/articles/dashboards-preattentive/) ✅ | ⚡ article | EN | 10 min | Nielsen Norman Group (2017, still the reference). Length/position beat area/angle (no pies, gauges, 3D); colour isn't magnitude; one task per dashboard | P1, P4, P3 |
| Fintech dashboard UX | [Financial dashboard design — Lazarev agency](https://www.lazarev.agency/articles/financial-dashboard-design) ✅ | ⚡ article | EN | 7 min | Jun 2026. Map each element to a user decision; progressive disclosure. Promotional slant | P4 |
| Explaining AI decisions in a UI | [People + AI Guidebook: Explainability + Trust (Google PAIR)](https://pair.withgoogle.com/chapter/People%20+%20AI%20Guidebook%20-%20Explainability%20+%20Trust.pdf) ✅ — read "Show model confidence" + "Explain for understanding" | 📘 chapter | EN | 15 min (sections only) | The standard design reference: explain at the moment of action; high-stakes outputs need the "why"; categorical confidence (high/med/low) tied to an action; counterfactual and "what-if" explanations. Also [Types of AI Explanations — NN/g, Jul 2026](https://www.nngroup.com/articles/ai-explanations/) ✅ (local vs global, static vs interactive; 5 min) | P1, P4 |
| SHAP → plain-language reason codes | [Model Explainability — Re-explained (Zest AI white paper)](https://www.zest.ai/wp-content/uploads/2025/08/Modelexplainability_reexplained.pdf) ✅ — read the reason-code + validation pages | 📘 white paper | EN | 15 min (sections) | US credit-underwriting vendor (2025). Rank Shapley contributions → **group related features into one reason** (30/60/90-day delinquencies → "Recent delinquencies") → validate by swapping in approved-applicant values and rescoring. Our `reasons[]` field should follow this. Counterfactual alternative: [optbinning counterfactuals](https://gnpalencia.org/optbinning/tutorials/tutorial_counterfactual_binary_target.html) 🔎 | P2, P1 |

### B. Forte brand (P4, P1)

| Concept | Resource | Format | Lang | Time | Why / quality signal | Who |
|---|---|---|---|---|---|---|
| Brand book | [forte.kz/logo](https://forte.kz/logo) ✅ | ⚡ official page | EN/RU | 8 min | Official. **Forte Magenta #A31551** (master logotype), **Sky Blue #61CFFF**, **Dark Magenta #4A0221**, **Coral Sunshine #FC9385**. Gradients: primary **Dark Magenta → Coral**, secondary **Magenta → Pink** (no hex given for pink). **Open Sans** 300/400/600/700/800 everywhere. Logo: Magenta master, clear space, never recolor; sub-brands keep the Forte logotype + a coloured descriptor (Solo, Premier, Business, Corporate, Leasing, Finance) | P4, P1 |
| Downloads | [Logo ZIP](https://forte.kz/docs/forte-logos.zip) · [Forte Presentation PPTX](https://forte.kz/docs/forte-presentation.pptx) · [Business Presentation PPTX](https://forte.kz/docs/forte-business-presentation.pptx) · [Open Sans ZIP](https://forte.kz/docs/open-sans.zip) — all linked from the ✅ brand page; each URL returned binary data (file exists) | 🛠 files | — | 5 min | ⚠️ I could not inspect the PPTX contents (binary; the research sandbox can't download from forte.kz). **P4: open both PPTX tonight and list their masters/layouts in `deck/README.md`** | P4 |
| Current style in the wild | [forte.kz homepage](https://forte.kz/) ✅ | ⚡ site | RU/KZ | 5 min | Short, plain, warm headings; products: Solo, Premier, Club, Market; the app is called **"Forte"** (also ForteBusiness). Rebrand: Feb 2025, new logo + gradient + slogan «Помогаем людям. Поддерживаем предпринимателей. Создаем будущее.» ([finratings.kz](https://finratings.kz/news/389-fortebank-prezentoval-novyi-logotip-i-slogan/) ✅ RU) | P4, P1 |
| Forte AI Hackathon 2025 | [bank.forte.kz/ru/ai-hakaton](https://bank.forte.kz/ru/ai-hakaton) ✅ RU + [the-tech.kz](https://the-tech.kz/fortebank-zapuskaet-forte-ai-hackathon-s-prizovym-fondom-5-millionov-tenge/) ✅ RU | ⚡ pages | RU | 5 min | Nov–Dec 2025, 5 M ₸, "MVP solving a real banking task" + how it plugs into bank processes. No published visuals, criteria or winners found | P4 |
| "Forte People" | — | — | — | — | ⚠️ **Not found** as a public ForteBank sub-brand: not on the brand page's sub-brand list, not in rebrand coverage, no careers site under that name. Treat the "forte PEOPLE" logo as a hint only (HR/people analytics possible, unconfirmed) | P4 |

### C. 90-second pitch (P4)

| Concept | Resource | Format | Lang | Time | Why / quality signal | Who |
|---|---|---|---|---|---|---|
| What judges want in a short demo | [How to win a hackathon: notes from the judging table — JetBrains](https://blog.jetbrains.com/ai/2026/06/how-to-win-a-hackathon-notes-from-the-judging-table/) ✅ | ⚡ blog | EN | 6 min | Jun 2026, quotes from judges: "show something working within about 90 seconds"; **one flow**; put the judge in the user's shoes; pre-fill data, remove stall points; be candid about gaps | P4 (team) |
| Clarity over precision (YC) | [The Y Combinator guide to pitching — Michael Seibel (SaaStr)](https://www.saastr.com/the-y-combinator-guide-to-perfectly-pitching-your-seed-stage-startup-with-michael-seibel) ✅ | ⚡ article | EN | 8 min | YC partner. "80% accurate, 100% understandable beats the reverse"; two-sentence description with one concrete example; plain slides, big text; cut weak sections entirely | P4 |
| Timed short-pitch flow | [Land your hackathon pitch — Xero Developer](https://devblog.xero.com/land-your-hackathon-pitch-a-playbook-for-winning-the-room-8b327de726ca) ✅ | ⚡ blog | EN | 5 min | Jun 2026. Demo is the centrepiece, skip sign-ups/settings, backup video, projector-legible slides. Its 3-min timeline compresses to v4's 90-s skeleton (drop the "tech" block) | P4 |
| Deck from a brand template | [Edit and re-apply a slide layout — Microsoft Support](https://support.microsoft.com/en-US/PowerPoint/edit-and-re-apply-a-slide-layout) ✅ | 🛠 docs | EN | 5 min | View → Slide Master → add placeholders → Home → Layout to re-apply. Build the 6 v4 slides as *layouts* tonight so tomorrow is drag-and-drop screenshots. (Google Slides: import the PPTX — themes come along) | P4 |
| Backup video on Ubuntu (fastest) | [GNOME screenshots & screencasts — Rocky Linux docs](https://docs.rockylinux.org/desktop/gnome/screenshot/) ✅ | 🛠 how-to | EN | 3 min | Built-in, nothing to install: **Print Screen → video toggle → Region/Screen → Capture**, or **Ctrl+Alt+Shift+R** start/stop; saves WebM to `~/Videos/Screencasts`. ⚠️ Older GNOME versions capped shortcut recordings at 30 s ([ubuntuhandbook, 2020](https://ubuntuhandbook.org/index.php/2020/01/record-ubuntu-desktop-built-in-screen-recorder) 🔎) — test a 90-s take tonight. No mic by default | P4, P1 |
| Backup video alternatives | [Kooha (Flathub, Wayland)](https://itsfoss.com/news/kooha-2-1-release.md) ✅ · [OBS quick start](https://obsproject.com/kb/quick-start-guide) 🔎 | ⚡ | EN | 5 min | Kooha: simple, Wayland-friendly, delay timer (article is 2022). OBS only if you need voice-over + scenes — more setup | P4 |
| ML metrics → money for a business jury | [From metrics to money — r-statistics.co](https://r-statistics.co/From-Metrics-to-Money.html) ✅ | ⚡ lesson | EN (R code; idea is language-free) | 10 min | Profit = b·TP − c·FP; set the threshold from costs; an AUC-0.83 model can make 0 profit at threshold 0.5. Gives the exact sentence pattern: "of every 100 we flag, N are real; each catch saves X ₸" | P4, P2 |
| Q&A under time pressure | [Pitch Q&A — Launchrock (Ryan Rutan)](https://www.launchrock.com/library/expert-advice/pitch-q-and-a) ✅ | ⚡ article | EN | 6 min | May 2026: acknowledge briefly, answer the question asked, give a number, "I don't know — here's how we'd check", stay calm, link to a strength only if natural | P4, P2, P3 |

### D. Deploy via tunnel (P1)

| Concept | Resource | Format | Lang | Time | Why / quality signal | Who |
|---|---|---|---|---|---|---|
| Cloudflare quick tunnel | [Quick Tunnels — Cloudflare docs](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/) ✅ | 🛠 docs | EN | 5 min | Updated 30 Sep 2026: **still no account or domain needed**. `cloudflared tunnel --url http://localhost:8000` → random `*.trycloudflare.com` (changes every run). Limits: 200 in-flight requests, **no SSE**, no uptime guarantee; dies when the process stops | P1 |
| ngrok | [Share localhost quickstart — ngrok docs](https://ngrok.com/docs/share-localhost/quickstart.md) ✅ | 🛠 docs | EN | 5 min | apt install → `ngrok config add-authtoken $TOKEN` → `ngrok http 8000`. Free plan includes one auto-assigned dev domain (stable URL → good for the QR code); custom domains are paid | P1 |
| Vite build served by FastAPI on one port | [StaticFiles — FastAPI reference](https://fastapi.tiangolo.com/reference/staticfiles/) ✅ | 🛠 docs | EN | 5 min | Official. Define API routes first, then `app.mount("/", StaticFiles(directory="web/dist", html=True), name="web")` — `html=True` serves `index.html`. One-screen app → no client-side routes needed. ⚠️ A blog says FastAPI 0.138 added `app.frontend()` for SPAs ([umesh-malik.com](https://umesh-malik.com/blog/fastapi-spa-app-frontend-explained) ✅) but the official release notes page didn't show the entry — use `StaticFiles` | P1 |

### E. "Niche technique? Name it, cite it" (P2, P3)

Use one **only if** it fixes something we actually see, then show before → after on slide 5. "Before → after" below = the metric to report; numbers are from the cited source only where marked.

| Technique | Paper (venue, year) | What it fixes | Package + quick start | Before → after to show | Who |
|---|---|---|---|---|---|
| Monotonic constraints in GBMs | Koklev, *What's the Price of Monotonicity? A Multi-Dataset Benchmark of Monotone-Constrained GB for Credit PD* ([arXiv 2512.17945](https://arxiv.org/pdf/2512.17945), Dec 2025) ✅; method: Auguste et al. ([arXiv 2011.00986](https://arxiv.org/abs/2011.00986v1), 2020) ✅ | Implausible shapes ("more income → more risk") that bankers reject; smoother per-client explanations | LightGBM `monotone_constraints` ([parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html) 🔎); same in XGBoost/CatBoost | **Sourced:** on Give Me Some Credit the AUC cost of monotonicity was only **0.2%** (avg 0.6–0.9% across 5 datasets); wrong-sign constraints cost 8.6%. Show AUC unconstrained → constrained + a PDP that's now monotone | P2 |
| Optimal binning (scorecard) | Navas-Palencia, *Optimal binning: mathematical programming formulation* ([arXiv 2001.08025](https://arxiv.org/abs/2001.08025v2), 2020/22) ✅ | Unstable, non-monotone manual bins; gives an interpretable challenger | `optbinning` ([scorecard tutorial](https://gnpalencia.org/optbinning/tutorials/tutorial_scorecard_binary_target.html) ✅) | Gini: GBM vs scorecard (small gap = "interpretable at little cost"); IV per feature | P2 |
| SHAP TreeExplainer | Lundberg et al., *From local explanations to global understanding with explainable AI for trees*, Nature Machine Intelligence 2020 ([arXiv 1905.04610](https://arxiv.org/abs/1905.04610v1) 🔎) | Fast, exact per-row attributions for tree models → `reasons[]` | `shap` ([intro](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html) ✅) | Standard — mention in passing only (v4 rule 5) | P2 |
| CatBoost ordered target statistics | Prokhorenkova et al., *CatBoost: unbiased boosting with categorical features*, NeurIPS 2018 ([proceedings](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html) 🔎 · [arXiv](https://arxiv.org/abs/1706.09516) ✅) | Target leakage ("prediction shift") from naive target encoding of high-cardinality categories (branch, employer, region) | `catboost` ([quickstart](https://catboost.ai/docs/en/concepts/python-quickstart) ✅) | CV vs holdout gap with naive target encoding → with CatBoost `cat_features` | P2 |
| Venn-Abers calibration | Vovk & Petej, *Venn-Abers Predictors*, UAI 2014 (arXiv 1211.0025, cited on the package page) | Miscalibrated probabilities → wrong ₸ estimates at the threshold; gives a probability interval | `venn-abers` 1.5.4, Aug 2026 ([PyPI](https://pypi.org/project/venn-abers) ✅, [GitHub](https://github.com/ip200/venn-abers)) — `VennAbersCalibrator(estimator=..., inductive=True, cal_size=0.2)` | Brier / log-loss and reliability curve: raw → isotonic → Venn-Abers (no published benchmark on the package page) | P2 |
| Conformal prediction | Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction and Distribution-Free UQ* ([arXiv 2107.07511](https://arxiv.org/html/2107.07511v6), 2021; FnT ML 2023) 🔎 | "How sure is the model for this client?" with a coverage guarantee → route uncertain cases to a human (AI-Law "human decides") | `mapie` v1 ([quick start](https://mapie.readthedocs.io/en/latest/content/getting-started/quick-start/) ✅) — `SplitConformalClassifier(confidence_level=0.9).conformalize(...)`, `predict_set` | Empirical coverage ≈ 90% on holdout; share of clients with a single-label set (auto-decide) vs two-label set (send to a human) | P2 |
| Adversarial validation | Pan et al. (Uber), *Adversarial Validation Approach to Concept Drift Problem in User Targeting Automation Systems* ([arXiv 2004.03045](https://arxiv.org/abs/2004.03045v1), 2020) ✅ | Train/test (or old/new period) drift that makes CV lie | No package needed: label train=0/test=1, fit LightGBM, read AUC + top features | Adversarial AUC before → after dropping/fixing the drifting feature (≈ 0.5 = no drift); the paper gives no headline numbers | P3, P2 |
| Worst-group evaluation (fairness) | Sagawa et al., *Distributionally Robust Neural Networks for Group Shifts*, ICLR 2020 ([arXiv 1911.08731](https://arxiv.org/abs/1911.08731v2)) 🔎 — cite for the *worst-group metric*; Group DRO training itself is for neural nets | Good average metric hiding a bad segment (region, age band, gender) — AI-Law discrimination question | `fairlearn` `MetricFrame` ([assessment guide](https://fairlearn.org/main/user_guide/assessment) ✅ → [MetricFrame API](https://fairlearn.org/main/api_reference/generated/fairlearn.metrics.MetricFrame.html) 🔎) | Overall AUC vs worst-group AUC (`group_min`) and the gap (`difference`) | P3, P2 |
| TabPFN (small data only) | Hollmann et al., *Accurate predictions on small data with a tabular foundation model*, Nature 2025 ([PubMed 39780007](https://pubmed.ncbi.nlm.nih.gov/39780007/) 🔎) | Strong baseline in seconds when n is small (≤ ~10k rows) | `tabpfn` 9.2.0, Oct 2026 ([PyPI](https://pypi.org/project/tabpfn/) ✅) | AUC: tuned LightGBM vs TabPFN. ⚠️ First use opens a **browser login + licence acceptance** (`TABPFN_TOKEN` for headless); default weights (TabPFN-3.5) are **non-commercial**; GPU recommended; CPU capped at 1–5k rows. **Skip unless it runs tonight** (v4 rule 11) | P2 |
| PD calibration literature | — | — | — | Not added: no single citable, maintained-code source found beyond sklearn calibration + Venn-Abers above | — |

### F. Conditional: people analytics / HR attrition (P2, P3) — only if the case is HR

| Concept | Resource | Format | Lang | Time | Why / quality signal | Who |
|---|---|---|---|---|---|---|
| Practice dataset | [IBM HR Analytics Employee Attrition (Kaggle)](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) 🔎 (linked from the ✅ tutorial below) | 🛠 dataset | EN | 5 min | ~1,470 employees, ~16% attrition, fictional (IBM data scientists). Small → good for tonight's pipeline dry run *if* P2 has spare time | P2, P3 |
| Attrition + SHAP tutorial | [Predict employee attrition with explainable AI (SHAP) — Analytics Vidhya](https://www.analyticsvidhya.com/blog/2025/07/predict-employee-attrition-with-shap/) ✅ | 🛠 tutorial | EN | 7 min | Jul 2025, full code: IBM data → XGBoost → SHAP summary. OverTime is the classic top driver — a "first idea is everyone's idea" warning | P2 |
| Time-to-attrition (survival) | [lifelines Quickstart](https://lifelines.readthedocs.io/en/stable/Quickstart.html) ✅ | 🛠 docs | EN | 10 min | Official (v0.30.3). `KaplanMeierFitter` for "how long until people leave" curves by segment; `CoxPHFitter` for drivers; handles still-employed people (censoring) correctly — a citable step beyond "classification" | P3, P2 |
| Survival worked example | [A survival analysis guide with Python — TDS](https://towardsdatascience.com/a-survival-analysis-guide-with-python-using-time-to-event-models-to-forecast-customer-lifetime/) ✅ | 🛠 article + repo | EN | 11 min (read KM + Cox parts) | Apr 2026, lifelines KM + log-rank + Cox per-customer predictions (churn, same maths as attrition) | P3 |
| Cost of attrition (₸ impact) | [This fixable problem costs U.S. businesses $1 trillion — Gallup](https://www.gallup.com/workplace/247391/fixable-problem-costs-businesses-trillion.aspx) ✅ | ⚡ article | EN | 5 min | Replacement cost **½–2× annual salary** (conservative); **52%** of leavers say it was preventable. ⚠️ 2019, US. Time-to-hire benchmark: SHRM 42 days / $4,129 per hire ([CBIA, 2016](https://www.cbia.com/news/economy/whats-average-cost-hiring-new-employee) ✅ — old) | P4, P3 |

### G. Workshop speakers' track record (P4)

| Concept | Resource | Format | Lang | Time | Why / quality signal | Who |
|---|---|---|---|---|---|---|
| Team "Richards" (HackNU 2025 Higgsfield ML track; Decentrathon 5.0 2nd, inDrive "Legislative Entropy") | — | — | — | — | ⚠️ **No public write-up, repo, deck or post found** under that team name (searched HackNU, Decentrathon, GitHub, LinkedIn). Ask them directly at the event | P4 |
| HackNU/25 context | [the-tech.kz announcement](https://the-tech.kz/v-astane-sostoitsya-krupnejshij-studencheskij-hakaton-hacknu-25/) ✅ RU · [Satbayev 2nd place, Higgsfield ML track (Aitys.ai)](https://satbayev.university/en/news/satbayev-university-team-took-2nd-place-at-hacknu-2025-with-the-project-aitysai) ✅ | ⚡ news | RU/EN | 3 min | 18–19 Oct 2025, 24 h at NU, tasks from Zaman Bank, Higgsfield, MyCar. 2nd place built a text → illustrated-story generator — shows the bar was a working end-to-end product | P4 |
| Decentrathon 5.0 inDrive cases + judging | [inDrive CTO on the cases — DigitalBusiness.kz](https://digitalbusiness.kz/2026-03-26/kak-s-pomoshchyu-ai-nahodit-liderov-i-raspredelyat-resursi-keysi-indrive-dlya-uchastnikov-decentrathon-5-0/) ✅ RU | ⚡ interview | RU | 8 min | **The criteria the speakers were judged on:** uniqueness over polish, **explainability**, **fairness/no bias**, validation (solves the user's real problem), **stability** under changed conditions. "Legislative Entropy" = contradicting laws/regulations → plain-language AI explanations, conflict detection, budget anomalies. Maps directly onto v4 | P4 (team) |
| Example Decentrathon 5.0 repo | [QwertyS — inVision U case study](https://github.com/pip00sya/invisionu-showcase) ✅ | ⚡ repo | EN | 5 min | Not a winner (no prize claimed), but a clean example of the expected shape: React + FastAPI, anonymised reviewer dashboard, criterion-based explanations, honest limitations section | P4, P1 |


---

## 4. KEEP-REFERENCE appendix — look up during the event, don't read tonight

Items carried from v1 that v4 still depends on. Grouped by when you'll need them. (Items promoted to tonight's lists are listed in section 1 instead.)

### Hour 0–3 (11:00–14:00): brief → checklist → tickets, first data look

| Item | Link | Why keep / when | Who |
|---|---|---|---|
| Claude Code best practices — "Let Claude interview you" + "Manage your session" sections | [code.claude.com/docs/en/best-practices](https://code.claude.com/docs/en/best-practices) ✅ | Fallback grill prompt (AskUserQuestion → SPEC.md) if `/grill-with-docs` misbehaves; /clear and /compact rules | P1 |
| Steering Claude Code (CLAUDE.md vs skills vs hooks vs subagents) | [claude.com/resources/…](https://claude.com/resources/articles/steering-claude-code-skills-hooks-rules-subagents-and-more) ✅ | 5-min refresher if `AGENTS.md` grows past 150 lines | P1 |
| CLAUDE.md / AGENTS.md (memory docs) | [code.claude.com/docs/en/memory](https://code.claude.com/docs/en/memory) ✅ | Claude can read `AGENTS.md` directly; check `/context` | P1 |
| Skills docs (Claude) | [code.claude.com/docs/en/skills](https://code.claude.com/docs/en/skills) ✅ | Frontmatter, `$ARGUMENTS`, `disable-model-invocation` — if a vendored skill needs a quick edit | P1 |
| Subagents docs | [code.claude.com/docs/en/sub-agents](https://code.claude.com/docs/en/sub-agents) ✅ | `model: haiku`, `maxTurns` for the reviewer/researcher agents | P1 |
| Hooks guide | [code.claude.com/docs/en/hooks-guide](https://code.claude.com/docs/en/hooks-guide) ✅ | Ruff PostToolUse hook; debugging Impeccable's hooks | P1 |
| Plan mode | [code.claude.com/docs/en/permission-modes](https://code.claude.com/docs/en/permission-modes) ✅ | Shift+Tab; only for 3+ file changes | All |
| Context window + Reduce token usage | [context-window](https://code.claude.com/docs/en/context-window) ✅ · [costs](https://code.claude.com/docs/en/costs) ✅ | When `/usage` > 70% before 01:00 | All |
| Pocock workflow map (grill → spec → tickets → review) | [contextstudios.ai](https://www.contextstudios.ai/blog/pocock-workflow-grill-me-spec-tickets-subagent-review) ✅ | 9-min map of MODE CASE steps if someone is lost | All |
| Codex: AGENTS.md, skills, worktrees, slash commands, best practices | [agents-md](https://learn.chatgpt.com/codex/agent-configuration/agents-md) ✅ · [build-skills](https://learn.chatgpt.com/docs/build-skills) ✅ · [worktrees](https://learn.chatgpt.com/codex/environments/git-worktrees) ✅ · [slash commands](https://learn.chatgpt.com/codex/reference/slash-commands) ✅ · [best practices](https://learn.chatgpt.com/guides/best-practices) ✅ | Codex users: `$skill`, `/status`, 32 KiB AGENTS.md cap, manual `.env` copy in CLI worktrees | Codex users |
| GitHub: issue dependencies, sub-issues, `gh pr create`, squash merges | [deps](https://docs.github.com/en/enterprise-cloud@latest/issues/tracking-your-work-with-issues/using-issues/creating-issue-dependencies) ✅ · [sub-issues](https://docs.github.com/en/enterprise-cloud@latest/issues/tracking-your-work-with-issues/using-issues/adding-sub-issues) 🔎 · [gh pr create](https://cli.github.com/manual/gh_pr_create) ✅ · [merging](https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/merging-a-pull-request) ✅ | Flags for draft PRs and blockers when the fallback (plain `gh issue create`) is needed | P1 |
| Conventional Commits · annotated tags | [conventionalcommits.org](https://www.conventionalcommits.org/en/v1.0.0/) ✅ · [Pro Git: Tagging](https://git-scm.com/book/en/v2/Git-Basics-Tagging) ✅ | `feat(#12): …`; `git push origin mvp0` (tags aren't pushed by default) | All |
| uv · ruff · pre-commit · nbstripout · jupytext | [uv projects](https://docs.astral.sh/uv/guides/projects/) ✅ · [ruff](https://docs.astral.sh/ruff/) ✅ · [pre-commit](https://pre-commit.com/) ✅ · [nbstripout](https://github.com/kynan/nbstripout) ✅ · [jupytext](https://jupytext.readthedocs.io/) 🔎 | Env breakage on a teammate's laptop | P1 |
| pandera with polars | [pandera polars docs](https://pandera.readthedocs.io/en/stable/polars.html) ✅ | Data-contract tests at the skeleton stage. ⚠️ No Hypothesis synthesis for polars | P1, P3 |
| Polars getting started | [docs.pola.rs](https://docs.pola.rs/user-guide/getting-started) ✅ | Syntax lookup | P3 |
| Datasheets for Datasets (question list) | [arXiv 1803.09010](https://arxiv.org/pdf/1803.09010) 🔎 | If the `DATA_CARD.md` template misses something (provenance, collection, known gaps) | P3 |
| Winners' framing: one actionable question; reframe to the decision | [Harvard SEAS 2022](https://seas.harvard.edu/news/harvard-team-wins-boston-regional-datathon-second-straight-year) ✅ · [Rotman 2020 recap](https://cd-www.rotman.utoronto.ca/Degrees/MastersPrograms/SpecializedProgramsBlog/MMA/MMA-Online-Datathon-2020-Recap) ✅ | Re-read at the 14:00 question lock (4 min each) | P4, all |
| Interview → SPEC.md fallback prompt | inside Best practices (above) ✅ | If mattpocock skills fail | P1 |
| Skills changelog v1.1 (renames) | [aihero.dev](https://www.aihero.dev/skills/skills-changelog-v1-1-wayfinder-to-spec-to-tickets-grilling-improvements) ✅ | Renames already applied in v4; check only if a skill name fails | P1 |
| Claude Code worktrees | [code.claude.com/docs/en/worktrees](https://code.claude.com/docs/en/worktrees) ✅ | v4 rule 8 already encodes `worktree-<slug>` and `.worktreeinclude`; look up on error | P1 |
| Use Claude Code with your Pro or Max plan | [support.claude.com](https://support.claude.com/en/articles/11145838-use-claude-code-with-your-pro-or-max-plan) ✅ | 5-hour + weekly limits, `/status`; no official message counts | All |
| Halyk Oiy rubric (KZ bank jury proxy) | [halykbank.kz](https://halykbank.kz/en/other/halyk-oiy-hackathon) ✅ | Six equal criteria incl. honesty & self-reflection — for `JURY.md` if time | P4 |
| Xero short-pitch playbook | [devblog.xero.com](https://devblog.xero.com/land-your-hackathon-pitch-a-playbook-for-winning-the-room-8b327de726ca) ✅ | Second opinion on the demo-centred flow | P4 |

### Signal hunt (14:00–00:00)

| Item | Link | Why keep / when | Who |
|---|---|---|---|
| skrub AggJoiner on a fraud dataset | [skrub example](https://skrub-data.org/stable/auto_examples/03_joining/0070_join_aggregation.html) ✅ | If the case has a one-to-many history table | P3 |
| Leakage taxonomy (Princeton) | [reproducible.cs.princeton.edu](https://reproducible.cs.princeton.edu/) ✅ | Answering "how do you know it isn't leakage?" | P3, P4 |
| PCS stability checklist (Rewolinski & Yu) | [arXiv 2508.00835](https://arxiv.org/html/2508.00835v2) ✅ — read "takeaways" only | Robustness gate item 2 (stability) | P3 |
| Forking paths / multiple comparisons | [Wikipedia](https://en.wikipedia.org/wiki/Forking_paths_problem) 🔎 | "We tested N hypotheses, K survive FDR" | P3 |
| E-value (unmeasured confounding) | [MetricGate post](https://metricgate.com/blogs/e-value-causal-bound-confounding/) 🔎 | Q&A "is it causal?" — one number | P3 |
| HDBSCAN demo · IsolationForest | [HDBSCAN demo](https://scikit-learn.org/1.6/auto_examples/cluster/plot_hdbscan.html) ✅ · [IsolationForest](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.IsolationForest.html) 🔎 | Named segments; anomaly ranking | P3 |
| Named-segment example (Wharton × Fulton Bank) | [case study](https://ai-analytics.wharton.upenn.edu/research/2021-fulton-bank-case-study/) ✅ | "Swing By / Loyal" style segment names; "16× more likely" phrasing | P3, P4 |
| PSI (drift between periods) | [NannyML](https://nannyml.com/blog/population-stability-index-psi) 🔎 · [Train in Data](https://blog.trainindata.com/population-stability-index-and-feature-selection-python/) 🔎 | "Holds in the latest period" check | P3, P2 |
| Kazakhstan fraud context (droppers art. 232-1, NBK anti-fraud centre) | [BES droppers](https://bes.media/news/kolichestvo-del-o-dropperah-viroslo-v-sem-raz-8576b1/) ✅ RU · [BES anti-fraud](https://bes.media/news/antifrod-tsentr-natsbanka-zablokiroval-25-mlrd-tenge-moshennicheskih-tranzaktsiy/) ✅ RU · [Group-IB mule indicators](https://www.group-ib.com/blog/money-mules/%20) ✅ | Only if the case is fraud/AML | P3, P4 |
| ARDFM bank-sector monthly release | [gov.kz/ardfm](https://www.gov.kz/memleket/entities/ardfm/press/news/details/1237225?lang=ru) 🔎 RU | Benchmark segment sizes against the market (needs JS) | P4 |

### Modeling (14:00–06:00)

| Item | Link | Why keep / when | Who |
|---|---|---|---|
| LightGBM Parameters Tuning | [lightgbm docs](https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html) ✅ | Over-fitting checklist | P2 |
| CatBoost quickstart + tutorials | [quickstart](https://catboost.ai/docs/en/concepts/python-quickstart) ✅ · [tutorials](https://github.com/catboost/tutorials/) 🔎 | Many categorical columns | P2 |
| Visualizing CV behaviour (GroupKFold, TimeSeriesSplit) | [sklearn example](https://scikit-learn.org/stable/auto%5Fexamples/model_selection/plot_cv_indices.html) ✅ | Picking the split at 11:30 | P2 |
| Molnar IML book, ch. 18 SHAP (limitations) | [christophm.github.io](https://christophm.github.io/interpretable-ml-book/) ✅ | Q&A "SHAP isn't causal" | P2 |
| "To SMOTE, or not to SMOTE?" | [arXiv 2201.08528](https://arxiv.org/abs/2201.08528v3) ✅ | Q&A "why didn't you rebalance?" — balancing doesn't help strong GBMs | P2 |
| AutoGluon quick start / Optuna | [AutoGluon](https://auto.gluon.ai/dev/tutorials/tabular/tabular-quick-start.html) ✅ · [Optuna](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/001_first.html) ✅ | Ceiling check / last-30-min tuning only — ⚠️ v4 rule: only tools tried tonight | P2 |
| IEEE-CIS 1st place (UID aggregates, time split) + Deotte posts | [NVIDIA blog](https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution) ✅ · [discussion 108575](https://www.kaggle.com/c/ieee-fraud-detection/discussion/108575) 🔎 | If transactional/fraud | P2 |
| Home Credit nested per-table models | [deepsense.ai](https://deepsense.ai/wait-so-loans-need-to-be-repaid-the-home-credit-risk-prediction-competition-on-kaggle/) ✅ | If credit with history tables | P2 |
| Personalised PageRank features (ING + UvA) | [arXiv 2402.09495](https://arxiv.org/html/2402.09495v1) ✅ | If there are links between participants | P2 |
| Gini explained (Gini = 2·AUC − 1, credit context) | [TDS](https://towardsdatascience.com/beyond-roc-auc-and-ks-gini-coefficient-explained-simply/) ✅ | Translating the metric for bankers | P2, P4 |
| RAG stack (only if the case needs text) | [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) ✅ · [bge-m3](https://huggingface.co/BAAI/bge-m3) ✅ · [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) ✅ · [RusBEIR](https://arxiv.org/abs/2504.12879v1) ✅ · [Docling chunking](https://docling-project.github.io/docling/concepts/chunking/) ✅ · [Ragas](https://docs.ragas.io/en/latest/getstarted/evaluation.html) ✅ | v4: use an API LLM; hybrid BM25 + bge-m3 + reranker if retrieval is needed | P2, P3 |
| Kazakh LLMs (only if "local LLM" comes up) | [ISSAI KazLLM](https://issai.nu.edu.kz/kazllm/) ✅ · [MDPI 2026 Kazakh benchmark](https://www.mdpi.com/2078-2489/17/9/921) ✅ | KazLLM is CC-BY-NC and unchanged since Dec 2024; Qwen3-8B leads KazMMLU | P4 |
| GenBI / text-to-SQL (only if asked) | [MotherDuck text-to-SQL agent](https://motherduck.com/blog/langchain-sql-agent-duckdb-motherduck/) ✅ | Read-only DuckDB connection; show the SQL | P2, P1 |

### Pitch & Q&A (06:00–11:00)

| Item | Link | Why keep / when | Who |
|---|---|---|---|
| BIS note: PD, LGD, EAD, expected loss | [bis.org](https://www.bis.org/bcbs/irbriskweight.pdf) ✅ — sections 1–3 | ₸-impact line if credit: EL = PD × LGD × EAD | P4, P2 |
| Experian swap sets | [experian.com](https://www.experian.com/blogs/insights/swap-set-measure-impact-model-changes/) ✅ | "Same bad rate, more approvals" — the "what changes on Monday" answer | P4 |
| SWD "Declutter this graph" | [storytellingwithdata.com](https://www.storytellingwithdata.com/blog/2017/3/29/declutter-this-graph) ✅ | Fix a busy chart before the freeze | P4, P3 |
| Streamlit "Create an app" | [docs.streamlit.io](https://docs.streamlit.io/get-started/tutorials/create-an-app) ✅ | **Emergency fallback UI only** (v4) | P1 |
| ForteBank background | [Wikipedia](https://en.wikipedia.org/wiki/ForteBank) 🔎 | Jury context | P4 |

---

## 5. DROPPED (v1 items not needed under v4)

| Item (v1 group) | Reason |
|---|---|
| shanraisshan/claude-code-best-practice (A) | 30–60 min; the template's settings are already decided |
| Matt Pocock "AI Coding Workflow" 1h36 workshop (A) | Too long; the skills are vendored and the flow is in v4 |
| Matt Pocock "Software Fundamentals for AI Coding" talk (A) | Nice background; no effect on tonight or tomorrow |
| Pragmatic Engineer × Pocock podcast (A) | 1h35; TDD/tracer-bullet takeaways already in v4 |
| Simon Willison on parallel agents (A) | Philosophy; v4 already sets small PRs and fast merges |
| Böckeler on spec-driven development tools (A) | The tool decision is made |
| claude-code-handbook-ru (A) | 33-star link catalog |
| Atlassian / trunkbaseddevelopment.com / Avito Habr / PostHog TBD explainers (B) | Generic; v4 rule 7 states the exact git rules |
| Cookiecutter Data Science (B) | Our template exists; CCDS v2 uses Black/flake8 |
| Traceability-matrix article (B) | `acceptance.md` format is fixed in v4 |
| Data Cards Playbook (C) | Datasheets kept as reference is enough |
| Leakage paper, full arXiv version (C) | 45 min; the Princeton summary page is kept |
| rsokl/testing-tutorial (half-day Hypothesis) (C) | Half a day; tests are limited to deterministic seams |
| EuroSciPy Hypothesis-pandas talk (C) | 30 min; not on the critical path |
| "pytest for pipelines" note (C) | Was a pointer, not a resource |
| mlcourse.ai (D) | Hours per topic |
| Short sklearn threshold example (D) | Duplicate of the cost-sensitive example kept tonight |
| SHAP interaction-values notebook on old mirror (D) | Outdated host; the main SHAP intro covers dependence colouring |
| optbinning FICO tutorial (D) | Duplicate of the scorecard tutorial (same HELOC data) |
| ID Finance scorecard Habr post (D) | 2018; optbinning tutorial covers the workflow |
| "How high does your Gini need to be?" paper (D) | 25 pages; one line of it is in the pitch-metrics item instead |
| NoxMoon Home Credit repo (D) | Practice repo; no time |
| Theodo Personalised-PageRank blog (D) | Weak; the ING paper is kept |
| IBM synthetic AML dataset (D) | Practice dataset; GMSC is tonight's practice set |
| Twyman's law chapter (E) | Paywalled; the robustness gate covers the reflex |
| Veridical Data Science book (E) | Book-length; the PCS checklist is kept |
| Facure "Causal Inference for the Brave and True" chapters (E) | > 30 min each; causal Q&A handled with honesty + E-value |
| Causal Inference: The Mixtape (E) | Book-length |
| DoWhy + Lalonde tutorial (E) | 43 min |
| Uplift: OTUS Habr post + scikit-uplift (E) | Not in v4's plan; scikit-uplift unmaintained |
| "Sensitivity analysis in practice" note (E) | Pointer only |
| DuckDB-NSQL (F) | Jan 2024 small model; v4 says use API LLMs |
| SQL-guard DEV post (F) | Read-only DB connection covers it |
| Storytelling with Data Talks at Google (~1 h) and book (G) | Too long for tonight; think-cell + declutter exercise cover the rules |
| "When decluttering isn't enough" (G) | Nice-to-have |
| Pyramid Principle (Kenney), SCQA note, Models substack (G) | v4 bans the problem statement; answer-first is already the 90-s skeleton |
| ASA DataFest judges' note (4 min, 2 slides) (G) | Different format; replaced by 90-s pitch sources |
| Harvey Mudd Citadel winner news (G) | External data angle already in v4 |
| Wiserfunding scoring-lifecycle article (H) | Low quality; not needed |
| v1 learning paths, team top 10 and 5 drills | Replaced by tonight's ≤ 75-min lists; v4 tonight tracks are the drills |

---

## 6. Gaps and uncertain items

### Things that contradict or update master prompt v4
| # | Item | What the sources say | Action tonight |
|---|---|---|---|
| 1 | Impeccable "Brand Commitments" | No such section in current docs (v4.6.0). Brand is carried by `DESIGN.md` (from `/impeccable document` after tokens exist) and constraints in `PRODUCT.md` (from `init`) | Put Forte tokens in CSS first, then `document`; write the brand rules into `init`'s constraints; commit both files |
| 2 | `/impeccable craft` | Listed in the README's command table but **not** in the docs' command list; docs use `/impeccable <plain request>` (Codex: `$impeccable`) | Try `craft` tonight; fall back to the plain form or `shape` |
| 3 | Impeccable prerequisites + telemetry | Getting-started says Node ≥ 22.18; telemetry on by default | `node -v` on all 4 laptops; `npx impeccable telemetry off` (bank data nearby) |
| 4 | Impeccable hooks in Claude Code | Since v4.6.0 the design hook installs into **user** settings, not the shared `.claude/settings.json` | Each Claude user runs the install/hook step on their own laptop |
| 5 | shadcn MCP for Codex | `mcp init` can't edit `~/.codex/config.toml`; add the `[mcp_servers.shadcn]` block by hand | As in table 3A |
| 6 | web-design-guidelines install | The repo README shows `npx skills add vercel-labs/agent-skills` (whole collection); the single-skill flag in v4 wasn't confirmed | Run v4's command; if it errors, install the collection and remove the rest |
| 7 | FastAPI SPA serving | `app.frontend()` (blog: FastAPI 0.138) not confirmed in official release notes | Use `StaticFiles(directory="web/dist", html=True)` mounted after API routes |
| 8 | Cloudflare quick tunnel | Still account-free (docs updated 30 Sep 2026), but **no SSE**, 200 in-flight requests, new URL each run | Don't stream responses through it; prefer ngrok's free dev domain for a stable QR code |
| 9 | Ubuntu screencast length | Older GNOME capped shortcut recordings at 30 s | Record a 90-s test tonight; if it stops at 30 s, use the Print-Screen UI or Kooha |
| 10 | TabPFN | Browser login + licence acceptance on first use; default weights non-commercial; GPU or ≤ 1–5k rows on CPU | Only use if it ran tonight; otherwise don't mention it |

### Not found / couldn't verify
- **Forte PPTX contents:** the four forte.kz downloads respond (binary), but the research sandbox couldn't download them to inspect layouts → P4 opens them tonight.
- **"Forte People":** no public sub-brand, HR brand or careers site under that name; not on the brand page's sub-brand list (Solo, Premier, Business, Corporate, Leasing, Finance). The HR-case hint stays an unverified assumption.
- **Team "Richards":** no public write-up, repo, deck or post found for their HackNU 2025 or Decentrathon 5.0 results; Decentrathon 5.0 winners lists weren't findable either (results article had no team names). Use the inDrive judging-criteria interview as the proxy and ask them in person.
- **Forte AI Hackathon 2025:** rules page found, but no published winners, visuals or criteria.
- **Impeccable video:** no official walkthrough video verified; only the docs case study and Bakaus's posts on X.
- **SHAP → reason codes with code:** the best source is a vendor white paper (concept, not code); the grouping/templating code is ours to write.
- **Before → after numbers for niche techniques:** only monotonic constraints has a sourced number (GMSC AUC cost 0.2%). For the rest, the table names the metric to report; we must produce our own number.
- **HR cost benchmarks** are US and old (Gallup 2019, SHRM 2016) — say so if used; no Kazakhstan figure found.
- **PD-calibration literature:** no compact, citable source with maintained code beyond sklearn calibration + Venn-Abers.

### Link status
- ✅ opened and checked on 16 Oct 2026 · 🔎 from search results or linked from an opened page, not opened myself. Carried-over v1 links keep their original marks.
- Blocked or rate-limited for automated fetching: some stat.gov.kz and gov.kz pages (need a browser), PubMed (captcha), Habr (rate limit earlier).