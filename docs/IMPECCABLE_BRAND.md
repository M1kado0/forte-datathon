# Impeccable — Brand Commitments (paste into `/impeccable init`)

**Brand:** ForteBank (Kazakhstan). Case partner of DATATHON '26. This is a datathon prototype shown to ForteBank staff.

**Mandatory — do not override with defaults:**
1. Font: **Open Sans** only (bundled via @fontsource). Weights: 800/700 headings, 600 labels, 400 body.
2. Palette (official, forte.kz/logo):
   - Forte Magenta `#A31551` — primary; the ONLY strong colour on a screen; primary buttons, highlights.
   - Dark Magenta `#4A0221` — headings, dark surfaces, hover state of primary.
   - Sky Blue `#61CFFF` — chart accent, sparingly.
   - Coral `#FC9385` — warm secondary accent.
   - Hero gradient: Dark Magenta → Coral, 135°. One gradient per screen, hero only. Keep it.
3. Neutrals (ours): bg `#FFFFFF`, muted bg `#F6F5F7`, text `#1A1A1A`, muted text `#6B6B70`, border `#E7E5EA`.
   Semantic: success `#1F9D63`, warning `#E8A317`, danger `#D93025` (never use magenta as "error").
4. Radius: cards 16px, buttons/inputs 12px. Shadow `0 4px 16px rgba(74,2,33,.08)`.
5. Logo: official file only (`web/public/forte-logo.svg`), Forte Magenta on white/light or on Dark Magenta.
   **Never recolour, stretch, or add effects.** Keep clear space around it.
6. Tokens live only in `web/src/index.css`; components use Tailwind classes mapped to them.

**Audience:** ForteBank staff (the jury: bankers, AI engineers, product people) and bank employees
or clients in Kazakhstan. UI language: **Russian**. Money: `1 250 000 ₸` (narrow no-break space thousands).

**Tone:** confident, warm, short, verb-led ("Проверить клиента", "Открыть отчёт"). No jargon on screen.

**Layout principles:** Google, not Yahoo. One primary action per screen. Above the fold: logo, product
name, one input, one magenta CTA. Result = big number + plain verdict + top-5 reasons + action with ₸ effect.

**Banned:** purple-blue gradients · glassmorphism · cards inside cards · grey text on colour ·
emoji or icon tile above every heading · lorem ipsum · rainbow charts (charts = grey + one accent) ·
dark mode toggles · decorative illustrations · more than one gradient per screen.

**Every screen needs:** loading skeleton, empty state, error state with retry, long-text handling, 390px mobile layout.
