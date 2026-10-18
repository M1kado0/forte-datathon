# Impeccable — Brand Commitments (paste into `/impeccable init`)

**Brand:** ForteBank (Kazakhstan). Case partner of DATATHON '26. This is a datathon prototype shown to ForteBank staff.

**Mandatory — do not override with defaults:**
1. Font: **Open Sans** for UI text (bundled via @fontsource). Weights: 700 headings, 600 labels, 400 body.
   Playfair Display 400 allowed for display titles (page and section h1/h2) only.
   Handjet 500 allowed for team names in the About section only.
2. Palette (official, forte.kz/logo):
   - Forte Magenta `#A31551` — primary; the ONLY strong colour on a screen; primary buttons, highlights.
   - Dark Magenta `#4A0221` — headings, dark surfaces. Result headings may use the neutral text colour.
   - Sky Blue `#61CFFF` — chart accent, sparingly.
   - Coral `#FC9385` — warm secondary accent.
   - Hero gradient: Dark Magenta → Coral, 135°, optional; at most one per screen.
3. Neutrals: white or warm off-white bg; text, muted text, border and semantic colours (success, warning,
   danger) must pass WCAG AA on their background. Never use magenta as "error".
4. Radius: cards 16px, buttons/inputs 8px. Shadow `0 4px 16px rgba(74,2,33,.08)`.
5. Logo: official file only (`web/public/forte-logo.svg`, or its PNG export `web/public/forte-logo.png`),
   Forte Magenta on white/light or on Dark Magenta.
   **Never recolour, stretch, or add effects.** Keep clear space around it.
6. Tokens live only in `web/src/index.css`; components use Tailwind classes mapped to them.

**Audience:** ForteBank staff (the jury: bankers, AI engineers, product people) and bank employees
or clients in Kazakhstan. UI language: **Russian**. Money: `1 250 000 ₸` (narrow no-break space thousands).

**Tone:** confident, warm, short, verb-led ("Проверить клиента", "Открыть отчёт"). No jargon on screen.

**Layout principles:** Google, not Yahoo. One primary action per screen. Above the fold: logo, product
name, the upload or one magenta CTA that opens it in place. Decoration never delays the upload by more than 1 s.
Result = verdict sentence, the "подсказка, не заключение" line, then per violation: type, timecode,
description, rule document + пункт, evidence frames.

**Motion:** respect `prefers-reduced-motion`; autoplay media muted and stopped while off-screen;
no identifiable people next to violation wording.

**Banned:** purple-blue gradients · frosted-glass panels behind text · cards inside cards · grey text on colour ·
emoji or icon tile above every heading · lorem ipsum · rainbow charts (charts = grey + one accent) ·
dark mode toggles · decoration that blocks or hides the upload · more than one gradient per screen.

**Every screen needs:** loading skeleton, empty state, error state with retry, long-text handling, 390px mobile layout.

## PR #96 typography exception (18 October 2026)

Per the author's confirmation that the reviewer approved these fonts, PR #96 keeps
Google Sans for display titles and evidence-viewer controls, and Handjet 500 for
the result verdict and violation headings as well as team names. Observations and
normative-source body text remain Open Sans. This scoped exception takes precedence
over item 1 above for these surfaces.

The author has designated the current design as final: preserve its layout,
palette, imagery and typography while correcting functional and accessibility
defects. The required first-screen logo, honest demo-mode indication and opaque
upload surface remain part of acceptance.
