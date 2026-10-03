# Design — sam-extra Forge Appearance

A locked design system for the Forge Neo appearance layer shipped by this
extension. It changes presentation only, with two documented build-time
placements on txt2img: (1) sam-extra's own Anima tuning scripts use a Forge
`Script.section` (`sam3_anima`, visible in Settings → Parameter order), and
(2) SAM3 Refine, Tile-Repair and Character Reference live in a "선택 이미지"
dock (`gr.Tabs`) under the gallery. Forge routes, generation behaviour,
settings semantics, script arguments, ui-config keys and extension element ids
stay intact.

## Genre

Atmospheric, with the restraint of a technical workbench. The interface is dark
because it is used for long image-generation sessions, not because it needs
decorative glow, glass, gradients, or motion.

## Macrostructure family

- App pages: **Workbench** — preserve Forge's existing top-level tabs and dense
  control hierarchy. SAM3's txt2img layout keeps its asymmetric
  Parameters / Scripts / Gallery columns and adds:
  - a full-width "켜진 기능" chip bar between the prompt band and the columns;
  - column 1 sections: core settings, 고정 (user pins, per browser), ANIMA 튜닝;
  - column 2 sections: 디테일러, 스크립트, and a collapsed 더 보기 (로라·제어,
    도구·실험; extensions with no registry entry land in 도구·실험);
  - column 3: gallery, then the 선택 이미지 dock, then Notebook (collapsed).

## Runtime DOM rules

- JS may move only whole containers (`#txt2img_settings`,
  `#txt2img_script_container`, the gallery section, `#txt2img_extra_tabs`) and,
  when the user pins one, a single always-on group root — restored exactly on
  unpin. `#script_list` and selectable script panels are never reparented.
- Runtime state is written only as `data-*` attributes or inline `order`, and
  drawn with CSS pseudo-elements. JS never adds a class to a Gradio-owned node
  (Gradio rewrites `class` wholesale) and never inserts or removes children at
  steady state, so Forge's childList observer stays quiet.
- No stacking-context property (`z-index`, `transform`, `filter`, `opacity < 1`,
  `contain`, `will-change`, `isolation`, `backdrop-filter`) on any layout
  ancestor: a Gradio popup inside one is trapped (v0.21.0 regression).
- Settings and extension pages: **Long Document** inside the same Workbench
  shell — existing section order and component ownership remain unchanged.
- Content/help surfaces: typography and dividers only; no decorative cards.

## Theme

The geometry, typography, spacing, and interaction language are shared. A user
may select one global palette at a time:

- **Forge Default** — no appearance override.
- **Graphite Ember** — neutral graphite surfaces with Forge's warm orange as
  the single signal colour.
- **Obsidian Violet** — violet-tinted near-black surfaces with a restrained
  violet signal colour.
- **Warm Espresso** — warm brown-black surfaces with amber signal colour.
- **OLED Mono** — near-black, low-chroma surfaces with an off-white signal.

The concrete OKLCH values and the Gradio variable bridge live in
[`tokens.css`](tokens.css). Palettes never mix between tabs.

## Typography

- Display: Source Sans Pro, weight 700, roman.
- Body: Source Sans Pro, weight 400.
- Mono: IBM Plex Mono, weight 400/600.
- Existing Forge font loading is retained to avoid an additional network
  dependency and layout shift.
- Data and numeric controls use tabular figures.

## Spacing

A named 4-point scale is defined in `tokens.css`. New appearance-layer rules use
the named tokens; existing Forge layout dimensions are not globally rewritten.

## Motion

- Easings: named exponential curves in `tokens.css`.
- State changes: colour and at most a one-pixel press translation.
- Page and tab content: no entrance animation.
- Focus indicators: instant.
- Reduced motion: spatial movement removed, state feedback retained.

## Microinteractions stance

- Silent success.
- No celebratory toasts.
- Hover is paired with keyboard focus.
- Focus rings are immediate and visible.
- Theme changes apply immediately after Settings are saved and persist through
  Forge's option store.
- An ON feature shows a pill in its accordion header ("켜짐", or "k/n 켜짐" for a
  panel holding several features) plus an accent border; a guessed ON control
  uses a dashed pill. Experimental features carry a "실험" tag, and a pinned row
  also shows its home group name. Chip reveal scrolls instantly, with no
  motion (reduced-motion safe).

## CTA voice

- Primary: solid signal colour, no gradient, compact radius, one-line label.
- Secondary: elevated neutral surface and visible boundary.
- Disabled/loading/error/success states retain text or another non-colour
  signal.

## Per-page allowances

- App pages use no decorative enrichment; function carries the page.
- The txt2img section layout can be turned off: Settings → SAM Extra Appearance
  → `sam3_layout_sections` (Forge restart), or `?sam3_lanes=off` for one page
  load.
- The smooth progress bar is opt-in: Settings → SAM Extra Progress Bar →
  `sam3_progress_enabled`. It inserts one `.sam3-progress` node per tab at
  mount, then writes only `data-*`/`aria-*` attributes, inline
  `--sam3-progress-*` properties and the `.data` of its two text nodes; Forge's
  `.progressDiv` is hidden, never removed.
- Existing preview images and generated galleries are content, not decoration.
- Third-party extensions may retain their own intentional brand surfaces, but
  common Gradio controls inherit this system.

## What pages MUST share

- Source Sans Pro / IBM Plex Mono.
- Surface elevation by lightness rather than shadow.
- The selected global palette and its one signal colour.
- Input, button, tab, focus, disabled, loading, error, and success language.
- Compact radii and the named spacing/motion tokens.

## What pages MAY differ on

- Existing Forge page composition and control density.
- Extension-specific information architecture.
- Generated-image and model-card content.

## Exports

### tokens.css

`tokens.css` is the canonical runtime export. It contains all colour, font,
spacing, radius, duration, easing, and elevation tokens plus the Gradio mapping.

### Tailwind v4 `@theme`

```css
@theme {
  --color-background: var(--sam3-color-paper);
  --color-foreground: var(--sam3-color-ink);
  --color-primary: var(--sam3-color-accent);
  --font-sans: var(--sam3-font-body);
  --font-mono: var(--sam3-font-mono);
  --spacing-md: var(--sam3-space-md);
  --radius-md: var(--sam3-radius-md);
}
```

### DTCG `tokens.json`

```json
{
  "color": {
    "paper": {"$value": "{sam3.color.paper}", "$type": "color"},
    "ink": {"$value": "{sam3.color.ink}", "$type": "color"},
    "accent": {"$value": "{sam3.color.accent}", "$type": "color"}
  },
  "space": {
    "md": {"$value": "1rem", "$type": "dimension"}
  }
}
```

### shadcn/ui CSS variables

```css
:root {
  --background: var(--sam3-color-paper);
  --foreground: var(--sam3-color-ink);
  --primary: var(--sam3-color-accent);
  --primary-foreground: var(--sam3-color-accent-ink);
  --muted: var(--sam3-color-paper-3);
  --muted-foreground: var(--sam3-color-muted);
  --border: var(--sam3-color-rule);
  --input: var(--sam3-color-rule);
  --ring: var(--sam3-color-focus);
}
```
