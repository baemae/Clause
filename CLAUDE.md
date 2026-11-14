# Project: Clause
Clause is an AI-powered search over your own documents (RAG). Ask a question
in plain language and get a clear answer with sources. Already deployed; this
task is only the landing page.

## Stack
- Landing page lives in `web/`: HTML, CSS, vanilla JS only.
- Don't touch `api/` or any Python files.
- No external images (CSS gradients only).
## Visual style
Soft pastel glassmorphism, matching the reference screenshots.
- Colors (use CSS variables): ink #1f2340, muted #5d6283, pink #ff9fb8,
  lilac #b7a5f7, sky #8fb8ff, mint #a6e6d6, soft blue-lilac page background.
- Cards: frosted glass (translucent white, 1px light border, backdrop-blur 18px),
  28px radius, soft purple-tinted shadow.
- Buttons: pill shape, pink-lilac-sky gradient.
- Fonts: Bricolage Grotesque (headings), DM Sans (body).
- Support dark mode via prefers-color-scheme.

## Sections
Nav, hero (headline, 2 buttons, 3 mini features, gradient art card),
features (3 tabs + stat tiles), recent work (4 gradient tiles), email CTA.

## Animations
- Hero headline words rise in one by one on load.
- Art card tilts with the mouse; orb floats; slow background blobs drift.
- Button gradient slides on hover; nav underline draws in.
- Tab cards slide in; progress bar fills; stats count up on scroll.
- Sections fade up on scroll; work tiles expand on hover.
- Form shows a toast on submit.

## Rules
- Mobile-friendly (stack columns under 820px).
- Visible keyboard focus; respect prefers-reduced-motion.
- Build in small steps and show me what changed. Don't rewrite whole files
  for small edits.