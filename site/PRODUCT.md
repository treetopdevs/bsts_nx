# Product

## Register

brand

## Users

Technically curious people — engineers, data scientists, product managers, and
researchers — who want to *understand* how BstsNx works, not just call an API.
They are evaluating whether a Bayesian structural time series library is
trustworthy and capable enough to adopt in an Elixir codebase. They arrive
skeptical of unverified performance claims and want to watch the model work,
live, on data where the true answer is already known.

## Product Purpose

A public marketing/tutorial/showcase site for BstsNx (Bayesian Structural Time
Series in pure Elixir). It exists to prove the library works by running it
live, server-side, on every page — not to describe it in prose. Every demo
plants a known truth in synthetic data, estimates it, then reveals and grades
the estimate in public. Success looks like a visitor leaving convinced the
library is statistically honest and capable, with a clear path (`/start`) to
trying it themselves.

## Brand Personality

Honest and rigorous. Plain-spoken like the source livebooks: "The black line
is what actually happened." Never hypes — "blazingly", "powerful",
"seamless", and "magic" are banned words. Uncertainty is a feature, not
something to hide: verdicts say things like "Honestly? Can't tell — the
interval straddles zero." The register is a technical report that happens to
be interactive, not a sales pitch.

## Anti-references

- Generic SaaS gradient-hero template: big number + small label + gradient
  accent, fabricated big-number stat blocks, stock illustrations.
- Enterprise navy-and-gold corporate / data-vendor look.
- Flashy fintech/crypto aesthetic: neon-on-black, glassmorphism, hype-driven
  copy.

## Design Principles

- Show, don't tell: every claim on the site is a live computation the visitor
  just watched run, never a screenshot or a hardcoded number.
- Plant the truth, grade in public: synthetic data with a known answer,
  revealed and graded after the estimate, on every demo.
- One accent, one meaning: the "lift" color is reserved exclusively for
  estimated effects; the "truth" color exclusively for planted ground truth.
  Semantic color is never decorative.
- Non-linear by design: no forced funnel. A hub with three doors (Questions,
  Engine, Trust) that cross-link into each other; the story does not require a
  single reading order.
- Honesty over confidence: uncertainty (wide intervals, "can't tell"
  verdicts, forbidden-live latency tiers) is shown, not smoothed over.

## Accessibility & Inclusion

Standard WCAG AA. Keep the existing implementation intact: visible focus
states, `prefers-reduced-motion` respected (the line-draw figure animation is
gated on it), responsive down to 375px, sufficient contrast on the
graph-paper palette.
