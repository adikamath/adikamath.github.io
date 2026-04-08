# Decision Log

This log records important product, design, and technical decisions for the blog.

Use it when we need to understand:
- why a decision was made
- when it was made
- when it took effect
- what parts of the site it impacted

## How To Use This Log

- Add a new entry for decisions that materially affect information architecture, content modeling, UX patterns, SEO/shareability, deployment, or operational debugging.
- Keep entries short. If a decision needs long analysis, link the supporting doc or PR instead of expanding the log.
- Do not use this log for routine refactors, copy edits, or low-risk visual tweaks.
- Append new decisions to the top so the latest decisions are easiest to find.

## Entry Template

```md
## DL-0000 Short Decision Title

- Status: Proposed | Accepted | Superseded
- Decision date: YYYY-MM-DD
- Effective date: YYYY-MM-DD or "On merge to main"
- Area: Architecture | Content | UX | Infra | Ops | SEO
- Decision: One sentence describing what we decided.
- Why: One or two sentences with the rationale.
- Impact: One or two sentences covering consequences, tradeoffs, or affected systems.
- Links: PR, issue, doc, or file paths if helpful.
```

## Decisions

## DL-0004 Improve page metadata and social previews with explicit post summaries and a site social card

- Status: Accepted
- Decision date: 2026-04-07
- Effective date: On merge to main
- Area: SEO
- Decision: Replace placeholder site metadata with explicit descriptions, normalize post front matter, and use a purpose-built 1200x630 social card as the default Open Graph image.
- Why: Search snippets and shared links were showing placeholder copy and inconsistent previews, which made the blog feel unfinished and reduced clarity before a visitor even landed on the site.
- Impact: Home, archive, topic, about, and post pages now emit cleaner metadata; posts use real summaries and images; and the default site preview is visually branded without changing the publishing workflow.
- Links: `_config.yml`, `_includes/header.html`, `assets/img/og-card.png`, `_posts/`

## DL-0003 Reframe the blog around guided discovery instead of a flat recent-post list

- Status: Accepted
- Decision date: 2026-04-07
- Effective date: On merge to main
- Area: UX
- Decision: Replace the flat homepage index with a simpler landing page built around a featured `Start Here` story, a streamlined `Recent Writing` list, richer archive/topic article cards, and a separate topics route in navigation.
- Why: The old homepage and archive worked as raw indexes, but they did not help first-time readers understand what the site covers or where to start; after iteration, homepage topic chips and extra landing-page actions added noise instead of clarity.
- Impact: The landing page is now more focused, article lists emphasize title and summary before inline metadata, topics remain available through navigation and the dedicated topics page, and tag destinations stay normalized through consistent slugs.
- Links: `_data/menus.yml`, `_includes/author.html`, `_layouts/home.html`, `archive.html`, `tags.html`, `_sass/klise/_layout.scss`

## DL-0002 Use progressive enhancement for navigation accessibility and long-form reading aids

- Status: Accepted
- Decision date: 2026-04-07
- Effective date: On merge to main
- Area: UX
- Decision: Replace the checkbox-driven mobile menu and anchor-based theme switch with button controls, then enhance post pages with read time, a generated table of contents, and back-to-top navigation.
- Why: The prior interactions were visually serviceable but semantically weak, and long technical posts needed more wayfinding without introducing risky build-time plugins.
- Impact: Mobile navigation now exposes better ARIA state, theme toggling is button-based, posts surface reading time consistently, and the table of contents is generated client-side only when enough headings exist.
- Links: `_includes/navbar.html`, `assets/js/main.js`, `_layouts/post.html`, `_sass/klise/_layout.scss`, `_sass/klise/_post.scss`

## DL-0001 Create and store a project decision log in the repo

- Status: Accepted
- Decision date: 2026-04-07
- Effective date: On merge to main
- Area: Ops
- Decision: Store the blog decision log in `docs/decision-log.md` as a Markdown file tracked in git.
- Why: The site now has explicit UX and metadata conventions that are easy to forget, and the repo needs one durable place to record why those choices were made.
- Impact: Future changes to navigation, SEO, post structure, and publishing polish can reference a single source of truth instead of rediscovering intent from diffs alone.
- Links: `docs/decision-log.md`
