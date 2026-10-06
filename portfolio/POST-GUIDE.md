# How to write the mettle portfolio post

This is a **brief**, not the post. It tells you how to write the one long-form
essay about mettle that goes on the portfolio site
(<https://adityam23.github.io>). Do not commit prose here — write the actual
post in a *separate* file (see "Where the post file goes" below).

> This guide file is deliberately safe to leave in `portfolio/`: the site
> aggregator only publishes `portfolio/*.md` files that **start with `---` YAML
> front matter**. This file starts with a `#` heading, so it is skipped and
> never appears on the site. Keep it that way — never put front matter at the
> top of this guide.

---

## 1. The voice you're matching

The site is styled as Unix **man pages**, and every post is a long-form
**essay / postmortem**, not a tutorial or a README. Before writing, read the two
reference posts to internalize the register:

- `_posts/2026-05-17-9-Classes-5-Weeks.md` — a technical build postmortem.
- `_posts/2026-08-03-ai-safety-rounding-error.md` — a researched argument.

Rules distilled from them:

- **First person, narrative.** You are telling the story of building mettle and
  the decisions behind it.
- **Section headings are sentences about the work, not labels.** e.g. "Treating
  VRAM as a scheduled resource" — not "Architecture".
- **Concrete detail, honestly.** Real constraints (the 6 GB card), real tradeoffs,
  what you tried that didn't work. The reference posts dwell on dead ends; do the
  same.
- **Essay, not how-to.** The home page says "all entries below are essays, not
  tutorials." Don't write install/usage steps; write about *why the design is the
  way it is* and what building it taught you.
- Don't open with a `---` horizontal rule or the layout reads it as front matter.
  Open with a Markdown `#` H1 that repeats the title.

## 2. The angle for mettle (fill these in — don't just describe features)

mettle is a local inference engine for GGUF models: Rust around an embedded
llama.cpp core (`llama-cpp-2` FFI), one daemon serving chat / completions /
embeddings / rerank / vision-OCR over OpenAI-, Ollama- and Anthropic-compatible
APIs, scheduling models against a fixed VRAM budget instead of dying on it. The
README is the *spec*; the post is the *story*. Pick a spine and commit.

The strongest spine, given the reference posts: **"Making a small GPU behave like
a scheduler, not a cliff."** The thesis in the README is that a 6 GB card can run
a chat generator + an embedding model + an OCR model *concurrently* because the
profile manager treats VRAM as a scheduled resource. That is the interesting
story — build the essay around it.

Questions the post should answer (turn each into narrative, not bullets):

- Why naive local inference "dies on" a small GPU — the failure mode you set out
  to eliminate (OOM on load, one model per card, no headroom).
- **Slots, not processes.** `mettled` embeds llama.cpp behind a channel-driven
  worker; models are slots loaded/unloaded atomically by alias. Why this design
  over spawning processes? What did it buy you and what was hard about the FFI
  boundary?
- **The degradation ladder.** A profile declares ordered slots with roles;
  admission compares header-derived cost against free VRAM minus a safety margin;
  under pressure it retries smaller context rungs down to each slot's `min_ctx`,
  then drops the slot. Walk through a concrete pressure scenario. This is the
  centerpiece — give it room.
- **Measuring instead of guessing.** Post-load footprints are measured and
  persisted in `state.json` (invalidated by binary version or GGUF mtime) so
  second activations fit precisely. Why estimation wasn't enough; the story of
  moving from predicted to measured VRAM.
- Chunked prefill (decode long prompts across bounded batches instead of
  reserving worst-case) and async profile activation over the SSE event bus —
  include if they carried weight in the design.
- **One core, many API dialects.** Why serve OpenAI + Ollama + Anthropic + Cohere
  rerank shapes from one engine? What was annoying about matching three
  ecosystems' quirks (logprobs, `reasoning_content`, chat templates from embedded
  jinja)?
- On benchmarks: the README deliberately publishes **no numbers** ("meaningless
  without your model and GPU") and ships `mettle bench` instead. That's a real
  opinion — you can defend it in the post rather than faking a leaderboard.
- The docket relationship: mettle is the supported local engine behind docket. If
  docket also gets a post, cross-link them (plain Markdown link).
- At least one honest dead end — a scheduling or FFI approach you abandoned, and
  why.

Keep any tables small and in service of the argument.

## 3. Where the post file goes — the two publishing paths

There are **two ways** to publish, and you must pick one. Both are documented in
full in the site repo's `PUBLISHING.md`; the summary:

### Path B — source lives in THIS repo (recommended for mettle)

Use this so the post travels with the code. Write the post as a **new** file:

```
portfolio/<name>.md      e.g. portfolio/vram-as-a-scheduled-resource.md
```

The filename is slugified for the URL (`VRAM Budget.md` → `vram-budget`). Give it
front matter (this is the real post, so front matter IS at the top):

```yaml
---
title: The Title Of The Post
description: One-line summary shown on the index and in the NAME section.
date: 2026-09-02
repo_url: https://github.com/adityam23/mettle
permalink: /projects/mettle/<slug>/
---
```

- `date:` is mandatory and must be `YYYY-MM-DD` (10 chars). Without it the
  aggregator stamps the build day and the URL changes on every build.
- Then register this repo once in the **site repo's** `projects.yml`:

  ```yaml
  projects:
    - repo: adityam23/music-gen
    - repo: adityam23/mettle
  #    branch: master   # only if the default branch isn't main
  ```

- **Publishing is triggered by the site repo, not this one.** Pushing to mettle
  alone does nothing. Push a change to `adityam23.github.io`, OR hit "Run
  workflow" on its Actions tab, OR re-enable the daily cron in its `build.yml`.
  The aggregator then clones mettle, copies `portfolio/*.md` into `_posts/`, and
  the post goes live.

### Path A — source lives in the SITE repo instead

If you'd rather keep this as a standalone essay decoupled from the mettle repo,
skip `portfolio/` entirely and create the file directly in the site repo at
`_posts/YYYY-MM-DD-<slug>.md` with the same front matter (drop `repo_url` if you
don't want the source link). The aggregator never touches hand-authored
`_posts/` files. Pushing the site repo publishes it. Use Path A only if you
*don't* want the post to live with mettle's code — for mettle, Path B is the
natural choice.

## 4. Images

If the post needs figures (a slot/profile diagram, the degradation ladder):

- Put them in `portfolio/assets/` in **this** repo and reference them
  **relatively**: `![alt](assets/ladder.png)` or `![alt](./assets/ladder.png)`.
  The aggregator copies them into `assets/projects/mettle/` on the site and
  rewrites the paths automatically.
- Absolute `https://` image URLs are left untouched, so both styles can coexist.
- Do **not** hand-place images in the site's `assets/projects/` — that folder is
  wiped (`rm -rf`) on every build.

## 5. Comments (giscus) — linking giscus.app to the right repo

Comments are [giscus](https://giscus.app): each thread is stored as a GitHub
Discussion. The site's `post` layout already contains the giscus `<script>`; it
only renders when the ID fields are present. You provide four values that the
layout injects into the script tag's `data-repo`, `data-repo-id`,
`data-category`, and `data-category-id` attributes.

**Which repo holds the discussions?** Recommended: centralize on the site repo
(`adityam23/adityam23.github.io`) so every post's comments live in one place and
are moderated together. giscus maps threads by page `pathname`, so each post
still gets its own thread even when they share one repo. (You *could* instead
point mettle's posts at `adityam23/mettle`'s own Discussions — same procedure,
just a different repo.)

### One-time setup on the repo that will hold discussions

1. **Enable Discussions**: repo **Settings → General → Features → check
   "Discussions"**. (giscus requires this — it will refuse otherwise.)
2. **Install the giscus GitHub App**: <https://github.com/apps/giscus> → Install
   → grant it access to that repo. (Required for giscus to read/write
   Discussions.)
3. The repo must be **public** (it is).
4. Make sure a Discussion **category** exists to post into. An
   "Announcements"-type category (maintainer-only new threads) is ideal so only
   giscus opens threads. `Announcements` is the layout default.

### Get the IDs from giscus.app

Go to <https://giscus.app> and fill the **Configuration** form:

| giscus.app field | What to enter |
|---|---|
| **Repository** | `adityam23/adityam23.github.io` (type `owner/repo`). giscus.app validates it live and shows a green check only if the repo is public, has the app installed, and has Discussions on — if you see an error, fix the step above it names. |
| **Page ↔ Discussions Mapping** | Choose **"Discussion title contains page `pathname`"**. The layout is hardcoded to `data-mapping="pathname"`, so this must match. |
| **Discussion Category** | `Announcements` (or the category you made). |
| Features / theme | Leave default — the layout already sets reactions, theme, input position. |

giscus.app then generates a full `<script>` snippet. **Do not copy the whole
snippet.** From it, take only the two computed IDs:

- `data-repo-id="R_…"`     → this is your **`giscus_repo_id`**
- `data-category-id="DIC_…"` → this is your **`giscus_category_id`**

(`data-repo` = the `owner/repo` slug you typed; `data-category` = the category
name — you already know both.)

### Put the four values where the layout reads them

**Recommended — set once for all posts** in the site repo's `_config.yml`
`defaults` block, so no post repeats them:

```yaml
defaults:
  - scope:
      path: ""
      type: "posts"
    values:
      layout: "post"
      giscus_repo: "adityam23/adityam23.github.io"
      giscus_repo_id: "R_xxxxxxxxxxxx"          # data-repo-id from giscus.app
      giscus_category: "Announcements"
      giscus_category_id: "DIC_xxxxxxxxxxxx"     # data-category-id from giscus.app
```

**Or per-post**, if you only want comments on some posts: put the same four keys
in this post's front matter instead.

The IDs are **not secrets** — they're visible in page source by design, safe to
commit. If the site owner has already done this one-time setup once, you don't
repeat it; comments will just work.

## 6. Final checklist

- [ ] Wrote the post as a *new* file (not this guide), front matter at the top.
- [ ] `date:` set to a real `YYYY-MM-DD`.
- [ ] `permalink:` under `/projects/mettle/…`, `repo_url:` set (Path B).
- [ ] Registered `adityam23/mettle` in the site repo's `projects.yml`.
- [ ] Details are real; at least one honest dead end included.
- [ ] Triggered a site-repo build (push there / Run workflow) — pushing mettle
      alone does nothing.
- [ ] Confirmed it's live at `https://adityam23.github.io/projects/mettle/<slug>/`.
