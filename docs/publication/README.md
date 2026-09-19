# Publication

**We are not writing the paper yet.** This directory holds the argument we intend to make, the
evidence that would support it, the figures that would carry it, and the runs still outstanding.
Nothing here is draft prose, and nothing here should become draft prose — when the story stops
moving, a manuscript gets written *from* these documents, somewhere else.

## The four documents

| doc | answers |
|---|---|
| **STORY.md** (local only) | What are we claiming? Three legs — the tool, why it is built that way, why the flow is solved — their strength, and what we must never claim. **Not in this repo**: it is the pre-publication claim document, held back until the paper is out. Everything it asserts is checked by `verify_claims.py`, whose ledger is public. |
| **[EVIDENCE.md](EVIDENCE.md)** | Is each number real? The claim-id ledger, the conventions that change numbers, and every retraction |
| **[FIGURES.md](FIGURES.md)** | What should we draw? Per leg: the argument each figure must carry, its generator, its placement |
| **[EXPERIMENTS.md](EXPERIMENTS.md)** | What should we run next? Ranked by what each run would settle |

Plus **[RELATED_WORK.md](RELATED_WORK.md)** — who else is in this space, what they have already
published, and the triage discipline that exists because we once missed a paper in our own target
venue.

## How they fit together

```
STORY.md          the claim          -- every leg names its evidence and its gaps
   |
   +-- EVIDENCE.md      can we prove it?    -- claim ids, artifacts, retractions
   +-- FIGURES.md       can we show it?     -- one figure per argument, not per result
   +-- EXPERIMENTS.md   what's missing?     -- the queue, ranked by what it settles
   +-- RELATED_WORK.md  has someone else?   -- and what remains ours
```

## The two rules that make this work

**1. A number without a claim id does not enter any document here.**

```bash
PYTHONPATH=. python scripts/publication/verify_claims.py --paper docs/publication/STORY.md
```

Exit code is the number of failures. Add the ledger row *before* quoting the number, not after.
[EVIDENCE.md](EVIDENCE.md) §1 explains the mechanism and why it exists.

**2. Status is stated, never implied.** Every leg in STORY.md (local only) carries an honest
strength label, and the weak ones say so in the same voice as the strong ones. A claim that reads
as settled when it rests on n=6 is the claim that loses a review.

## The three legs, in one line each

| leg | claim |
|---|---|
| **0 — the tool** | mesh in (injured or intact), clot field out, 58.9 s against ~48 h; scored by **BATC**, a burden-adjusted metric designed for this problem; shipped as a double-click app |
| **1 — why it is built this way** | what to solve vs. learn is empirical and flips *inside* one component, and the architecture choices are measured against from-scratch and depth-matched controls |
| **2 — why the flow is solved** | an oracle closed loop is flat, so one solve suffices; one solve caps any surrogate at 7.9% of runtime; so the choice is decided on accuracy, and FEM wins it |

## What was superseded

| was | now |
|---|---|
| `docs/PAPER.md` — manuscript-shaped narrative | archived to [../archive/PAPER.md](../archive/PAPER.md). Its argument is STORY.md (local only); its number-checking discipline is [EVIDENCE.md](EVIDENCE.md); its section-to-evidence map is split across [FIGURES.md](FIGURES.md) and [EVIDENCE.md](EVIDENCE.md) |
| `docs/PHYSICS_ABLATION_PLAN.md` | archived to [../archive/PHYSICS_ABLATION_PLAN.md](../archive/PHYSICS_ABLATION_PLAN.md). Results live in STORY.md (local only) Legs 1 and 3 |
| `docs/PUBLICATION_PLAN.md`, `docs/PUBLICATION_NOTES.md` | left in place as **local-only working journals** (gitignored by deliberate choice). Their durable content — retractions, conventions, open runs — was carried into [EVIDENCE.md](EVIDENCE.md) and [EXPERIMENTS.md](EXPERIMENTS.md) |

Source-code comments that cite the old filenames as provenance still resolve: the tracked ones
are under `docs/archive/`, the journals are where they always were.
