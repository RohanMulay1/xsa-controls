# Anonymization checklist for double-blind submission

**Nothing here has been applied.** This is the inventory and the procedure.
Anonymizing in place would break CI, the cross-repository drift check and the
manifest's commit provenance, so the anonymous artifact is produced as a
separate archive from a scratch clone and the working repositories are left
alone.

Verified against `xsa-controls` at 34 commits and the sibling report
repository, 2026-09-05, by grepping tracked content, git metadata and the
rendered PDF. Re-run the commands in **Verifying** before submitting, because
this list goes stale the moment a commit adds a URL.

## What actually carries identity

The scan result is worth stating plainly, because it decides how much work
this is: **no tracked file in `xsa-controls` contains a name, an email or a
personal URL.** Every identifying string is in metadata or in the sibling
report repository.

### 1. Git metadata — `xsa-controls`, all 34 commits

| Surface | Value | Where |
|---|---|---|
| Author name | `Rohan Mulay`, `RohanMulay1` | every commit |
| Author email | `rohanm1307@gmail.com`, `diyaj11205@gmail.com` | every commit |
| Committer | `GitHub <noreply@github.com>` | merge commits |

Three distinct identities appear because early commits used a different
address. All three must be rewritten, not just the most common one.

### 2. Repository and remote identity

| Surface | Value |
|---|---|
| Clone URL | `github.com/RohanMulay1/xsa-controls` |
| Sibling repo referenced in prose | `github.com/ishaannk/crpa`, PR #1 |
| Merge commit subjects | "Merge pull request #N from RohanMulay1/..." |

The merge subjects are the easy one to miss: they name the fork owner inside
the commit message, so an author rewrite alone does not remove them.

### 3. Files naming infrastructure

| File | What it carries |
|---|---|
| `RUN_IN_PROGRESS.md` | RunPod pod id, GPU vendor, hourly rate |
| `AUDIT.md`, `FINAL_STATUS.md`, `README.md` | provider name and rates |

A cloud provider and an hourly rate are not identifying on their own. They
are listed because together with a timestamp they narrow the field, and
because a reviewer who recognises the spend pattern from a public thread has
effectively deanonymised the work.

### 4. The sibling report repository

| File | Line | What |
|---|---|---|
| `build_pdf.py` | 184 | `author="Rohan Mulay"` — **written into the PDF metadata** |
| `README.md` | 9 | `https://github.com/RohanMulay1/xsa-controls` |
| `.github/workflows/ci.yml` | 26, 31 | `RohanMulay1/crpa`, `RohanMulay1/xsa-controls` |
| `QA_CHECKLIST.md` | — | repository paths and owner |
| `report_values.py` | `SOURCES` | source commit hashes, which resolve to public repos |
| `attention-research-results.pdf` | metadata | `/Author: Rohan Mulay` |

**The PDF is the one that matters most.** A reader does not need to open the
repository: `pdfinfo` on the submitted artifact prints the author. Everything
else requires someone to go looking.

### 5. Not present, checked anyway

No `claude.ai/code/session` links, no `Co-Authored-By` trailers, no API keys,
no SSH host keys, no absolute home-directory paths in tracked files. The
LICENSE says `Copyright (c) 2026` with no name.

## Producing the anonymous archive

Run from a scratch directory. Never in the working repositories.

```bash
# 1. Scratch clone, full history
git clone --no-local ~/xsa-controls /tmp/anon-xsac
cd /tmp/anon-xsac

# 2. Rewrite every author and committer identity, all three variants
git filter-repo --force \
  --name-callback 'return b"Anonymous Author"' \
  --email-callback 'return b"anon@example.invalid"'

# 3. Strip the fork owner out of merge subjects
git filter-repo --force --message-callback '
  return message.replace(b"RohanMulay1/", b"anon/").replace(b"ishaannk/", b"anon/")'

# 4. Content substitutions
grep -rl "RohanMulay1\|ishaannk" --exclude-dir=.git . \
  | xargs -r sed -i 's#github.com/RohanMulay1#github.com/ANONYMISED#g; s#ishaannk#ANONYMISED#g'

# 5. Infrastructure detail: replace the provider with a generic description
sed -i 's/RunPod/a commercial GPU cloud/g; s/pod xlbfc44jppqxyd/pod [id redacted]/g' \
  RUN_IN_PROGRESS.md AUDIT.md FINAL_STATUS.md README.md

# 6. The report repository, same treatment plus the PDF author
git clone --no-local ~/attention-research-report /tmp/anon-report
cd /tmp/anon-report
sed -i 's/author="Rohan Mulay"/author="Anonymous"/' build_pdf.py
sed -i 's#RohanMulay1#ANONYMISED#g' README.md QA_CHECKLIST.md .github/workflows/ci.yml
python build_pdf.py            # regenerate so the PDF metadata is rewritten

# 7. Archive
cd /tmp && zip -r anon-artifact.zip anon-xsac anon-report -x '*/.git/*'
```

Step 7 excludes `.git` deliberately. Shipping history is the largest
deanonymisation surface and a reviewer needs the tree, not the log. If the
venue requires history, ship the rewritten `.git` from steps 2-3 and verify
it with the commands below.

## Verifying before submission

Every one of these must return nothing.

```bash
cd /tmp/anon-xsac   # and again in /tmp/anon-report
git log --format='%an %ae %cn %ce' | sort -u | grep -viE 'anonymous|example.invalid'
git log --format='%s%n%b' | grep -iE 'rohan|ishaannk|RohanMulay1'
grep -rniE 'rohan|ishaannk|RohanMulay1|rohanm1307|diyaj11205' --exclude-dir=.git .
grep -rn 'claude.ai/code/session' --exclude-dir=.git .
```

And on the PDF specifically, because it is submitted on its own:

```bash
python -c "import fitz; print(fitz.open('attention-research-results.pdf').metadata)"
pdfinfo attention-research-results.pdf | grep -i author
```

`report_values.py` keeps the source commit hashes under `SOURCES`. Those are
40-hex strings that resolve to public repositories if anyone tries them.
Either blank them for the anonymous build or accept them: they are only
identifying to someone who already suspects which repositories to check.

## What this costs

The drift check in the report repository's CI checks out both source
repositories by their public name. In an anonymous clone that step will fail.
That is the correct trade: the check exists to keep the two repositories
honest during development, and the submitted artifact is a snapshot rather
than a living build.
