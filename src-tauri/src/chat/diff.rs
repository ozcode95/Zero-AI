//! Minimal line-level diff for the chat file-edit cards.
//!
//! Hand-rolled (rather than pulling a crate) to keep the dependency
//! surface small. Produces a unified-diff-style body with `+` / `-` / ` `
//! line prefixes and `@@` hunk headers, plus added/removed line counts for
//! the `+X -Y` badge the UI renders.
//!
//! The expensive part — a full LCS dynamic-programming table — is bounded:
//! we first strip the common prefix and suffix (so a one-line edit to a
//! huge file diffs in linear time), and only run the quadratic LCS on the
//! differing middle when it is small enough. Past that bound we fall back
//! to a "replace the whole middle" script, which is cheap and still
//! readable.

/// Number of unchanged context lines kept around each change in the
/// rendered hunks. Matches the conventional `diff -U3`.
const CONTEXT: usize = 3;

/// Cap on the rendered diff body. A single huge `fs.write` shouldn't be
/// able to blow out the chat bubble; the `+X -Y` counts still reflect the
/// full change even when the body is clipped.
const MAX_BODY_LINES: usize = 600;

/// Upper bound on the LCS table size (`old_mid.len() * new_mid.len()`).
/// Above this we use the linear replace-middle fallback. 4M cells of
/// `u32` is ~16 MiB transient — fine for an interactive edit, and only
/// hit by wholesale rewrites of very large files.
const MAX_LCS_CELLS: usize = 4_000_000;

/// Added / removed line tallies for a single file edit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DiffStat {
    pub added: usize,
    pub removed: usize,
}

/// A computed file diff: the `+X -Y` tally plus a unified-diff body ready
/// to drop inside a ```` ```diff ```` fence.
#[derive(Debug, Clone)]
pub struct FileDiff {
    pub stat: DiffStat,
    pub body: String,
    /// True when `body` was clipped at [`MAX_BODY_LINES`]; the UI appends a
    /// "diff truncated" hint.
    pub truncated: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Op {
    Equal,
    Delete,
    Insert,
}

/// Compute the line-level diff between `old` and `new`.
pub fn compute(old: &str, new: &str) -> FileDiff {
    let old_lines: Vec<&str> = split_lines(old);
    let new_lines: Vec<&str> = split_lines(new);
    let ops = diff_ops(&old_lines, &new_lines);

    let mut stat = DiffStat::default();
    for op in &ops {
        match op {
            Op::Delete => stat.removed += 1,
            Op::Insert => stat.added += 1,
            Op::Equal => {}
        }
    }

    let (body, truncated) = render_unified(&old_lines, &new_lines, &ops);
    FileDiff {
        stat,
        body,
        truncated,
    }
}

/// Split into lines without the trailing terminators. `str::lines` handles
/// both `\n` and `\r\n`; an empty input yields no lines.
fn split_lines(s: &str) -> Vec<&str> {
    if s.is_empty() {
        Vec::new()
    } else {
        s.lines().collect()
    }
}

/// Produce one [`Op`] per old/new line in output order. Trims the common
/// prefix/suffix first so the quadratic LCS only runs on the changed
/// middle; falls back to "delete-all + insert-all" of the middle when even
/// that is too large.
fn diff_ops(old: &[&str], new: &[&str]) -> Vec<Op> {
    // Common prefix.
    let mut p = 0;
    while p < old.len() && p < new.len() && old[p] == new[p] {
        p += 1;
    }
    // Common suffix (not overlapping the prefix).
    let mut s = 0;
    while s < old.len() - p && s < new.len() - p && old[old.len() - 1 - s] == new[new.len() - 1 - s]
    {
        s += 1;
    }

    let old_mid = &old[p..old.len() - s];
    let new_mid = &new[p..new.len() - s];

    let mut ops = Vec::with_capacity(old.len().max(new.len()));
    ops.extend(std::iter::repeat(Op::Equal).take(p));

    if old_mid.is_empty() {
        ops.extend(std::iter::repeat(Op::Insert).take(new_mid.len()));
    } else if new_mid.is_empty() {
        ops.extend(std::iter::repeat(Op::Delete).take(old_mid.len()));
    } else if old_mid.len().saturating_mul(new_mid.len()) <= MAX_LCS_CELLS {
        ops.extend(lcs_ops(old_mid, new_mid));
    } else {
        // Too big for the LCS table: replace the whole middle.
        ops.extend(std::iter::repeat(Op::Delete).take(old_mid.len()));
        ops.extend(std::iter::repeat(Op::Insert).take(new_mid.len()));
    }

    ops.extend(std::iter::repeat(Op::Equal).take(s));
    ops
}

/// Classic LCS backtrack producing an edit script that prefers deletions
/// before insertions on ties (stable, readable hunks).
fn lcs_ops(a: &[&str], b: &[&str]) -> Vec<Op> {
    let n = a.len();
    let m = b.len();
    // dp[i][j] = LCS length of a[i..] vs b[j..]. One extra row/col of zeros.
    let mut dp = vec![vec![0u32; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            dp[i][j] = if a[i] == b[j] {
                dp[i + 1][j + 1] + 1
            } else {
                dp[i + 1][j].max(dp[i][j + 1])
            };
        }
    }

    let mut ops = Vec::with_capacity(n + m);
    let (mut i, mut j) = (0usize, 0usize);
    while i < n && j < m {
        if a[i] == b[j] {
            ops.push(Op::Equal);
            i += 1;
            j += 1;
        } else if dp[i + 1][j] >= dp[i][j + 1] {
            ops.push(Op::Delete);
            i += 1;
        } else {
            ops.push(Op::Insert);
            j += 1;
        }
    }
    while i < n {
        ops.push(Op::Delete);
        i += 1;
    }
    while j < m {
        ops.push(Op::Insert);
        j += 1;
    }
    ops
}

/// Render the edit script as unified-diff hunks with [`CONTEXT`] lines of
/// surrounding context. Returns `(body, truncated)`.
fn render_unified(old: &[&str], new: &[&str], ops: &[Op]) -> (String, bool) {
    // Materialise each op into a tagged line, tracking how many old/new
    // lines precede it (so hunk headers get correct 1-based line numbers).
    struct Item<'a> {
        op: Op,
        text: &'a str,
        old_before: usize,
        new_before: usize,
    }
    let mut items: Vec<Item> = Vec::with_capacity(ops.len());
    let (mut oi, mut ni) = (0usize, 0usize);
    for &op in ops {
        match op {
            Op::Equal => {
                items.push(Item {
                    op,
                    text: old[oi],
                    old_before: oi,
                    new_before: ni,
                });
                oi += 1;
                ni += 1;
            }
            Op::Delete => {
                items.push(Item {
                    op,
                    text: old[oi],
                    old_before: oi,
                    new_before: ni,
                });
                oi += 1;
            }
            Op::Insert => {
                items.push(Item {
                    op,
                    text: new[ni],
                    old_before: oi,
                    new_before: ni,
                });
                ni += 1;
            }
        }
    }

    // Indices of changed lines.
    let changed: Vec<usize> = items
        .iter()
        .enumerate()
        .filter(|(_, it)| it.op != Op::Equal)
        .map(|(i, _)| i)
        .collect();
    if changed.is_empty() {
        return (String::new(), false);
    }

    // Group changes into hunks, merging when separated by <= 2*CONTEXT
    // equal lines (so adjacent edits share one hunk instead of repeating
    // context).
    let mut hunks: Vec<(usize, usize)> = Vec::new(); // inclusive item ranges
    let mut start = changed[0];
    let mut end = changed[0];
    for &c in &changed[1..] {
        if c - end <= 2 * CONTEXT {
            end = c;
        } else {
            hunks.push((start, end));
            start = c;
            end = c;
        }
    }
    hunks.push((start, end));

    let mut out = String::new();
    let mut body_lines = 0usize;
    let mut truncated = false;

    'hunks: for (h_start, h_end) in hunks {
        let lo = h_start.saturating_sub(CONTEXT);
        let hi = (h_end + CONTEXT).min(items.len() - 1);

        let mut old_count = 0usize;
        let mut new_count = 0usize;
        for it in &items[lo..=hi] {
            match it.op {
                Op::Equal => {
                    old_count += 1;
                    new_count += 1;
                }
                Op::Delete => old_count += 1,
                Op::Insert => new_count += 1,
            }
        }
        let old_before = items[lo].old_before;
        let new_before = items[lo].new_before;
        let old_start = if old_count > 0 {
            old_before + 1
        } else {
            old_before
        };
        let new_start = if new_count > 0 {
            new_before + 1
        } else {
            new_before
        };

        out.push_str(&format!(
            "@@ -{old_start},{old_count} +{new_start},{new_count} @@\n"
        ));
        body_lines += 1;

        for it in &items[lo..=hi] {
            if body_lines >= MAX_BODY_LINES {
                truncated = true;
                break 'hunks;
            }
            let prefix = match it.op {
                Op::Equal => ' ',
                Op::Delete => '-',
                Op::Insert => '+',
            };
            out.push(prefix);
            out.push_str(it.text);
            out.push('\n');
            body_lines += 1;
        }
    }

    // Drop the trailing newline so the ```diff fence closes cleanly.
    while out.ends_with('\n') {
        out.pop();
    }
    (out, truncated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_change_yields_empty_body_and_zero_stat() {
        let d = compute("a\nb\nc\n", "a\nb\nc\n");
        assert_eq!(
            d.stat,
            DiffStat {
                added: 0,
                removed: 0
            }
        );
        assert!(d.body.is_empty());
        assert!(!d.truncated);
    }

    #[test]
    fn new_file_counts_all_additions() {
        let d = compute("", "line1\nline2\nline3\n");
        assert_eq!(
            d.stat,
            DiffStat {
                added: 3,
                removed: 0
            }
        );
        assert!(d.body.contains("+line1"));
        assert!(d.body.contains("+line3"));
    }

    #[test]
    fn deleted_file_counts_all_removals() {
        let d = compute("a\nb\n", "");
        assert_eq!(
            d.stat,
            DiffStat {
                added: 0,
                removed: 2
            }
        );
        assert!(d.body.contains("-a"));
        assert!(d.body.contains("-b"));
    }

    #[test]
    fn single_line_change_in_middle() {
        let old = "one\ntwo\nthree\nfour\nfive\n";
        let new = "one\ntwo\nTWO_AND_A_HALF\nfour\nfive\n";
        let d = compute(old, new);
        assert_eq!(
            d.stat,
            DiffStat {
                added: 1,
                removed: 1
            }
        );
        // The unchanged head/tail are trimmed; the changed line shows both
        // a removal and an addition with context around it.
        assert!(d.body.contains("-three"));
        assert!(d.body.contains("+TWO_AND_A_HALF"));
        assert!(d.body.contains(" two")); // context line, space-prefixed
        assert!(d.body.contains("@@"));
    }

    #[test]
    fn pure_insertion_keeps_surrounding_context() {
        let old = "a\nb\nc\n";
        let new = "a\nb\ninserted\nc\n";
        let d = compute(old, new);
        assert_eq!(
            d.stat,
            DiffStat {
                added: 1,
                removed: 0
            }
        );
        assert!(d.body.contains("+inserted"));
    }

    #[test]
    fn distant_edits_split_into_two_hunks() {
        // 1..=20, edit line 2 and line 19 → two hunks, not one giant block.
        let old: String = (1..=20).map(|n| format!("L{n}\n")).collect();
        let mut lines: Vec<String> = (1..=20).map(|n| format!("L{n}")).collect();
        lines[1] = "L2_EDIT".into();
        lines[18] = "L19_EDIT".into();
        let new = format!("{}\n", lines.join("\n"));
        let d = compute(&old, &new);
        assert_eq!(
            d.stat,
            DiffStat {
                added: 2,
                removed: 2
            }
        );
        let hunk_headers = d.body.matches("@@ -").count();
        assert_eq!(hunk_headers, 2, "expected two separate hunks");
    }

    #[test]
    fn huge_rewrite_is_truncated_but_counts_are_exact() {
        let old: String = (0..2000).map(|n| format!("old{n}\n")).collect();
        let new: String = (0..2000).map(|n| format!("new{n}\n")).collect();
        let d = compute(&old, &new);
        assert_eq!(d.stat.added, 2000);
        assert_eq!(d.stat.removed, 2000);
        assert!(d.truncated);
        assert!(d.body.lines().count() <= MAX_BODY_LINES + 1);
    }

    #[test]
    fn hunk_header_line_numbers_are_one_based() {
        let old = "a\nb\nc\n";
        let new = "a\nB\nc\n";
        let d = compute(old, new);
        // Change on line 2 (1-based), 3 lines of context clamp to the file.
        assert!(d.body.starts_with("@@ -1,3 +1,3 @@"));
    }
}
