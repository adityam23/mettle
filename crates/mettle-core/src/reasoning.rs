//! Reasoning-channel separation for `/v1/chat/completions`.
//!
//! The contract (agent-docs/downstream-consumers.md, "Serving different models
//! properly") requires `choices[].message.content` to be the final answer only.
//! Reasoning models (`<think>`-style Qwen distills and `<analysis>`/`<answer>`
//! distills) leak their thinking channel into `content` unless the backend
//! splits it out into `message.reasoning_content`.
//!
//! This module is the single splitter used by both the streaming and
//! non-streaming chat paths. It is deliberately tag-driven and text-based:
//! it runs on whatever the sampler produced, independent of how the prompt
//! was framed.

/// Result of splitting a complete generation into reasoning + answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Split {
    /// Thinking-channel text without its tags, when the model emitted one.
    pub reasoning: Option<String>,
    /// Final answer text — what `message.content` must carry.
    pub content: String,
}

const THINK_TAGS: [&str; 3] = ["think", "thinking", "reasoning"];
/// Bare tag names whose opener may start a thinking block (`<analysis>`
/// routes through the answer-tag layout below). Used for opener matching and
/// for the "could this still become an opener?" ambiguity check.
const ALL_TAGS: [&str; 4] = ["think", "thinking", "reasoning", "analysis"];

/// Split a complete generation into (reasoning, answer).
///
/// Recognised layouts:
/// - `<think>…</think>answer` (also `<thinking>`, `<reasoning>`)
/// - `<analysis>…</analysis><answer>…</answer>` (Claude-distill style)
/// - an *unclosed* opener: everything after `<think>` is reasoning and content
///   stays empty — truncated generations must not leak thinking into content
/// - no tags at all → unchanged content, no reasoning
pub fn split_reasoning(text: &str) -> Split {
    let rest = text.trim_start();
    for tag in THINK_TAGS {
        if let Some(body) = strip_open(rest, tag) {
            let (reasoning, answer) = split_at_close(body, tag);
            let reasoning = reasoning.trim();
            return Split {
                reasoning: (!reasoning.is_empty()).then(|| reasoning.to_string()),
                content: answer.to_string(),
            };
        }
    }
    if let Some(body) = strip_open(rest, "analysis") {
        let (reasoning, after_close) = split_at_close(body, "analysis");
        let reasoning = reasoning.trim();
        // Answer lives in <answer>…</answer>; anything outside the pair is
        // ignored — the consumer treats content strictly as the answer field.
        let answer = if let Some((_, open_end)) = find_tag(after_close, "answer") {
            let (body, _) = split_at_close(&after_close[open_end..], "answer");
            body
        } else {
            ""
        };
        return Split {
            reasoning: (!reasoning.is_empty()).then(|| reasoning.to_string()),
            content: answer.to_string(),
        };
    }
    Split {
        reasoning: None,
        content: text.to_string(),
    }
}

/// Strip an opening tag `<tag>`.
fn strip_open<'a>(s: &'a str, tag: &str) -> Option<&'a str> {
    s.strip_prefix(&format!("<{tag}>"))
}

/// Split `s` at the first closing tag. Returns (before, after-closing-tag).
/// Without a closing tag, returns the whole input and an empty slice — the
/// unclosed-think case keeps reasoning out of content.
fn split_at_close<'a>(s: &'a str, tag: &str) -> (&'a str, &'a str) {
    match find_tag(s, &format!("/{tag}")) {
        Some((start, end)) => (&s[..start], &s[end..]),
        None => (s, ""),
    }
}

/// Find `<needle>` anywhere in `s`; returns (start, end-exclusive) offsets.
fn find_tag(s: &str, needle: &str) -> Option<(usize, usize)> {
    let pat = format!("<{needle}>");
    let start = s.find(&pat)?;
    Some((start, start + pat.len()))
}

/// Streaming filter: feeds incremental deltas, returns only answer-side text,
/// and never emits a byte twice or drops answer bytes when a structural tag
/// spans chunk boundaries.
///
/// Reasoning deltas are consumed silently — the streamed surface carries the
/// answer channel only, mirroring the non-streaming shape where thinking rides
/// on `message.reasoning_content`, never inside streamed `content`.
#[derive(Debug)]
pub struct ReasoningFilter {
    phase: Phase,
    buf: String,
    /// Bytes of `buf` already returned to the caller (Answer phase only).
    emitted: usize,
    /// Whether the most recent [`Self::push`] emission included bytes held
    /// back from earlier pushes. Callers streaming per-token logprobs must
    /// not attach one to such a chunk: its text spans several tokens, so any
    /// single-token logprob would be a fabricated pairing.
    merged_last: bool,
}

#[derive(Debug, Clone, Copy)]
enum Phase {
    /// Buffering until the presence/absence of a reasoning opener is decided.
    Detect,
    /// Inside a thinking block tagged `tag`; text is dropped, watching for
    /// `</tag>`.
    Reasoning { tag: &'static str },
    /// `</analysis>` seen: discard filler until `<answer>` opens.
    WaitForAnswer,
    /// Streaming answer text; optionally stops at (and drops) `</closer>`.
    Answer {
        closer: Option<&'static str>,
        spent: bool,
    },
}

impl Default for ReasoningFilter {
    fn default() -> Self {
        Self {
            phase: Phase::Detect,
            buf: String::new(),
            emitted: 0,
            merged_last: false,
        }
    }
}

impl ReasoningFilter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one delta; returns the answer-side text to stream now (may be
    /// empty). Call [`Self::finish`] at end of stream to flush held-back text.
    pub fn push(&mut self, delta: &str) -> String {
        let prev_buf_len = self.buf.len();
        self.merged_last = false;
        let out = match self.phase {
            Phase::Detect => {
                self.buf.push_str(delta);
                self.detect(prev_buf_len)
            }
            Phase::Reasoning { tag } => {
                self.buf.push_str(delta);
                self.leave_reasoning(tag, prev_buf_len)
            }
            Phase::WaitForAnswer => {
                self.buf.push_str(delta);
                self.wait_for_answer(prev_buf_len)
            }
            Phase::Answer { .. } => {
                self.buf.push_str(delta);
                let prev_emitted = self.emitted;
                let out = self.answer_text(prev_buf_len);
                // Exact offsets: the emission includes bytes buffered by an
                // earlier push iff it started before this push's bytes began.
                if !out.is_empty() && prev_emitted < prev_buf_len {
                    self.merged_last = true;
                }
                out
            }
        };
        out
    }

    /// Whether the last [`Self::push`] emitted text accumulated from earlier
    /// pushes. Such a chunk spans several tokens; attach no per-token logprob.
    pub fn last_push_was_merged(&self) -> bool {
        self.merged_last
    }

    /// End of stream: flush whatever legitimately remains. An unclosed
    /// thinking block must never leak into `content`.
    pub fn finish(&mut self) -> String {
        match &self.phase {
            Phase::Detect => split_reasoning(&std::mem::take(&mut self.buf)).content,
            Phase::Reasoning { .. } | Phase::WaitForAnswer => String::new(),
            Phase::Answer { spent, .. } => {
                if *spent {
                    String::new()
                } else {
                    self.take_unemitted()
                }
            }
        }
    }

    fn take_unemitted(&mut self) -> String {
        let out = self.buf[self.emitted.min(self.buf.len())..].to_string();
        self.emitted = self.buf.len();
        out
    }

    fn detect(&mut self, prev_buf_len: usize) -> String {
        let rest = self.buf.trim_start();
        for tag in ALL_TAGS {
            if let Some(body) = strip_open(rest, tag) {
                self.buf = body.to_string();
                self.phase = Phase::Reasoning { tag };
                return self.leave_reasoning(tag, prev_buf_len);
            }
        }
        // Still ambiguous? Hold back until the opener question settles.
        if ALL_TAGS.iter().any(|t| format!("<{t}").starts_with(rest)) {
            return String::new();
        }
        // No reasoning channel: everything buffered is answer text. Merged
        // unless it all arrived with this one push.
        self.merged_last = prev_buf_len > 0;
        self.phase = Phase::Answer {
            closer: None,
            spent: false,
        };
        std::mem::take(&mut self.buf)
    }

    /// Consume buffered reasoning text up to (and including) the closing tag;
    /// advance the phase past it. Nothing in this phase is ever emitted.
    fn leave_reasoning(&mut self, tag: &'static str, prev_buf_len: usize) -> String {
        match find_tag(&self.buf, &format!("/{tag}")) {
            Some((_, end)) => {
                // Output starts at `end`; it is pure current-push bytes iff
                // the closer finished within this push.
                self.merged_last = end < prev_buf_len;
                let rest = self.buf[end..].to_string();
                self.buf = rest;
                if tag == "analysis" {
                    self.phase = Phase::WaitForAnswer;
                    return self.wait_for_answer(prev_buf_len);
                }
                self.phase = Phase::Answer {
                    closer: None,
                    spent: false,
                };
                self.emitted = 0;
                std::mem::take(&mut self.buf)
            }
            None => String::new(), // still thinking
        }
    }

    fn wait_for_answer(&mut self, prev_buf_len: usize) -> String {
        let trimmed = self.buf.trim_start();
        if trimmed.is_empty() || "<answer>".starts_with(trimmed) {
            // Whitespace, or text that may still grow into `<answer>`: wait.
            self.buf.clear();
            return String::new();
        }
        if let Some(body) = trimmed.strip_prefix("<answer>") {
            self.buf = body.to_string();
            self.phase = Phase::Answer {
                closer: Some("answer"),
                spent: false,
            };
            self.emitted = 0;
            let out = self.answer_text(prev_buf_len);
            if !out.is_empty() && prev_buf_len > 0 {
                // The <answer> opener may span pushes; be conservative.
                self.merged_last = true;
            }
            return out;
        }
        // Model skipped the answer tag: what follows is ordinary answer text.
        self.merged_last = prev_buf_len > 0;
        self.phase = Phase::Answer {
            closer: None,
            spent: false,
        };
        self.emitted = 0;
        std::mem::take(&mut self.buf)
    }

    fn answer_text(&mut self, prev_buf_len: usize) -> String {
        let Phase::Answer { closer, .. } = self.phase else {
            unreachable!("answer_text outside Answer phase");
        };
        let Some(tag) = closer else {
            return self.take_unemitted();
        };
        let needle = format!("</{tag}>");
        // A closing tag may span pushes: rescan a window that reaches back
        // far enough to catch one straddling the previous boundary.
        let scan_from = floor_boundary(&self.buf, self.emitted.saturating_sub(needle.len() - 1));
        if let Some((rel_start, _)) = find_tag(&self.buf[scan_from..], &format!("/{tag}")) {
            self.phase = Phase::Answer {
                closer: Some(tag),
                spent: true,
            };
            let out = self.buf[self.emitted..scan_from + rel_start].to_string();
            self.emitted = self.buf.len();
            return out;
        }
        // Withhold a trailing partial `</tag>` so it is never streamed early;
        // the bytes stay in `buf` (nothing is dropped).
        let bytes = self.buf.as_bytes();
        let withhold = (1..needle.len())
            .rev()
            .find(|k| bytes.len() >= *k && bytes[bytes.len() - *k..] == needle.as_bytes()[..*k])
            .unwrap_or(0);
        let end = self.buf.len() - withhold;
        if self.emitted >= end {
            return String::new();
        }
        let out = self.buf[self.emitted..end].to_string();
        self.emitted = end;
        out
    }
}

/// Largest char boundary ≤ `i` (`str::floor_char_boundary` is nightly-only;
/// MSRV here is stable 1.80).
fn floor_boundary(s: &str, mut i: usize) -> usize {
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_answer_untouched() {
        let s = split_reasoning("Just the answer.");
        assert_eq!(s.reasoning, None);
        assert_eq!(s.content, "Just the answer.");
    }

    #[test]
    fn think_style_separated() {
        let s = split_reasoning("<think>chain of thought</think>The capital is Paris.");
        assert_eq!(s.reasoning.as_deref(), Some("chain of thought"));
        assert_eq!(s.content, "The capital is Paris.");
    }

    #[test]
    fn unclosed_think_keeps_content_clean() {
        let s = split_reasoning("<think>rambling without end");
        assert_eq!(s.reasoning.as_deref(), Some("rambling without end"));
        assert_eq!(s.content, "");
    }

    #[test]
    fn angle_bracket_in_plain_answer_not_mangled() {
        let s = split_reasoning("use a<b and x <y> carefully");
        assert_eq!(s.reasoning, None);
        assert_eq!(s.content, "use a<b and x <y> carefully");
    }

    #[test]
    fn analysis_answer_layout() {
        let text = "<analysis>need FY2018 capex</analysis><answer>$1.2B</answer>";
        let s = split_reasoning(text);
        assert_eq!(s.reasoning.as_deref(), Some("need FY2018 capex"));
        assert_eq!(s.content, "$1.2B");
    }

    #[test]
    fn analysis_without_answer_tag_yields_empty_content() {
        let s = split_reasoning("<analysis>x</analysis>");
        assert_eq!(s.reasoning.as_deref(), Some("x"));
        assert_eq!(s.content, "");
    }

    #[test]
    fn filter_plain_text_streams_through() {
        let mut f = ReasoningFilter::new();
        assert_eq!(f.push("Plain"), "Plain");
        assert_eq!(f.push(" answer"), " answer");
        assert_eq!(f.finish(), "");
    }

    #[test]
    fn filter_streams_answer_after_think() {
        let mut f = ReasoningFilter::new();
        assert_eq!(f.push("<th"), "");
        assert_eq!(f.push("ink>secret "), "");
        assert_eq!(f.push("</thi"), "");
        assert_eq!(f.push("nk>Hello"), "Hello");
        assert_eq!(f.push(" world"), " world");
        assert_eq!(f.finish(), "");
    }

    #[test]
    fn filter_flushes_non_reasoning_open() {
        let mut f = ReasoningFilter::new();
        assert_eq!(f.push("The <b>answer"), "The <b>answer");
        assert_eq!(f.finish(), "");
    }

    #[test]
    fn filter_unclosed_think_never_leaks() {
        let mut f = ReasoningFilter::new();
        assert_eq!(f.push("<think>hidden thoughts"), "");
        assert_eq!(f.finish(), "");
    }

    #[test]
    fn filter_analysis_answer_stream() {
        let mut f = ReasoningFilter::new();
        assert_eq!(f.push("<analysis>why</analy"), "");
        assert_eq!(f.push("sis><answer>42"), "42");
        // closing tag is dropped, including a split tail
        assert_eq!(f.push("</an"), "");
        assert_eq!(f.push("swer> tail"), "");
        assert_eq!(f.finish(), "");
    }

    #[test]
    fn filter_answer_with_angle_bracket_not_eaten() {
        // "</b>" is not the answer-closing tag: must stream through intact.
        let mut f = ReasoningFilter::new();
        assert_eq!(f.push("hi </b> there"), "hi </b> there");
        assert_eq!(f.push("</b> ok"), "</b> ok");
        assert_eq!(f.finish(), "");
    }

    #[test]
    fn filter_holds_back_partial_opener_then_flushes() {
        // A message that merely begins with '<' but is not a reasoning tag
        // must reach content once disproved.
        let mut f = ReasoningFilter::new();
        assert_eq!(f.push("<t"), "");
        assert_eq!(f.push("d>hello"), "<td>hello");
        assert_eq!(f.finish(), "");
    }

    #[test]
    fn filter_split_closer_still_detected() {
        let mut f = ReasoningFilter::new();
        assert_eq!(f.push("<think>t</thi"), "");
        // the closer completed in this delta, answer follows next
        assert_eq!(f.push("nk>"), "");
        assert_eq!(f.push("ans"), "ans");
        assert_eq!(f.finish(), "");
    }

    #[test]
    fn merged_flag_off_for_pure_deltas() {
        let mut f = ReasoningFilter::new();
        let _ = f.push("<th");
        let _ = f.push("ink>s");
        let _ = f.push("</think>Hello");
        assert!(!f.last_push_was_merged());
    }

    #[test]
    fn merged_flag_on_when_disproved_opener_flushes_backlog() {
        let mut f = ReasoningFilter::new();
        let _ = f.push("<t");
        let _ = f.push("d>x");
        assert!(f.last_push_was_merged());
    }

    #[test]
    fn merged_flag_on_when_partial_closer_resolves_as_text() {
        // analysis-style: "</an" withheld, then disproved by what follows —
        // the emission spans two pushes, so no per-token logprob may attach.
        let mut f = ReasoningFilter::new();
        let _ = f.push("<analysis>w</analysis><answer>ab</an");
        assert_eq!(f.push("cX"), "</ancX");
        assert!(f.last_push_was_merged());
    }
}
