//! How the synthesis logs a filled hole: at `warn` when the upstream that
//! answered kept the request's `safeguards` field, at `debug` otherwise.

/// Marks a response relayed from a provider that kept the request's
/// `safeguards` field (`adapters::anthropic::keeps_safeguards`), so a response
/// with this mark was expected to answer it. Set by the Anthropic adapter's
/// `relay_response`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FirstPartyRelay;

/// Whether the upstream that answered kept the request's `safeguards` field.
pub(super) enum FirstParty {
    /// Fixed when the response was built: it carries [`FirstPartyRelay`].
    Marked(bool),
    /// Read when a hole is filled: a committed chain names its winner only as
    /// the stream is relayed.
    Winner(Box<dyn Fn() -> bool + Send + Sync>),
}

/// How a filled hole is logged.
pub(super) struct HoleReport {
    first_party: FirstParty,
    /// The span current when the synthesis was set up. A relayed stream is
    /// read after the handler returned, outside that span, so the event names
    /// it as its parent explicitly; its fields (the session id among them)
    /// stay on the log line.
    span: tracing::Span,
}

impl HoleReport {
    pub(super) fn new(first_party: FirstParty) -> Self {
        Self {
            first_party,
            span: tracing::Span::current(),
        }
    }

    /// Log one filled hole. `tool_uses` is the number of `tool_use` ids seen.
    pub(super) fn filled(&self, streaming: bool, tool_uses: usize) {
        let first_party = match &self.first_party {
            FirstParty::Marked(marked) => *marked,
            FirstParty::Winner(winner) => winner(),
        };
        if first_party {
            tracing::warn!(
                parent: &self.span,
                streaming,
                tool_uses,
                "a first-party upstream answered a request carrying `safeguards` \
                 without `safeguard_results`; synthesized per-tool-use `unavailable` results"
            );
        } else {
            tracing::debug!(
                parent: &self.span,
                streaming,
                tool_uses,
                "synthesized safeguard_results on a relayed message"
            );
        }
    }
}
