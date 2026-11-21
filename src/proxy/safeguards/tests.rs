use axum::{
    body::{Body, Bytes},
    http::{header::CONTENT_TYPE, StatusCode},
    response::Response,
};
use futures_util::{stream, StreamExt};
use serde_json::{json, Value};

use super::{
    requested_types, synthesize, synthesize_for_chain, transform_stream, FirstParty,
    FirstPartyRelay, HoleReport, MAX_FRAME_BYTES, MIN_SYNTHESIS_BYTES,
};

type Item = Result<Bytes, std::convert::Infallible>;

const LIMIT: usize = 64 * 1024 * 1024;

fn chunk(text: &str) -> Item {
    Ok(Bytes::from(text.to_owned()))
}

fn types() -> Vec<String> {
    vec!["dangerous_tool_use".to_string()]
}

async fn relay(chunks: Vec<Item>) -> String {
    let report = HoleReport::new(FirstParty::Marked(false));
    let out: Vec<_> = transform_stream(stream::iter(chunks), types(), report)
        .collect()
        .await;
    out.into_iter()
        .map(|item| String::from_utf8(item.unwrap().to_vec()).unwrap())
        .collect()
}

/// The `data:` payload of the first frame whose event is `message_delta`.
fn message_delta(relayed: &str) -> Value {
    relayed
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter_map(|payload| serde_json::from_str::<Value>(payload).ok())
        .find(|value| value["type"] == "message_delta")
        .expect("the relay carries a message_delta frame")
}

fn sse_response(body: &str) -> Response {
    Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, "text/event-stream")
        .body(Body::from(body.to_owned()))
        .unwrap()
}

fn json_response(status: StatusCode, body: &str) -> Response {
    Response::builder()
        .status(status)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_owned()))
        .unwrap()
}

async fn body_string(response: Response) -> String {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    String::from_utf8(bytes.to_vec()).unwrap()
}

const DELTA: &str = "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":7}}\n\n";

#[test]
fn requested_types_reads_the_safeguards_array() {
    let request = json!({
        "model": "claude-sonnet-4-6",
        "safeguards": [{"type": "dangerous_tool_use", "classifier_context": {"cwd": "/tmp"}}],
    });
    assert_eq!(requested_types(&request), vec!["dangerous_tool_use"]);
    assert!(requested_types(&json!({"model": "claude-sonnet-4-6"})).is_empty());
}

#[tokio::test]
async fn message_delta_gains_results_for_every_tool_use_seen() {
    let relayed = relay(vec![
        chunk("event: message_start\ndata: {\"type\":\"message_start\"}\n\n"),
        chunk("event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_a\",\"name\":\"Bash\"}}\n\n"),
        chunk("event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n"),
        chunk(DELTA),
    ])
    .await;

    let delta = message_delta(&relayed);
    assert_eq!(delta["delta"]["stop_reason"], "tool_use");
    let results = &delta["delta"]["safeguard_results"];
    assert_eq!(results[0]["type"], "dangerous_tool_use");
    assert_eq!(results[0]["status"]["type"], "available");
    assert_eq!(
        results[0]["status"]["tool_uses"],
        json!({"toolu_a": {"type": "unavailable", "reason": "error"}}),
        "only the tool_use block contributes an id"
    );
}

#[tokio::test]
async fn a_payload_split_over_several_data_lines_is_parsed_as_one_event() {
    // The SSE spec joins an event's `data:` lines with newlines. Parsed one at
    // a time, neither line is valid JSON, so the tool id would go unrecorded
    // and the delta unrewritten — the session-wide fallback this module
    // prevents. The rewritten delta folds back into a single `data:` line.
    let relayed = relay(vec![
        chunk("event: content_block_start\ndata: {\"type\":\"content_block_start\",\ndata: \"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_multi\"}}\n\n"),
        chunk("event: message_delta\ndata: {\"type\":\"message_delta\",\ndata: \"delta\":{\"stop_reason\":\"tool_use\"}}\n\n"),
    ])
    .await;

    let delta = message_delta(&relayed);
    assert_eq!(delta["delta"]["stop_reason"], "tool_use");
    assert_eq!(
        delta["delta"]["safeguard_results"][0]["status"]["tool_uses"],
        json!({"toolu_multi": {"type": "unavailable", "reason": "error"}}),
        "the id from the split content_block_start must reach the results"
    );
    // The start event is not rewritten, so it relays byte-for-byte with both of
    // its data lines; only the rewritten delta folds into a single one.
    let (start, delta_event) = relayed.split_once("event: message_delta").unwrap();
    assert_eq!(
        start.matches("data:").count(),
        2,
        "an unrewritten frame keeps its own framing\n{relayed}"
    );
    assert_eq!(
        delta_event.matches("data:").count(),
        1,
        "the re-serialized delta occupies one data line\n{relayed}"
    );
}

#[tokio::test]
async fn frames_split_across_chunks_still_contribute_their_ids() {
    let relayed = relay(vec![
        chunk("event: content_block_start\ndata: {\"type\":\"content_block_st"),
        chunk("art\",\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_split\"}}\n"),
        chunk("\nevent: message_del"),
        chunk(
            "ta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"}}\n\n",
        ),
    ])
    .await;

    let delta = message_delta(&relayed);
    assert_eq!(
        delta["delta"]["safeguard_results"][0]["status"]["tool_uses"],
        json!({"toolu_split": {"type": "unavailable", "reason": "error"}})
    );
}

#[tokio::test]
async fn a_turn_without_tool_uses_reports_an_empty_map() {
    let relayed = relay(vec![chunk(DELTA)]).await;
    let delta = message_delta(&relayed);
    assert_eq!(
        delta["delta"]["safeguard_results"][0]["status"]["tool_uses"],
        json!({})
    );
}

#[tokio::test]
async fn an_upstream_verdict_is_relayed_byte_for_byte() {
    let answered = "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"safeguard_results\":[{\"type\":\"dangerous_tool_use\",\"status\":{\"type\":\"available\",\"tool_uses\":{}}}]}}\n\n";
    assert_eq!(relay(vec![chunk(answered)]).await, answered);
}

#[tokio::test]
async fn every_other_frame_is_relayed_byte_for_byte() {
    let other = concat!(
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"model\":\"m\"}}\n\n",
        "event: ping\ndata: {\"type\":\"ping\"}\n\n",
        ": keepalive\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"hi\"}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    );
    assert_eq!(relay(vec![chunk(other)]).await, other);
}

#[tokio::test]
async fn an_oversized_frame_passes_through_unmodified() {
    // The `message_delta` is preceded, inside one frame, by more bytes than the
    // parse bound allows: it is forwarded rather than buffered and rewritten.
    let padding = "x".repeat(MAX_FRAME_BYTES + 1);
    let frame = format!("event: message_delta\ndata: {{\"type\":\"message_delta\",\"delta\":{{\"pad\":\"{padding}\"}}}}\n\n");
    assert_eq!(relay(vec![chunk(&frame)]).await, frame);
}

#[tokio::test]
async fn an_oversized_frame_split_across_chunks_passes_through_unmodified() {
    // The head is released before its boundary arrives, so the tail that
    // follows is not a frame of its own and must not be parsed as one — were it
    // re-serialized, the `message_delta` it happens to contain would be
    // rewritten into the middle of another frame.
    let padding = "x".repeat(MAX_FRAME_BYTES + 1);
    let head = format!("event: message_delta\ndata: {{\"pad\":\"{padding}\",");
    let tail = "\"type\":\"message_delta\",\"delta\":{}}

";
    let relayed = relay(vec![chunk(&head), chunk(tail)]).await;
    assert_eq!(relayed, format!("{head}{tail}"));
}

#[tokio::test]
async fn a_request_without_safeguards_leaves_the_stream_verbatim() {
    let body = format!("event: message_start\ndata: {{\"type\":\"message_start\"}}\n\n{DELTA}");
    let response = synthesize(sse_response(&body), &[], LIMIT).await;
    assert_eq!(body_string(response).await, body);
}

#[tokio::test]
async fn a_non_streaming_message_gains_results() {
    let body = r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"tool_use","id":"toolu_b","name":"Bash"},{"type":"text","text":"ok"}]}"#;
    let response = synthesize(json_response(StatusCode::OK, body), &types(), LIMIT).await;
    let value: Value = serde_json::from_str(&body_string(response).await).unwrap();

    assert_eq!(value["id"], "msg_1");
    assert_eq!(
        value["safeguard_results"],
        json!([{
            "type": "dangerous_tool_use",
            "status": {
                "type": "available",
                "tool_uses": {"toolu_b": {"type": "unavailable", "reason": "error"}},
            },
        }])
    );
}

#[tokio::test]
async fn a_non_streaming_upstream_verdict_is_relayed_unchanged() {
    let body = r#"{"type":"message","content":[],"safeguard_results":[{"type":"dangerous_tool_use","status":{"type":"unavailable","reason":"disabled"}}]}"#;
    let response = synthesize(json_response(StatusCode::OK, body), &types(), LIMIT).await;
    assert_eq!(body_string(response).await, body);
}

#[tokio::test]
async fn a_non_2xx_response_is_relayed_unchanged() {
    let body = r#"{"type":"error","error":{"type":"api_error","message":"boom"}}"#;
    let response = synthesize(
        json_response(StatusCode::INTERNAL_SERVER_ERROR, body),
        &types(),
        LIMIT,
    )
    .await;
    assert_eq!(body_string(response).await, body);
}

#[tokio::test]
async fn a_body_past_the_synthesis_budget_is_relayed_unchanged() {
    // Still bounded: past the resolved budget the body is relayed as it
    // arrived rather than buffered whole.
    let padding = "x".repeat(MIN_SYNTHESIS_BYTES);
    let body = format!(r#"{{"type":"message","content":[],"pad":"{padding}"}}"#);
    let response = synthesize(json_response(StatusCode::OK, &body), &types(), 4).await;
    assert_eq!(body_string(response).await, body);
}

#[tokio::test]
async fn a_lowered_inbound_limit_does_not_disable_synthesis() {
    // The budget reaching this module is `server.limits.max_request_bytes`, an
    // inbound upload cap. Before the floor, an operator lowering it relayed
    // every response without `safeguard_results`, which retires the client's
    // server classifier for the whole session (#622 review).
    let body = r#"{"type":"message","content":[{"type":"tool_use","id":"toolu_c"}]}"#;
    let response = synthesize(json_response(StatusCode::OK, body), &types(), 4).await;
    let value: Value = serde_json::from_str(&body_string(response).await).unwrap();

    assert_eq!(
        value["safeguard_results"][0]["status"]["tool_uses"],
        json!({"toolu_c": {"type": "unavailable", "reason": "error"}}),
        "a 4-byte inbound limit must not suppress the response-side answer"
    );
}

/// Capture the formatted log output for the rest of the test. The tests run on
/// tokio's current-thread runtime, so the thread-local default covers every
/// poll of the relayed body.
fn capture_logs() -> (
    tracing::subscriber::DefaultGuard,
    std::sync::Arc<std::sync::Mutex<Vec<u8>>>,
) {
    use std::sync::{Arc, Mutex};

    struct Writer(Arc<Mutex<Vec<u8>>>);
    impl std::io::Write for Writer {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    let output = Arc::new(Mutex::new(Vec::new()));
    let writer_output = Arc::clone(&output);
    let subscriber = tracing_subscriber::fmt()
        .with_writer(move || Writer(Arc::clone(&writer_output)))
        .with_ansi(false)
        .without_time()
        .finish();
    (tracing::subscriber::set_default(subscriber), output)
}

fn logged(output: &std::sync::Mutex<Vec<u8>>) -> String {
    String::from_utf8(output.lock().unwrap().clone()).unwrap()
}

fn first_party(mut response: Response) -> Response {
    response.extensions_mut().insert(FirstPartyRelay);
    response
}

const TOOL_START: &str = "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_a\",\"name\":\"Bash\"}}\n\n";

const FIRST_PARTY_HOLE: &str = "a first-party upstream answered";

#[tokio::test]
async fn a_first_party_stream_hole_warns_under_the_request_span() {
    use tracing::Instrument;

    let (_guard, output) = capture_logs();
    let body = format!("{TOOL_START}{DELTA}");
    let span = tracing::info_span!("proxy_request", session_id = "session-1");
    let response = synthesize(first_party(sse_response(&body)), &types(), LIMIT)
        .instrument(span)
        .await;
    // Read outside the span, as hyper reads a relayed body after the handler
    // returned.
    let relayed = body_string(response).await;

    assert_eq!(
        message_delta(&relayed)["delta"]["safeguard_results"][0]["status"]["tool_uses"],
        json!({"toolu_a": {"type": "unavailable", "reason": "error"}}),
        "the hole is still filled"
    );
    let logs = logged(&output);
    let line = logs
        .lines()
        .find(|line| line.contains(FIRST_PARTY_HOLE))
        .unwrap_or_else(|| panic!("no first-party warning in:\n{logs}"));
    assert!(line.contains("WARN"), "{line}");
    assert!(line.contains("session-1"), "{line}");
    assert!(line.contains("streaming=true"), "{line}");
    assert!(line.contains("tool_uses=1"), "{line}");
}

#[tokio::test]
async fn a_first_party_json_hole_warns() {
    let (_guard, output) = capture_logs();
    let body = r#"{"type":"message","content":[{"type":"tool_use","id":"toolu_b"}]}"#;
    let response = synthesize(
        first_party(json_response(StatusCode::OK, body)),
        &types(),
        LIMIT,
    )
    .await;
    let value: Value = serde_json::from_str(&body_string(response).await).unwrap();

    assert!(
        value.get("safeguard_results").is_some(),
        "the hole is still filled"
    );
    let logs = logged(&output);
    let line = logs
        .lines()
        .find(|line| line.contains(FIRST_PARTY_HOLE))
        .unwrap_or_else(|| panic!("no first-party warning in:\n{logs}"));
    assert!(line.contains("WARN"), "{line}");
    assert!(line.contains("streaming=false"), "{line}");
    assert!(line.contains("tool_uses=1"), "{line}");
}

#[tokio::test]
async fn a_translated_hole_does_not_warn() {
    let (_guard, output) = capture_logs();
    let body = format!("{TOOL_START}{DELTA}");
    let relayed = body_string(synthesize(sse_response(&body), &types(), LIMIT).await).await;

    assert!(message_delta(&relayed)["delta"]
        .get("safeguard_results")
        .is_some());
    assert_eq!(
        logged(&output),
        "",
        "a translated route fills holes every turn"
    );
}

#[tokio::test]
async fn a_first_party_verdict_does_not_warn() {
    let (_guard, output) = capture_logs();
    let delta = "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\",\"safeguard_results\":[{\"type\":\"dangerous_tool_use\",\"status\":{\"type\":\"available\",\"tool_uses\":{\"toolu_a\":{\"type\":\"evaluated\",\"outcome\":\"not_flagged\"}}}}]}}\n\n";
    let body = format!("{TOOL_START}{delta}");
    let relayed =
        body_string(synthesize(first_party(sse_response(&body)), &types(), LIMIT).await).await;

    assert_eq!(relayed, body);
    assert_eq!(logged(&output), "");
}

#[tokio::test]
async fn a_chain_won_by_a_first_party_upstream_warns() {
    let (_guard, output) = capture_logs();
    let body = format!("{TOOL_START}{DELTA}");
    let response = synthesize_for_chain(sse_response(&body), &types(), LIMIT, || true).await;
    let relayed = body_string(response).await;

    assert!(message_delta(&relayed)["delta"]
        .get("safeguard_results")
        .is_some());
    let logs = logged(&output);
    assert!(
        logs.lines()
            .any(|line| line.contains("WARN") && line.contains(FIRST_PARTY_HOLE)),
        "no first-party warning in:\n{logs}"
    );
}

#[tokio::test]
async fn a_chain_won_by_a_translated_upstream_does_not_warn() {
    let (_guard, output) = capture_logs();
    let body = format!("{TOOL_START}{DELTA}");
    let response = synthesize_for_chain(sse_response(&body), &types(), LIMIT, || false).await;
    let relayed = body_string(response).await;

    assert!(message_delta(&relayed)["delta"]
        .get("safeguard_results")
        .is_some());
    assert_eq!(logged(&output), "");
}
