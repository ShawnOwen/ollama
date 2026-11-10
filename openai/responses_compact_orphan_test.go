package openai

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

// orphanedSummaryCallID matches the shape seen in Codex histories that older
// releases left unable to compact: a prior summary output kept in retained
// without the synthetic call that produced it.
const (
	orphanedSummaryCallID = "call_ollama_compaction_2fa260fb1a51"
	summaryCallIDPrefix   = "call_ollama_compaction_"
)

func compactionItemJSON(t *testing.T, payload OllamaCompactionPayload) ResponsesCompactionItem {
	t.Helper()
	payload.Type = OllamaCompactionPayloadType
	payload.Version = OllamaCompactionPayloadVersion
	b, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	return ResponsesCompactionItem{Type: "compaction", EncryptedContent: string(b)}
}

func poisonedCompactionItem(t *testing.T, orphanCallID string) ResponsesCompactionItem {
	t.Helper()
	return compactionItemJSON(t, OllamaCompactionPayload{
		Summary: "latest summary",
		Retained: []api.Message{
			{Role: "user", Content: "build the pipeline"},
			{Role: "tool", ToolCallID: orphanCallID, Content: "orphaned prior summary text"},
			{Role: "assistant", ToolCalls: []api.ToolCall{{
				ID: "call_shell", Function: api.ToolCallFunction{Name: "exec_command", Arguments: api.NewToolCallFunctionArguments()},
			}}},
			{Role: "tool", ToolCallID: "call_shell", Content: "exit 0"},
		},
	})
}

func compactionBody(t *testing.T, triggered bool, input ...any) []byte {
	t.Helper()
	if triggered {
		input = append(input, ResponsesCompactionTrigger{Type: "compaction_trigger"})
	}
	body, err := json.Marshal(map[string]any{"model": "test", "stream": triggered, "input": input})
	if err != nil {
		t.Fatal(err)
	}
	return body
}

func prepareCompaction(t *testing.T, triggered bool, body []byte) (*ResponsesCompactionPlan, error) {
	t.Helper()
	if triggered {
		plan, requested, err := PrepareTriggeredCompaction(body)
		if !requested {
			t.Fatal("compaction_trigger was not recognized")
		}
		return plan, err
	}
	return PrepareStandaloneCompaction(body)
}

func refsWithCallID(plan *ResponsesCompactionPlan, callID string) []string {
	var refs []string
	for _, item := range plan.items {
		if item.Message.ToolCallID == callID {
			refs = append(refs, item.Ref)
		}
		for _, call := range item.Message.ToolCalls {
			if call.ID == callID {
				refs = append(refs, item.Ref)
			}
		}
	}
	return refs
}

func allRefs(plan *ResponsesCompactionPlan) []string {
	refs := make([]string, 0, len(plan.items))
	for _, item := range plan.items {
		refs = append(refs, item.Ref)
	}
	return refs
}

func assertNoUnpairedToolResults(t *testing.T, retained []api.Message) {
	t.Helper()
	calls := make(map[string]struct{})
	for _, message := range retained {
		for _, call := range message.ToolCalls {
			calls[call.ID] = struct{}{}
		}
	}
	for i, message := range retained {
		if message.Role != "tool" || message.ToolCallID == "" {
			continue
		}
		if strings.HasPrefix(message.ToolCallID, summaryCallIDPrefix) {
			t.Errorf("retained[%d] keeps prior summary output %q", i, message.ToolCallID)
		}
		if _, ok := calls[message.ToolCallID]; !ok {
			t.Errorf("retained[%d] is a tool result with no matching call %q", i, message.ToolCallID)
		}
	}
}

// The malformed write: the summarizer selects a prior summary's output but
// not its synthetic call. The new payload must not keep the output alone, and
// the history it produces must compact again through both entry points.
func TestCompactionDoesNotRetainPriorSummaryOutputWithoutCall(t *testing.T) {
	prior := compactionItemJSON(t, OllamaCompactionPayload{
		Summary:  "first summary",
		Retained: []api.Message{{Role: "user", Content: "original task"}},
	})
	body := compactionBody(t, true, prior, map[string]any{"type": "message", "role": "user", "content": "new work"})
	plan, err := prepareCompaction(t, true, body)
	if err != nil {
		t.Fatal(err)
	}

	var summaryOutputRef string
	for _, item := range plan.items {
		if item.Type == "function_call_output" && strings.HasPrefix(item.Message.ToolCallID, summaryCallIDPrefix) {
			summaryOutputRef = item.Ref
		}
	}
	if summaryOutputRef == "" {
		t.Fatal("expanded history has no prior summary output")
	}

	result, err := plan.Complete(compactionResponseBody(t, map[string]any{
		"summary": "second summary", "retain_item_ids": []string{summaryOutputRef},
	}))
	if err != nil {
		t.Fatal(err)
	}
	payload := decodeResultPayload(t, result)
	assertNoUnpairedToolResults(t, payload.Retained)
	if strings.Contains(result.Item.EncryptedContent, "first summary") {
		t.Fatalf("prior summary was nested in the new payload: %s", result.Item.EncryptedContent)
	}

	for _, triggered := range []bool{true, false} {
		next := compactionBody(t, triggered, result.Item, map[string]any{"type": "message", "role": "user", "content": "later work"})
		if _, err := prepareCompaction(t, triggered, next); err != nil {
			t.Fatalf("triggered=%v: next compaction failed: %v", triggered, err)
		}
	}
}

// The two stalls: a history that an older release already saved with an
// orphaned summary output must compact through the Codex trigger on
// POST /v1/responses and through POST /v1/responses/compact.
func TestCompactionRecoversHistoryWithOrphanedPriorSummary(t *testing.T) {
	for _, mode := range []struct {
		name      string
		triggered bool
	}{
		{"triggered", true},
		{"standalone", false},
	} {
		t.Run(mode.name, func(t *testing.T) {
			body := compactionBody(t, mode.triggered,
				poisonedCompactionItem(t, orphanedSummaryCallID),
				map[string]any{"type": "message", "role": "user", "content": "hi"},
			)
			plan, err := prepareCompaction(t, mode.triggered, body)
			if err != nil {
				t.Fatalf("prepare failed on orphaned prior summary: %v", err)
			}

			request, err := plan.SummaryRequest("")
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(summaryTranscriptText(t, request), "orphaned prior summary text") {
				t.Fatal("orphaned summary content was hidden from the summarizer")
			}

			if refs := refsWithCallID(plan, orphanedSummaryCallID); len(refs) != 1 {
				t.Fatalf("orphan refs=%v, want 1", refs)
			}
			result, err := plan.Complete(compactionResponseBody(t, map[string]any{
				"summary": "recovered summary", "retain_item_ids": allRefs(plan),
			}))
			if err != nil {
				t.Fatal(err)
			}
			payload := decodeResultPayload(t, result)
			assertNoUnpairedToolResults(t, payload.Retained)
			var keptShell bool
			for _, message := range payload.Retained {
				if message.Role == "tool" && message.ToolCallID == "call_shell" {
					keptShell = true
				}
			}
			if !keptShell {
				t.Fatalf("ordinary tool pair was dropped: %+v", payload.Retained)
			}

			next := compactionBody(t, mode.triggered, result.Item, map[string]any{"type": "message", "role": "user", "content": "continue"})
			if _, err := prepareCompaction(t, mode.triggered, next); err != nil {
				t.Fatalf("history is still stuck after recovery: %v", err)
			}
		})
	}
}

// Only synthetic summary outputs are tolerated; any other orphaned tool result
// is still invalid history.
func TestCompactionStillRejectsOrdinaryOrphanedToolOutputs(t *testing.T) {
	for _, triggered := range []bool{true, false} {
		body := compactionBody(t, triggered,
			poisonedCompactionItem(t, "call_not_a_summary"),
			map[string]any{"type": "message", "role": "user", "content": "hi"},
		)
		_, err := prepareCompaction(t, triggered, body)
		if err == nil || !strings.Contains(err.Error(), "no matching call") {
			t.Fatalf("triggered=%v: expected no matching call error, got %v", triggered, err)
		}
	}
}
