// Types mirroring src/agents/models.py (Citation, RunResult) on the backend.

export interface Citation {
	review_id: string;
	claim: string;
	grounded: boolean;
	confidence: number | null;
	reason: string | null;
}

export interface Claim {
	text: string;
	review_id: string;
}

export interface Recommendation {
	place_name: string;
	claims: Claim[];
	rating: number | null;
	address: string | null;
}

export interface RunResult {
	answer: string; // rendered prose fallback, not rendered directly by the UI
	recommendations: Recommendation[];
	refusal: string | null;
	seen_review_ids: string[];
	citations: Citation[];
	revised: boolean;
	tool_calls_made: string[];
}

export interface StatusEvent {
	stage: string;
	label: string;
}

/**
 * POSTs a query to the /api/chat SSE endpoint and invokes callbacks as
 * "status" (progress) and "result" (final answer) events arrive.
 *
 * This is not token-level LLM streaming: the backend only knows the final
 * answer once citation verification (and any revision pass) completes, so
 * the stream instead reports which pipeline stage is currently running.
 */
export async function streamChat(
	query: string,
	onStatus: (event: StatusEvent) => void,
	onResult: (result: RunResult) => void
): Promise<void> {
	const response = await fetch('/api/chat', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ query })
	});

	if (!response.ok || !response.body) {
		throw new Error(`Chat request failed: ${response.status} ${response.statusText}`);
	}

	const reader = response.body.getReader();
	const decoder = new TextDecoder();
	let buffer = '';

	while (true) {
		const { done, value } = await reader.read();
		if (done) break;
		buffer += decoder.decode(value, { stream: true });

		let separatorIndex: number;
		while ((separatorIndex = buffer.indexOf('\n\n')) !== -1) {
			const rawEvent = buffer.slice(0, separatorIndex);
			buffer = buffer.slice(separatorIndex + 2);

			let eventName = 'message';
			let data = '';
			for (const line of rawEvent.split('\n')) {
				if (line.startsWith('event: ')) eventName = line.slice('event: '.length);
				else if (line.startsWith('data: ')) data = line.slice('data: '.length);
			}
			if (!data) continue;

			const parsed = JSON.parse(data);
			if (eventName === 'status') onStatus(parsed as StatusEvent);
			else if (eventName === 'result') onResult(parsed as RunResult);
		}
	}
}
