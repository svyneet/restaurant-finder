<script lang="ts">
	import { streamChat, type RunResult } from '$lib/api';
	import { EXAMPLE_PROMPTS, type ChatMessage } from '$lib/chat';
	import GroundingReport from '$lib/GroundingReport.svelte';

	const STORAGE_KEY = 'chat-history';

	// Real counts from data/districtwise/*.json (Mitte, Kreuzberg, Neukölln,
	// Charlottenburg-Wilmersdorf) — update if the dataset is re-scraped.
	const DATASET_STATS = [
		{ label: 'Places indexed', value: '423' },
		{ label: 'Reviews indexed', value: '5,792' },
		{ label: 'Districts covered', value: '4' }
	];

	const PIPELINE_STEPS = [
		{
			no: '01',
			title: 'Search',
			body: 'Reviews are embedded and indexed with llama_index — a vector search over meaning, not just keywords, finds passages that match your question.'
		},
		{
			no: '02',
			title: 'Cite',
			body: 'The agent may only make a restaurant-specific claim by attaching it to a real reviewId, as structured output rather than free text.'
		},
		{
			no: '03',
			title: 'Verify',
			body: "Each citation is checked against the real review text it points to with a local entailment model — not trusted on the agent's word."
		},
		{
			no: '04',
			title: 'Drop or refuse',
			body: 'A claim that fails verification is dropped from the answer, rather than left to stand.'
		}
	];

	function loadHistory(): ChatMessage[] {
		if (typeof sessionStorage === 'undefined') return [];
		try {
			const raw = sessionStorage.getItem(STORAGE_KEY);
			return raw ? (JSON.parse(raw) as ChatMessage[]) : [];
		} catch {
			return [];
		}
	}

	let history = $state<ChatMessage[]>(loadHistory());
	let input = $state('');
	let statusLabel = $state('');
	let busy = $state(false);
	let error = $state('');

	$effect(() => {
		sessionStorage.setItem(STORAGE_KEY, JSON.stringify(history));
	});

	async function send(query: string) {
		if (!query.trim() || busy) return;
		error = '';
		history.push({ role: 'user', content: query });
		input = '';
		busy = true;
		statusLabel = 'Starting...';

		try {
			await streamChat(
				query,
				(status) => (statusLabel = status.label),
				(result: RunResult) => {
					history.push({
						role: 'assistant',
						content: result.answer,
						recommendations: result.recommendations,
						refusal: result.refusal,
						citations: result.citations,
						toolCallsMade: result.tool_calls_made,
						revised: result.revised
					});
				}
			);
		} catch (e) {
			error = e instanceof Error ? e.message : 'Something went wrong talking to the backend.';
		} finally {
			busy = false;
			statusLabel = '';
		}
	}

	function onSubmit(e: SubmitEvent) {
		e.preventDefault();
		send(input);
	}

	function citationGrounded(msg: ChatMessage, reviewId: string, claimText: string): boolean {
		const citation = (msg.citations ?? []).find(
			(c) => c.review_id === reviewId && c.claim === claimText
		);
		return citation?.grounded ?? true;
	}
</script>

<svelte:head>
	<title>Berlin Restaurant Kiezscout</title>
</svelte:head>

<div class="shop">
	<div class="layout">
		<aside>
			<div class="brand">
				<span class="mark">B</span>
				<div>
					<h1>Kiezscout</h1>
					<span class="sub">Berlin restaurant finder</span>
				</div>
			</div>

			<div class="board">
				<span class="board-title">The dataset, at a glance</span>
				<ul class="stats">
					{#each DATASET_STATS as stat (stat.label)}
						<li><span class="num">{stat.value}</span><span class="lbl">{stat.label}</span></li>
					{/each}
				</ul>
				<p class="fine">Mitte · Kreuzberg · Neukölln · Charlottenburg-Wilmersdorf</p>
			</div>

			<div class="board">
				<span class="board-title">How an answer gets grounded</span>
				<ol class="pipeline">
					{#each PIPELINE_STEPS as step (step.no)}
						<li>
							<span class="no">{step.no}</span>
							<div>
								<strong>{step.title}</strong>
								<p>{step.body}</p>
							</div>
						</li>
					{/each}
				</ol>
				<p class="fine">
					If the dataset doesn't cover it — try asking about ramen — the agent says so instead of
					guessing. Offline evals additionally score answers with an LLM judge for faithfulness and
					relevancy, on top of the deterministic checks above.
				</p>
			</div>

			<div class="board">
				<span class="board-title">Today's specials</span>
				<ul class="menu">
					{#each EXAMPLE_PROMPTS as prompt (prompt)}
						<li>
							<button disabled={busy} onclick={() => send(prompt)}>
								<span class="q">{prompt}</span>
								<span class="leader" aria-hidden="true"></span>
								<span class="go">ask</span>
							</button>
						</li>
					{/each}
				</ul>
			</div>
		</aside>

		<main>
			<header class="hero">
				<span class="eyebrow">Multi-agent RAG · grounded answers</span>
				<h2>What's good to eat in Berlin?</h2>
				<p class="caption">
					Every restaurant-specific claim is cited to a real Google Maps review and independently
					re-checked before it reaches you — see the pipeline in the margin.
				</p>
			</header>

			<div class="history">
				{#each history as msg, i (i)}
					<div class="row {msg.role}">
						<span class="who">{msg.role === 'user' ? 'you' : 'kiezscout'}</span>
						<div class="ticket">
							{#if msg.role === 'user'}
								<p>{msg.content}</p>
							{:else if msg.refusal}
								<p>{msg.refusal}</p>
							{:else if msg.recommendations?.length}
								<div class="recommendations">
									{#each msg.recommendations as rec (rec.place_name)}
										<div class="recommendation">
											<h3 class="place-name">{rec.place_name}</h3>
											<ul class="claims">
												{#each rec.claims as claim, i (claim.review_id + i)}
													<li>
														{claim.text}
														<sup
															class="cite"
															class:ok={citationGrounded(msg, claim.review_id, claim.text)}
															class:bad={!citationGrounded(msg, claim.review_id, claim.text)}
															>[{i + 1}]</sup
														>
													</li>
												{/each}
											</ul>
											<p class="rec-meta">
												{#if rec.rating != null}<span>Rated {rec.rating}/5</span>{/if}
												{#if rec.address}<span class="address">{rec.address}</span>{/if}
											</p>
										</div>
									{/each}
								</div>
							{:else}
								<p>{msg.content}</p>
							{/if}
							{#if msg.role === 'assistant'}
								{#if msg.toolCallsMade?.length}
									<p class="tools">tools used · {msg.toolCallsMade.join(', ')}</p>
								{/if}
								<GroundingReport citations={msg.citations ?? []} />
								{#if msg.revised}
									<p class="revised-note">
										<span class="stamp">revised</span> claims that failed verification were removed.
									</p>
								{/if}
							{/if}
						</div>
					</div>
				{/each}

				{#if busy}
					<div class="row assistant">
						<span class="who">kiezscout</span>
						<div class="ticket">
							<p class="thinking"><span class="spinner"></span>{statusLabel}</p>
						</div>
					</div>
				{/if}

				{#if error}
					<p class="error">{error}</p>
				{/if}
			</div>

			<form class="composer" onsubmit={onSubmit}>
				<input
					type="text"
					bind:value={input}
					placeholder="Ask about restaurants in Berlin, e.g. 'Is there a good Italian restaurant?'"
					disabled={busy}
				/>
				<button type="submit" disabled={busy || !input.trim()}>Ask</button>
			</form>
		</main>
	</div>
</div>

<style>
	:global(:root) {
		/* Imbisskiosk, refined — warm paper, one serif display, a single
		   restrained burnt-orange accent. Structure comes from rules and
		   whitespace rather than heavy boxes. */
		--paper: #f3ede0;
		--paper-raised: #faf7f0;
		--ink: #201d16;
		--ink-soft: #55503f;
		--muted: #948c76;
		--accent: #bd4a1f;
		--accent-soft: #e8ceb9;
		--accent-ink: #fbf3ea;
		--border: #201d16;
		--border-soft: #ddd3bd;
		--shadow-tint: 32 24 8;
		--ok-ink: #4c5c2c;
		--ok-bg: #e9ecdc;
		--bad-ink: #8a3418;
		--bad-bg: #f6e2d5;

		--font-display: 'Fraunces', 'Iowan Old Style', Georgia, serif;
		--font-body: 'Work Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
		--font-mono: 'Space Mono', ui-monospace, 'SF Mono', 'Courier New', monospace;
	}

	@media (prefers-color-scheme: dark) {
		:global(:root) {
			--paper: #17140f;
			--paper-raised: #1e1a13;
			--ink: #f1ece0;
			--ink-soft: #cbc2a9;
			--muted: #8d846c;
			--accent: #e8703c;
			--accent-soft: #4a2d1a;
			--accent-ink: #1c0f06;
			--border: #eee8d8;
			--border-soft: #38321f;
			--shadow-tint: 0 0 0;
			--ok-ink: #cfe0a4;
			--ok-bg: #26301a;
			--bad-ink: #f3c1a4;
			--bad-bg: #3a2015;
		}
	}

	:global(body) {
		margin: 0;
		font-family: var(--font-body);
		color: var(--ink);
		background: var(--paper);
	}

	.shop {
		min-height: 100vh;
	}

	.layout {
		display: grid;
		grid-template-columns: 306px 1fr;
		min-height: 100vh;
	}

	aside {
		border-right: 1px solid var(--border-soft);
		padding: 2.25rem 1.65rem 3rem;
		display: flex;
		flex-direction: column;
		gap: 2.25rem;
	}

	.brand {
		display: flex;
		align-items: center;
		gap: 0.8rem;
	}

	.mark {
		width: 38px;
		height: 38px;
		flex-shrink: 0;
		border-radius: 3px;
		background: var(--accent);
		color: var(--accent-ink);
		font-family: var(--font-display);
		font-weight: 600;
		font-style: italic;
		font-size: 1.2rem;
		display: flex;
		align-items: center;
		justify-content: center;
	}

	.brand h1 {
		font-family: var(--font-display);
		font-weight: 600;
		font-size: 1.15rem;
		margin: 0;
	}

	.brand .sub {
		display: block;
		font-size: 0.68rem;
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: var(--muted);
		margin-top: 0.15rem;
	}

	.board {
		padding-top: 1rem;
		border-top: 1px solid var(--border-soft);
	}

	.board-title {
		display: block;
		font-family: var(--font-display);
		font-style: italic;
		font-weight: 500;
		font-size: 0.92rem;
		color: var(--ink);
		margin-bottom: 1rem;
	}

	.stats {
		list-style: none;
		margin: 0 0 0.85rem;
		padding: 0;
		display: grid;
		grid-template-columns: repeat(3, minmax(0, 1fr));
		gap: 0.5rem;
	}

	.stats li {
		display: flex;
		flex-direction: column;
		gap: 0.2rem;
	}

	.stats .num {
		font-family: var(--font-display);
		font-weight: 600;
		font-size: 1.3rem;
		font-variant-numeric: tabular-nums;
	}

	.stats .lbl {
		font-size: 0.66rem;
		color: var(--muted);
		line-height: 1.3;
	}

	.fine {
		font-size: 0.75rem;
		color: var(--muted);
		line-height: 1.55;
		margin: 0;
	}

	.pipeline {
		list-style: none;
		margin: 0 0 0.9rem;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
	}

	.pipeline li {
		display: flex;
		gap: 0.7rem;
		min-width: 0;
	}

	.pipeline li > div {
		min-width: 0;
	}

	.pipeline .no {
		font-family: var(--font-mono);
		font-size: 0.68rem;
		color: var(--accent);
		flex-shrink: 0;
		padding-top: 0.15rem;
	}

	.pipeline strong {
		display: block;
		font-weight: 600;
		font-size: 0.84rem;
		margin-bottom: 0.15rem;
	}

	.pipeline p {
		margin: 0;
		font-size: 0.78rem;
		color: var(--ink-soft);
		line-height: 1.5;
	}

	.menu {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
	}

	.menu button {
		width: 100%;
		display: flex;
		align-items: baseline;
		gap: 0.5rem;
		text-align: left;
		background: none;
		border: none;
		border-bottom: 1px solid var(--border-soft);
		padding: 0.55rem 0;
		font: inherit;
		font-size: 0.83rem;
		color: var(--ink);
		cursor: pointer;
	}

	.menu .q {
		min-width: 0;
	}

	.menu li:first-child button {
		padding-top: 0;
	}

	.menu button:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}

	.menu .leader {
		flex: 1;
		border-bottom: 1px dotted var(--border-soft);
		margin-bottom: 0.3rem;
	}

	.menu .go {
		flex-shrink: 0;
		font-size: 0.68rem;
		font-style: italic;
		font-family: var(--font-display);
		color: var(--muted);
	}

	.menu button:hover:not(:disabled) .go {
		color: var(--accent);
	}

	main {
		display: flex;
		flex-direction: column;
		padding: 2.75rem 3rem;
		max-width: 900px;
	}

	.hero {
		margin-bottom: 2rem;
	}

	.eyebrow {
		display: inline-block;
		font-family: var(--font-mono);
		font-size: 0.68rem;
		letter-spacing: 0.06em;
		text-transform: uppercase;
		color: var(--accent);
		margin-bottom: 0.65rem;
	}

	.hero h2 {
		font-family: var(--font-display);
		font-style: italic;
		font-weight: 600;
		font-size: clamp(1.55rem, 4.4vw + 0.9rem, 2.15rem);
		margin: 0 0 0.6rem;
		text-wrap: balance;
	}

	.caption {
		color: var(--ink-soft);
		font-size: 0.95rem;
		line-height: 1.6;
		margin: 0;
		max-width: 58ch;
	}

	.history {
		flex: 1;
		display: flex;
		flex-direction: column;
		gap: 1.25rem;
		margin-bottom: 1.5rem;
	}

	.row {
		display: flex;
		flex-direction: column;
		gap: 0.35rem;
	}

	.row.user {
		align-items: flex-end;
	}

	.who {
		font-family: var(--font-mono);
		font-size: 0.66rem;
		text-transform: uppercase;
		letter-spacing: 0.08em;
		color: var(--muted);
	}

	.ticket {
		max-width: 82%;
		padding: 1rem 1.25rem;
		border: 1px solid var(--border-soft);
		background: var(--paper-raised);
		box-shadow: 0 1px 2px rgb(var(--shadow-tint) / 5%);
	}

	.row.user .ticket {
		background: var(--accent);
		color: var(--accent-ink);
		border-color: var(--accent);
	}

	.row.user .who {
		color: var(--muted);
	}

	.ticket p {
		margin: 0 0 0.5rem;
		white-space: pre-wrap;
		overflow-wrap: anywhere;
		line-height: 1.6;
		font-size: 0.95rem;
	}

	.ticket p:last-child {
		margin-bottom: 0;
	}

	.recommendations {
		display: flex;
		flex-direction: column;
		gap: 0.9rem;
		margin: 0 0 0.5rem;
	}

	.recommendation:not(:last-child) {
		padding-bottom: 0.9rem;
		border-bottom: 1px solid var(--border-soft);
	}

	.place-name {
		margin: 0 0 0.35rem;
		font-family: var(--font-display);
		font-size: 1.02rem;
		font-weight: 600;
	}

	.claims {
		margin: 0;
		padding: 0;
		list-style: none;
		display: flex;
		flex-direction: column;
		gap: 0.3rem;
	}

	.claims li {
		font-size: 0.95rem;
		line-height: 1.6;
	}

	.cite {
		font-family: var(--font-mono);
		font-size: 0.65rem;
		margin-left: 0.1rem;
	}

	.cite.ok {
		color: var(--ok-ink);
	}

	.cite.bad {
		color: var(--bad-ink);
	}

	.rec-meta {
		margin: 0.4rem 0 0;
		display: flex;
		gap: 0.75rem;
		font-size: 0.78rem;
		color: var(--ink-soft);
	}

	.rec-meta .address {
		font-family: var(--font-mono);
	}

	.tools {
		font-family: var(--font-mono);
		font-size: 0.7rem;
		color: var(--muted);
	}

	.stamp {
		display: inline-block;
		white-space: nowrap;
		border: 1px solid var(--accent);
		color: var(--accent);
		font-family: var(--font-display);
		font-style: italic;
		font-weight: 600;
		font-size: 0.68rem;
		padding: 0.05rem 0.45rem;
		margin-right: 0.4rem;
		border-radius: 2px;
	}

	.revised-note {
		font-size: 0.83rem;
		color: var(--ink-soft);
		display: flex;
		align-items: flex-start;
		flex-wrap: wrap;
		row-gap: 0.3rem;
	}

	.thinking {
		color: var(--muted);
		display: flex;
		align-items: center;
		gap: 0.55rem;
	}

	.spinner {
		width: 0.8rem;
		height: 0.8rem;
		border-radius: 50%;
		border: 2px solid var(--border-soft);
		border-top-color: var(--accent);
		animation: spin 0.8s linear infinite;
	}

	@keyframes spin {
		to {
			transform: rotate(360deg);
		}
	}

	@media (prefers-reduced-motion: reduce) {
		.spinner {
			animation: none;
		}
	}

	.error {
		color: var(--bad-ink);
		background: var(--bad-bg);
		border: 1px solid var(--bad-ink);
		padding: 0.65rem 0.9rem;
		font-size: 0.85rem;
	}

	.composer {
		display: flex;
		gap: 0.6rem;
	}

	input {
		flex: 1;
		min-width: 0;
		padding: 0.9rem 1.05rem;
		border: 1px solid var(--border-soft);
		background: var(--paper-raised);
		font-size: 0.95rem;
		font-family: inherit;
		color: var(--ink);
		outline-offset: 2px;
	}

	input:focus-visible {
		border-color: var(--ink);
	}

	input::placeholder {
		color: var(--muted);
	}

	.composer button {
		flex-shrink: 0;
		padding: 0 1.75rem;
		border: 1px solid var(--ink);
		background: var(--ink);
		color: var(--paper);
		font-family: var(--font-display);
		font-style: italic;
		font-weight: 600;
		font-size: 0.92rem;
		cursor: pointer;
		transition:
			background 0.15s ease,
			border-color 0.15s ease;
	}

	.composer button:hover:not(:disabled) {
		background: var(--accent);
		border-color: var(--accent);
		color: var(--accent-ink);
	}

	.composer button:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}

	@media (max-width: 760px) {
		.layout {
			grid-template-columns: minmax(0, 1fr);
		}

		aside,
		main {
			min-width: 0;
		}

		aside {
			border-right: none;
			border-bottom: 1px solid var(--border-soft);
			padding: 1.5rem 1.35rem 2rem;
			gap: 1.75rem;
		}

		main {
			padding: 1.75rem 1.35rem 2.25rem;
		}

		.hero {
			margin-bottom: 1.5rem;
		}

		.ticket {
			max-width: 100%;
		}

		.stats {
			gap: 0.35rem;
		}

		.stats .num {
			font-size: 1.15rem;
		}

		.composer {
			gap: 0.5rem;
		}

		.composer button {
			padding: 0 1.15rem;
		}
	}

	@media (max-width: 420px) {
		.brand {
			gap: 0.6rem;
		}

		.mark {
			width: 34px;
			height: 34px;
			font-size: 1.05rem;
		}

		.stats {
			grid-template-columns: repeat(3, 1fr);
			gap: 0.3rem;
		}

		.stats .num {
			font-size: 1rem;
		}

		.stats .lbl {
			font-size: 0.6rem;
		}

		input {
			padding: 0.8rem 0.9rem;
			font-size: 0.9rem;
		}

		.composer button {
			padding: 0 0.9rem;
			font-size: 0.85rem;
		}
	}
</style>
