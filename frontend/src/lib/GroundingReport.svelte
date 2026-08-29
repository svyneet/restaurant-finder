<script lang="ts">
	import type { Citation } from '$lib/api';

	let { citations }: { citations: Citation[] } = $props();
	let open = $state(false);
</script>

{#if citations.length === 0}
	<p class="empty">No citations made in this answer.</p>
{:else}
	<details bind:open>
		<summary
			>Grounding report — {citations.length} citation{citations.length === 1 ? '' : 's'} checked</summary
		>
		<ul>
			{#each citations as c (c.review_id)}
				<li>
					<span class="verdict" class:ok={c.grounded} class:bad={!c.grounded}>
						{c.grounded ? 'Grounded' : 'Not grounded'}
					</span>
					<div class="detail">
						<code>REVIEW:{c.review_id}</code>
						<p>{c.reason ?? `confidence=${c.confidence?.toFixed(3) ?? 'n/a'}`}</p>
					</div>
				</li>
			{/each}
		</ul>
	</details>
{/if}

<style>
	.empty {
		font-size: 0.8rem;
		color: var(--muted);
		margin: 0.25rem 0;
	}

	details {
		margin: 0.75rem 0 0;
		font-size: 0.85rem;
		border-top: 1px solid var(--border-soft);
		padding-top: 0.7rem;
	}

	summary {
		cursor: pointer;
		color: var(--ink-soft);
		font-family: var(--font-display);
		font-style: italic;
		font-weight: 600;
		font-size: 0.85rem;
	}

	summary:hover {
		color: var(--accent);
	}

	ul {
		list-style: none;
		margin: 0.75rem 0 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: 0.6rem;
	}

	li {
		display: flex;
		align-items: flex-start;
		gap: 0.6rem;
	}

	.verdict {
		flex-shrink: 0;
		display: inline-block;
		font-family: var(--font-mono);
		font-size: 0.62rem;
		letter-spacing: 0.03em;
		text-transform: uppercase;
		padding: 0.2rem 0.4rem;
		border-radius: 2px;
		margin-top: 0.05rem;
	}

	.verdict.ok {
		background: var(--ok-bg);
		color: var(--ok-ink);
	}

	.verdict.bad {
		background: var(--bad-bg);
		color: var(--bad-ink);
	}

	.detail {
		min-width: 0;
	}

	.detail code {
		font-family: var(--font-mono);
		font-size: 0.72rem;
		color: var(--ink-soft);
		overflow-wrap: anywhere;
	}

	.detail p {
		margin: 0.3rem 0 0;
		font-size: 0.82rem;
		color: var(--ink-soft);
		line-height: 1.45;
	}
</style>
