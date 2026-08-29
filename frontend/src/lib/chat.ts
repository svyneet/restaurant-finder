export interface ChatMessage {
	role: 'user' | 'assistant';
	content: string;
	recommendations?: import('./api').Recommendation[];
	refusal?: string | null;
	citations?: import('./api').Citation[];
	toolCallsMade?: string[];
	revised?: boolean;
}

export const EXAMPLE_PROMPTS = [
	'What is the best place for sushi in Berlin?',
	'Is there a good Italian restaurant?',
	"Where's good for Middle Eastern or halal food?",
	'Is there an Indian restaurant?',
	'What are good lunch options?',
	'Where can I get Israeli food?'
];
