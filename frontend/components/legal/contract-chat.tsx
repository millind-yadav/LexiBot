'use client';

import { useCallback, useMemo, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Loader2, Send, Upload, FileText, Trash2 } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Card } from '@/components/ui/card';
import { toast } from '@/components/toast';

type Message = {
  role: 'user' | 'assistant';
  content: string;
  plan?: Array<Record<string, any>>;
  trace?: Array<Record<string, any>>;
};

const SUGGESTED_PROMPTS = [
  'Summarize the obligations and renewal terms in this contract.',
  'Highlight potential termination risks and associated notice periods.',
  'List the indemnification clauses and explain who bears the liability.',
  'Which sections describe payment milestones and late payment penalties?',
];

function buildContextFromContract(contractText: string | null) {
  if (!contractText || !contractText.trim()) {
    return undefined;
  }

  return {
    documents: [
      {
        text: contractText,
        metadata: { source: 'uploaded_contract' },
      },
    ],
  };
}

export function ContractChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [contractText, setContractText] = useState<string | null>(null);
  const [contractName, setContractName] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const hasMessages = messages.length > 0;

  const suggestions = useMemo(() => SUGGESTED_PROMPTS, []);

  const handleUpload = useCallback(async (file: File) => {
    if (file.type && file.type !== 'text/plain') {
      toast({
        type: 'error',
        description: 'Only plain text (.txt) files are supported right now.',
      });
      return;
    }

    try {
      const text = await file.text();
      if (!text.trim()) {
        toast({ type: 'error', description: 'Uploaded file appears to be empty.' });
        return;
      }
      setContractText(text);
      setContractName(file.name);
      toast({
        type: 'success',
        description: `Loaded ${file.name}. All upcoming questions will reference this contract.`,
      });
    } catch (error) {
      console.error('Failed to read contract file:', error);
      toast({ type: 'error', description: 'Could not read the uploaded file.' });
    }
  }, []);

  const handleFileChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) return;
      void handleUpload(file);
      event.target.value = '';
    },
    [handleUpload],
  );

  const handleSubmit = useCallback(
    async (event?: React.FormEvent<HTMLFormElement>) => {
      event?.preventDefault();
      const trimmed = input.trim();
      if (!trimmed) {
        toast({ type: 'error', description: 'Ask a question about the contract to get started.' });
        return;
      }

      setMessages((prev) => [...prev, { role: 'user', content: trimmed }]);
      setInput('');
      setIsLoading(true);

      try {
        const response = await fetch('/api/legal/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            message: trimmed,
            context: buildContextFromContract(contractText),
          }),
        });

        if (!response.ok) {
          const errorBody = await response.json().catch(() => null);
          const errorMessage = errorBody?.error || 'Failed to analyze contract. Please try again.';
          throw new Error(errorMessage);
        }

        const result = await response.json();
        const assistantMessage: Message = {
          role: 'assistant',
          content: result.answer ?? 'I was unable to generate an answer for that request.',
          plan: result.plan,
          trace: result.trace,
        };
        setMessages((prev) => [...prev, assistantMessage]);
      } catch (error) {
        console.error('Contract chat error', error);
        toast({
          type: 'error',
          description:
            error instanceof Error ? error.message : 'Something went wrong while contacting the agent.',
        });
        setMessages((prev) => prev.slice(0, -1));
      } finally {
        setIsLoading(false);
      }
    },
    [contractText, input],
  );

  const handleSuggestion = useCallback((suggestion: string) => {
    setInput(suggestion);
  }, []);

  const clearContract = useCallback(() => {
    setContractText(null);
    setContractName(null);
  }, []);

  return (
    <div className="flex flex-1 flex-col gap-6">
      <section className="rounded-3xl bg-gradient-to-br from-blue-600/10 via-blue-600/5 to-indigo-600/10 p-8 shadow-sm backdrop-blur">
        <div className="mb-4 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-blue-700">
          <span className="inline-flex h-8 w-8 items-center justify-center rounded-full bg-white/80 text-blue-700">
            ⚖️
          </span>
          LexiBot Legal AI
        </div>
        <h1 className="mb-3 text-3xl font-bold text-slate-900 md:text-4xl">Welcome to your legal co-pilot</h1>
        <p className="mb-6 text-lg text-slate-600 md:text-xl">
          Upload a contract and ask natural-language questions. LexiBot will reference the uploaded
          text to surface obligations, risks, and negotiation insights.
        </p>
        <div className="flex flex-wrap items-center gap-3 text-sm text-blue-700 md:text-base">
          <span className="flex items-center gap-2 rounded-full bg-white/80 px-4 py-2 shadow-sm">
            Ask contract questions in natural language
          </span>
          <span className="flex items-center gap-2 rounded-full bg-white/80 px-4 py-2 shadow-sm">
            Surface redlines and critical clauses instantly
          </span>
        </div>
      </section>

      <section className="rounded-3xl border border-blue-100 bg-white/95 p-6 text-slate-900 shadow-xl backdrop-blur">
        <header className="flex flex-col gap-2 border-b border-blue-100 pb-4 md:flex-row md:items-center md:justify-between">
          <div>
            <h2 className="text-xl font-semibold text-blue-900">Legal Contract Analyzer</h2>
            <p className="text-sm text-blue-700/80">
              Upload contract text (optional) and chat with the agent to investigate obligations, risks, and key terms.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <input
              ref={fileInputRef}
              type="file"
              accept=".txt"
              className="hidden"
              onChange={handleFileChange}
            />
            <Button
              variant="outline"
              className="flex items-center gap-2 rounded-full border-blue-200 text-blue-900 hover:bg-blue-50"
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload className="h-4 w-4" />
              Upload contract (.txt)
            </Button>
            {contractName && (
              <Button
                variant="ghost"
                className="flex items-center gap-1 text-xs text-blue-700 hover:text-blue-900"
                onClick={clearContract}
              >
                <Trash2 className="h-4 w-4" />
                Remove
              </Button>
            )}
          </div>
        </header>

        {contractName && (
          <Card className="mt-4 border-blue-100 bg-blue-50/70 p-4 text-sm text-blue-900">
            <div className="flex items-start gap-3">
              <FileText className="h-5 w-5 shrink-0" />
              <div>
                <p className="font-medium">{contractName}</p>
                <p className="text-blue-700/80">
                  Contract loaded. Answers will reference this document until you remove or replace it.
                </p>
              </div>
            </div>
          </Card>
        )}

        <div className="mt-6">
          <p className="mb-2 text-xs font-semibold uppercase tracking-widest text-blue-600/80">
            Quick legal prompts
          </p>
          <div className="flex flex-wrap gap-2">
            {suggestions.map((suggestion) => (
              <Button
                key={suggestion}
                type="button"
                variant="outline"
                className="h-auto whitespace-normal rounded-full border-blue-200 bg-white px-4 py-2 text-left text-sm text-blue-900 shadow-sm transition hover:bg-blue-50"
                onClick={() => handleSuggestion(suggestion)}
              >
                {suggestion}
              </Button>
            ))}
          </div>
        </div>

        <div className="mt-6 flex flex-col gap-4 rounded-2xl border border-blue-100 bg-white/80 p-6">
          <div className="flex h-80 flex-col gap-4 overflow-y-auto rounded-2xl bg-gradient-to-b from-white via-white to-blue-50 p-4">
            {!hasMessages && (
              <div className="mx-auto max-w-xl text-center text-sm text-blue-700/80">
                <p className="font-medium">Start the conversation</p>
                <p>
                  Upload a contract (optional) or paste its key sections, then ask follow-up questions to inspect obligations,
                  renewal clauses, or negotiation risks.
                </p>
              </div>
            )}

            <AnimatePresence>
              {messages.map((message, index) => (
                <motion.div
                  key={`message-${index}`}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ duration: 0.2 }}
                  className={`flex ${
                    message.role === 'user' ? 'justify-end' : 'justify-start'
                  }`}
                >
                  <div
                    className={`max-w-xl rounded-2xl px-4 py-3 text-sm shadow-md ${
                      message.role === 'user'
                        ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white'
                        : 'bg-white text-slate-900'
                    }`}
                  >
                    <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
                    {message.role === 'assistant' && message.plan && message.plan.length > 0 && (
                      <details className="mt-3 rounded-lg bg-blue-50/80 p-3 text-blue-900">
                        <summary className="cursor-pointer text-sm font-semibold">Execution plan</summary>
                        <ol className="mt-2 list-decimal space-y-1 pl-4 text-sm">
                          {message.plan.map((step, stepIndex) => (
                            <li key={`plan-${stepIndex}`}>{step.description ?? JSON.stringify(step)}</li>
                          ))}
                        </ol>
                      </details>
                    )}
                    {message.role === 'assistant' && message.trace && message.trace.length > 0 && (
                      <details className="mt-3 rounded-lg bg-blue-50/80 p-3 text-blue-900">
                        <summary className="cursor-pointer text-sm font-semibold">Tool trace</summary>
                        <ul className="mt-2 space-y-2 text-sm">
                          {message.trace.map((entry, traceIndex) => (
                            <li key={`trace-${traceIndex}`} className="rounded-md bg-white/80 p-2">
                              <code className="text-xs text-blue-800/90">
                                {JSON.stringify(entry, null, 2)}
                              </code>
                            </li>
                          ))}
                        </ul>
                      </details>
                    )}
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>

            {isLoading && (
              <div className="flex items-center gap-2 rounded-full bg-white/80 px-4 py-2 text-sm text-blue-700 shadow-sm">
                <Loader2 className="h-4 w-4 animate-spin" /> Analyzing contract...
              </div>
            )}
          </div>

          <form onSubmit={handleSubmit} className="flex flex-col gap-3">
            <Textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Ask about obligations, renewal clauses, risks, or negotiation strategy..."
              className="min-h-[120px] rounded-2xl border border-blue-200 bg-white/90 text-sm text-slate-900 shadow-sm focus:border-blue-300 focus-visible:outline-none"
            />
            <div className="flex flex-col items-stretch gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div className="text-xs text-blue-700/80">
                Tip: Upload a contract or paste its text before asking questions to ground the agent in your document.
              </div>
              <Button
                type="submit"
                disabled={isLoading}
                className="inline-flex items-center gap-2 rounded-full bg-gradient-to-r from-blue-600 to-indigo-600 px-6 py-2 text-sm font-semibold text-white shadow-lg transition hover:from-blue-500 hover:to-indigo-500 disabled:cursor-not-allowed disabled:from-slate-300 disabled:to-slate-300"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Thinking
                  </>
                ) : (
                  <>
                    <Send className="h-4 w-4" />
                    Analyze
                  </>
                )}
              </Button>
            </div>
          </form>
        </div>
      </section>
    </div>
  );
}
