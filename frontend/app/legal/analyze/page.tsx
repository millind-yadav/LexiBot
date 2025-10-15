import Link from 'next/link';

import { ContractAnalyzer } from '@/components/legal/contract-analyzer';

const PREDEFINED_PROMPTS = [
  'Summarize the obligations and renewal terms in this contract.',
  'Highlight potential termination risks and associated notice periods.',
  'List the indemnification clauses and explain who bears the liability.',
  'Which sections describe payment milestones and late payment penalties?',
];

export default function AnalyzeContractsPage() {
  return (
    <div className="flex min-h-screen flex-col bg-background">
      <header className="border-b border-border/80 bg-card/60 backdrop-blur">
        <div className="mx-auto flex w-full max-w-5xl items-center justify-between px-4 py-6 md:px-6">
          <div className="flex items-center gap-3 text-sm">
            <span className="inline-flex h-9 w-9 items-center justify-center rounded-2xl border border-border/60 bg-card text-sm font-semibold text-foreground">
              LA
            </span>
            <div className="flex flex-col">
              <span className="text-xs uppercase tracking-[0.28em] text-muted-foreground">LexiBot Legal AI</span>
              <span className="text-base font-semibold text-foreground">Contract Analysis Studio</span>
            </div>
          </div>
          <nav className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
            <Link href="/" className="rounded-full border border-border/60 px-3 py-1 hover:bg-muted/40">
              Home
            </Link>
            <Link href="/legal" className="rounded-full border border-border/60 px-3 py-1 hover:bg-muted/40">
              Legal AI
            </Link>
            <span className="rounded-full bg-primary/10 px-3 py-1 text-primary">Analyze Contracts</span>
          </nav>
        </div>
      </header>

      <main className="flex flex-1 justify-center px-4 py-8 md:px-6 md:py-12">
        <div className="flex w-full max-w-4xl flex-col gap-6">
          <section className="rounded-3xl border border-border/70 bg-card/80 p-6 shadow-lg backdrop-blur">
            <h1 className="text-2xl font-semibold text-foreground md:text-3xl">
              Ask questions about your contract like you would in chat.
            </h1>
            <p className="mt-3 max-w-3xl text-sm text-muted-foreground md:text-base">
              Upload a .txt contract or paste its contents, then explore obligations, renewal language, and potential
              risks using the prompts below or your own follow-up questions. The layout mirrors the chat workspace so
              your team can stay in a familiar flow.
            </p>
          </section>

          <section className="rounded-3xl border border-border/70 bg-card/90 p-6 shadow-xl">
            <ContractAnalyzer variant="embedded" suggestions={PREDEFINED_PROMPTS} />
          </section>
        </div>
      </main>
    </div>
  );
}
