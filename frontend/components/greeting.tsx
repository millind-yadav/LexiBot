import { motion } from 'framer-motion';
import { MessageCircle, Scale } from 'lucide-react';

export const Greeting = () => {
  return (
    <div
      key="overview"
      className="mx-auto mt-6 flex size-full max-w-4xl flex-col justify-center px-4 md:mt-12 md:px-8"
    >
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: 10 }}
        transition={{ delay: 0.5 }}
        className="rounded-3xl bg-gradient-to-br from-blue-600/10 via-blue-600/5 to-indigo-600/10 p-8 shadow-sm backdrop-blur"
      >
        <div className="mb-4 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-blue-700">
          <Scale className="h-5 w-5" />
          LexiBot Legal AI
        </div>
        <div className="mb-3 text-3xl font-bold text-slate-900 md:text-4xl">
          Welcome to your legal co-pilot
        </div>
        <div className="mb-6 text-lg text-slate-600 md:text-xl">
          Review contracts, surface redlines, and draft precise clauses with the same AI that powers the homepage experience.
        </div>
        <div className="flex flex-wrap gap-3 text-sm text-blue-700 md:text-base">
          <span className="flex items-center gap-2 rounded-full bg-white/70 px-4 py-2 shadow-sm">
            <MessageCircle className="h-4 w-4" />
            Ask contract questions in natural language
          </span>
          <span className="flex items-center gap-2 rounded-full bg-white/70 px-4 py-2 shadow-sm">
            Review AI-suggested clauses instantly
          </span>
        </div>
      </motion.div>
    </div>
  );
};
