import { ChatInterface } from './components/ChatInterface';
import { Scale } from 'lucide-react';

function App() {
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center py-6 px-4">
      <header className="mb-6 text-center">
        <div className="flex items-center justify-center gap-2 mb-1">
          <Scale className="w-8 h-8 text-blue-700" />
          <h1 className="text-2xl font-bold text-gray-800">LexiBot</h1>
        </div>
        <p className="text-sm text-gray-500">AI-Powered Legal Assistant</p>
      </header>

      <main className="w-full max-w-5xl flex flex-col items-center flex-1">
        <ChatInterface />
      </main>
    </div>
  );
}

export default App;
