'use client';

import { useState, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Loader2, Upload, FileText, AlertCircle, CheckCircle } from 'lucide-react';
import { toast } from '@/components/toast';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { cn } from '@/lib/utils';
import { Suggestions, Suggestion } from '@/components/elements/suggestion';

interface AnalysisResult {
  success: boolean;
  analysis: Record<string, string>;
  confidence_score: number;
  processing_time: number;
  model_version: string;
  risk_level?: 'low' | 'medium' | 'high';
}

interface ContractAnalyzerProps {
  onAnalysisComplete?: (result: AnalysisResult) => void;
  variant?: 'standalone' | 'embedded';
  className?: string;
  suggestions?: string[];
}

export function ContractAnalyzer({ onAnalysisComplete, variant = 'standalone', className, suggestions }: ContractAnalyzerProps) {
  const [contractText, setContractText] = useState('');
  const [analysisType, setAnalysisType] = useState('comprehensive');
  const [customQuestions, setCustomQuestions] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);

  const promptList = suggestions ?? [
    'Summarize the key obligations in this services agreement.',
    'Highlight termination risks and notice periods.',
    'Draft a mutual NDA clause with California governing law.',
    'Compare indemnification language between two contracts.',
  ];

  const handleSuggestionClick = useCallback((prompt: string) => {
    setAnalysisType('specific');
    setCustomQuestions((current) => {
      if (!current.trim()) return prompt;
      const lines = current.split('\n').map((line) => line.trim());
      if (lines.includes(prompt)) {
        toast({ type: 'success', description: 'Suggestion already added to custom questions.' });
        return current;
      }
      toast({ type: 'success', description: 'Added to custom questions. Adjust as needed.' });
      return `${current}\n${prompt}`.trim();
    });
  }, []);

  const analyzeContract = useCallback(async () => {
    if (!contractText.trim()) {
      toast({ type: 'error', description: 'Please provide contract text to analyze' });
      return;
    }

    setIsAnalyzing(true);
    setUploadProgress(0);

    try {
      // Simulate progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => Math.min(prev + 10, 90));
      }, 200);

      const questions = customQuestions
        .split('\n')
        .map(q => q.trim())
        .filter(q => q.length > 0);

      const response = await fetch('/api/legal/analyze-contract', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: contractText,
          analysis_type: analysisType,
          questions: questions.length > 0 ? questions : undefined,
        }),
      });

      clearInterval(progressInterval);
      setUploadProgress(100);

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`);
      }

      const result: AnalysisResult = await response.json();
      
      // Determine risk level based on analysis content
      const analysisText = Object.values(result.analysis).join(' ').toLowerCase();
      let riskLevel: 'low' | 'medium' | 'high' = 'low';
      
      if (analysisText.includes('high risk') || analysisText.includes('severe') || analysisText.includes('critical')) {
        riskLevel = 'high';
      } else if (analysisText.includes('medium risk') || analysisText.includes('caution') || analysisText.includes('moderate')) {
        riskLevel = 'medium';
      }
      
      result.risk_level = riskLevel;
      setAnalysisResult(result);
      onAnalysisComplete?.(result);
      
      toast({ type: 'success', description: `Analysis completed with ${(result.confidence_score * 100).toFixed(1)}% confidence` });
    } catch (error) {
      console.error('Analysis error:', error);
      toast({ type: 'error', description: 'Failed to analyze contract. Please try again.' });
    } finally {
      setIsAnalyzing(false);
      setUploadProgress(0);
    }
  }, [contractText, analysisType, customQuestions, onAnalysisComplete]);

  const handleFileUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (file.type === 'text/plain') {
        const reader = new FileReader();
        reader.onload = (e) => {
          setContractText(e.target?.result as string);
          toast({ type: 'success', description: 'Contract file loaded successfully' });
        };
        reader.readAsText(file);
      } else {
        toast({ type: 'error', description: 'Please upload a text file (.txt). PDF support coming soon!' });
      }
    }
  }, []);

  const getRiskBadgeColor = (riskLevel?: string) => {
    switch (riskLevel) {
      case 'high': return 'destructive';
      case 'medium': return 'default';
      case 'low': return 'secondary';
      default: return 'outline';
    }
  };

  const formContent = (
    <>
      <Tabs defaultValue="text" className="w-full">
        <TabsList className="rounded-full border border-blue-200 bg-white/80 p-1 text-blue-700 shadow-sm">
          <TabsTrigger value="text" className="rounded-full px-4 py-2 text-sm font-medium">
            Paste Text
          </TabsTrigger>
          <TabsTrigger value="upload" className="rounded-full px-4 py-2 text-sm font-medium">
            Upload File
          </TabsTrigger>
        </TabsList>

        <TabsContent value="text" className="space-y-4">
          <Textarea
            placeholder="Paste your contract text here..."
            value={contractText}
            onChange={(e) => setContractText(e.target.value)}
            className="min-h-[200px] rounded-2xl border border-blue-100 bg-white/95 text-slate-900 shadow-sm focus:border-blue-300 focus:ring-0 focus-visible:outline-none"
          />
        </TabsContent>

        <TabsContent value="upload" className="space-y-4">
          <div className="rounded-2xl border-2 border-dashed border-blue-100 bg-white/70 p-6 text-center shadow-sm">
            <input
              type="file"
              accept=".txt"
              onChange={handleFileUpload}
              className="hidden"
              id="contract-upload"
            />
            <label htmlFor="contract-upload" className="cursor-pointer text-blue-800/80 hover:text-blue-900">
              <Upload className="mx-auto mb-4 h-12 w-12 text-blue-400" />
              <p className="text-sm">Click to upload contract file (.txt)</p>
            </label>
          </div>
          {contractText && (
            <div className="text-sm text-blue-700/90">
              Contract loaded ({contractText.length} characters)
            </div>
          )}
        </TabsContent>
      </Tabs>

      <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-2">
        <div>
          <label className="text-sm font-medium text-blue-900">Analysis Type</label>
          <Select value={analysisType} onValueChange={setAnalysisType}>
            <SelectTrigger className="mt-2 h-11 rounded-xl border-blue-100 bg-white/95 text-blue-900 shadow-sm focus:border-blue-300 focus:ring-0 focus:ring-offset-0">
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="border-blue-100 bg-white/95 text-blue-900">
              <SelectItem value="comprehensive">Comprehensive Analysis</SelectItem>
              <SelectItem value="quick">Quick Overview</SelectItem>
              <SelectItem value="specific">Custom Questions</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {analysisType === 'specific' && (
        <div className="mt-4">
          <label className="text-sm font-medium text-blue-900">
            Custom Questions (one per line)
          </label>
          <Textarea
            placeholder={`Who are the main parties?
What are the key risks?
What are the termination conditions?`}
            value={customQuestions}
            onChange={(e) => setCustomQuestions(e.target.value)}
            className="mt-2 rounded-2xl border border-blue-100 bg-white/95 text-slate-900 shadow-sm focus:border-blue-300 focus:ring-0 focus-visible:outline-none"
          />
        </div>
      )}

      {isAnalyzing && (
        <div className="mt-4 space-y-2 rounded-xl border border-blue-100 bg-white/80 p-3 text-blue-900">
          <div className="flex items-center gap-2">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span className="text-sm">Analyzing contract...</span>
          </div>
          <Progress value={uploadProgress} className="h-2 rounded-full bg-blue-100" />
        </div>
      )}

      <Button
        onClick={analyzeContract}
        disabled={!contractText.trim() || isAnalyzing}
        className="mt-6 w-full rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-md transition hover:from-blue-500 hover:to-indigo-500 disabled:from-slate-300 disabled:to-slate-300"
      >
        {isAnalyzing ? 'Analyzing...' : 'Analyze Contract'}
      </Button>
    </>
  );

  const resultsContent = analysisResult && (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          Analysis Results
          <div className="flex items-center gap-2">
            <Badge variant={getRiskBadgeColor(analysisResult.risk_level)}>
              {analysisResult.risk_level?.toUpperCase() || 'UNKNOWN'} RISK
            </Badge>
            <Badge variant="outline">
              {(analysisResult.confidence_score * 100).toFixed(1)}% confidence
            </Badge>
          </div>
        </CardTitle>
        <CardDescription>
          Processed in {analysisResult.processing_time.toFixed(2)}s • Model: {analysisResult.model_version}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {Object.entries(analysisResult.analysis).map(([question, answer], index) => (
          <div key={index} className="rounded-lg border p-4">
            <h4 className="mb-2 text-sm font-medium text-gray-600">{question}</h4>
            <div className="prose prose-sm max-w-none">
              <p className="whitespace-pre-wrap">{answer}</p>
            </div>
          </div>
        ))}

        <div className="flex items-center justify-between border-t pt-4">
          <div className="flex items-center gap-2">
            {analysisResult.risk_level === 'low' && (
              <CheckCircle className="h-4 w-4 text-green-500" />
            )}
            {analysisResult.risk_level === 'medium' && (
              <AlertCircle className="h-4 w-4 text-yellow-500" />
            )}
            {analysisResult.risk_level === 'high' && (
              <AlertCircle className="h-4 w-4 text-red-500" />
            )}
            <span className="text-sm text-gray-600">
              Risk Assessment: {analysisResult.risk_level?.toUpperCase()}
            </span>
          </div>

          <Button variant="outline" size="sm">
            Export Report
          </Button>
        </div>
      </CardContent>
    </Card>
  );

  if (variant === 'embedded') {
    return (
      <div className={cn('w-full space-y-8', className)}>
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-blue-900">
            <FileText className="h-5 w-5" />
            <h2 className="text-xl font-semibold">Legal Contract Analyzer</h2>
          </div>
          <p className="text-sm text-blue-700/90">
            AI-powered contract analysis using your fine-tuned legal model
          </p>
        </div>
        {promptList.length > 0 && (
          <div className="rounded-3xl border border-blue-100/60 bg-gradient-to-br from-blue-50/80 via-white to-indigo-50/80 p-4 shadow-sm">
            <p className="mb-3 text-xs font-semibold uppercase tracking-widest text-blue-600/80">
              Quick legal prompts
            </p>
            <Suggestions className="gap-3">
              {promptList.map((prompt) => (
                <Suggestion
                  key={prompt}
                  suggestion={prompt}
                  onClick={handleSuggestionClick}
                  className="flex h-auto min-w-[220px] items-center justify-start whitespace-normal rounded-full border border-blue-200 bg-white/90 px-4 py-2 text-sm font-medium text-blue-900 shadow-sm transition hover:bg-blue-100"
                />
              ))}
            </Suggestions>
            <p className="mt-3 text-xs text-blue-700/70">
              Selecting a prompt adds it to the custom questions list for this analysis.
            </p>
          </div>
        )}
        <div className="rounded-3xl border border-blue-100 bg-gradient-to-b from-white via-white to-blue-50 p-6 shadow-lg">
          <div className="space-y-6">{formContent}</div>
        </div>
        {analysisResult && (
          <div className="space-y-4 rounded-2xl border border-blue-100 bg-gradient-to-b from-white via-white to-blue-100/70 p-6 shadow-lg">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <h3 className="text-lg font-semibold text-blue-900">Analysis Results</h3>
                <p className="text-sm text-blue-700/90">
                  Processed in {analysisResult.processing_time.toFixed(2)}s · Model: {analysisResult.model_version}
                </p>
              </div>
              <div className="flex items-center gap-2">
                <Badge variant={getRiskBadgeColor(analysisResult.risk_level)}>
                  {analysisResult.risk_level?.toUpperCase() || 'UNKNOWN'} RISK
                </Badge>
                <Badge variant="outline">
                  {(analysisResult.confidence_score * 100).toFixed(1)}% confidence
                </Badge>
              </div>
            </div>
            <div className="space-y-3">
              {Object.entries(analysisResult.analysis).map(([question, answer], index) => (
                <div key={index} className="rounded-xl border border-blue-100 bg-white/90 p-4 shadow-sm">
                  <h4 className="mb-2 text-sm font-medium text-blue-800/90">{question}</h4>
                  <p className="whitespace-pre-wrap text-sm text-slate-700">{answer}</p>
                </div>
              ))}
            </div>
            <div className="flex items-center justify-between border-t border-blue-100 pt-4">
              <div className="flex items-center gap-2 text-sm text-blue-800/90">
                {analysisResult.risk_level === 'low' && (
                  <CheckCircle className="h-4 w-4 text-emerald-500" />
                )}
                {analysisResult.risk_level === 'medium' && (
                  <AlertCircle className="h-4 w-4 text-amber-500" />
                )}
                {analysisResult.risk_level === 'high' && (
                  <AlertCircle className="h-4 w-4 text-red-500" />
                )}
                <span>Risk Assessment: {analysisResult.risk_level?.toUpperCase()}</span>
              </div>
              <Button variant="outline" size="sm" className="border-blue-200 text-blue-800 hover:bg-blue-100">
                Export Report
              </Button>
            </div>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className={cn('mx-auto w-full max-w-4xl space-y-6', className)}>
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileText className="h-5 w-5" />
            Legal Contract Analyzer
          </CardTitle>
          <CardDescription>
            AI-powered contract analysis using your fine-tuned legal model
          </CardDescription>
        </CardHeader>
        <CardContent>{formContent}</CardContent>
      </Card>

      {resultsContent}
    </div>
  );
}