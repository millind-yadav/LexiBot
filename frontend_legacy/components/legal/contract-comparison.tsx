'use client';

import { useState, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Textarea } from '@/components/ui/textarea';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Copy, GitCompare, AlertTriangle, Info } from 'lucide-react';
import { toast } from '@/components/toast';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';

interface ComparisonResult {
  success: boolean;
  analysis: {
    summary: string;
    key_differences: string[];
    risk_assessment: string;
    recommendations: string[];
  };
  confidence_score: number;
  processing_time: number;
}

export function ContractComparison() {
  const [contractA, setContractA] = useState('');
  const [contractB, setContractB] = useState('');
  const [isComparing, setIsComparing] = useState(false);
  const [comparisonResult, setComparisonResult] = useState<ComparisonResult | null>(null);

  const compareContracts = useCallback(async () => {
    if (!contractA.trim() || !contractB.trim()) {
      toast({ type: 'error', description: 'Please provide both contracts to compare' });
      return;
    }

    setIsComparing(true);

    try {
      const response = await fetch('/api/legal/compare-contracts', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          contract_a: contractA,
          contract_b: contractB,
          focus_areas: ['Payment Terms', 'Termination Clauses', 'Liability', 'Intellectual Property']
        }),
      });

      if (!response.ok) {
        throw new Error(`Comparison failed: ${response.statusText}`);
      }

      const result: ComparisonResult = await response.json();
      setComparisonResult(result);
      toast({ type: 'success', description: 'Contracts compared successfully' });
    } catch (error) {
      console.error('Comparison error:', error);
      toast({ type: 'error', description: 'Failed to compare contracts. Please try again.' });
    } finally {
      setIsComparing(false);
    }
  }, [contractA, contractB]);

  const copyToClipboard = useCallback(async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      toast({ type: 'success', description: 'Copied to clipboard' });
    } catch (error) {
      toast({ type: 'error', description: 'Failed to copy to clipboard' });
    }
  }, []);

  return (
    <div className="w-full max-w-6xl mx-auto space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <GitCompare className="w-5 h-5" />
            Contract Comparison
          </CardTitle>
          <CardDescription>
            Compare two contracts side-by-side and identify key differences
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">Contract A</label>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => copyToClipboard(contractA)}
                  disabled={!contractA}
                >
                  <Copy className="w-4 h-4" />
                </Button>
              </div>
              <Textarea
                placeholder="Paste first contract here..."
                value={contractA}
                onChange={(e) => setContractA(e.target.value)}
                className="min-h-[300px] font-mono text-sm"
              />
              <div className="text-xs text-gray-500">
                {contractA.length} characters
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">Contract B</label>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => copyToClipboard(contractB)}
                  disabled={!contractB}
                >
                  <Copy className="w-4 h-4" />
                </Button>
              </div>
              <Textarea
                placeholder="Paste second contract here..."
                value={contractB}
                onChange={(e) => setContractB(e.target.value)}
                className="min-h-[300px] font-mono text-sm"
              />
              <div className="text-xs text-gray-500">
                {contractB.length} characters
              </div>
            </div>
          </div>

          <Button 
            onClick={compareContracts} 
            disabled={!contractA.trim() || !contractB.trim() || isComparing}
            className="w-full mt-6"
          >
            {isComparing ? 'Comparing Contracts...' : 'Compare Contracts'}
          </Button>
        </CardContent>
      </Card>

      {comparisonResult && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              Comparison Results
              <Badge variant="outline">
                {(comparisonResult.confidence_score * 100).toFixed(1)}% confidence
              </Badge>
            </CardTitle>
            <CardDescription>
              Analysis completed in {comparisonResult.processing_time.toFixed(2)} seconds
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Summary */}
            <div className="space-y-2">
              <h3 className="flex items-center gap-2 font-semibold">
                <Info className="w-4 h-4" />
                Executive Summary
              </h3>
              <div className="p-4 bg-blue-50 rounded-lg border">
                <p className="text-sm whitespace-pre-wrap">{comparisonResult.analysis.summary}</p>
              </div>
            </div>

            <Separator />

            {/* Key Differences */}
            <div className="space-y-2">
              <h3 className="flex items-center gap-2 font-semibold">
                <GitCompare className="w-4 h-4" />
                Key Differences
              </h3>
              <div className="space-y-2">
                {comparisonResult.analysis.key_differences?.map((difference, index) => (
                  <Collapsible key={index}>
                    <CollapsibleTrigger asChild>
                      <Button variant="ghost" className="w-full justify-start p-2 h-auto">
                        <Badge variant="outline" className="mr-2">
                          #{index + 1}
                        </Badge>
                        <span className="text-left text-sm">{difference.substring(0, 100)}...</span>
                      </Button>
                    </CollapsibleTrigger>
                    <CollapsibleContent>
                      <div className="p-4 bg-gray-50 rounded-lg border ml-4">
                        <p className="text-sm whitespace-pre-wrap">{difference}</p>
                      </div>
                    </CollapsibleContent>
                  </Collapsible>
                ))}
              </div>
            </div>

            <Separator />

            {/* Risk Assessment */}
            <div className="space-y-2">
              <h3 className="flex items-center gap-2 font-semibold">
                <AlertTriangle className="w-4 h-4 text-yellow-600" />
                Risk Assessment
              </h3>
              <div className="p-4 bg-yellow-50 rounded-lg border border-yellow-200">
                <p className="text-sm whitespace-pre-wrap">{comparisonResult.analysis.risk_assessment}</p>
              </div>
            </div>

            <Separator />

            {/* Recommendations */}
            <div className="space-y-2">
              <h3 className="font-semibold">Recommendations</h3>
              <div className="space-y-2">
                {comparisonResult.analysis.recommendations?.map((recommendation, index) => (
                  <div key={index} className="flex items-start gap-2 p-3 bg-green-50 rounded-lg border border-green-200">
                    <Badge variant="secondary" className="mt-0.5">
                      {index + 1}
                    </Badge>
                    <p className="text-sm flex-1">{recommendation}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Export Options */}
            <div className="flex gap-2 pt-4 border-t">
              <Button variant="outline" size="sm">
                Export PDF Report
              </Button>
              <Button variant="outline" size="sm">
                Save to Dashboard
              </Button>
              <Button 
                variant="outline" 
                size="sm"
                onClick={() => copyToClipboard(JSON.stringify(comparisonResult, null, 2))}
              >
                Copy Raw Data
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}