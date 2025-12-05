import { ContractAnalyzer } from '@/components/legal/contract-analyzer';
import { ContractComparison } from '@/components/legal/contract-comparison';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Scale, FileText, GitCompare, Bot, Zap, Shield, Clock, Target, CheckCircle, ArrowRight, Users, TrendingUp, Award } from 'lucide-react';

export default function LegalAIPage() {
  return (
    <div className="min-h-screen bg-background">
      {/* Hero Section */}
      <div className="relative overflow-hidden bg-gradient-to-br from-blue-600 via-blue-700 to-indigo-800 text-white">
        <div className="absolute inset-0 bg-black/10"></div>
        <div className="relative container mx-auto px-4 py-16 md:py-24">
          <div className="max-w-4xl mx-auto text-center">
            <div className="flex items-center justify-center gap-2 mb-6">
              <Scale className="w-12 h-12" />
              <span className="text-2xl font-bold">LexiBot</span>
            </div>
            
            <h1 className="text-4xl md:text-6xl font-bold mb-6 leading-tight">
              AI-Powered Legal
              <span className="text-blue-200"> Document Analysis</span>
            </h1>
            
            <p className="text-xl md:text-2xl mb-8 text-blue-100 max-w-3xl mx-auto">
              Transform your legal workflow with our fine-tuned Llama 3.2-3B model. 
              Analyze contracts, extract key terms, and identify risks in seconds.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center mb-12">
              <Button size="lg" className="bg-white text-blue-700 hover:bg-blue-50 px-8 py-3">
                <FileText className="w-5 h-5 mr-2" />
                Try Contract Analysis
              </Button>
              <Button variant="outline" size="lg" className="border-white text-white hover:bg-white/10 px-8 py-3">
                <Bot className="w-5 h-5 mr-2" />
                Watch Demo
              </Button>
            </div>

            <div className="flex items-center justify-center gap-6 text-sm text-blue-200">
              <div className="flex items-center gap-1">
                <CheckCircle className="w-4 h-4" />
                95%+ Accuracy
              </div>
              <div className="flex items-center gap-1">
                <CheckCircle className="w-4 h-4" />
                5x Faster Analysis
              </div>
              <div className="flex items-center gap-1">
                <CheckCircle className="w-4 h-4" />
                Enterprise Ready
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Stats Section */}
      <div className="bg-gray-50 py-12">
        <div className="container mx-auto px-4">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8 text-center">
            <div>
              <div className="text-3xl font-bold text-blue-600 mb-2">500K+</div>
              <div className="text-gray-600">Contracts Analyzed</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-blue-600 mb-2">95%</div>
              <div className="text-gray-600">Accuracy Rate</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-blue-600 mb-2">5x</div>
              <div className="text-gray-600">Faster Than Manual</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-blue-600 mb-2">24/7</div>
              <div className="text-gray-600">AI Availability</div>
            </div>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="py-16">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              Powerful Legal AI Features
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Built with cutting-edge AI technology and trained specifically for legal document analysis
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
            <Card className="relative group hover:shadow-lg transition-shadow">
              <CardHeader>
                <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4 group-hover:bg-blue-200 transition-colors">
                  <FileText className="w-6 h-6 text-blue-600" />
                </div>
                <CardTitle>Smart Contract Analysis</CardTitle>
                <CardDescription>
                  Comprehensive legal document analysis with risk assessment and key term extraction using our fine-tuned model
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2 text-sm text-gray-600">
                  <li className="flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    Party identification
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    Risk assessment
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    Key terms extraction
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    Compliance checking
                  </li>
                </ul>
              </CardContent>
            </Card>

            <Card className="relative group hover:shadow-lg transition-shadow">
              <CardHeader>
                <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mb-4 group-hover:bg-green-200 transition-colors">
                  <GitCompare className="w-6 h-6 text-green-600" />
                </div>
                <CardTitle>Document Comparison</CardTitle>
                <CardDescription>
                  Side-by-side contract comparison with intelligent difference highlighting and risk analysis
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2 text-sm text-gray-600">
                  <li className="flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    Clause-by-clause comparison
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    Difference highlighting
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    Risk impact analysis
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    Recommendation engine
                  </li>
                </ul>
              </CardContent>
            </Card>

            <Card className="relative group hover:shadow-lg transition-shadow">
              <CardHeader>
                <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mb-4 group-hover:bg-purple-200 transition-colors">
                  <Bot className="w-6 h-6 text-purple-600" />
                </div>
                <CardTitle>AI Agent Assistant</CardTitle>
                <CardDescription>
                  Interactive legal AI assistant with multi-step reasoning capabilities for complex legal queries
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2 text-sm text-gray-600">
                  <li className="flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    Natural language queries
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    Multi-step reasoning
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    Legal research assistance
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    Document drafting help
                  </li>
                </ul>
              </CardContent>
            </Card>
          </div>

          {/* Benefits Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-16">
            <Card className="bg-gradient-to-r from-blue-50 to-indigo-50 border-blue-200">
              <CardHeader>
                <div className="flex items-center gap-3">
                  <Clock className="w-8 h-8 text-blue-600" />
                  <div>
                    <CardTitle className="text-blue-900">Save Time & Money</CardTitle>
                    <CardDescription className="text-blue-700">
                      Reduce legal document review time by up to 80% with AI-powered analysis
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>
            </Card>

            <Card className="bg-gradient-to-r from-green-50 to-emerald-50 border-green-200">
              <CardHeader>
                <div className="flex items-center gap-3">
                  <Shield className="w-8 h-8 text-green-600" />
                  <div>
                    <CardTitle className="text-green-900">Minimize Legal Risks</CardTitle>
                    <CardDescription className="text-green-700">
                      Identify potential risks and compliance issues before they become problems
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>
            </Card>

            <Card className="bg-gradient-to-r from-purple-50 to-violet-50 border-purple-200">
              <CardHeader>
                <div className="flex items-center gap-3">
                  <Target className="w-8 h-8 text-purple-600" />
                  <div>
                    <CardTitle className="text-purple-900">Increase Accuracy</CardTitle>
                    <CardDescription className="text-purple-700">
                      Leverage AI trained on legal documents for consistent, accurate analysis
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>
            </Card>

            <Card className="bg-gradient-to-r from-orange-50 to-red-50 border-orange-200">
              <CardHeader>
                <div className="flex items-center gap-3">
                  <TrendingUp className="w-8 h-8 text-orange-600" />
                  <div>
                    <CardTitle className="text-orange-900">Scale Your Practice</CardTitle>
                    <CardDescription className="text-orange-700">
                      Handle more contracts and clients with AI-powered efficiency
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>
            </Card>
          </div>
        </div>
      </div>

      {/* Technology Section */}
      <div className="bg-gray-50 py-16">
        <div className="container mx-auto px-4">
          <div className="max-w-3xl mx-auto text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              Powered by Advanced AI Technology
            </h2>
            <p className="text-xl text-gray-600">
              Built on Meta's Llama 3.2-3B model, fine-tuned specifically for legal document analysis using the CUAD dataset
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Award className="w-8 h-8 text-blue-600" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Fine-tuned Model</h3>
              <p className="text-gray-600">
                Specialized training on 500+ legal contracts for domain expertise
              </p>
            </div>

            <div className="text-center">
              <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Zap className="w-8 h-8 text-green-600" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Lightning Fast</h3>
              <p className="text-gray-600">
                Optimized inference pipeline delivers results in under 5 seconds
              </p>
            </div>

            <div className="text-center">
              <div className="w-16 h-16 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Users className="w-8 h-8 text-purple-600" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Enterprise Ready</h3>
              <p className="text-gray-600">
                Scalable architecture with enterprise security and compliance
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="bg-gradient-to-r from-blue-600 to-indigo-700 text-white py-16">
        <div className="container mx-auto px-4 text-center">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            Ready to Transform Your Legal Workflow?
          </h2>
          <p className="text-xl mb-8 text-blue-100 max-w-2xl mx-auto">
            Join leading law firms and legal departments using AI to streamline contract analysis
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button size="lg" className="bg-white text-blue-700 hover:bg-blue-50 px-8 py-3">
              Start Free Analysis
              <ArrowRight className="w-5 h-5 ml-2" />
            </Button>
            <Button variant="outline" size="lg" className="border-white text-white hover:bg-white/10 px-8 py-3">
              Schedule Demo
            </Button>
          </div>
        </div>
      </div>

      {/* Interactive Tools Section */}
      <div className="py-16">
        <div className="container mx-auto px-4">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              Try Our AI Tools
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Experience the power of legal AI with our interactive tools
            </p>
          </div>

          <Tabs defaultValue="analyze" className="w-full">
            <TabsList className="grid w-full grid-cols-3 max-w-md mx-auto">
              <TabsTrigger value="analyze" className="flex items-center gap-2">
                <FileText className="w-4 h-4" />
                <span className="hidden sm:inline">Analysis</span>
              </TabsTrigger>
              <TabsTrigger value="compare" className="flex items-center gap-2">
                <GitCompare className="w-4 h-4" />
                <span className="hidden sm:inline">Compare</span>
              </TabsTrigger>
              <TabsTrigger value="chat" className="flex items-center gap-2">
                <Bot className="w-4 h-4" />
                <span className="hidden sm:inline">AI Chat</span>
              </TabsTrigger>
            </TabsList>

            <div className="mt-8">
              <TabsContent value="analyze" className="space-y-4">
                <ContractAnalyzer />
              </TabsContent>

              <TabsContent value="compare" className="space-y-4">
                <ContractComparison />
              </TabsContent>

              <TabsContent value="chat" className="space-y-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Bot className="w-5 h-5" />
                      Legal AI Agent Chat
                    </CardTitle>
                    <CardDescription>
                      Interactive legal assistant with multi-step reasoning capabilities
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="text-center py-12 text-muted-foreground">
                      <Bot className="w-12 h-12 mx-auto mb-4 opacity-50" />
                      <p className="text-lg mb-2">AI Agent Chat Coming Soon</p>
                      <p className="text-sm">
                        Advanced conversational AI for complex legal queries and multi-step analysis
                      </p>
                      <Button className="mt-4" variant="outline">
                        Join Waitlist
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </div>
          </Tabs>
        </div>
      </div>
    </div>
  );
}