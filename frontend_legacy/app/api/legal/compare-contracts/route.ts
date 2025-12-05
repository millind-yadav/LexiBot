import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { contract_a, contract_b, focus_areas } = body;

    if (!contract_a || !contract_b) {
      return NextResponse.json(
        { success: false, error: 'Both contracts are required for comparison' },
        { status: 400 }
      );
    }

    // TODO: Replace with your actual API endpoint
    const API_BASE_URL = process.env.LEGAL_API_URL || 'http://localhost:8000/api/v1';
    const API_TOKEN = process.env.LEGAL_API_TOKEN || 'your-secret-token';

    const response = await fetch(`${API_BASE_URL}/compare-contracts`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${API_TOKEN}`,
      },
      body: JSON.stringify({
        contract_a,
        contract_b,
        focus_areas,
      }),
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.statusText}`);
    }

    const result = await response.json();
    
    return NextResponse.json(result);
  } catch (error) {
    console.error('Contract comparison error:', error);
    
    // For development/testing, return mock data
    if (process.env.NODE_ENV === 'development') {
      const mockResult = {
        success: true,
        analysis: {
          summary: "The contract comparison reveals significant differences in liability allocation, payment terms, and termination conditions. Contract A appears more balanced while Contract B heavily favors the service provider with limited client protections.",
          key_differences: [
            "Payment Terms: Contract A requires 50% upfront payment vs Contract B's 30% upfront. Contract A has milestone-based payments while Contract B uses time-based billing.",
            "Liability Clauses: Contract A caps liability at contract value ($100,000) while Contract B has unlimited liability exposure for the client.",
            "Termination Rights: Contract A allows 30-day termination for convenience with 25% penalty. Contract B requires 90-day notice and 75% penalty, making termination much more expensive.",
            "Intellectual Property: Contract A grants client full IP ownership vs Contract B where provider retains IP rights and grants limited license.",
            "Warranty Period: Contract A provides 12-month warranty vs Contract B's 90-day warranty, significantly different support coverage."
          ],
          risk_assessment: "MEDIUM to HIGH RISK differences identified. Contract B presents higher financial and legal risks for the client due to unlimited liability, restrictive termination terms, and limited IP rights. The payment structure in Contract B also increases cash flow risk. Contract A offers more balanced risk allocation and better protection for both parties.",
          recommendations: [
            "If choosing Contract B, negotiate liability cap similar to Contract A's structure to limit financial exposure",
            "Request modification of termination penalty in Contract B from 75% to 25% to match industry standards",
            "Clarify IP ownership rights in Contract B - consider negotiating for at least partial IP ownership or broader license terms",
            "Extend warranty period in Contract B to match Contract A's 12-month coverage for better long-term protection",
            "Consider Contract A's milestone-based payment structure as it provides better project oversight and reduced payment risk"
          ]
        },
        confidence_score: 0.89,
        processing_time: 4.2
      };
      
      // Simulate processing delay
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      return NextResponse.json(mockResult);
    }
    
    return NextResponse.json(
      { 
        success: false, 
        error: 'Failed to compare contracts',
        details: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}