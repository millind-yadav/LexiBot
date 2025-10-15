import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { text, analysis_type = 'comprehensive', questions } = body;

    if (!text || text.trim().length === 0) {
      return NextResponse.json(
        { success: false, error: 'Contract text is required' },
        { status: 400 }
      );
    }

    // TODO: Replace with your actual API endpoint
    const API_BASE_URL = process.env.LEGAL_API_URL || 'http://localhost:8000/api/v1';
    const API_TOKEN = process.env.LEGAL_API_TOKEN || 'your-secret-token';

    const response = await fetch(`${API_BASE_URL}/analyze-contract`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${API_TOKEN}`,
      },
      body: JSON.stringify({
        text,
        analysis_type,
        questions,
      }),
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.statusText}`);
    }

    const result = await response.json();
    
    return NextResponse.json(result);
  } catch (error) {
    console.error('Contract analysis error:', error);
    
    // For development/testing, return mock data
    if (process.env.NODE_ENV === 'development') {
      const mockResult = {
        success: true,
        analysis: {
          "Who are the main parties?": "Based on the contract analysis, the main parties are Company A (the service provider) and Company B (the client). Company A is responsible for delivering software development services, while Company B will provide requirements and payment.",
          "What are the key terms?": "Key terms include: 1) Service delivery within 6 months, 2) Payment schedule of $50,000 in 3 installments, 3) Intellectual property rights remain with Company A, 4) 90-day warranty period, 5) Confidentiality obligations for both parties.",
          "What are the potential risks?": "HIGH RISK: The contract contains several concerning clauses: 1) Unlimited liability for Company B, 2) No force majeure protection, 3) Automatic renewal clause without clear termination rights, 4) Broad indemnification requirements. These could expose significant financial and legal risks.",
          "Are there any termination clauses?": "The contract includes a 30-day termination clause with written notice. However, there are significant penalties for early termination by Company B, including payment of 75% of remaining contract value. No termination rights for convenience are provided."
        },
        confidence_score: 0.92,
        processing_time: 3.4,
        model_version: "lexibot-v1.0"
      };
      
      // Simulate processing delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      return NextResponse.json(mockResult);
    }
    
    return NextResponse.json(
      { 
        success: false, 
        error: 'Failed to analyze contract',
        details: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}