import { NextResponse } from 'next/server';

/**
 * This route previously proxied chat requests to the backend agent. The legal analyzer page
 * now uses the contract analysis endpoints directly, so the proxy is intentionally disabled
 * to prevent accidental usage.
 */
export async function POST() {
  return NextResponse.json(
    {
      error:
        'The legal chat proxy has been disabled. Use /api/legal/analyze-contract or related endpoints instead.',
    },
    { status: 410 },
  );
}
