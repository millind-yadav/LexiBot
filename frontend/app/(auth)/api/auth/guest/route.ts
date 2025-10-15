import { signIn } from '@/app/(auth)/auth';
import { NextResponse } from 'next/server';

const authDisabled =
  process.env.ENABLE_AUTH === '0' || process.env.ENABLE_AUTH === 'false';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const redirectUrl = searchParams.get('redirectUrl') || '/';

  if (authDisabled) {
    return NextResponse.redirect(new URL(redirectUrl, request.url));
  }

  return signIn('guest', { redirect: true, redirectTo: redirectUrl, redirectBaseUrl: request.url });
}
