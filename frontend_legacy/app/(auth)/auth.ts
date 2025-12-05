import { compare } from 'bcrypt-ts';
import NextAuth, { type DefaultSession } from 'next-auth';
import type { DefaultJWT } from 'next-auth/jwt';
import Credentials from 'next-auth/providers/credentials';
import { NextResponse } from 'next/server';

import { createGuestUser, getUser } from '@/lib/db/queries';
import { DUMMY_PASSWORD, isDevelopmentEnvironment } from '@/lib/constants';
import { authConfig } from './auth.config';

export type UserType = 'guest' | 'regular';

declare module 'next-auth' {
  interface Session extends DefaultSession {
    user: {
      id: string;
      type: UserType;
    } & DefaultSession['user'];
  }

  interface User {
    id?: string;
    email?: string | null;
    type: UserType;
  }
}

declare module 'next-auth/jwt' {
  interface JWT extends DefaultJWT {
    id: string;
    type: UserType;
  }
}

const authDisabled =
  process.env.ENABLE_AUTH === '0' || process.env.ENABLE_AUTH === 'false';

let GET: (request: any) => Promise<Response>;
let POST: (request: any) => Promise<Response>;
let authImpl: () => Promise<DefaultSession | null>;
let signInImpl: (
  provider?: string,
  options?: {
    redirect?: boolean;
    redirectTo?: string;
    redirectBaseUrl?: string;
  },
) => Promise<Response | undefined>;
let signOutImpl: (options?: { redirectTo?: string }) => Promise<void | Response>;

if (authDisabled) {
  const defaultSession: DefaultSession = {
    user: {
      id: process.env.DEFAULT_GUEST_ID ?? 'guest',
      email: process.env.DEFAULT_GUEST_EMAIL ?? 'guest@example.com',
      type: 'guest',
      name: 'Guest User',
    } as DefaultSession['user'] & { type: UserType },
    expires: new Date(Date.now() + 86400000).toISOString(),
  };

  const disabledResponse = async () =>
    NextResponse.json(
      { error: 'Authentication is disabled in this environment.' },
      { status: 404 },
    );

  GET = disabledResponse;
  POST = disabledResponse;

  authImpl = async () => defaultSession;

  signInImpl = async (_provider, options) => {
    if (options?.redirect) {
      const baseUrl =
        options.redirectBaseUrl ??
        process.env.NEXT_PUBLIC_APP_URL ??
        (isDevelopmentEnvironment ? 'http://localhost:3000' : '/');
      const destination = options.redirectTo ?? '/';
      return NextResponse.redirect(new URL(destination, baseUrl));
    }

    return undefined;
  };

  signOutImpl = async () => undefined;
} else {
  const nextAuthExports = NextAuth({
    ...authConfig,
    providers: [
      Credentials({
        credentials: {},
        async authorize({ email, password }: any) {
          const users = await getUser(email);

          if (users.length === 0) {
            await compare(password, DUMMY_PASSWORD);
            return null;
          }

          const [user] = users;

          if (!user.password) {
            await compare(password, DUMMY_PASSWORD);
            return null;
          }

          const passwordsMatch = await compare(password, user.password);

          if (!passwordsMatch) return null;

          return { ...user, type: 'regular' };
        },
      }),
      Credentials({
        id: 'guest',
        credentials: {},
        async authorize() {
          const [guestUser] = await createGuestUser();
          return { ...guestUser, type: 'guest' };
        },
      }),
    ],
    callbacks: {
      async jwt({ token, user }) {
        if (user) {
          token.id = user.id as string;
          token.type = user.type;
        }

        return token;
      },
      async session({ session, token }) {
        if (session.user) {
          session.user.id = token.id;
          session.user.type = token.type;
        }

        return session;
      },
    },
  });

  ({ GET, POST } = nextAuthExports.handlers);
  authImpl = nextAuthExports.auth;
  signInImpl = nextAuthExports.signIn;
  signOutImpl = nextAuthExports.signOut;
}

export const auth = authImpl;
export const signIn = signInImpl;
export const signOut = signOutImpl;
export { GET, POST };
