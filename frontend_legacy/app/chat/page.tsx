import { cookies } from 'next/headers';

import { Chat } from '@/components/chat';
import { DEFAULT_CHAT_MODEL } from '@/lib/ai/models';
import { generateUUID } from '@/lib/utils';
import { DataStreamHandler } from '@/components/data-stream-handler';
import { auth } from '../(auth)/auth';
import { redirect } from 'next/navigation';

type ChatSession = Awaited<ReturnType<typeof auth>> & {
  user: {
    id: string;
    type: 'guest' | 'regular';
  };
};

const authDisabled =
  process.env.ENABLE_AUTH === '0' || process.env.ENABLE_AUTH === 'false';

export default async function Page() {
  let session = (await auth()) as ChatSession | null;

  if (!session) {
    if (authDisabled) {
      session = {
        user: {
          id: 'guest',
          email: 'guest@example.com',
          type: 'guest',
          name: 'Guest User',
        } as any,
        expires: new Date(Date.now() + 86400000).toISOString(),
      };
    } else {
      redirect('/api/auth/guest');
    }
  }

  const id = generateUUID();

  const cookieStore = await cookies();
  const modelIdFromCookie = cookieStore.get('chat-model');

  if (!modelIdFromCookie) {
    return (
      <>
        <Chat
          key={id}
          id={id}
          initialMessages={[]}
          initialChatModel={DEFAULT_CHAT_MODEL}
          initialVisibilityType="private"
          isReadonly={false}
          session={session}
          autoResume={false}
        />
        <DataStreamHandler />
      </>
    );
  }

  return (
    <>
      <Chat
        key={id}
        id={id}
        initialMessages={[]}
        initialChatModel={modelIdFromCookie.value}
        initialVisibilityType="private"
        isReadonly={false}
        session={session}
        autoResume={false}
      />
      <DataStreamHandler />
    </>
  );
}
