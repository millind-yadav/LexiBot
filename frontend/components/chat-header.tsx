'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useWindowSize } from 'usehooks-ts';

import { SidebarToggle } from '@/components/sidebar-toggle';
import { Button } from '@/components/ui/button';
import { PlusIcon, VercelIcon } from './icons';
import { useSidebar } from './ui/sidebar';
import { memo } from 'react';
import { type VisibilityType, VisibilitySelector } from './visibility-selector';
import type { Session } from 'next-auth';

function PureChatHeader({
  chatId,
  selectedVisibilityType,
  isReadonly,
  session,
}: {
  chatId: string;
  selectedVisibilityType: VisibilityType;
  isReadonly: boolean;
  session: Session;
}) {
  const router = useRouter();
  const { open } = useSidebar();

  const { width: windowWidth } = useWindowSize();

  return (
    <header className="sticky top-0 z-20 flex items-center gap-2 rounded-t-3xl bg-gradient-to-r from-blue-600 to-indigo-700 px-3 py-2 text-white shadow-md md:px-4">
      <SidebarToggle className="border-white/30 bg-white/10 text-white hover:bg-white/20 hover:text-white" />

      {(!open || windowWidth < 768) && (
        <Button
          variant="outline"
          className="order-2 ml-auto h-8 border-white/30 bg-white/10 px-3 text-white hover:bg-white/20 md:order-1 md:ml-0 md:h-fit md:px-3"
          onClick={() => {
            router.push('/');
            router.refresh();
          }}
        >
          <PlusIcon />
          <span className="md:sr-only">New Chat</span>
        </Button>
      )}

      {!isReadonly && (
        <VisibilitySelector
          chatId={chatId}
          selectedVisibilityType={selectedVisibilityType}
          className="order-1 border-white/30 bg-white/10 text-white hover:bg-white/20 md:order-2"
        />
      )}

      <div className="order-4 ml-auto hidden items-center text-sm font-medium text-blue-100 md:flex">
        {session?.user?.name ? `Welcome, ${session.user.name}` : 'LexiBot Assistant'}
      </div>

      <Button
        className="order-5 hidden border-white/30 bg-white/10 px-3 text-white hover:bg-white/20 md:flex md:h-fit"
        asChild
      >
        <Link
          href={`https://vercel.com/templates/next.js/nextjs-ai-chatbot`}
          target="_noblank"
          rel="noreferrer"
        >
          <VercelIcon size={16} />
          Deploy with Vercel
        </Link>
      </Button>
    </header>
  );
}

export const ChatHeader = memo(PureChatHeader, (prevProps, nextProps) => {
  return (
    prevProps.chatId === nextProps.chatId &&
    prevProps.selectedVisibilityType === nextProps.selectedVisibilityType &&
    prevProps.isReadonly === nextProps.isReadonly
  );
});
