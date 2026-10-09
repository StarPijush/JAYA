"use client";

import { motion } from "framer-motion";

export type ChatMessage = {
  id: string;
  text: string;
};

type ConversationPanelProps = {
  messages: ChatMessage[];
};

/**
 * Open conversation canvas — no shared container card.
 * Renders only messages the user has actually sent in this session.
 * Assistant replies will render here once a backend is connected.
 */
export default function ConversationPanel({ messages }: ConversationPanelProps) {
  return (
    <ol aria-label="Messages" className="flex w-full flex-col gap-4 py-6">
      {messages.map((message) => (
        <motion.li
          key={message.id}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.2, ease: "easeOut" }}
          className="ml-auto w-fit max-w-[85%] break-words rounded-2xl bg-surface-container-high px-4 py-2.5 text-[15px] leading-6 whitespace-pre-wrap text-on-surface"
        >
          {message.text}
        </motion.li>
      ))}
    </ol>
  );
}
