import { redirect } from "next/navigation";

/** Primary chatbot lives at `/chat`; keep `/` as the entry redirect. */
export default function Home() {
  redirect("/chat");
}
