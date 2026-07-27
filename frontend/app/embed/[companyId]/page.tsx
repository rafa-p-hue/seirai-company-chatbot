import { redirect } from "next/navigation";

type Props = {
  params: Promise<{ companyId: string }>;
};

export default async function CompanyEmbedPage({ params }: Props) {
  const { companyId } = await params;
  redirect(`/embed/seirai?company_id=${encodeURIComponent(companyId)}`);
}
