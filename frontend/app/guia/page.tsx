import type { Metadata } from "next";
import QuickGuide from "../../components/QuickGuide";

export const metadata: Metadata = { title: "Guia rápido" };

export default function GuidePage() {
  return <QuickGuide />;
}
