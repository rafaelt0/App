import type { Metadata } from "next";
import Welcome from "../../components/Welcome";

export const metadata: Metadata = { title: "Boas-vindas" };

export default function WelcomePage() {
  return <Welcome />;
}
