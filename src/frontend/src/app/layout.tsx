import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CDS Agent — Clinical Decision Support",
  description:
    "Agentic clinical decision support powered by MedGemma (HAI-DEF)",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
