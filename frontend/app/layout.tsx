import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Providers } from "./providers";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Canadian Civic Bylaw & Zoning Assistant",
  description:
    "Ask questions about Canadian municipal zoning bylaws. Answers cite the official bylaw section they come from. Currently covering Atlantic Canada, expanding nationwide.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  // `lang` is hardcoded to "en" for Phase 1. The Phase 0 bilingual lock
  // means Phase 4 must drive this from the selected language so screen
  // readers pronounce French bylaw text correctly.
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
