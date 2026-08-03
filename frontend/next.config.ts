import type { NextConfig } from "next";

const backendUrl =
  process.env.RAG_API_URL?.replace(/\/$/, "") || "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  // Dev opens via either localhost or 127.0.0.1; allow both for HMR/assets.
  allowedDevOrigins: ["127.0.0.1", "localhost"],
  async rewrites() {
    return [
      {
        source: "/rag-api/:path*",
        destination: `${backendUrl}/:path*`,
      },
    ];
  },
};

export default nextConfig;
