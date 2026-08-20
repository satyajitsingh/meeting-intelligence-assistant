import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Emits a self-contained server bundle for the Docker runtime stage.
  // Harmless for `next dev` and `next start`.
  output: "standalone",
};

export default nextConfig;
