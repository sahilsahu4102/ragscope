/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  reactStrictMode: true,
  // Lint runs as a separate CI job; don't block production builds on it.
  eslint: { ignoreDuringBuilds: true },
  // Allow the docker-compose dev origin.
  allowedDevOrigins: ["localhost:3000"],
};

export default nextConfig;
