/** @type {import('next').NextConfig} */
const nextConfig = {
    reactStrictMode: true,
    images: {
      domains: ['localhost'],
    },
    typescript: {
      ignoreBuildErrors: false,
    },
  }
  
  module.exports = nextConfig