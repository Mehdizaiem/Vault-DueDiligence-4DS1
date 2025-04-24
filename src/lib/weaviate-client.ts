// lib/weaviate-client.ts
import weaviate from 'weaviate-ts-client';

export function getWeaviateClient() {
  const client = weaviate.client({
    scheme: process.env.WEAVIATE_SCHEME || 'http',
    host: process.env.WEAVIATE_HOST || 'localhost:9090',
  });
  
  return client;
}