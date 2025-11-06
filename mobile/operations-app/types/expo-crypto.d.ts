declare module "expo-crypto" {
  export function getRandomBytesAsync(length: number): Promise<Uint8Array>;
  export function randomUUID(): string;
}

