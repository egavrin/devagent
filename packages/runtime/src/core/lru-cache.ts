/**
 * Simple LRU (Least Recently Used) cache.
 * Fixed-size, O(1) get/set via Map insertion order.
 * Used for caching expensive lookups: git root, tool descriptions, file state.
 *
 * Inspired by claude-code-src multi-level caching pattern.
 */

// ─── LRUCache ───────────────────────────────────────────────

/**
 * Generic LRU cache with configurable max size.
 * When the cache exceeds maxSize, the least recently used entry is evicted.
 * Thread-safe for single-threaded JS runtimes (Map preserves insertion order).
 */
export class LRUCache<K, V> {
  private readonly cache = new Map<K, V>();
  private readonly maxSize: number;
  private hits = 0;
  private misses = 0;

  constructor(maxSize: number) {
    if (maxSize < 1) throw new Error("LRUCache maxSize must be >= 1");
    this.maxSize = maxSize;
  }

  /**
   * Get a value from the cache. Returns undefined if not found.
   * Accessing a key moves it to the most-recently-used position.
   */
  get(key: K): V | undefined {
    const value = this.cache.get(key);
    if (value === undefined) {
      this.misses++;
      return undefined;
    }
    this.hits++;
    // Move to end (most recently used) by re-inserting
    this.cache.delete(key);
    this.cache.set(key, value);
    return value;
  }

  /**
   * Set a value in the cache.
   * If the cache is full, evicts the least recently used entry.
   */
  set(key: K, value: V): void {
    // If key already exists, delete it first to update insertion order
    if (this.cache.has(key)) {
      this.cache.delete(key);
    } else if (this.cache.size >= this.maxSize) {
      // Evict the least recently used (first entry in Map)
      const firstKey = this.cache.keys().next().value;
      if (firstKey !== undefined) {
        this.cache.delete(firstKey);
      }
    }
    this.cache.set(key, value);
  }

  /**
   * Check if a key exists in the cache (without updating recency).
   */
  has(key: K): boolean {
    return this.cache.has(key);
  }

  /**
   * Delete a specific key from the cache.
   */
  delete(key: K): boolean {
    return this.cache.delete(key);
  }

  /**
   * Clear all entries from the cache.
   */
  clear(): void {
    this.cache.clear();
    this.hits = 0;
    this.misses = 0;
  }

  /**
   * Current number of entries in the cache.
   */
  get size(): number {
    return this.cache.size;
  }

  /**
   * Cache hit/miss statistics.
   */
  get stats(): { hits: number; misses: number; hitRate: number; size: number } {
    const total = this.hits + this.misses;
    return {
      hits: this.hits,
      misses: this.misses,
      hitRate: total > 0 ? this.hits / total : 0,
      size: this.cache.size,
    };
  }
}

// ─── Memoize ────────────────────────────────────────────────

/**
 * Memoize a function with LRU caching.
 * The key function extracts a cache key from the arguments.
 */
export function memoizeWithLRU<Args extends ReadonlyArray<unknown>, Result>(
  fn: (...args: Args) => Result,
  keyFn: (...args: Args) => string,
  maxSize: number = 50,
): ((...args: Args) => Result) & { cache: LRUCache<string, Result>; clear: () => void } {
  const cache = new LRUCache<string, Result>(maxSize);

  const memoized = (...args: Args): Result => {
    const key = keyFn(...args);
    const cached = cache.get(key);
    if (cached !== undefined) return cached;

    const result = fn(...args);
    cache.set(key, result);
    return result;
  };

  memoized.cache = cache;
  memoized.clear = () => cache.clear();

  return memoized;
}

/**
 * Memoize an async function with LRU caching.
 * Deduplicates concurrent calls to the same key (only one in-flight promise per key).
 */
export function memoizeAsyncWithLRU<Args extends ReadonlyArray<unknown>, Result>(
  fn: (...args: Args) => Promise<Result>,
  keyFn: (...args: Args) => string,
  maxSize: number = 50,
): ((...args: Args) => Promise<Result>) & { cache: LRUCache<string, Result>; inflight: Map<string, Promise<Result>>; clear: () => void } {
  const cache = new LRUCache<string, Result>(maxSize);
  const inflight = new Map<string, Promise<Result>>();

  const memoized = async (...args: Args): Promise<Result> => {
    const key = keyFn(...args);

    // Check cache first
    const cached = cache.get(key);
    if (cached !== undefined) return cached;

    // Deduplicate concurrent calls
    const existing = inflight.get(key);
    if (existing) return existing;

    const promise = fn(...args).then((result) => {
      cache.set(key, result);
      inflight.delete(key);
      return result;
    }).catch((err) => {
      inflight.delete(key);
      throw err;
    });

    inflight.set(key, promise);
    return promise;
  };

  memoized.cache = cache;
  memoized.inflight = inflight;
  memoized.clear = () => {
    cache.clear();
    inflight.clear();
  };

  return memoized;
}
