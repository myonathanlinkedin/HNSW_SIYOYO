using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace HnswSiyoyoProject
{
    /// <summary>
    /// Parallel processing framework leveraging .NET 9's improved parallel processing capabilities.
    /// Provides efficient batch processing for HNSW operations.
    /// </summary>
    public static class ParallelProcessor
    {
        /// <summary>
        /// Processes items in parallel batches.
        /// </summary>
        /// <typeparam name="T">Type of items to process</typeparam>
        /// <param name="items">Items to process</param>
        /// <param name="action">Action to perform on each item</param>
        /// <param name="batchSize">Size of each batch</param>
        public static void ProcessBatch<T>(IEnumerable<T> items, Action<T> action, int batchSize = 1000)
        {
            if (items == null)
                throw new ArgumentNullException(nameof(items));
            
            if (action == null)
                throw new ArgumentNullException(nameof(action));
            
            if (batchSize <= 0)
                throw new ArgumentException("Batch size must be positive", nameof(batchSize));

            var batches = items.Chunk(batchSize);
            
            Parallel.ForEach(batches, batch =>
            {
                foreach (var item in batch)
                {
                    action(item);
                }
            });
        }

        /// <summary>
        /// Processes items in parallel with a result selector.
        /// </summary>
        /// <typeparam name="T">Type of input items</typeparam>
        /// <typeparam name="TResult">Type of result items</typeparam>
        /// <param name="items">Items to process</param>
        /// <param name="selector">Function to transform each item</param>
        /// <param name="batchSize">Size of each batch</param>
        /// <returns>Transformed results</returns>
        public static List<TResult> ProcessBatchWithResult<T, TResult>(IEnumerable<T> items, Func<T, TResult> selector, int batchSize = 1000)
        {
            if (items == null)
                throw new ArgumentNullException(nameof(items));
            
            if (selector == null)
                throw new ArgumentNullException(nameof(selector));
            
            if (batchSize <= 0)
                throw new ArgumentException("Batch size must be positive", nameof(batchSize));

            var batches = items.Chunk(batchSize);
            var results = new List<TResult>();
            var lockObject = new object();

            Parallel.ForEach(batches, batch =>
            {
                var batchResults = batch.Select(selector).ToList();
                
                lock (lockObject)
                {
                    results.AddRange(batchResults);
                }
            });

            return results;
        }

        /// <summary>
        /// Processes vectors in parallel for similarity calculations.
        /// </summary>
        /// <param name="query">Query vector</param>
        /// <param name="vectors">Vectors to compare against</param>
        /// <param name="similarityFunc">Similarity calculation function</param>
        /// <param name="batchSize">Size of each batch</param>
        /// <returns>Array of similarity scores</returns>
        public static float[] ProcessSimilarityBatch(float[] query, float[][] vectors, Func<float[], float[], float> similarityFunc, int batchSize = 1000)
        {
            if (query == null)
                throw new ArgumentNullException(nameof(query));
            
            if (vectors == null)
                throw new ArgumentNullException(nameof(vectors));
            
            if (similarityFunc == null)
                throw new ArgumentNullException(nameof(similarityFunc));

            var results = new float[vectors.Length];
            var batches = Enumerable.Range(0, vectors.Length).Chunk(batchSize);

            Parallel.ForEach(batches, batch =>
            {
                foreach (var index in batch)
                {
                    results[index] = similarityFunc(query, vectors[index]);
                }
            });

            return results;
        }

        /// <summary>
        /// Processes multiple queries in parallel.
        /// </summary>
        /// <param name="queries">Query vectors</param>
        /// <param name="vectors">Dataset vectors</param>
        /// <param name="similarityFunc">Similarity calculation function</param>
        /// <param name="batchSize">Size of each batch</param>
        /// <returns>Matrix of similarity scores</returns>
        public static float[][] ProcessMultipleQueries(float[][] queries, float[][] vectors, Func<float[], float[], float> similarityFunc, int batchSize = 1000)
        {
            if (queries == null)
                throw new ArgumentNullException(nameof(queries));
            
            if (vectors == null)
                throw new ArgumentNullException(nameof(vectors));
            
            if (similarityFunc == null)
                throw new ArgumentNullException(nameof(similarityFunc));

            var results = new float[queries.Length][];
            
            Parallel.For(0, queries.Length, i =>
            {
                results[i] = ProcessSimilarityBatch(queries[i], vectors, similarityFunc, batchSize);
            });

            return results;
        }

        /// <summary>
        /// Gets the optimal batch size based on system capabilities.
        /// </summary>
        /// <returns>Optimal batch size</returns>
        public static int GetOptimalBatchSize()
        {
            var processorCount = Environment.ProcessorCount;
            var memorySize = GC.GetTotalMemory(false);
            
            // Simple heuristic: batch size should be proportional to processor count
            // but not too large to avoid memory pressure
            return Math.Min(1000, Math.Max(100, processorCount * 50));
        }

        /// <summary>
        /// Processes items with progress reporting.
        /// </summary>
        /// <typeparam name="T">Type of items to process</typeparam>
        /// <param name="items">Items to process</param>
        /// <param name="action">Action to perform on each item</param>
        /// <param name="progressCallback">Progress callback</param>
        /// <param name="batchSize">Size of each batch</param>
        public static void ProcessBatchWithProgress<T>(IEnumerable<T> items, Action<T> action, Action<int, int> progressCallback, int batchSize = 1000)
        {
            if (items == null)
                throw new ArgumentNullException(nameof(items));
            
            if (action == null)
                throw new ArgumentNullException(nameof(action));

            var itemsList = items.ToList();
            var totalItems = itemsList.Count;
            var processedItems = 0;
            var lockObject = new object();

            var batches = itemsList.Chunk(batchSize);
            
            Parallel.ForEach(batches, batch =>
            {
                foreach (var item in batch)
                {
                    action(item);
                }

                lock (lockObject)
                {
                    processedItems += batch.Count();
                    progressCallback?.Invoke(processedItems, totalItems);
                }
            });
        }
    }
} 