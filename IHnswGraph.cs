using System;
using System.Collections.Generic;

namespace HnswSiyoyoProject
{
    /// <summary>
    /// Interface defining the contract for HNSW graph implementations.
    /// Provides methods for inserting vectors and performing similarity search.
    /// </summary>
    public interface IHnswGraph
    {
        /// <summary>
        /// Inserts a vector into the graph.
        /// </summary>
        /// <param name="vector">The vector to insert</param>
        void Insert(float[] vector);

        /// <summary>
        /// Searches for the k nearest neighbors of the query vector.
        /// </summary>
        /// <param name="query">The query vector</param>
        /// <param name="k">Number of neighbors to return</param>
        /// <param name="ef">Search parameter controlling accuracy vs speed trade-off</param>
        /// <returns>List of indices of the k nearest neighbors</returns>
        List<int> Search(float[] query, int k, int ef);

        /// <summary>
        /// Gets the current memory usage in bytes.
        /// </summary>
        /// <returns>Memory usage in bytes</returns>
        int GetMemoryUsage();

        /// <summary>
        /// Clears all data from the graph.
        /// </summary>
        void Clear();

        /// <summary>
        /// Gets the number of vectors currently in the graph.
        /// </summary>
        /// <returns>Number of vectors</returns>
        int Count { get; }

        /// <summary>
        /// Gets the maximum level in the graph.
        /// </summary>
        /// <returns>Maximum level</returns>
        int MaxLevel { get; }
    }
} 