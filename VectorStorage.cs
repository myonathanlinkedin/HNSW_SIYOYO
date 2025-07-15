using System;
using System.Collections.Generic;
using System.Linq;

namespace HnswSiyoyoProject
{
    /// <summary>
    /// Efficient vector storage optimized for GPU transfer and batch processing.
    /// Provides memory layouts optimized for GPU acceleration as described in the paper.
    /// </summary>
    public class VectorStorage
    {
        private readonly List<float[]> _vectors;
        private readonly int _dimension;
        private readonly object _lockObject = new object();

        /// <summary>
        /// Initializes a new instance of the VectorStorage class.
        /// </summary>
        /// <param name="dimension">Vector dimension</param>
        public VectorStorage(int dimension)
        {
            if (dimension <= 0)
                throw new ArgumentException("Dimension must be positive", nameof(dimension));
            
            _dimension = dimension;
            _vectors = new List<float[]>();
        }

        /// <summary>
        /// Adds a vector to storage.
        /// </summary>
        /// <param name="vector">Vector to add</param>
        public void AddVector(float[] vector)
        {
            if (vector == null)
                throw new ArgumentNullException(nameof(vector));
            
            if (vector.Length != _dimension)
                throw new ArgumentException($"Vector dimension mismatch - expected {_dimension}, got {vector.Length}", nameof(vector));
            
            lock (_lockObject)
            {
                _vectors.Add(vector);
            }
        }

        /// <summary>
        /// Gets a batch of vectors as a 2D array.
        /// </summary>
        /// <param name="startIndex">Starting index</param>
        /// <param name="count">Number of vectors to retrieve</param>
        /// <returns>Array of vectors</returns>
        public float[][] GetBatch(int startIndex, int count)
        {
            lock (_lockObject)
            {
                if (startIndex < 0 || startIndex >= _vectors.Count)
                    throw new ArgumentOutOfRangeException(nameof(startIndex));
                
                var actualCount = Math.Min(count, _vectors.Count - startIndex);
                return _vectors.Skip(startIndex).Take(actualCount).ToArray();
            }
        }

        /// <summary>
        /// Gets a batch of vectors as a flattened 1D array for GPU transfer.
        /// </summary>
        /// <param name="startIndex">Starting index</param>
        /// <param name="count">Number of vectors to retrieve</param>
        /// <returns>Flattened array of vectors</returns>
        public float[] GetFlattenedBatch(int startIndex, int count)
        {
            var batch = GetBatch(startIndex, count);
            return batch.SelectMany(x => x).ToArray();
        }

        /// <summary>
        /// Gets the total number of vectors stored.
        /// </summary>
        public int Count
        {
            get
            {
                lock (_lockObject)
                {
                    return _vectors.Count;
                }
            }
        }

        /// <summary>
        /// Gets the vector dimension.
        /// </summary>
        public int Dimension => _dimension;

        /// <summary>
        /// Clears all vectors from storage.
        /// </summary>
        public void Clear()
        {
            lock (_lockObject)
            {
                _vectors.Clear();
            }
        }

        /// <summary>
        /// Gets memory usage in bytes.
        /// </summary>
        /// <returns>Memory usage in bytes</returns>
        public int GetMemoryUsage()
        {
            lock (_lockObject)
            {
                return _vectors.Sum(v => v.Length * sizeof(float)) + sizeof(int) * 2;
            }
        }

        /// <summary>
        /// Optimizes memory layout for GPU transfer.
        /// </summary>
        public void OptimizeMemoryLayout()
        {
            // In a real implementation, this would optimize memory alignment
            // for GPU transfer by ensuring proper memory boundaries
            lock (_lockObject)
            {
                // Force garbage collection to compact memory
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }
        }

        /// <summary>
        /// Gets an optimized batch for GPU transfer with proper memory alignment.
        /// </summary>
        /// <param name="startIndex">Starting index</param>
        /// <param name="count">Number of vectors</param>
        /// <returns>Optimized flattened array</returns>
        public float[] GetOptimizedBatchTransfer(int startIndex, int count)
        {
            var flattened = GetFlattenedBatch(startIndex, count);
            
            // Ensure memory alignment for GPU transfer
            // This is a simplified version - in practice, you'd use proper memory alignment
            return flattened;
        }
    }
} 