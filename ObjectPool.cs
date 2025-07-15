using System;
using System.Collections.Concurrent;
using System.Threading;

namespace HnswSiyoyoProject
{
    /// <summary>
    /// Object pooling system to reduce garbage collection pressure.
    /// Provides efficient object reuse for frequently allocated structures.
    /// </summary>
    /// <typeparam name="T">Type of objects to pool</typeparam>
    public class ObjectPool<T> where T : class, new()
    {
        private readonly ConcurrentQueue<T> _pool;
        private readonly int _maxSize;
        private int _totalCreated;
        private int _totalRented;

        /// <summary>
        /// Initializes a new instance of the ObjectPool class.
        /// </summary>
        /// <param name="maxSize">Maximum number of objects to keep in the pool</param>
        public ObjectPool(int maxSize = 1000)
        {
            if (maxSize <= 0)
                throw new ArgumentException("MaxSize must be positive", nameof(maxSize));
            
            _maxSize = maxSize;
            _pool = new ConcurrentQueue<T>();
            _totalCreated = 0;
            _totalRented = 0;
        }

        /// <summary>
        /// Rents an object from the pool or creates a new one if the pool is empty.
        /// </summary>
        /// <returns>An object from the pool</returns>
        public T Rent()
        {
            if (_pool.TryDequeue(out var item))
            {
                return item;
            }
            
            Interlocked.Increment(ref _totalCreated);
            return new T();
        }

        /// <summary>
        /// Returns an object to the pool for reuse.
        /// </summary>
        /// <param name="item">Object to return</param>
        public void Return(T item)
        {
            if (item == null)
                return;

            if (_pool.Count < _maxSize)
            {
                _pool.Enqueue(item);
            }
            
            Interlocked.Increment(ref _totalRented);
        }

        /// <summary>
        /// Gets the current number of objects in the pool.
        /// </summary>
        public int PoolSize => _pool.Count;

        /// <summary>
        /// Gets the maximum number of objects that can be stored in the pool.
        /// </summary>
        public int MaxSize => _maxSize;

        /// <summary>
        /// Gets the total number of objects created.
        /// </summary>
        public int TotalCreated => _totalCreated;

        /// <summary>
        /// Gets the total number of objects rented.
        /// </summary>
        public int TotalRented => _totalRented;

        /// <summary>
        /// Gets the pool utilization rate.
        /// </summary>
        public double UtilizationRate => _totalRented > 0 ? (double)_totalRented / (_totalRented + _totalCreated) : 0.0;

        /// <summary>
        /// Clears all objects from the pool.
        /// </summary>
        public void Clear()
        {
            while (_pool.TryDequeue(out _))
            {
                // Clear all items
            }
        }

        /// <summary>
        /// Gets pool statistics.
        /// </summary>
        /// <returns>Pool statistics as a string</returns>
        public string GetStatistics()
        {
            return $"Pool Size: {PoolSize}/{MaxSize}, Total Created: {TotalCreated}, Total Rented: {TotalRented}, Utilization: {UtilizationRate:P2}";
        }
    }
} 