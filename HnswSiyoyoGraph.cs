using System;
using System.Collections.Generic;
using System.Linq;

namespace HnswSiyoyoProject
{
    /// <summary>
    /// HNSW Siyoyo variant with adaptive neighbor constraints and vector normalization.
    /// Introduces layer-adaptive neighbor scaling and normalized vector storage.
    /// </summary>
    public class HnswSiyoyoGraph : IHnswGraph
    {
        private readonly List<float[]> _vectors;
        private readonly Dictionary<int, Dictionary<int, List<int>>> _layers;
        private readonly Random _random;
        private readonly int _m0;
        private readonly int _efConstruction;
        private readonly float _lambda;
        private int _entryPoint;
        private int _maxLevel;

        public int Count => _vectors.Count;
        public int MaxLevel => _maxLevel;

        /// <summary>
        /// Initializes a new instance of the HnswSiyoyoGraph class.
        /// </summary>
        /// <param name="m0">Base number of connections per element per layer</param>
        /// <param name="efConstruction">Size of the dynamic candidate list during construction</param>
        /// <param name="seed">Random seed for reproducibility</param>
        public HnswSiyoyoGraph(int m0 = 16, int efConstruction = 200, int seed = 42)
        {
            _vectors = new List<float[]>();
            _layers = new Dictionary<int, Dictionary<int, List<int>>>();
            _random = new Random(seed);
            _m0 = m0;
            _efConstruction = efConstruction;
            _lambda = 1.0f / (float)Math.Log(m0);
            _entryPoint = -1;
            _maxLevel = -1;
        }

        public void Insert(float[] vector)
        {
            if (vector == null || vector.Length == 0)
                throw new ArgumentException("Vector cannot be null or empty");

            var normalizedVector = NormalizeVector(vector);
            var vectorIndex = _vectors.Count;
            _vectors.Add(normalizedVector);

            var level = SampleLevel();
            
            if (_entryPoint == -1)
            {
                _entryPoint = vectorIndex;
                _maxLevel = level;
                InsertIntoAllLayers(vectorIndex, level);
                return;
            }

            var current = _entryPoint;
            
            // Search for the nearest neighbor at each level above the insertion level
            for (int l = _maxLevel; l > level; l--)
            {
                current = SearchLayer(normalizedVector, current, 1, l)[0].index;
            }

            // Insert at each level from the insertion level down to 0
            for (int l = Math.Min(level, _maxLevel); l >= 0; l--)
            {
                var candidates = SearchLayer(normalizedVector, current, _efConstruction, l);
                var adaptiveM = CalculateAdaptiveM(l);
                var selected = SelectNeighbors(candidates, adaptiveM);
                LinkMutual(vectorIndex, selected, l);
            }

            if (level > _maxLevel)
            {
                _entryPoint = vectorIndex;
                _maxLevel = level;
            }
        }

        public List<int> Search(float[] query, int k, int ef)
        {
            if (query == null || query.Length == 0)
                throw new ArgumentException("Query vector cannot be null or empty");

            if (_entryPoint == -1)
                return new List<int>();

            var normalizedQuery = NormalizeVector(query);
            var current = _entryPoint;

            // Search at each level from max level down to 1
            for (int l = _maxLevel; l > 0; l--)
            {
                var top1 = SearchLayer(normalizedQuery, current, 1, l);
                if (top1.Count > 0)
                    current = top1[0].index;
            }

            // Search at level 0
            var candidates = SearchLayer(normalizedQuery, current, ef, 0);
            // Return top-k results
            return candidates.OrderByDescending(c => c.similarity).Take(k).Select(c => c.index).ToList();
        }

        public int GetMemoryUsage()
        {
            var vectorMemory = _vectors.Sum(v => v.Length * sizeof(float));
            var layerMemory = _layers.Values.Sum(layer => 
                layer.Values.Sum(neighbors => neighbors.Count * sizeof(int)));
            
            return vectorMemory + layerMemory + sizeof(int) * 4; // Additional overhead
        }

        public void Clear()
        {
            _vectors.Clear();
            _layers.Clear();
            _entryPoint = -1;
            _maxLevel = -1;
        }

        /// <summary>
        /// Normalizes a vector to unit length.
        /// </summary>
        /// <param name="vector">The vector to normalize</param>
        /// <returns>Normalized vector</returns>
        private float[] NormalizeVector(float[] vector)
        {
            var norm = (float)Math.Sqrt(vector.Sum(x => x * x));
            if (norm == 0)
                return vector;

            return vector.Select(x => x / norm).ToArray();
        }

        /// <summary>
        /// Calculates adaptive neighbor constraints based on layer level.
        /// </summary>
        /// <param name="level">The layer level</param>
        /// <returns>Adaptive neighbor count</returns>
        private int CalculateAdaptiveM(int level)
        {
            return Math.Max(1, (int)(_m0 * Math.Exp(-_lambda * level)));
        }

        private int SampleLevel()
        {
            var u = _random.NextDouble();
            return (int)(-Math.Log(u) / _lambda);
        }

        private void InsertIntoAllLayers(int vectorIndex, int level)
        {
            for (int l = 0; l <= level; l++)
            {
                if (!_layers.ContainsKey(l))
                    _layers[l] = new Dictionary<int, List<int>>();
                
                _layers[l][vectorIndex] = new List<int>();
            }
        }

        private List<(int index, float similarity)> SearchLayer(float[] query, int entryPoint, int ef, int level)
        {
            var candidates = new List<(int index, float similarity)>();
            var visited = new HashSet<int>();
            var searchSet = new List<(int index, float similarity)>();

            // Start with entry point
            var entrySimilarity = DotProduct(query, _vectors[entryPoint]);
            searchSet.Add((entryPoint, entrySimilarity));
            visited.Add(entryPoint);

            while (searchSet.Count > 0)
            {
                // Find the closest unvisited element
                var bestIndex = -1;
                var bestSimilarity = -1.0f;
                
                for (int i = 0; i < searchSet.Count; i++)
                {
                    if (searchSet[i].similarity > bestSimilarity)
                    {
                        bestIndex = i;
                        bestSimilarity = searchSet[i].similarity;
                    }
                }

                if (bestIndex == -1) break;

                var current = searchSet[bestIndex];
                searchSet.RemoveAt(bestIndex);
                candidates.Add(current);

                // Stop if we have enough candidates
                if (candidates.Count >= ef) break;

                // Explore neighbors of current element
                if (_layers.ContainsKey(level) && _layers[level].ContainsKey(current.index))
                {
                    foreach (var neighborIndex in _layers[level][current.index])
                    {
                        if (!visited.Contains(neighborIndex))
                        {
                            visited.Add(neighborIndex);
                            var neighborSimilarity = DotProduct(query, _vectors[neighborIndex]);
                            searchSet.Add((neighborIndex, neighborSimilarity));
                        }
                    }
                }
            }

            return candidates.OrderByDescending(c => c.similarity).ToList();
        }

        private List<int> SelectNeighbors(List<(int index, float similarity)> candidates, int m)
        {
            return candidates.Take(m).Select(c => c.index).ToList();
        }

        private void LinkMutual(int vectorIndex, List<int> neighbors, int level)
        {
            if (!_layers.ContainsKey(level))
                _layers[level] = new Dictionary<int, List<int>>();

            if (!_layers[level].ContainsKey(vectorIndex))
                _layers[level][vectorIndex] = new List<int>();

            foreach (var neighbor in neighbors)
            {
                if (!_layers[level].ContainsKey(neighbor))
                    _layers[level][neighbor] = new List<int>();

                if (!_layers[level][vectorIndex].Contains(neighbor))
                    _layers[level][vectorIndex].Add(neighbor);

                if (!_layers[level][neighbor].Contains(vectorIndex))
                    _layers[level][neighbor].Add(vectorIndex);
            }
        }

        /// <summary>
        /// Computes dot product between two normalized vectors.
        /// Since vectors are normalized, this is equivalent to cosine similarity.
        /// </summary>
        /// <param name="a">First normalized vector</param>
        /// <param name="b">Second normalized vector</param>
        /// <returns>Dot product (cosine similarity)</returns>
        private float DotProduct(float[] a, float[] b)
        {
            if (a.Length != b.Length)
                throw new ArgumentException("Vectors must have the same length");

            float dotProduct = 0;
            for (int i = 0; i < a.Length; i++)
            {
                dotProduct += a[i] * b[i];
            }

            return dotProduct;
        }

        private void InsertSorted(List<(int index, float similarity)> list, (int index, float similarity) item)
        {
            var index = list.BinarySearch(item, new SimilarityComparer());
            if (index < 0)
                index = ~index;
            list.Insert(index, item);
        }

        private class SimilarityComparer : IComparer<(int index, float similarity)>
        {
            public int Compare((int index, float similarity) x, (int index, float similarity) y)
            {
                return y.similarity.CompareTo(x.similarity); // Descending order
            }
        }

        /// <summary>
        /// Gets layer statistics for analysis.
        /// </summary>
        /// <returns>Dictionary mapping layer numbers to average neighbor counts</returns>
        public Dictionary<int, double> GetLayerStatistics()
        {
            var stats = new Dictionary<int, double>();
            
            foreach (var layer in _layers.Keys.OrderBy(k => k))
            {
                var totalNeighbors = 0;
                var nodeCount = 0;
                
                foreach (var node in _layers[layer].Keys)
                {
                    totalNeighbors += _layers[layer][node].Count;
                    nodeCount++;
                }
                
                var avgNeighbors = nodeCount > 0 ? (double)totalNeighbors / nodeCount : 0.0;
                stats[layer] = avgNeighbors;
            }
            
            return stats;
        }

        /// <summary>
        /// Gets the adaptive neighbor count for a specific layer.
        /// </summary>
        /// <param name="level">Layer level</param>
        /// <returns>Adaptive neighbor count</returns>
        public int GetAdaptiveM(int level)
        {
            return CalculateAdaptiveM(level);
        }
    }
} 