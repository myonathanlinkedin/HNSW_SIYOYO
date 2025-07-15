using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

namespace HnswSiyoyoProject
{
    /// <summary>
    /// Comprehensive benchmarking framework for HNSW implementations.
    /// Provides performance evaluation, recall analysis, and result export capabilities.
    /// </summary>
    public class BenchmarkRunner
    {
        private readonly IHnswGraph _standardGraph;
        private readonly IHnswGraph _siyoyoGraph;
        private readonly GpuSimilarityHelper _gpuHelper;
        private readonly string _resultsPath;
        private readonly string _plotsPath;

        public BenchmarkRunner(string resultsPath = "results", string plotsPath = "plots")
        {
            _standardGraph = new HnswGraphBase();
            _siyoyoGraph = new HnswSiyoyoGraph();
            _gpuHelper = new GpuSimilarityHelper();
            _resultsPath = resultsPath;
            _plotsPath = plotsPath;

            // Create directories if they don't exist
            Directory.CreateDirectory(_resultsPath);
            Directory.CreateDirectory(_plotsPath);
        }

        /// <summary>
        /// Runs a comprehensive benchmark comparing standard HNSW and Siyoyo variant.
        /// </summary>
        /// <param name="dataset">Training dataset</param>
        /// <param name="queries">Test queries</param>
        /// <param name="k">Number of neighbors to retrieve</param>
        /// <param name="efValues">Array of ef values to test</param>
        /// <returns>Benchmark results</returns>
        public BenchmarkResults RunBenchmark(float[][] dataset, float[][] queries, int k = 10, int[] efValues = null)
        {
            // Initialize enhanced components as described in the paper
            var vectorStorage = new VectorStorage(dataset[0].Length);
            var objectPool = new ObjectPool<List<(int index, float similarity)>>(1000);
            var analytics = new GraphAnalytics(new HnswSiyoyoGraph());
            
            // Use VectorStorage for efficient data management
            ParallelProcessor.ProcessBatch(dataset, vector => vectorStorage.AddVector(vector));
            Console.WriteLine($"VectorStorage loaded: {vectorStorage.Count} vectors");
            
            // Use ObjectPool for memory optimization
            Console.WriteLine($"ObjectPool statistics: {objectPool.GetStatistics()}");

            if (efValues == null)
                efValues = new[] { 32, 64, 128, 256 };

            var results = new BenchmarkResults();
            var stopwatch = Stopwatch.StartNew();

            Console.WriteLine("Starting HNSW Siyoyo benchmark...");
            Console.WriteLine($"Dataset size: {dataset.Length} vectors");
            Console.WriteLine($"Query count: {queries.Length}");
            Console.WriteLine($"GPU info: {_gpuHelper.GetGpuInfo()}");

            // Dimension check for dataset
            int expectedDim = dataset[0].Length;
            foreach (var vector in dataset)
            {
                if (vector.Length != expectedDim)
                    throw new ArgumentException($"All dataset vectors must have the same length: expected {expectedDim}, got {vector.Length}");
            }
            foreach (var query in queries)
            {
                if (query.Length != expectedDim)
                    throw new ArgumentException($"All query vectors must have the same length as dataset: expected {expectedDim}, got {query.Length}");
            }

            // Create new graph instances for this benchmark to avoid dimension conflicts
            var standardGraph = new HnswGraphBase();
            var siyoyoGraph = new HnswSiyoyoGraph();

            // Insert data into both graphs
            Console.WriteLine("Inserting data into graphs...");
            var insertTimes = new List<long>();
            
            foreach (var vector in dataset)
            {
                stopwatch.Restart();
                standardGraph.Insert(vector);
                stopwatch.Stop();
                insertTimes.Add(stopwatch.ElapsedTicks);
            }

            results.AverageInsertTime = insertTimes.Average() / TimeSpan.TicksPerMillisecond;
            results.MemoryUsageStandard = standardGraph.GetMemoryUsage();

            // Clear and insert into Siyoyo graph
            insertTimes.Clear();
            
            foreach (var vector in dataset)
            {
                stopwatch.Restart();
                siyoyoGraph.Insert(vector);
                stopwatch.Stop();
                insertTimes.Add(stopwatch.ElapsedTicks);
            }

            results.AverageInsertTimeSiyoyo = insertTimes.Average() / TimeSpan.TicksPerMillisecond;
            results.MemoryUsageSiyoyo = siyoyoGraph.GetMemoryUsage();

            // Run queries with different ef values
            foreach (var ef in efValues)
            {
                Console.WriteLine($"Testing with ef={ef}...");
                
                var queryTimes = new List<long>();
                var siyoyoQueryTimes = new List<long>();
                var recalls = new List<double>();
                var recallsSiyoyo = new List<double>();

                foreach (var query in queries)
                {
                    // Test standard HNSW
                    stopwatch.Restart();
                    var standardResults = standardGraph.Search(query, k, ef);
                    stopwatch.Stop();
                    queryTimes.Add(stopwatch.ElapsedTicks);

                    // Calculate recall using GPU baseline
                    var gpuBaseline = _gpuHelper.ComputeTopKNeighbors(query, dataset, k);
                    var recall = _gpuHelper.ComputeRecall(standardResults, gpuBaseline, k);
                    recalls.Add(recall);

                    // Test Siyoyo variant
                    stopwatch.Restart();
                    var siyoyoResults = siyoyoGraph.Search(query, k, ef);
                    stopwatch.Stop();
                    siyoyoQueryTimes.Add(stopwatch.ElapsedTicks);
                    var siyoyoRecall = _gpuHelper.ComputeRecall(siyoyoResults, gpuBaseline, k);
                    recallsSiyoyo.Add(siyoyoRecall);
                }

                results.AddEfResult(ef, 
                    queryTimes.Average() / TimeSpan.TicksPerMillisecond,
                    siyoyoQueryTimes.Average() / TimeSpan.TicksPerMillisecond,
                    recalls.Average(),
                    recallsSiyoyo.Average());
            }

            // Generate layer statistics for Siyoyo
            results.LayerStatistics = siyoyoGraph.GetLayerStatistics();

            // Use GraphAnalytics for comprehensive analysis
            var graphAnalytics = new GraphAnalytics(siyoyoGraph);
            Console.WriteLine("Graph Analytics:");
            Console.WriteLine(graphAnalytics.GetComprehensiveStatistics());
            
            // Test memory optimization
            vectorStorage.OptimizeMemoryLayout();
            Console.WriteLine($"Memory optimization completed. VectorStorage memory usage: {vectorStorage.GetMemoryUsage()} bytes");

            Console.WriteLine("Benchmark completed successfully!");
            return results;
        }

        /// <summary>
        /// Runs a parameter sensitivity study.
        /// </summary>
        /// <param name="dataset">Training dataset</param>
        /// <param name="queries">Test queries</param>
        /// <param name="mValues">Array of M values to test</param>
        /// <returns>Parameter sensitivity results</returns>
        public ParameterSensitivityResults RunParameterSensitivity(float[][] dataset, float[][] queries, int[] mValues)
        {
            var results = new ParameterSensitivityResults();

            foreach (var m in mValues)
            {
                Console.WriteLine($"Testing with M={m}...");
                
                var standardGraph = new HnswGraphBase(m: m);
                var siyoyoGraph = new HnswSiyoyoGraph(m0: m);

                // Insert data
                foreach (var vector in dataset)
                {
                    standardGraph.Insert(vector);
                    siyoyoGraph.Insert(vector);
                }

                // Run queries
                var standardTimes = new List<long>();
                var siyoyoTimes = new List<long>();
                var standardRecalls = new List<double>();
                var siyoyoRecalls = new List<double>();

                foreach (var query in queries)
                {
                    var stopwatch = Stopwatch.StartNew();
                    var standardResults = standardGraph.Search(query, 10, 64);
                    stopwatch.Stop();
                    standardTimes.Add(stopwatch.ElapsedTicks);

                    stopwatch.Restart();
                    var siyoyoResults = siyoyoGraph.Search(query, 10, 64);
                    stopwatch.Stop();
                    siyoyoTimes.Add(stopwatch.ElapsedTicks);

                    var gpuBaseline = _gpuHelper.ComputeTopKNeighbors(query, dataset, 10);
                    standardRecalls.Add(_gpuHelper.ComputeRecall(standardResults, gpuBaseline, 10));
                    siyoyoRecalls.Add(_gpuHelper.ComputeRecall(siyoyoResults, gpuBaseline, 10));
                }

                results.AddMResult(m,
                    standardTimes.Average() / TimeSpan.TicksPerMillisecond,
                    siyoyoTimes.Average() / TimeSpan.TicksPerMillisecond,
                    standardRecalls.Average(),
                    siyoyoRecalls.Average(),
                    standardGraph.GetMemoryUsage(),
                    siyoyoGraph.GetMemoryUsage());
            }

            return results;
        }

        /// <summary>
        /// Runs advanced benchmarking with detailed performance metrics.
        /// </summary>
        /// <param name="dataset">Training dataset</param>
        /// <param name="queries">Test queries</param>
        /// <param name="k">Number of neighbors to retrieve</param>
        /// <returns>Advanced benchmark results</returns>
        public AdvancedBenchmarkResults RunAdvancedBenchmark(float[][] dataset, float[][] queries, int k = 10)
        {
            var results = new AdvancedBenchmarkResults();
            var stopwatch = Stopwatch.StartNew();

            // Create graph instances
            var standardGraph = new HnswGraphBase();
            var siyoyoGraph = new HnswSiyoyoGraph();

            // Measure insertion performance with progress reporting
            Console.WriteLine("Inserting data with progress tracking...");
            var insertProgress = 0;
            var totalInserts = dataset.Length;

            foreach (var vector in dataset)
            {
                standardGraph.Insert(vector);
                siyoyoGraph.Insert(vector);
                
                insertProgress++;
                if (insertProgress % 100 == 0)
                {
                    Console.WriteLine($"Inserted {insertProgress}/{totalInserts} vectors");
                }
            }

            // Measure query performance with latency percentiles
            var queryLatencies = new List<double>();
            var recalls = new List<double>();

            foreach (var query in queries)
            {
                stopwatch.Restart();
                var results_standard = standardGraph.Search(query, k, 64);
                stopwatch.Stop();
                queryLatencies.Add(stopwatch.ElapsedMilliseconds);

                var gpuBaseline = _gpuHelper.ComputeTopKNeighbors(query, dataset, k);
                recalls.Add(_gpuHelper.ComputeRecall(results_standard, gpuBaseline, k));
            }

            // Calculate advanced metrics
            results.AverageQueryLatency = queryLatencies.Average();
            results.P95Latency = CalculatePercentile(queryLatencies.ToArray(), 95);
            results.P99Latency = CalculatePercentile(queryLatencies.ToArray(), 99);
            results.AverageRecall = recalls.Average();
            results.Throughput = queries.Length / (queryLatencies.Sum() / 1000.0); // queries per second
            results.MemoryUsage = standardGraph.GetMemoryUsage();

            return results;
        }

        /// <summary>
        /// Calculates percentile from an array of values.
        /// </summary>
        /// <param name="values">Array of values</param>
        /// <param name="percentile">Percentile to calculate (0-100)</param>
        /// <returns>Value at the specified percentile</returns>
        private double CalculatePercentile(double[] values, double percentile)
        {
            if (values == null || values.Length == 0)
                return 0.0;

            Array.Sort(values);
            var index = (int)Math.Ceiling((percentile / 100.0) * values.Length) - 1;
            return values[Math.Max(0, index)];
        }

        /// <summary>
        /// Exports benchmark results to CSV format.
        /// </summary>
        /// <param name="results">Benchmark results</param>
        /// <param name="filename">Output filename</param>
        public void ExportResults(BenchmarkResults results, string filename = "benchmark_results.csv")
        {
            var filePath = Path.Combine(_resultsPath, filename);
            var csv = new StringBuilder();

            // Header
            csv.AppendLine("ef,avg_query_time_standard,avg_query_time_siyoyo,recall_standard,recall_siyoyo");

            // Data
            foreach (var efResult in results.EfResults)
            {
                csv.AppendLine($"{efResult.Ef},{efResult.AverageQueryTimeStandard.ToString("F3", System.Globalization.CultureInfo.InvariantCulture)},{efResult.AverageQueryTimeSiyoyo.ToString("F3", System.Globalization.CultureInfo.InvariantCulture)}," +
                             $"{efResult.RecallStandard.ToString("F3", System.Globalization.CultureInfo.InvariantCulture)},{efResult.RecallSiyoyo.ToString("F3", System.Globalization.CultureInfo.InvariantCulture)}");
            }

            File.WriteAllText(filePath, csv.ToString());
            Console.WriteLine($"Results exported to {filePath}");
        }

        /// <summary>
        /// Exports parameter sensitivity results to CSV format.
        /// </summary>
        /// <param name="results">Parameter sensitivity results</param>
        /// <param name="filename">Output filename</param>
        public void ExportParameterSensitivity(ParameterSensitivityResults results, string filename = "parameter_sensitivity.csv")
        {
            var filePath = Path.Combine(_resultsPath, filename);
            var csv = new StringBuilder();

            // Header
            csv.AppendLine("m,query_time_standard,query_time_siyoyo,recall_standard,recall_siyoyo,memory_standard,memory_siyoyo");

            // Data
            foreach (var mResult in results.MResults)
            {
                csv.AppendLine($"{mResult.M},{mResult.QueryTimeStandard.ToString("F3", System.Globalization.CultureInfo.InvariantCulture)},{mResult.QueryTimeSiyoyo.ToString("F3", System.Globalization.CultureInfo.InvariantCulture)}," +
                             $"{mResult.RecallStandard.ToString("F3", System.Globalization.CultureInfo.InvariantCulture)},{mResult.RecallSiyoyo.ToString("F3", System.Globalization.CultureInfo.InvariantCulture)}," +
                             $"{mResult.MemoryStandard},{mResult.MemorySiyoyo}");
            }

            File.WriteAllText(filePath, csv.ToString());
            Console.WriteLine($"Parameter sensitivity results exported to {filePath}");
        }

        /// <summary>
        /// Generates a summary report of the benchmark results.
        /// </summary>
        /// <param name="results">Benchmark results</param>
        /// <returns>Summary report string</returns>
        public string GenerateSummaryReport(BenchmarkResults results)
        {
            var report = new StringBuilder();
            report.AppendLine("=== HNSW Siyoyo Benchmark Summary ===");
            report.AppendLine();
            report.AppendLine($"Average Insert Time (Standard): {results.AverageInsertTime.ToString("F3", CultureInfo.InvariantCulture)} ms");
            report.AppendLine($"Average Insert Time (Siyoyo): {results.AverageInsertTimeSiyoyo.ToString("F3", CultureInfo.InvariantCulture)} ms");
            report.AppendLine($"Memory Usage (Standard): {(results.MemoryUsageStandard / 1024.0).ToString("F2", CultureInfo.InvariantCulture)} KB");
            report.AppendLine($"Memory Usage (Siyoyo): {(results.MemoryUsageSiyoyo / 1024.0).ToString("F2", CultureInfo.InvariantCulture)} KB");
            report.AppendLine($"Memory Reduction: {((double)(results.MemoryUsageStandard - results.MemoryUsageSiyoyo) / results.MemoryUsageStandard * 100).ToString("F1", CultureInfo.InvariantCulture)}%");
            report.AppendLine();

            if (results.EfResults.Any())
            {
                var bestEf = results.EfResults.OrderByDescending(r => r.RecallSiyoyo).First();
                report.AppendLine($"Best Performance (ef={bestEf.Ef}):");
                report.AppendLine($"  Query Time (Standard): {bestEf.AverageQueryTimeStandard.ToString("F3", CultureInfo.InvariantCulture)} ms");
                report.AppendLine($"  Query Time (Siyoyo): {bestEf.AverageQueryTimeSiyoyo.ToString("F3", CultureInfo.InvariantCulture)} ms");
                report.AppendLine($"  Recall (Standard): {bestEf.RecallStandard.ToString("F3", CultureInfo.InvariantCulture)}");
                report.AppendLine($"  Recall (Siyoyo): {bestEf.RecallSiyoyo.ToString("F3", CultureInfo.InvariantCulture)}");
            }

            if (results.LayerStatistics != null)
            {
                report.AppendLine();
                report.AppendLine("Layer Statistics (Siyoyo):");
                foreach (var stat in results.LayerStatistics.OrderBy(s => s.Key))
                {
                    report.AppendLine($"  Layer {stat.Key}: {stat.Value.ToString("F1", CultureInfo.InvariantCulture)} avg neighbors");
                }
            }

            return report.ToString();
        }

        /// <summary>
        /// Runs a quick performance test with synthetic data.
        /// </summary>
        /// <param name="datasetSize">Number of vectors in dataset</param>
        /// <param name="queryCount">Number of test queries</param>
        /// <param name="dimension">Vector dimension</param>
        /// <returns>Quick test results</returns>
        public QuickTestResults RunQuickTest(int datasetSize = 1000, int queryCount = 100, int dimension = 128)
        {
            Console.WriteLine($"Running quick test with {datasetSize} vectors, {queryCount} queries, dimension {dimension}...");

            var dataset = _gpuHelper.GenerateSyntheticData(datasetSize, dimension, "uniform");
            var queries = _gpuHelper.GenerateSyntheticData(queryCount, dimension, "uniform");

            var results = RunBenchmark(dataset, queries);
            var summary = GenerateSummaryReport(results);

            Console.WriteLine(summary);

            return new QuickTestResults
            {
                DatasetSize = datasetSize,
                QueryCount = queryCount,
                Dimension = dimension,
                Summary = summary,
                Results = results
            };
        }
    }
}

/// <summary>
/// Results from benchmark runs.
/// </summary>
public class BenchmarkResults
{
    public double AverageInsertTime { get; set; }
    public double AverageInsertTimeSiyoyo { get; set; }
    public int MemoryUsageStandard { get; set; }
    public int MemoryUsageSiyoyo { get; set; }
    public List<EfResult> EfResults { get; set; } = new List<EfResult>();
    public Dictionary<int, double> LayerStatistics { get; set; }

    public void AddEfResult(int ef, double queryTimeStandard, double queryTimeSiyoyo, double recallStandard, double recallSiyoyo)
    {
        EfResults.Add(new EfResult
        {
            Ef = ef,
            AverageQueryTimeStandard = queryTimeStandard,
            AverageQueryTimeSiyoyo = queryTimeSiyoyo,
            RecallStandard = recallStandard,
            RecallSiyoyo = recallSiyoyo
        });
    }
}

/// <summary>
/// Results for a specific ef value.
/// </summary>
public class EfResult
{
    public int Ef { get; set; }
    public double AverageQueryTimeStandard { get; set; }
    public double AverageQueryTimeSiyoyo { get; set; }
    public double RecallStandard { get; set; }
    public double RecallSiyoyo { get; set; }
}

/// <summary>
/// Parameter sensitivity results.
/// </summary>
public class ParameterSensitivityResults
{
    public List<MResult> MResults { get; set; } = new List<MResult>();

    public void AddMResult(int m, double queryTimeStandard, double queryTimeSiyoyo, 
        double recallStandard, double recallSiyoyo, int memoryStandard, int memorySiyoyo)
    {
        MResults.Add(new MResult
        {
            M = m,
            QueryTimeStandard = queryTimeStandard,
            QueryTimeSiyoyo = queryTimeSiyoyo,
            RecallStandard = recallStandard,
            RecallSiyoyo = recallSiyoyo,
            MemoryStandard = memoryStandard,
            MemorySiyoyo = memorySiyoyo
        });
    }
}

/// <summary>
/// Results for a specific M value.
/// </summary>
public class MResult
{
    public int M { get; set; }
    public double QueryTimeStandard { get; set; }
    public double QueryTimeSiyoyo { get; set; }
    public double RecallStandard { get; set; }
    public double RecallSiyoyo { get; set; }
    public int MemoryStandard { get; set; }
    public int MemorySiyoyo { get; set; }
}

/// <summary>
/// Quick test results for rapid evaluation.
/// </summary>
public class QuickTestResults
{
    public int DatasetSize { get; set; }
    public int QueryCount { get; set; }
    public int Dimension { get; set; }
    public string Summary { get; set; }
    public BenchmarkResults Results { get; set; }
}

/// <summary>
/// Advanced benchmark results with detailed performance metrics.
/// </summary>
public class AdvancedBenchmarkResults
{
    public double AverageQueryLatency { get; set; }
    public double P95Latency { get; set; }
    public double P99Latency { get; set; }
    public double AverageRecall { get; set; }
    public double Throughput { get; set; } // queries per second
    public int MemoryUsage { get; set; }
    public Dictionary<string, double> DetailedMetrics { get; set; } = new Dictionary<string, double>();

    public void AddDetailedMetric(string name, double value)
    {
        DetailedMetrics[name] = value;
    }

    public string GetSummary()
    {
        var summary = new System.Text.StringBuilder();
        summary.AppendLine("=== Advanced Benchmark Results ===");
        summary.AppendLine($"Average Query Latency: {AverageQueryLatency:F3} ms");
        summary.AppendLine($"P95 Latency: {P95Latency:F3} ms");
        summary.AppendLine($"P99 Latency: {P99Latency:F3} ms");
        summary.AppendLine($"Average Recall: {AverageRecall:F3}");
        summary.AppendLine($"Throughput: {Throughput:F1} queries/sec");
        summary.AppendLine($"Memory Usage: {MemoryUsage / 1024.0:F1} KB");
        
        return summary.ToString();
    }
} 