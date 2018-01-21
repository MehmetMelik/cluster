package tr.edu.boun.spark.clustering;

import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class KMeansClustering2 extends ClusterData {

    public static void main(String[] args) {
        long lStartTime = System.nanoTime();
        SparkSession spark = SparkSession
                .builder()
                .appName("SWE578")
                .config("spark.config.option", "value")
                .getOrCreate();
        Dataset<Row> graphDF = getRowDataset(spark);
        KMeansModel kMeansModel = createModel(graphDF);
        double WCSS = kMeansModel.computeCost(graphDF);
        System.out.println("WCSS = " + WCSS);
        printClusterCenters(kMeansModel);
        Dataset<Row> transformedSet =  kMeansModel.transform(graphDF);
        printClusteringResults(transformedSet);
        spark.stop();
        long lEndTime = System.nanoTime();
        long elapsedTime = lEndTime - lStartTime;
        System.out.println("ElapsedTime: " + elapsedTime + " nano seconds");
    }

    private static KMeansModel createModel(Dataset<Row> graphDF) {
        KMeans kMeans = new KMeans().setK(3).setSeed(1)
                .setFeaturesCol("features");
        return kMeans.fit(graphDF);
    }

    private static void printClusterCenters(KMeansModel kMeansModel) {
        for (int i = 0; i < kMeansModel.clusterCenters().length; i++) {
            Vector clusterCenter = kMeansModel.clusterCenters()[i];
            double[] centerPoint = clusterCenter.toArray();
            System.out.println("Cluster Center " + i +
                    ": [ 'x': " + centerPoint[0] +
                    ", 'y': " + centerPoint[1] + " ]");
        }
    }
}
