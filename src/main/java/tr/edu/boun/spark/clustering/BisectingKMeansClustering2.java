package tr.edu.boun.spark.clustering;

import org.apache.spark.ml.clustering.BisectingKMeans;
import org.apache.spark.ml.clustering.BisectingKMeansModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class BisectingKMeansClustering2 extends ClusterData {

    public static void main(String[] args) {
        long lStartTime = System.nanoTime();
        SparkSession spark = SparkSession
                .builder()
                .appName("SWE578")
                .config("spark.config.option", "value")
                .getOrCreate();
        Dataset<Row> graphDF = getRowDataset(spark);
        BisectingKMeansModel bisectingKMeansModel = createModel(graphDF);
        double WCSS = bisectingKMeansModel.computeCost(graphDF);
        System.out.println("WCSS = " + WCSS);
        printClusterCenters(bisectingKMeansModel);
        Dataset<Row> transformedSet =  bisectingKMeansModel.transform(graphDF);
        printClusteringResults(transformedSet);
        spark.stop();
        long lEndTime = System.nanoTime();
        long elapsedTime = lEndTime - lStartTime;
        System.out.println("ElapsedTime: " + elapsedTime + " nano seconds");
    }

    private static BisectingKMeansModel createModel(Dataset<Row> graphDF) {
        BisectingKMeans bisectingKMeans = new BisectingKMeans().setK(3).setSeed(1)
                .setFeaturesCol("features");
        return bisectingKMeans.fit(graphDF);
    }

    private static void printClusterCenters(BisectingKMeansModel model) {
        for (int i = 0; i < model.clusterCenters().length; i++) {
            Vector clusterCenter = model.clusterCenters()[i];
            double[] centerPoint = clusterCenter.toArray();
            System.out.println("Cluster Center " + i +
                    ": [ 'x': " + centerPoint[0] +
                    ", 'y': " + centerPoint[1] + " ]");
        }
    }
}