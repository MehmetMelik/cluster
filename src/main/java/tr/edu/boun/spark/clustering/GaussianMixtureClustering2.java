package tr.edu.boun.spark.clustering;

import org.apache.spark.ml.clustering.GaussianMixture;
import org.apache.spark.ml.clustering.GaussianMixtureModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.stat.distribution.MultivariateGaussian;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class GaussianMixtureClustering2 extends ClusterData {

    public static void main(String[] args) {
        long lStartTime = System.nanoTime();
        SparkSession spark = SparkSession
                .builder()
                .appName("SWE578")
                .config("spark.config.option", "value")
                .getOrCreate();
        Dataset<Row> graphDF = getRowDataset(spark);
        GaussianMixtureModel gaussianMixtureModel = createModel(graphDF);
        printClusterCenters(gaussianMixtureModel);
        Dataset<Row> transformedSet =  gaussianMixtureModel.transform(graphDF);
        printClusteringResults(transformedSet);
        spark.stop();
        long lEndTime = System.nanoTime();
        long elapsedTime = lEndTime - lStartTime;
        System.out.println("ElapsedTime: " + elapsedTime + " nano seconds");
    }

    private static GaussianMixtureModel createModel(Dataset<Row> graphDF) {
        GaussianMixture gmm = new GaussianMixture()
                .setK(3).setFeaturesCol("features");
        return gmm.fit(graphDF);
    }

    private static void printClusterCenters(GaussianMixtureModel model) {
        int i = 0;
        MultivariateGaussian[] gaussians = model.gaussians();
        for (MultivariateGaussian mg : gaussians) {
            Vector clusterCenter = mg.mean();
            double[] centerPoint = clusterCenter.toArray();
            System.out.println("Cluster Center " + i +
                    ": [ 'x': " + centerPoint[0] +
                    ", 'y': " + centerPoint[1] + " ]");
            i++;
        }
    }
}
