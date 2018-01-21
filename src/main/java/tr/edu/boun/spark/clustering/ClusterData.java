package tr.edu.boun.spark.clustering;

import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

class ClusterData {
    static Dataset<Row> getRowDataset(SparkSession spark) {
        List<Row> coordinateData = spark.read()
                .textFile("ml/sample-data.ml").javaRDD().map(line -> {
            String[] parts = line.split(" ");
            return RowFactory.create(Integer.parseInt(parts[2].trim()),
                    Vectors.dense(Double.parseDouble(parts[0].trim()),
                            Double.parseDouble(parts[1].trim())));
        }).collect();

        StructType schema = new StructType(new StructField[]{
                new StructField("label", DataTypes.IntegerType,
                        false, Metadata.empty()),
                new StructField("features", new VectorUDT(),
                        false, Metadata.empty())
        });

        return (Dataset<Row>) spark.createDataFrame(coordinateData, schema);
    }

    static void printClusteringResults(Dataset<Row> transformedSet) {
        Map<String, Integer> labelCount = new HashMap<>();
        transformedSet.collectAsList().forEach(row -> {
            String classValue = "(" + row.getInt(0) + ":"
                    + (row.getInt(2)+1) + ")";
            labelCount.compute(classValue, (k, v) -> (v==null) ? 1: v+1 );
        });

        labelCount.forEach((key, value) ->
                System.out.println("k:" + key + " v:" + value));
    }
}
