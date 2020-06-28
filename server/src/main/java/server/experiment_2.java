package server;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class experiment_2 {

    private static Set<Integer> selected_clients = new HashSet<>();
    private static int numInputs = 45;
    private static int numOutputs = 10;
    public static int batchSize = 10;
    private static int layer = 2;
    private static double alpha = 0.0;
    public static MultiLayerNetwork model = null;
    public static Map<Integer, Map<String, INDArray>> cache = new HashMap<>();
    public static String filenameTest = "res/dataset_1_2/test.csv";
    private static final String serverModel = "res/model/test_2.zip";

    /****************************  server  ****************************/
    public static void initModel() {
        int seed = 100;
        double learningRate = 0.01;
        int numHiddenNodes = 1000;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(learningRate, 0.9))
                .list()
                .layer(new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .build();

        model = new MultiLayerNetwork(conf);

        model.init();
        System.out.println("initModel finish\n");
    }

    public static void AverageWeights() throws IOException {

        //original model
        Map<String, INDArray> avgParamTable = model.paramTable();
        Map<String, INDArray> localParamTable = null;

        for (Map.Entry<String, INDArray> entry : avgParamTable.entrySet()) {
            String key = entry.getKey();
            INDArray value = entry.getValue();
            avgParamTable.replace(key, value.mul(alpha));
        }

        //average
        int K = cache.size();
        for (Map.Entry<Integer, Map<String, INDArray>> entry : cache.entrySet()) {
            localParamTable = entry.getValue();
            for (Map.Entry<String, INDArray> iter : localParamTable.entrySet()) {
                String key = iter.getKey();
                INDArray value = iter.getValue();
                value = avgParamTable.get(key).add(value.mul(1.0 - alpha).div(K));
                avgParamTable.replace(key, value);
            }
        }
        model.setParamTable(avgParamTable);

        ModelSerializer.writeModel(model, serverModel, true);

        //clear cache
        cache.clear();
        System.out.println("AverageWeights of " + K + " clients finish");

    }

    public static void evaluateModel() throws IOException, InterruptedException {
        //Load the test/evaluation data:
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File(filenameTest)));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, batchSize, 0, 10);

        //eval
        Evaluation eval = new Evaluation(numOutputs);
        while (testIter.hasNext()) {
            DataSet t = testIter.next();
            INDArray features = t.getFeatures();
            INDArray labels = t.getLabels();

            INDArray predicted = model.output(features);

            System.out.println("labels:");
            System.out.println(labels);
            System.out.println("predicted:");
            System.out.println(predicted);

            eval.eval(labels, predicted);
        }

        // Print the evaluation statistics
        System.out.println(eval.stats());

        //print out to file
        File file = new File("Evaluation_apr_16.txt");
        FileWriter fr = new FileWriter(file, true);
        fr.write(eval.stats());
        fr.close();
    }


    /****************************  clients  ****************************/
    private static class Client implements Runnable {
        private final int id;

        private static final int nEpochs = 1;

        private static MultiLayerNetwork localModel = null;

        private Client(int id) {
            this.id = id;
        }

        public static FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(5e-5))
                .seed(100)
                .build();


        public void run() {
            System.out.println("Hello from client: " + id);

            localModel = model;

            //load train data
            RecordReader rr = new CSVRecordReader();
            String filenameTrain = "res/dataset_1_2/client" + "_" + id + ".csv";
            try {
                rr.initialize(new FileSplit(new File(filenameTrain)));
            } catch (IOException | InterruptedException e) {
                e.printStackTrace();
            }
            DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, 0, 10);

            //train
            localModel.fit(trainIter, nEpochs);

            //upload weight and bias
//            Map<String, INDArray> paramTable = localModel.paramTable();
//            Map<String, INDArray> map = new HashMap<>();
//            map.put("weight", paramTable.get(String.format("%d_W", layer)));
//            map.put("bias", paramTable.get(String.format("%d_b", layer)));
            cache.put(id, localModel.paramTable());
        }
    }


    /****************************  select clients  ****************************/
    private static int getRandomNumberInRange(int min, int max) {
        if (min >= max) {
            throw new IllegalArgumentException("max must be greater than min");
        }
        Random r = new Random();
        return r.nextInt((max - min) + 1) + min;
    }

    public static void random_select(int K) {
        int lb = 1;
        int ub = 1000;
        while (selected_clients.size() < K) {
            selected_clients.add(getRandomNumberInRange(lb, ub));
        }
    }


    public static void main(String args[]) throws InterruptedException, IOException {

        int K = 1000;
        double C = 0.05;
        int round = 4000;

        initModel();

        for (int t = 0; t < round; t++) {

            long startTime = System.currentTimeMillis();

            System.out.println("\n\nround:" + t);

            int m = (int) Math.max(C * K, 1);
            selected_clients.clear();
            random_select(m);
            ExecutorService executor = Executors.newFixedThreadPool(m);

            Iterator iter = selected_clients.iterator();
            while (iter.hasNext()) {
                int id = (int) iter.next();
                Runnable client = new Client(id);
                executor.execute(client);
            }
            executor.shutdown();

            // Wait until all threads are finish
            while (!executor.isTerminated()) {
            }

            System.out.println("\nFinished all threads");

            AverageWeights();
            evaluateModel();
            long endTime = System.currentTimeMillis();
            double timeTaken = (endTime - startTime);
            timeTaken /= 1000;
            System.out.println(String.format("Round %,d finished in %,.4fs", t, timeTaken));
        }

    }

}

