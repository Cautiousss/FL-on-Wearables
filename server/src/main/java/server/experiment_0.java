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
import org.deeplearning4j.nn.transferlearning.TransferLearning;
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

public class experiment_0 {

    private static Set<Integer> selected_clients = new HashSet<>();

    public static int batchSize;
    public static int nEpochs;
    public static int experiment;

    private static int numInputs = 45;
    private static int numOutputs = 10;

    private static int layer = 2;
    private static double alpha = 0.0;
    public static MultiLayerNetwork model = null;
    public static Map<Integer, Map<String, INDArray>> cache = new HashMap<>();
    public static String filenameTest = "res/newest_dataset/test.csv";
    public static final String serverModel = "res/model/model_"+experiment+".zip";

    /****************************  server  ****************************/
    public static void initModel() {
        int seed = 100;
        double learningRate = 0.05;
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
        Map<String, INDArray> paramTable = model.paramTable();
        INDArray weight = paramTable.get(String.format("%d_W", layer));
        INDArray bias = paramTable.get(String.format("%d_b", layer));
        INDArray avgWeights = weight.mul(alpha);
        INDArray avgBias = bias.mul(alpha);

        //average
        int K = cache.size();
        for (Map.Entry<Integer, Map<String, INDArray>> entry : cache.entrySet()) {
            paramTable = entry.getValue();
            weight = paramTable.get("weight");
            bias = paramTable.get("bias");
            avgWeights = avgWeights.add(weight.mul(1.0 - alpha).div(K));
            avgBias = avgBias.add(bias.mul(1.0 - alpha).div(K));
        }

        model.setParam(String.format("%d_W", layer), avgWeights);
        model.setParam(String.format("%d_b", layer), avgBias);
        ModelSerializer.writeModel(model, serverModel, true);

        //clear cache
        cache.clear();
        System.out.println("AverageWeights of " + K + " clients finish");

    }

    public static void evaluateModel(int round) throws IOException, InterruptedException {
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
            INDArray predicted = model.output(features, false);

//            System.out.println("labels:");
//            System.out.println(labels);
//            System.out.println("predicted:");
//            System.out.println(predicted);

            eval.eval(labels, predicted);
        }

        // Print the evaluation statistics
        System.out.println(eval.stats());

        //print out to file

        File file = new File("result_"+experiment+".txt");
        FileWriter fr = new FileWriter(file, true);
        fr.write("\nround:" + round);
        fr.write(eval.stats());
        fr.close();
    }


    /****************************  clients  ****************************/
    private static class Client {

        private final int id;


        private Client(int id) {
            this.id = id;
        }

        public static FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(5e-5))
                .seed(100)
                .build();

        public void localUpdate() {
            System.out.println("Hello from client: " + id);

            //transfer model
            MultiLayerNetwork transferred_model = new TransferLearning.Builder(model)
                    .fineTuneConfiguration(fineTuneConf)
                    .setFeatureExtractor(1)
                    .build();

            //load train data
            RecordReader rr = new CSVRecordReader();
            String filenameTrain = "res/newest_dataset/client" + "_" + id + ".csv";
            try {
                rr.initialize(new FileSplit(new File(filenameTrain)));
            } catch (IOException | InterruptedException e) {
                e.printStackTrace();
            }
            DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, 0, 10);

            //train
            transferred_model.fit(trainIter, nEpochs);

            //upload weight and bias
            Map<String, INDArray> paramTable = transferred_model.paramTable();
            Map<String, INDArray> map = new HashMap<>();
            map.put("weight", paramTable.get(String.format("%d_W", layer)));
            map.put("bias", paramTable.get(String.format("%d_b", layer)));
            cache.put(id, map);
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
        double C = 0.1;
        int round = 1000;
        experiment=Integer.parseInt(args[0]);
        batchSize = Integer.parseInt(args[1]);
        nEpochs = Integer.parseInt(args[2]);

        System.out.println("serverModel:"+serverModel);
        System.out.println("batchSize:" + batchSize);
        System.out.println("nEpochs:" + nEpochs);

        File file = new File(serverModel);
        if(file.exists()) {
            model = ModelSerializer.restoreMultiLayerNetwork(serverModel, false);
        }
        else{
            initModel();
        }

        for (int t = 0; t < round; t++) {
            System.out.println("\nround:" + t);

            int m = (int) Math.max(C * K, 1);
            selected_clients.clear();
            random_select(m);

            for (int i : selected_clients) {
                Client client = new Client(i);
                client.localUpdate();
            }

            AverageWeights();
            evaluateModel(t);

        }
    }

}
