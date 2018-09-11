import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.InceptionResNetV1;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/*Порядок классов:
    0 - car
    1 - flower
    2 - airplnae

 */

public class Main {
    private static int picturesPerIteration = 180;
    private static int iterationsAmount = 3;
    private static final double correctValue = 0.8, wrongValue = 0.1;
    private static int car_correct = 500;
    private static int flower_correct = 840;
    private static int airplane_correct = 530;
    static List<File> carFiles, flowerFiles, airplaneFiles;
    private static final Logger log = LoggerFactory.getLogger(Main.class);
    public static void main(String[] args) {
        System.setProperty("org.bytedeco.javacpp.maxphysicalbytes", "0");
        System.setProperty("org.bytedeco.javacpp.maxbytes", "0");
        CudaEnvironment.getInstance().getConfiguration().allowMultiGPU(true).setMaximumDeviceCache(2L * 1024L * 1024L * 1024L).allowCrossDeviceAccess(true);
        carFiles = getFiles("C:\\Users\\Rubil\\Documents\\NeuralNetworkProjects\\images\\car\\");
        flowerFiles = getFiles("C:\\Users\\Rubil\\Documents\\NeuralNetworkProjects\\images\\flower\\");
        airplaneFiles = getFiles("C:\\Users\\Rubil\\Documents\\NeuralNetworkProjects\\images\\airplane\\");
        int nChannels = 1;
        int outputNum = 3;
        int batchSize = 128;
        int nEpochs = 10;
        int inputHeight = 160, inputWidth  = 160;
        int seed = (new Random()).nextInt();
        InceptionResNetV1 zooModel = InceptionResNetV1.builder()
                .numClasses(outputNum)
                .seed(seed)
                .inputShape(new int[]{nChannels, inputHeight, inputWidth})
                .build();
        ComputationGraph model = zooModel.init();
        model.init();
        System.out.println("Training model....");
        model.setListeners(new ScoreIterationListener(100));
        long timeX = System.currentTimeMillis();
        //MultipleEpochsIterator mnistMultiEpochIterator = new MultipleEpochsIterator(nEpochs, mnistTrain);
        for( int i=0; i<nEpochs; i++ ) {
            System.out.println("new Epoch " + i);
            long time1 = System.currentTimeMillis();
            ParallelWrapper wrapper = new ParallelWrapper.Builder(model)
                    .prefetchBuffer(128)
                    .workers(3)
                    .trainingMode(ParallelWrapper.TrainingMode.SHARED_GRADIENTS)
                    .averagingFrequency(2)
                    .reportScoreAfterAveraging(true)
                    .build();
            for (int j = 0; j < iterationsAmount; j++) {
                DataSetIterator Train =setImagesTrain(new ImageDataSetIterator(batchSize, inputHeight, inputWidth, 3), j);
                wrapper.fit(Train);
            }
            long time2 = System.currentTimeMillis();
            System.out.printf("*** Completed epoch {}, time: {} ***", i, (time2 - time1)/1000 + " sec");
        }
        long timeY = System.currentTimeMillis();

        System.out.printf("*** Training complete, time: {} ***", (timeY - timeX));
        DataSetIterator Test = new ImageDataSetIterator(batchSize, inputHeight, inputWidth, 3);
        Test = setImagesTest((ImageDataSetIterator) Test);
        log.info("Evaluate model....");
        Evaluation eval = model.evaluate(Test);
        log.info(eval.stats());
        Test.reset();

        System.out.println("****************Example finished********************");
    }

    static ArrayList<File> getFiles(String path){
        ArrayList<File> ret = new ArrayList<>();
        File file = new File(path);
        for(File f: file.listFiles()){
            if (f.isDirectory()){continue;}
            ret.add(f);
        }
        return ret;
    }

    static ImageDataSetIterator setImagesTrain(ImageDataSetIterator input, int iteration) {
        ArrayList<String> paths = new ArrayList<>();
        ArrayList<double[]> outs = new ArrayList<>();
        for (int i = picturesPerIteration*iteration; i < picturesPerIteration*(iteration+1); i++) {
            paths.add(carFiles.get(i).getAbsolutePath());
            outs.add(new double[]{correctValue, wrongValue, wrongValue});
        }
        input.addDataString((List<String>) paths.clone(), (List<double[]>) outs.clone());
        paths = new ArrayList<>(); outs = new ArrayList<>();
        for (int i = picturesPerIteration*iteration; i < picturesPerIteration*(iteration+1); i++) {
            paths.add(flowerFiles.get(i).getAbsolutePath());
            outs.add(new double[]{wrongValue, correctValue, wrongValue});
        }
        input.addDataString((List<String>) paths.clone(), (List<double[]>) outs.clone());
        paths = new ArrayList<>(); outs = new ArrayList<>();
        for (int i = picturesPerIteration*iteration; i < picturesPerIteration*(iteration+1); i++) {
            paths.add(airplaneFiles.get(i).getAbsolutePath());
            outs.add(new double[]{wrongValue, wrongValue, correctValue});
        }
        input.addDataString((List<String>) paths.clone(), (List<double[]>) outs.clone());
        return input;
    }


    static ImageDataSetIterator setImagesTest(ImageDataSetIterator input) {
        ArrayList<String> paths = new ArrayList<>();
        ArrayList<double[]> outs = new ArrayList<>();
        for (int i = car_correct; i < carFiles.size(); i++) {
            paths.add(carFiles.get(i).getAbsolutePath());
            outs.add(new double[]{correctValue, wrongValue, wrongValue});
        }
        input.addDataString((List<String>) paths.clone(), (List<double[]>) outs.clone());
        paths = new ArrayList<>(); outs = new ArrayList<>();
        for (int i = flower_correct; i < flowerFiles.size(); i++) {
            paths.add(flowerFiles.get(i).getAbsolutePath());
            outs.add(new double[]{wrongValue, correctValue, wrongValue});
        }
        input.addDataString((List<String>) paths.clone(), (List<double[]>) outs.clone());
        paths = new ArrayList<>(); outs = new ArrayList<>();
        for (int i = airplane_correct; i < airplaneFiles.size(); i++) {
            paths.add(airplaneFiles.get(i).getAbsolutePath());
            outs.add(new double[]{wrongValue, wrongValue, correctValue});
        }
        input.addDataString((List<String>) paths.clone(), (List<double[]>) outs.clone());
        return input;
    }
}
