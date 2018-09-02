import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.parallelism.ParallelWrapper;
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
    private static int picturesPerIteration = 60;
    private static int iterationsAmount = 9;
    private static int car_correct = 500;
    private static int flower_correct = 840;
    private static int airplane_correct = 530;
    static List<File> carFiles, flowerFiles, airplaneFiles;
    private static final Logger log = LoggerFactory.getLogger(Main.class);
    public static void main(String[] args) throws Exception{
        System.setProperty("org.bytedeco.javacpp.maxphysicalbytes", "0");
        System.setProperty("org.bytedeco.javacpp.maxbytes", "0");
        CudaEnvironment.getInstance().getConfiguration().allowMultiGPU(true).setMaximumDeviceCache(2L * 1024L * 1024L * 1024L).allowCrossDeviceAccess(true);
        carFiles = getFiles("C:\\Users\\Rubil\\Documents\\NeuralNetworkProjects\\images\\car");
        flowerFiles = getFiles("C:\\Users\\Rubil\\Documents\\NeuralNetworkProjects\\images\\flower");
        airplaneFiles = getFiles("C:\\Users\\Rubil\\Documents\\NeuralNetworkProjects\\images\\airplane\\");
        int nChannels = 1;
        int outputNum = 3;
        int batchSize = 128;
        int nEpochs = 5;
        int inputHeight = 100, inputWidth  = 100;
        int seed = (new Random()).nextInt();
        System.out.println("Building model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(0.0005)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs.Builder().learningRate(.01).build())
                .biasUpdater(new Nesterovs.Builder().learningRate(0.02).build())
                .list()
                .layer(0, new ConvolutionLayer.Builder(11, 11)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(128)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(8, 8)
                        //Note that nIn need not be specified in later layers
                        .stride(1, 1)
                        .nOut(256)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4, new ConvolutionLayer.Builder(4, 4)
                        .stride(1, 1)
                        .nOut(128)
                        .activation(Activation.IDENTITY).build())
                .layer(5, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(300).build())
                .layer(6, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(100,100,1)) //See note below
                .backprop(true).pretrain(false).build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        System.out.println("Loading data....");
        List<DataSetIterator> data = new ArrayList<>();
        for (int i = 0; i < iterationsAmount; i++) {
            DataSetIterator Train = new ImageDataSetIterator(batchSize, inputHeight, inputWidth, 3);
            data.add(setImagesTrain((ImageDataSetIterator) Train, i));
        }
        DataSetIterator Test = new ImageDataSetIterator(batchSize, inputHeight, inputWidth, 3);
        Test = setImagesTest((ImageDataSetIterator) Test);

        // ParallelWrapper will take care of load balancing between GPUs.
        ParallelWrapper wrapper = new ParallelWrapper.Builder(model)
            // DataSets prefetching options. Set this value with respect to number of actual devices
            .prefetchBuffer(128)

            // set number of workers equal to number of available devices. x1-x2 are good values to start with
            .workers(3)

            // rare averaging improves performance, but might reduce model accuracy
            .averagingFrequency(2)

            // if set to TRUE, on every averaging model score will be reported
            .reportScoreAfterAveraging(true)

            .build();
        System.out.println("Training model....");
        model.setListeners(new ScoreIterationListener(100));
        long timeX = System.currentTimeMillis();

        // optionally you might want to use MultipleEpochsIterator instead of manually iterating/resetting over your iterator
        //MultipleEpochsIterator mnistMultiEpochIterator = new MultipleEpochsIterator(nEpochs, mnistTrain);

        for( int i=0; i<nEpochs; i++ ) {
            System.out.println("new Epoch " + i);
            long time1 = System.currentTimeMillis();
            // Please note: we're feeding ParallelWrapper with iterator, not model directly
//            wrapper.fit(mnistMultiEpochIterator);
            for (int j = 0; j < iterationsAmount; j++) {
                wrapper.fit(data.get(j));
            }
            long time2 = System.currentTimeMillis();
            System.out.printf("*** Completed epoch {}, time: {} ***", i, (time2 - time1)/1000 + " sec");
        }
        long timeY = System.currentTimeMillis();

        System.out.printf("*** Training complete, time: {} ***", (timeY - timeX));

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum);
        while(Test.hasNext()){
            DataSet ds = Test.next();
            INDArray output = model.output(ds.getFeatureMatrix(), false);
            eval.eval(ds.getLabels(), output);
        }
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
            outs.add(new double[]{1.0, 0.0, 0.0});
        }
        input.addDataString((List<String>) paths.clone(), (List<double[]>) outs.clone());
        paths = new ArrayList<>(); outs = new ArrayList<>();
        for (int i = picturesPerIteration*iteration; i < picturesPerIteration*(iteration+1); i++) {
            paths.add(flowerFiles.get(i).getAbsolutePath());
            outs.add(new double[]{0.0, 1.0, 0.0});
        }
        input.addDataString((List<String>) paths.clone(), (List<double[]>) outs.clone());
        paths = new ArrayList<>(); outs = new ArrayList<>();
        for (int i = picturesPerIteration*iteration; i < picturesPerIteration*(iteration+1); i++) {
            paths.add(airplaneFiles.get(i).getAbsolutePath());
            outs.add(new double[]{0.0, 0.0, 1.0});
        }
        input.addDataString((List<String>) paths.clone(), (List<double[]>) outs.clone());
        return input;
    }


    static ImageDataSetIterator setImagesTest(ImageDataSetIterator input) {
        ArrayList<String> paths = new ArrayList<>();
        ArrayList<double[]> outs = new ArrayList<>();
        for (int i = car_correct; i < carFiles.size(); i++) {
            paths.add(carFiles.get(i).getAbsolutePath());
            outs.add(new double[]{1.0, 0.0, 0.0});
        }
        input.addDataString((List<String>) paths.clone(), (List<double[]>) outs.clone());
        paths = new ArrayList<>(); outs = new ArrayList<>();
        for (int i = flower_correct; i < flowerFiles.size(); i++) {
            paths.add(flowerFiles.get(i).getAbsolutePath());
            outs.add(new double[]{0.0, 1.0, 0.0});
        }
        input.addDataString((List<String>) paths.clone(), (List<double[]>) outs.clone());
        paths = new ArrayList<>(); outs = new ArrayList<>();
        for (int i = airplane_correct; i < airplaneFiles.size(); i++) {
            paths.add(airplaneFiles.get(i).getAbsolutePath());
            outs.add(new double[]{0.0, 0.0, 1.0});
        }
        input.addDataString((List<String>) paths.clone(), (List<double[]>) outs.clone());
        return input;
    }
}
