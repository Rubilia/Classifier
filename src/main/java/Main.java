import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.InceptionResNetV1;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/*Порядок классов:
    0 - cat
    1 - dog
 */

public class Main {
    private static int picturesPerIteration = 900;
    private static final double correctValue = 0.9, wrongValue = 0.1;
    static List<File> catFiels, dogFiels;
    private static final Logger log = LoggerFactory.getLogger(Main.class);
    public static void main(String[] args) {
        System.setProperty("org.bytedeco.javacpp.maxphysicalbytes", "0");
        System.setProperty("org.bytedeco.javacpp.maxbytes", "0");
        CudaEnvironment.getInstance().getConfiguration().allowMultiGPU(true).setMaximumDeviceCache(2L * 1024L * 1024L * 1024L).allowCrossDeviceAccess(true);
        getFiles("C:\\Users\\Rubil\\Documents\\NeuralNetworkProjects\\Cat_Dog\\train\\");
        int nChannels = 3;
        int outputNum = 2;
        int batchSize = 3;
        int nEpochs = 100;
        int inputHeight = 160, inputWidth  = 160;
        int seed = (new Random()).nextInt();
        log.info("Building a model....");
        ZooModel inception = InceptionResNetV1.builder().numClasses(2).seed(seed).build();
        inception.setInputShape(new int[][]{{nChannels, inputHeight, inputWidth}, {batchSize}});
        ComputationGraph model = inception.init();
        model.setListeners(new PerformanceListener(1, true));
        log.info("Loading training data....");
        DataSetIterator Train;
        System.out.println("Training model....");
        long timeX = System.currentTimeMillis();
        for( int i=0; i<nEpochs; i++ ) {
            System.out.println("new Epoch " + i);
            long time1 = System.currentTimeMillis();
            ParallelWrapper wrapper = new ParallelWrapper.Builder(model)
                    .prefetchBuffer(batchSize)
                    .workers(4)
                    .averagingFrequency(2)
                    .reportScoreAfterAveraging(true)
                    .build();
            for (int j = 0; j < nEpochs; j++) {
                for (int k = 0; k < (catFiels.size()+dogFiels.size())/picturesPerIteration-1; k++) {
                    Train = setImagesTrain(new ImageDataSetIterator(batchSize, inputHeight, inputWidth, outputNum), 0, picturesPerIteration);
                    wrapper.fit(Train);
                    Train.reset();
                }
            }
            long time2 = System.currentTimeMillis();
            System.out.printf("*** Completed epoch {}, time: {} ***", i, (time2 - time1)/1000 + " sec");
        }
        long timeY = System.currentTimeMillis();
        System.out.printf("*** Training complete, time: {} ***", (timeY - timeX));
        log.info("Getting data for test");
        DataSetIterator Test = setImagesTrain(new ImageDataSetIterator(batchSize, inputHeight, inputWidth, outputNum), 10000, 10500);
        log.info("Evaluate model....");
        Evaluation eval = model.evaluate(Test);
        log.info(eval.stats());
        System.out.println("****************Example finished********************");
    }

    static void getFiles(String path){
        catFiels = new ArrayList<>();
        dogFiels = new ArrayList<>();
        File file = new File(path);
        for(File f: file.listFiles()){
            if (f.getName().contains("cat")) {catFiels.add(f);}
            else {dogFiels.add(f);}
        }
    }

    static ImageDataSetIterator setImagesTrain(ImageDataSetIterator input, int begin, int amont) {
        ArrayList<String> paths = new ArrayList<>();
        ArrayList<double[]> outs = new ArrayList<>();
        for (int i = begin; i < amont; i++) {
            paths.add(catFiels.get(i).getAbsolutePath());
            outs.add(new double[]{correctValue, wrongValue});
        }
        input.addDataString((List<String>) paths.clone(), (List<double[]>) outs.clone());
        paths = new ArrayList<>(); outs = new ArrayList<>();
        for (int i = begin; i < amont; i++) {
            paths.add(dogFiels.get(i).getAbsolutePath());
            outs.add(new double[]{wrongValue, correctValue});
        }
        input.addDataString((List<String>) paths.clone(), (List<double[]>) outs.clone());
        return input;
    }

}
