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
    private static int picturesPerIteration = 500;
    private static final double correctValue = 0.9, wrongValue = 0.1;
    static List<File> catFiels, dogFiels;
    private static final Logger log = LoggerFactory.getLogger(Main.class);
    public static void main(String[] args) {
        System.setProperty("org.bytedeco.javacpp.maxphysicalbytes", "0");
        System.setProperty("org.bytedeco.javacpp.maxbytes", "0");
        for (int i = 0; i < 10; i++) {
            run();
            catFiels.clear();
            dogFiels.clear();
        }
        long s = 0;
        for (long i : times){
            s+=i;
        }
        System.out.println("avg time: " + s/times.size());
    }
    static ArrayList<Long> times =new ArrayList<>();
    public static void run() {
        long time = System.currentTimeMillis();
        CudaEnvironment.getInstance().getConfiguration().allowMultiGPU(true).setMaximumDeviceCache(2L * 1024L * 1024L * 1024L).allowCrossDeviceAccess(true);
        getFiles("C:\\Users\\Rubil\\Documents\\NeuralNetworkProjects\\Cat_Dog\\train\\");
        int nChannels = 3;
        int outputNum = 2;
        int batchSize = 3;
        int nEpochs = 1;
        int inputHeight = 160, inputWidth  = 160;
        int seed = (new Random()).nextInt();
        ZooModel inception = InceptionResNetV1.builder().numClasses(2).seed(seed).build();
        inception.setInputShape(new int[][]{{nChannels, inputHeight, inputWidth}, {batchSize}});
        ComputationGraph model = inception.init();
        DataSetIterator Train;
        for( int i=0; i<nEpochs; i++ ) {
            ParallelWrapper wrapper = new ParallelWrapper.Builder(model)
                    .prefetchBuffer(batchSize)
                    .workers(4)
                    .averagingFrequency(2)
                    .reportScoreAfterAveraging(false)
                    .build();
            for (int j = 0; j < nEpochs; j++) {
                for (int k = 0; k < 1; k++) {
                    Train = setImagesTrain(new ImageDataSetIterator(batchSize, inputHeight, inputWidth, outputNum), 0, picturesPerIteration);
                    wrapper.fit(Train);
                    Train.reset();
                }
            }
        }
        DataSetIterator Test = setImagesTrain(new ImageDataSetIterator(batchSize, inputHeight, inputWidth, outputNum), 10000, 11000);
        Evaluation eval = model.evaluate(Test);
        long endTime = System.currentTimeMillis()-time;
        System.out.println("total time: " + endTime);
        times.add(endTime);
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
